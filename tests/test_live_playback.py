"""Regression tests for live timing. No model download or MIDI hardware needed."""
import queue
import threading
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mido
import midi_io
from amt import make_event, parse_events, _cached_logits
from anticipation.vocab import MAX_DUR, DUR_OFFSET
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from live_duet import LiveDuet


class Source:
    def __init__(self, events=()):
        self.events = list(events)

    def poll(self, playhead):
        result, self.events = self.events, []
        return result

    def exhausted(self):
        return False


class TimingTests(unittest.TestCase):
    def duet(self, **kwargs):
        defaults = dict(model=None, melody_source=Source(), melody_len_s=None,
                        bpm=60, lookahead_beats=3, commit_beats=1,
                        listen_first_beats=8, top_p=.95, accomp_instrs=(40,), t0=1)
        defaults.update(kwargs)
        duet = LiveDuet(**defaults)
        self.addCleanup(duet.pool.shutdown)
        duet.log = Mock()
        return duet

    def test_fast_generation_waits_for_performer(self):
        duet = self.duet()
        duet.committed_horizon = 60
        with patch.object(duet.pool, 'submit') as submit:
            duet._maybe_kick_generation(10)
            submit.assert_not_called()
        self.assertIsNone(duet.pending)

    def test_recovers_at_current_playhead_after_underrun(self):
        duet = self.duet()
        with patch.object(duet.pool, 'submit', return_value=Mock()) as submit:
            duet._maybe_kick_generation(12)
        self.assertEqual(duet.pending[1:3], (12, 15))
        submit.assert_called_once()

    def test_new_melody_reaches_next_generation(self):
        source = Source([(9, .5, 67)])
        duet = self.duet(melody_source=source)
        duet.committed_horizon = 11
        duet._reveal_melody(10)
        with patch.object(duet.pool, 'submit', return_value=Mock()) as submit:
            duet._maybe_kick_generation(10)
        snapshot = submit.call_args.args[5]
        self.assertEqual(list(parse_events(snapshot))[0][3], 67)

    def test_held_note_revises_one_history_event(self):
        source = Source([dict(id=0, onset=1, duration=.1, pitch=60, complete=False)])
        duet = self.duet(melody_source=source)
        duet._reveal_melody(1)
        source.events = [dict(id=0, onset=1, duration=3, pitch=60, complete=True)]
        duet._reveal_melody(4)
        self.assertEqual(len(duet.history), 3)
        self.assertAlmostEqual(list(parse_events(duet.history))[0][1], 3)
        self.assertEqual(len(duet.played), 1)
        self.assertEqual(duet._melody_refs, {})

    def test_monophonic_history_and_log_both_trim(self):
        duet = self.duet()
        duet._commit_accompaniment([(1, 3, 40, 60)])
        duet._commit_accompaniment([(2, 1, 40, 62)])
        self.assertAlmostEqual(list(parse_events(duet.history))[0][1], 1)
        self.assertEqual(duet.played[0][1], 1)

    def test_collection_drops_expired_and_counts_only_committed_time(self):
        duet = self.duet()
        # At t=10.5, the first event has expired; the second is still sounding.
        result = make_event(10, .1, 40, 60) + make_event(10.2, 1, 40, 62)
        future = Future()
        future.set_result((result, 1))
        duet.pending = (future, 10, 13, 11, 0)
        with patch('live_duet.time.monotonic', return_value=11.5):
            duet._collect_generation()
        self.assertEqual(duet.dropped_expired, 1)
        self.assertEqual(len(duet.played), 1)
        self.assertAlmostEqual(duet.played[0][0], 10.5)
        self.assertAlmostEqual(duet.played[0][1], .7)
        self.assertEqual(duet.gen_stats, [(1, .5)])

    def test_commit_includes_start_and_excludes_end(self):
        duet = self.duet()
        result = make_event(10, .2, 40, 60) + make_event(11, .2, 40, 62)
        future = Future()
        future.set_result((result, 1))
        duet.pending = (future, 10, 13, 9, 0)
        with patch('live_duet.time.monotonic', return_value=10):
            duet._collect_generation()
        self.assertEqual([n[3] for n in duet.played], [60])


class MidiTests(unittest.TestCase):
    def source(self):
        with patch('midi_io.mido.open_input', return_value=Mock()):
            return midi_io.MidiKeyboardInput('test', 0)

    def player(self, monophonic=True):
        player = midi_io.MidiPlayer.__new__(midi_io.MidiPlayer)
        player.channel_by_instr = {40: 1, 41: 2}
        player.monophonic = monophonic
        player._lock = threading.Lock()
        player._pending = []
        player._active_counts = {}
        player._port = Mock()
        player.dropped_expired = 0
        return player

    def messages(self, player):
        return [(c.args[0].type, c.args[0].note, c.args[0].channel)
                for c in player._port.send.call_args_list]

    def test_note_visible_before_release_and_velocity_preserved(self):
        source = self.source()
        with patch('midi_io.time.monotonic', return_value=1):
            source._on_message(mido.Message('note_on', note=60, velocity=101))
        event = source.poll(1)[0]
        self.assertEqual((event['pitch'], event['velocity'], event['complete']), (60, 101, False))
        held = source.poll(3)[0]
        self.assertEqual(held['id'], event['id'])
        self.assertGreater(held['duration'], 2)
        with patch('midi_io.time.monotonic', return_value=4):
            source._on_message(mido.Message('note_off', note=60))
        final = source.poll(4)[0]
        self.assertEqual((final['id'], final['duration'], final['complete']), (event['id'], 3, True))
        self.assertEqual(source.poll(5), [])

    def test_same_pitch_different_channels_and_retrigger_have_distinct_ids(self):
        source = self.source()
        with patch('midi_io.time.monotonic', return_value=1):
            source._on_message(mido.Message('note_on', note=60, channel=0))
            source._on_message(mido.Message('note_on', note=60, channel=1))
        events = source.poll(1)
        self.assertEqual(len({e['id'] for e in events}), 2)
        with patch('midi_io.time.monotonic', return_value=2):
            source._on_message(mido.Message('note_on', note=60, channel=0))
        events = source.poll(2)
        self.assertEqual(len(events), 3)
        self.assertEqual(sum(e['complete'] for e in events), 1)

    def test_mono_output_stops_previous_voice_across_batches(self):
        player = self.player()
        player.schedule(1, 3, 40, 60)
        player._tick(1)
        player.schedule(2, 1, 40, 62)
        player._tick(2)
        self.assertEqual(self.messages(player), [('note_on',60,1),('note_off',60,1),('note_on',62,1)])
        player._tick(4)
        self.assertEqual(self.messages(player)[-1], ('note_off',62,1))
        self.assertEqual(player._pending, [])

    def test_expired_note_never_sounds(self):
        player = self.player()
        player.schedule(1, .1, 40, 60)
        player._tick(2)
        self.assertEqual(self.messages(player), [])
        self.assertEqual(player.dropped_expired, 1)

    def test_different_instruments_remain_independent(self):
        player = self.player()
        player.schedule(1, 3, 40, 60)
        player.schedule(1, 3, 41, 65)
        player._tick(1)
        self.assertEqual(len(player._pending), 2)
        self.assertEqual(len(self.messages(player)), 2)

    def test_polyphonic_same_pitch_old_end_does_not_cut_new_note(self):
        player = self.player(monophonic=False)
        player.schedule(1, 2, 40, 60)
        player._tick(1)
        player.schedule(2, 2, 40, 60)
        player._tick(2)
        player._tick(3)
        self.assertEqual([m[0] for m in self.messages(player)], ['note_on','note_on'])
        player._tick(4)
        self.assertEqual(self.messages(player)[-1], ('note_off',60,1))


class ModelTests(unittest.TestCase):
    def test_cached_logits_match_full_context_after_append_and_rebase(self):
        torch.manual_seed(4)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=64,
                                          n_embd=16, n_layer=1, n_head=2)).eval()
        cache = {}
        with torch.inference_mode():
            for prefix in ([1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 3, 4], [1, 3, 4]):
                expected = model(torch.tensor([prefix]), use_cache=False).logits[0, -1]
                actual = _cached_logits(model, prefix, cache)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_long_held_note_stays_in_duration_vocabulary(self):
        token = make_event(0, 100, 0, 60)[1]
        self.assertLess(token, DUR_OFFSET + MAX_DUR)


if __name__ == '__main__':
    unittest.main()
