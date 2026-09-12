# Duet.AI — a live AI duet partner

Prepared on an M-series Mac (MPS), 2026-09-11/12. Tests the reframe in
`proposal.md`: inference latency (100 ms-1 s+) isn't a latency problem, it's
a *scheduling* problem, solved by writing accompaniment a few beats ahead of
the live playhead and hiding think-time behind that buffer (the ReaLJam
protocol: lookahead, commit, listen-first).

Model: `stanford-crfm/music-small-800k` (128M params, Apache 2.0) via
Stanford CRFM's [`anticipation`](https://github.com/jthickstun/anticipation)
package. Not on PyPI — install from GitHub (see below).

## What it does

- `melody.py` — a synthetic scale-constrained melody (built with `musicpy`),
  standing in for a live player.
- `amt.py` — the model interface: event↔token encoding, a logit mask that
  restricts generation to melody (piano) plus a configurable set of
  companion instruments (`INSTRUMENT_PRESETS` — see `--voices` below)
  instead of letting a Lakh-trained checkpoint free-associate across all
  128, and a `temperature` knob (scales raw logits before `top_p`
  truncation) alongside the existing `accomp_bias`. **Default voices are a
  string ensemble** (violin, viola, cello) — empirically verified against a
  solo piano prompt, individually and together (checking the three don't
  starve each other out); see "Findings" below.
- `live_duet.py` — the scheduler. Runs a real wall-clock transport; the
  melody is only ever revealed up to the current playhead (no peeking at its
  own future), and a background thread continuously asks the model to write
  `--lookahead-beats` of accompaniment past the last committed point, of
  which only `--commit-beats` is frozen. The rest is discarded and
  regenerated once more real melody has arrived. If the model doesn't
  finish before its own deadline, that's logged as a genuine underrun, not
  hidden. If a window comes back with zero companion notes at all (a
  legitimate sample, just an unwanted one), it's retried up to
  `MAX_GENERATION_ATTEMPTS` times before being accepted as silence. By
  default each companion instrument is kept independently monophonic (no
  self-overlap, "one violin") by trimming a note's tail if that same
  instrument's next note starts before it ends — the model has no such
  constraint on its own; `--multi-voice` disables this per instrument
  ("multiple violins", a section instead of a soloist). Different
  instruments in an ensemble may always sound together regardless. Output
  is a MIDI file of exactly what was decided live. `LiveDuet` itself doesn't
  know or care where the melody comes from -- see `midi_io.py`/`live_midi.py`
  below.
- `midi_io.py` / `live_midi.py` — the same `LiveDuet` scheduler wired to a
  real MIDI keyboard instead of a synthetic melody: `midi_io.py` is pure
  hardware plumbing (open a port, turn note-on/note-off into
  `(onset_s, dur_s, pitch)`, schedule companion notes to a real output at
  the right wall-clock time); `live_midi.py` is the CLI entry point. See
  **SETUP.md** for how to actually wire up a keyboard and hear the result.

## Run

```bash
pip install torch transformers musicpy
pip install git+https://github.com/jthickstun/anticipation.git
python live_duet.py --notes 32 --bpm 80 --lookahead-beats 2.5 --commit-beats 1.75
python live_duet.py --notes 32 --bpm 80 --voices sax --temperature 1.4 --role lead --multi-voice
# writes output/live_duet.mid in this directory by default; override with --outdir
```

Flags: `--bpm`, `--key`/`--mode`, `--notes` (melody length), `--seed`,
`--lookahead-beats`, `--commit-beats`, `--top-p`, `--accomp-bias` (logit
bias toward the kept instrument(s) — free, since the alternative is always
discarded anyway). Four independent, composable knobs:

- `--voices {strings,violin,guitar,sax,brass,keys,orchestral,ambient}` —
  *which* instrument(s) play (default `strings`: violin/viola/cello). All
  eight are `amt.INSTRUMENT_PRESETS`, each empirically verified to produce
  real output against a solo piano prompt (see "Findings").
- `--multi-voice` — *how many notes at once* a given instrument may play:
  one at a time (default) or overlapping, a section rather than a soloist.
- `--temperature` (default 1.0) — how wild: scales the raw logits before
  `top_p` truncation. Below 1.0 sharpens toward the model's most confident
  guesses; above 1.0 flattens the distribution into wilder, less coherent
  pitch/rhythm choices. Distinct from `--top-p` (how much of the
  probability tail gets truncated) and `--accomp-bias` (a fixed preference
  for *which* instrument, not how sharply any of them is sampled).
- `--role {follow,lead}` (default `follow`) — `follow` waits through
  `--listen-first-beats` (default 8) before playing, always reacting to
  melody already heard; `lead` starts immediately
  (`--listen-first-beats` defaults to 0). Both still only ever generate
  from melody already revealed, so this doesn't reverse who the human/AI
  parts are — it only changes whether the companion waits for a cue to
  start. An explicit `--listen-first-beats` always overrides the role's
  default.

All flags compose freely.

For an actual physical MIDI keyboard instead of the synthetic melody, see
**SETUP.md** and run `live_midi.py` instead (same flags, plus `--midi-in`/
`--midi-out`).

## Findings

- **A real live session went completely silent for its entire length --
  root-caused and fixed.** Every single window committed 0 notes (both
  retry attempts, every window, from the very first) in an actual session
  with a musician playing real MIDI. Reproduced directly: took the exact
  melody from that session's log and replayed it, first independently per
  window (worked fine most of the time) then *sequentially* -- each window
  building on the real (empty) accompaniment history left by the previous
  one, matching what actually happened live -- and hit the same lock-in:
  once a few windows in a row commit nothing, the model has no precedent
  for that instrument anywhere in the growing context and increasingly
  favors continuing its absence, so retrying with the same now-stuck
  context (`MAX_GENERATION_ATTEMPTS`) doesn't help. Confirmed directly
  against a stuck context pulled from that reproduction: `accomp_bias=2.0`
  (the base default) failed 4/4 trials there; 4.0-6.0 reliably broke the
  lock (1-8 notes/trial, never zero). Fixed by escalating the bias by
  `SILENCE_BIAS_STEP` per consecutive silent window (capped at
  `MAX_SILENCE_BIAS_STEPS`), resetting the moment something commits --
  verified against the same stuck sequence: 3 silent windows in a row,
  then broke through and never got stuck again.
- **A broader instrument sweep (3 trials each, solo-masked against the same
  piano prompt) found several more usable voices beyond the string
  section:** electric piano (69 notes across 3 trials), nylon guitar (219 —
  by far the strongest single voice found), string-ensemble patch (43),
  trumpet (47), alto sax (109), flute (20), new-age pad (25), harp (20).
  `INSTRUMENT_PRESETS` (`--voices`) packages these into 8 named options —
  a solo violin, the string trio, and five more solo/paired options — so a
  caller doesn't need to know GM program numbers or which ones actually
  produce output.
- **Temperature is a real, distinct knob from `top_p`/`accomp_bias`,
  verified not just plausible.** Scaling logits by `1/temperature` before
  `top_p` truncation measurably changes both density and pitch spread on
  the same prompt: temperature 0.6 produced 227 notes with pitch std 13.3
  across 3 trials, 1.0 produced 30 notes with std 9.1, 1.6 produced only 10
  notes but with std 19.3 (wider, more scattered pitch choices) — lower
  temperature trends denser/safer, higher trends sparser/wilder, not a
  placebo knob.
- **"Follow vs. lead" is implemented honestly as what's actually
  changeable, not a fake role reversal.** The model has no "leading" mode
  and always generates from melody already revealed — there's no
  architecture change that would let the companion take over the main
  line without retraining. What's real and cheap: whether it waits through
  `listen_first` before playing at all. `--role lead` sets that to 0 (starts
  immediately) instead of the polite 8-beat default; verified the companion
  actually starts within one commit cycle at t≈1s instead of waiting ~6s.
  Genre control was considered and dropped: this checkpoint has no genre
  conditioning signal at all (Lakh MIDI isn't cleanly genre-labeled for it),
  so anything called "genre" would have to be a fake proxy (e.g. biasing
  pitch range or density to loosely gesture at a style) — that's decorative,
  not a real feature, and out of scope without fine-tuning (which
  `proposal.md` already flags as the highest-risk, lowest-return item here).
- **Live (unbounded) sessions had an unbounded performance regression --
  found while building `live_midi.py`, fixed.** `_maybe_kick_generation`
  handed the model's preprocessing (`ops.sort`/`clip`/`pad`, all O(history
  length)) the *entire* accumulated history every cycle. Invisible for a
  ~20s synthetic demo, but a real bug for a live session with no fixed
  length: verified with a scripted end-to-end test (a real MIDI performance
  sent over a virtual port into `live_midi.py`) that generation time
  escalated 19s → 36s → 53s per call as the session ran past ~100s of music
  time -- the model was drowning in a growing list most of which it
  couldn't even use (`amt._add_token` already only looks at its own
  trailing ~1017-token window internally). Fixed by clipping the history
  handed to the model to a trailing `HISTORY_LOOKBACK_S` (90s, generously
  above what that token window could span at any realistic density) and
  re-basing it to start near zero -- clipping alone isn't enough, since
  `ops.pad` pads silence from absolute time zero, so an old absolute
  timestamp keeps the O(session length) scaling even after trimming the
  event list itself. Re-verified with the same scripted test past 190s of
  music time: generation time stayed in the 0.3-1.3s range throughout,
  realtime factor 2.44x, 0 underruns, 297 companion notes actually
  delivered over the (virtual) output port.
- **The output used to run much longer than the input melody -- fixed.**
  `_maybe_kick_generation` had no upper bound on `committed_horizon`, so once
  the melody ended it kept pipelining new lookahead windows every cycle
  regardless -- there was no more melody to inform them, but nothing said
  "stop". Since the outer loop's only exit condition was wall-clock time
  reaching `melody_len_s + tail_s`, and the model usually runs faster than
  real time, `committed_horizon` (music time already planned) could race far
  ahead of the wall clock before the loop noticed it should stop: one 19.1s
  melody produced a 33.5s MIDI file. Capping `_maybe_kick_generation` at
  `committed_horizon >= melody_len_s + tail_s` fixed it (same melody now
  produces 21.85s) and also finishes faster (9.4s wall time vs. 17.4s,
  fewer wasted inference calls: 13 vs. 21) since it stops working the moment
  there's nothing left to usefully generate for.
- **Realtime factor 1.3x-3.6x** across runs on an M-series Mac/MPS, for
  ~1.5-2.5 beat commit windows at 80 BPM — net faster than real time on
  average, but with **high per-call variance** (0.1 s-9 s for similarly
  sized chunks). A standing buffer (priming the first chunk a full
  lookahead early, plus continuous pipelining once the loop is running)
  absorbs most of that variance; a few underruns still happen on a loaded
  machine. This is the actual go/no-go number `proposal.md` asks for from
  the first two hours of work — worth re-measuring on real target hardware
  before committing to this model for a live demo.
- **The base checkpoint needs help to act like a duet, not a Lakh song.**
  Given only a sparse prompt, unmasked generation sprayed notes across a
  dozen-plus unrelated GM instruments — Lakh MIDI is full of multi-track
  songs, so that's an in-distribution sample, just not a useful one here.
  Masking down to a small instrument set fixes the spraying, but the model
  still sometimes chooses to write *zero* companion notes in a window (also
  a legitimate sample) — several in a row leaves the companion audibly
  silent (one run had a 9.6s gap out of a 24.7s piece).
- **A logit bias toward the kept instrument(s) helps but isn't reliable
  enough on its own.** Sweeping `--accomp-bias` from 2 to 10 across several
  melodies raised *average* note density, but individual runs still landed
  double-digit-second silent stretches regardless, and the relationship
  wasn't even monotonic (bias 8 was sometimes worse than bias 6). What
  actually closes the gap: when a window comes back with literally nothing,
  just ask again. Retrying up to `MAX_GENERATION_ATTEMPTS` (2, tuned
  empirically) cut the longest observed silent gap from 9.6s to ~6-7s across
  test seeds, at essentially no quality cost since resampling doesn't touch
  the musical decision, only whether an empty answer gets accepted. The
  tradeoff is real, though: each retry multiplies that window's generation
  time, and pushing the retry count to 4 was enough to wreck the realtime
  factor on one melody (16 of 17 windows underran, factor 1.37x vs. 2-3x
  typical at 2 attempts) — silence-avoidance and real-time viability trade
  directly against each other on this hardware, they don't come for free
  together.
- **Instrument pairing is not arbitrary, but a proper string section does
  work.** A sweep over candidate GM instruments against the same piano
  prompt found wildly different note yields (e.g. violin ≈40+ notes per
  window vs. bass ≈0). Viola and cello were flagged as unverified in an
  earlier pass; checked directly (solo-masked individually, then all three
  masked in together as a real ensemble) and both produce real output --
  individually 8-63 notes per 4-beat window across trials, and together
  (violin/viola/cello all competing for probability mass under one mask)
  7-80 notes total per window with no voice starving the others out. This
  is now the default (`STRING_ENSEMBLE_ACCOMP_INSTRS`); `--solo` drops back
  to one violin.
- **A lone violin was consistently too sparse to hear against a real
  performance.** In an actual session with a musician playing densely
  (chords, continuous notes), one excerpt had ~150 human notes against only
  16 companion notes in the same span, and half of *those* were trimmed to
  under 150ms by the monophony logic (see below) -- audible as clicks at
  best. The 3-voice ensemble is the fix: more total companion notes, and
  since each instrument is independently monophonic, the ensemble's
  *combined* texture can still be `--multi-voice`-off (each individual
  voice clean) while having far more presence than one voice alone.
- **The ensemble costs more compute than solo, and it shows.** Three
  instruments competing for probability mass under one mask is a harder
  generation problem than one -- realtime factor dropped to ~0.6-1.0x with
  the default CLI settings (`--lookahead-beats 3 --commit-beats 1.5`),
  down from the 2-4x solo saw. Worth tuning lookahead/commit down (or
  raising them for more buffer cushion) per your hardware if you see
  underruns; not yet retuned after this change.
- **Monophony is a choice, not a constraint of the model.** `--multi-voice`
  simply skips the trim/dedup step at commit time — same generated notes,
  just not forced into a single line. Verified on one run: 34 violin notes
  / 0 self-overlaps with the default, vs. 43 violin notes / 32 self-overlaps
  with `--multi-voice` on the same melody/seed. Composes cleanly with the
  string ensemble too (each instrument's polyphony is independent).
- **It doesn't just double the melody.** Checked one run's committed notes:
  41 accompaniment vs. 32 melody notes, only 1/41 at the exact same pitch,
  4/41 sharing a pitch class (unison/octave), and a pitch range (48-90)
  extending both above and below the melody's (60-72). No explicit
  diversity constraint is applied — this fell out of the instrument split
  and the base model's learned behavior on real multi-track songs, not
  something enforced. `proposal.md`'s best-of-k reranking would be the
  principled way to *guarantee* this rather than rely on it.
- **The synthetic-melody demo (`live_duet.py`) has no input latency**
  (`proposal.md`'s delay #1) because the "performer" is synthetic Python
  events, not a transcribed instrument — it only exercises delay #2
  (inference/scheduling), which was the point. `live_midi.py` (see
  **SETUP.md**) closes that gap for real MIDI input, which needs no
  transcription step at all; audio input (a microphone) would need real
  pitch tracking first, with its own latency and error modes on top of
  everything measured here.

## Next steps

- Re-measure realtime factor on whatever hardware this actually ends up
  running on (this was CPU/MPS on a laptop).
- Best-of-k reranking (per `proposal.md`) instead of the flat accompaniment
  bias, for an actual quality/distinctiveness signal rather than a fixed
  nudge.
- Audio input (microphone) via real pitch tracking, as a second input path
  alongside `live_midi.py`'s MIDI keyboard support.
