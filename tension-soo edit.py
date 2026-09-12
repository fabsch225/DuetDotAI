"""
A live, incrementally-updated tension score for the duet's combined output
(melody + companion), meant to drive Project LYDIA's Morpho blend/morph
parameter over MIDI CC (see morpho_cc.py) -- so the AI partner's *timbre*,
not just its notes, responds to what's being played.

Deliberately built from pitch-class-set primitives (interval-class content,
melodic leap size, simultaneity dissonance, note density) instead of a
learned model: cheap to compute per note, and every term is something a
music theorist can reason about and retune by ear during rehearsal, rather
than a black box. Nothing here talks to MIDI or knows about the scheduler --
RollingTension.update() takes exactly the same (onset_s, dur_s, role, pitch)
shape that live_duet.LiveDuet already calls on_played with, so it drops
straight into that callback.
"""

from collections import deque

# Consonance weight per interval class (0-6, octave-invariant: a m2 and a
# M7 are both ic1, a m3 and a M6 both ic3). ic1 (m2/M7) and ic6 (tritone)
# are the most dissonant; unison/octave (ic0) is fully consonant. Standard
# pitch-class-set intuition, not a fitted value -- these are the first
# thing to retune by ear against a real melody, not a hidden constant.
IC_DISSONANCE = {0: 0.0, 1: 1.0, 2: 0.5, 3: 0.2, 4: 0.15, 5: 0.05, 6: 0.9}

# Notes/second treated as "maximally dense" for the density term -- purely
# a normalization constant, not a claim about any particular tempo. Two
# notes/sec is already a fairly busy single line; tune upward if the
# performance is characteristically denser than that (e.g. fast passagework)
# so density_tension doesn't just pin at 1.0 the whole time.
DENSITY_REFERENCE_NPS = 2.0


def interval_class(semitones):
    """Octave-invariant interval class (0-6) for a signed interval in
    semitones. 0 = unison/octave, 6 = tritone."""
    semitones = abs(round(semitones)) % 12
    return min(semitones, 12 - semitones)


def _leap_tension(semitones):
    """Wide melodic leaps read as tense even when the interval class itself
    is consonant (a bare octave leap is still a bigger gesture than a
    step) -- normalized against one octave, capped at 1.0 beyond that."""
    return min(abs(semitones) / 12.0, 1.0)


class RollingTension:
    """Keeps a trailing window of recently *committed* notes (both voices)
    and turns it into a single smoothed tension value in [0, 1] every time
    a new note arrives.

    window_s: how far back "recent" reaches for the density/vertical-overlap
        terms. Shorter reacts faster to what just happened; longer is more
        about the texture's overall temperature. A phrase or two (~4-8s at
        a moderate tempo) is a reasonable start.
    weights: relative contribution of each term (melodic interval-class
        dissonance, leap size, vertical dissonance between overlapping
        melody/companion notes, note density). Not required to sum to 1 --
        the combined score is renormalized by the weight total, then
        clipped to [0, 1].
    ema: exponential-moving-average coefficient applied to the *output*
        score itself (separate from morpho_cc.MorphoCC's own smoothing on
        the MIDI side) so a single very consonant or dissonant note doesn't
        yank the score around -- set to 1.0 to disable and use the raw
        per-note score.
    """

    def __init__(self, window_s=6.0, weights=None, ema=0.4):
        self.window_s = window_s
        self.weights = weights or {
            "melodic_ic": 0.30,
            "leap": 0.15,
            "vertical_ic": 0.35,
            "density": 0.20,
        }
        self.ema = ema
        self._events = deque()  # (onset_s, dur_s, role, pitch), oldest first
        self._last_pitch_by_role = {}  # role -> last pitch seen, for melodic intervals
        self._value = 0.0

    def update(self, onset_s, dur_s, role, pitch):
        """Feed one newly-committed note (melody or companion) in and
        return the updated smoothed tension. Same call shape as
        live_duet.LiveDuet's on_played, so wire it in directly:

            def on_played(onset_s, dur_s, role, pitch):
                t = rolling_tension.update(onset_s, dur_s, role, pitch)
                morpho.set_tension(t)
                ... (existing scheduling of the note to a synth/output) ...
        """
        self._events.append((onset_s, dur_s, role, pitch))
        self._evict_older_than(onset_s - self.window_s)

        melodic = self._melodic_ic_term(role, pitch)
        leap = self._leap_term(role, pitch)
        vertical = self._vertical_ic_term(onset_s, dur_s, role, pitch)
        density = self._density_term(onset_s)

        self._last_pitch_by_role[role] = pitch

        w = self.weights
        total_w = sum(w.values()) or 1.0
        raw = (
            w["melodic_ic"] * melodic
            + w["leap"] * leap
            + w["vertical_ic"] * vertical
            + w["density"] * density
        ) / total_w
        raw = max(0.0, min(1.0, raw))

        self._value += self.ema * (raw - self._value)
        return self._value

    def _evict_older_than(self, cutoff_s):
        while self._events and self._events[0][0] < cutoff_s:
            self._events.popleft()

    def _melodic_ic_term(self, role, pitch):
        prev = self._last_pitch_by_role.get(role)
        if prev is None:
            return 0.0
        return IC_DISSONANCE[interval_class(pitch - prev)]

    def _leap_term(self, role, pitch):
        prev = self._last_pitch_by_role.get(role)
        if prev is None:
            return 0.0
        return _leap_tension(pitch - prev)

    def _vertical_ic_term(self, onset_s, dur_s, role, pitch):
        """Dissonance against every other voice currently sounding at
        onset_s (i.e. an existing note whose [onset, onset+dur) span covers
        this new note's onset). Averaged if more than one voice overlaps
        (e.g. a multi-voice string ensemble); 0.0 if this note is alone."""
        overlapping = [
            IC_DISSONANCE[interval_class(pitch - other_pitch)]
            for other_onset, other_dur, other_role, other_pitch in self._events
            if other_role != role
            and other_onset <= onset_s < other_onset + other_dur
        ]
        return sum(overlapping) / len(overlapping) if overlapping else 0.0

    def _density_term(self, onset_s):
        window_start = onset_s - self.window_s
        n = sum(1 for (t, _, _, _) in self._events if t >= window_start)
        nps = n / self.window_s
        return min(nps / DENSITY_REFERENCE_NPS, 1.0)
