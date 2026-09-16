"""
Quick, hardware-free sanity check for tension.RollingTension: feeds a few
hand-picked note sequences through it and prints the resulting tension
trace, so the weights in tension.py can be sanity-checked/tuned by eye
before any MIDI hardware is involved.

Run: python test_tension.py
"""

from tension import RollingTension


def run_case(name, events):
    print(f"\n=== {name} ===")
    rt = RollingTension()
    for onset_s, dur_s, role, pitch in events:
        t = rt.update(onset_s, dur_s, role, pitch)
        print(f"  t={onset_s:5.2f}s role={role:12s} pitch={pitch:3d}  -> tension={t:.3f}")


# Stepwise, consonant melody alone -- expect tension to stay low.
run_case("stepwise consonant melody (C D E F G)", [
    (0.0, 0.5, "melody", 60),
    (0.5, 0.5, "melody", 62),
    (1.0, 0.5, "melody", 64),
    (1.5, 0.5, "melody", 65),
    (2.0, 0.5, "melody", 67),
])

# Wide chromatic leaps -- expect tension to climb.
run_case("wide chromatic leaps", [
    (0.0, 0.5, "melody", 60),
    (0.5, 0.5, "melody", 73),   # +13, minor 9th-ish leap
    (1.0, 0.5, "melody", 61),   # -12 leap, ic1 landing
    (1.5, 0.5, "melody", 66),   # tritone from previous
])

# Melody + companion moving in parallel major thirds (both voices step by
# a whole tone, preserving the third) -- vertical term should stay low
# throughout, since we don't want the companion's *own* melodic motion
# (a separate term) to confound this case.
run_case("melody + companion in parallel thirds", [
    (0.0, 1.0, "melody", 60),
    (0.0, 1.0, "accomp:40", 64),   # major third above, ic4 (low dissonance)
    (1.0, 1.0, "melody", 62),
    (1.0, 1.0, "accomp:40", 66),   # major third above again, ic4 (low dissonance)
])

# Melody + companion a semitone apart -- vertical term should spike.
run_case("melody + companion a semitone apart (ic1 clash)", [
    (0.0, 1.0, "melody", 60),
    (0.0, 1.0, "accomp:40", 61),   # minor second, ic1 -- max dissonance
    (1.0, 1.0, "melody", 60),
    (1.0, 1.0, "accomp:40", 61),
])

# Dense burst of notes -- density term should push tension up even if the
# pitches themselves are unremarkable.
run_case("dense burst (8 notes in 1s)", [
    (i * 0.125, 0.1, "melody", 60 + (i % 3)) for i in range(8)
])
