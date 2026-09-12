# Run the duet locally

## This Mac

The local environment and AMT checkpoint have been prepared in `.venv/` and
`.model-cache/`. From Terminal:

```sh
cd /Users/weronikazygis/Desktop/projects/DuetDotAI
./run_local.sh --list-ports
```

Connect the USB MIDI keyboard. In Audio MIDI Setup, open MIDI Studio, enable the
IAC Driver, and note the port names printed by the command above. Start with:

```sh
./run_local.sh --midi-in "Your keyboard port" --midi-out "IAC Driver Bus 1" \
  --solo --bpm 80 --lookahead-beats 4 --commit-beats 2 --no-program-change
```

Use your actual port names (unique substrings work). The companion listens for
8 beats before entering. Play to an 80 BPM metronome. Ctrl+C stops playback;
the process may take a little longer to exit while a model call finishes.
The session is exported to `output/live_midi_session.mid`.

`--no-program-change` preserves the instrument preset selected in your DAW.
It is useful for your musician's custom instrument. Without it, the app sends
General MIDI program changes for violin/viola/cello. The model's instrument
labels and your chosen playback sound can be different.

## Hear it in GarageBand

The app sends companion MIDI; it does not render audio or echo your own piano.
Create a software instrument track, select an audible instrument, and ensure it
receives the companion's virtual MIDI output. GarageBand has limited MIDI input
filtering/routing: do not assume it can independently filter each track by input
port. Confirm that playing the human keyboard does not also trigger the
companion sound. If necessary use the keyboard's own audio for the human part
and a host with explicit port/channel filtering for the companion.

Audio Unit (AU) instrument plugins can render MIDI on a software instrument
track. A VST-only plugin needs a compatible host; MIDI files are note sequences,
not instrument plugins. A sample library needs its compatible sampler.

For Morpho: load the AU effect AFTER the software instrument on the companion
track. Morpho processes audio, so feeding it MIDI alone will not make sound.
Keep the human piano on a separate audio path. Start with a clear pitched model
and modest wet/dry blend, then audition more experimental sounds.

## LYDIA connection

Use LYDIA as an audio processor for the generated companion:

Keyboard MIDI -> AMT -> companion synth -> audio output -> LYDIA audio input
-> LYDIA audio output -> mixer/speakers.

Mix the human piano separately. Do not feed LYDIA's processed output back into
its own input. An interface with separate outputs helps send only the companion
to LYDIA. Phase I uses an external USB audio interface; Phase II has integrated
audio I/O. Ask the Roland team which prototype you have, which connectors and
levels to use, and which models/macros it exposes. Knobs/MIDI parameter maps
and model-loading support depend on that prototype. The laptop still generates
notes; this setup does not put AMT on LYDIA.

## Test without a keyboard

```sh
./run_local.sh --test
./run_local.sh --demo
```

The demo runs the real AMT model on a synthetic melody and saves
`output/live_duet.mid`; it does not play audio. Drag that MIDI into your DAW to
listen. Apple Silicon acceleration is chosen automatically when available.
The first model load/warmup happens before the timed performance.

## Install on another Mac

Use Python 3.10-3.12 and Git. In the project directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_local.sh --demo
```

The first demo downloads the checkpoint into `.model-cache/`; later runs use the
cache. The prepared environment on this Mac reuses its existing Python packages.

## Current behavior and limits

- No Space, Temperature, or Genre controls were added in this update.
- Live note attacks reach the model before release. Held notes have one stable
  identity and an estimated duration updated while held, then finalized on key
  release. MIDI velocity is retained at input but output remains velocity 80.
- Sustain-pedal CC is not modeled; duration represents physical key-down time.
- Model duration tokens cap very long held notes to the vocabulary's maximum.
- Committed music stays within lookahead plus one commit window of playback.
  This trades response speed against compute cushion; it does not eliminate
  musical reaction delay. BPM is fixed, not inferred from playing.
- Expired generated notes are dropped instead of firing in a late burst.
  Partly late notes keep their original end time. Recovery advances to the
  current playhead instead of repeatedly generating already-expired windows.
- Monophonic playback stops the previous note on the same output channel.
  Different ensemble channels stay independent; `--multi-voice` enables overlap.
- Performance reports count committed timeline duration, excluding discarded
  lookahead. Historical README benchmarks used a different, optimistic measure.
- Solo is the recommended first soundcheck. Ensemble generation may need more
  compute. A short passing run is not a guarantee of a two-minute live performance.

Official references:
- https://neutone.jp/morpho
- https://articles.roland.com/project-lydia-phase-ii-neural-sampling-evolved/
- https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/CoreAudioOverview/WhatisCoreAudio.html

## Validation on this Mac, September 12

- 15 regression tests passed in the installed environment.
- Real AMT ran with MPS acceleration.
- A 120-second end-to-end test used isolated CoreMIDI input and output ports:
  142 input notes, 45 companion note attacks, 78 generation calls, zero missed
  note deadlines, zero expired notes, no simultaneous voices on the solo
  channel, and no notes left on after shutdown.
- This verifies timing and MIDI delivery for that test, not musical quality or
  physical keyboard/DAW/LYDIA latency. No external MIDI devices were present.
