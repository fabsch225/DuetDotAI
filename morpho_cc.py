"""
Sends the live tension score (see tension.py) to Project LYDIA's Neutone
Morpho unit as a MIDI CC, so the AI partner's *timbre* -- not just its
notes -- reacts to what's being played. As thin as midi_io.MidiPlayer on
purpose: this module only knows about MIDI CC messages, smoothing, and
rate-limiting -- not about tension itself or the scheduler.

Requires the same python-rtmidi backend as midi_io.py (`pip install
python-rtmidi`), and a MIDI connection from this machine to the Morpho unit
(direct USB MIDI if it has a class-compliant port, or a MIDI interface into
its MIDI-in if not -- confirm which with whoever's running the hardware).

CC number and channel are NOT guessed here -- they depend entirely on how
Morpho's blend/morph parameter is mapped on the actual unit, which needs
confirming against Project LYDIA's own documentation or the Roland/Neutone
team directly. Pass whatever they confirm via --morpho-cc / --morpho-channel
in live_midi.py.
"""

import threading
import time

import mido


class MorphoCC:
    """Owns a MIDI output port to the Morpho unit and continuously pushes a
    smoothed, rate-limited CC value toward the latest tension target. Runs
    its own thread (like midi_io.MidiPlayer) so a burst of notes -- and
    therefore tension updates -- never blocks, and is never blocked by, the
    scheduler's main loop.

    smoothing: exponential-moving-average coefficient (0-1) applied on the
        MIDI side, independent of tension.RollingTension's own smoothing.
        Higher = snappier CC response to a jump in tension; lower = slower,
        smoother morph sweeps. There's no correct value -- retune by ear
        against how fast Morpho's own morph actually sounds good moving.
    min_val/max_val: the CC value (0-127) sent for tension=0.0 and
        tension=1.0. Useful if the blend should never fully bottom out (e.g.
        keep a minimum "wet" amount always audible) -- not a hardware limit,
        just a musical choice.
    min_interval_s: floor on time between actually-sent CC messages, so a
        fast-changing smoothed value doesn't flood the port with every
        1-unit step -- most MIDI gear does not need or want CC at audio
        rate.
    """

    def __init__(self, port_name, cc_number, channel=0, min_val=0, max_val=127,
                 smoothing=0.3, min_interval_s=0.03):
        self._port = mido.open_output(port_name)
        self.cc_number = cc_number
        self.channel = channel
        self.min_val = min_val
        self.max_val = max_val
        self.smoothing = smoothing
        self.min_interval_s = min_interval_s

        self._lock = threading.Lock()
        self._target = 0.0
        self._smoothed = 0.0
        self._last_sent_val = None
        self._last_sent_time = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_tension(self, tension):
        """Non-blocking: called from the scheduler thread (e.g. inside
        on_played, right after tension.RollingTension.update()) every time
        a new note updates the score. Just updates the target the
        background thread is continuously chasing."""
        with self._lock:
            self._target = max(0.0, min(1.0, tension))

    def _run(self, poll_interval=0.01):
        while not self._stop.is_set():
            with self._lock:
                target = self._target
            self._smoothed += self.smoothing * (target - self._smoothed)
            cc_val = round(self.min_val + self._smoothed * (self.max_val - self.min_val))
            now = time.monotonic()
            if cc_val != self._last_sent_val and now - self._last_sent_time >= self.min_interval_s:
                self._port.send(mido.Message(
                    "control_change", channel=self.channel,
                    control=self.cc_number, value=cc_val,
                ))
                self._last_sent_val = cc_val
                self._last_sent_time = now
            time.sleep(poll_interval)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._port.close()
