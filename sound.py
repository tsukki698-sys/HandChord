"""Instruments for Handchord. Every chord is rendered ONCE into a 2-second stereo loop.

Two kinds of instrument
  sustained   (Pad, Synth, Organ, Choir, Pure): every frequency is snapped to a multiple
              of 0.5 Hz so the wave fits whole cycles in the loop -> perfectly seamless.
  percussive  (Piano, Keys, Guitar, D-Guitar, Bass, 808): a struck note that fades away.
              The loop is simply "strike, ring, fade", and it re-strikes every 2 seconds
              while you hold the chord. A tiny fade at the very end avoids any click.

How each sound is built (all from sine waves, no sample files):
  Piano     - 10 partials, slightly stretched, higher partials die faster, two strings
  Keys      - electric piano: sine + soft 2nd harmonic + a short "tine" ping + tremolo
  Guitar    - plucked string: bright at first, mellow later, notes strummed low to high
  D-Guitar  - "digital" guitar: longer sustain, then pushed through tanh() distortion
  Bass      - the chord's root note, two octaves down, with a little saturation
  808       - root note, two octaves down, with a pitch drop at the start + a click
  Synth     - detuned sawtooth-like voices
  Pad       - soft detuned voices with reverb          Organ - drawbar harmonics + tremolo
  Choir     - voice: harmonics shaped by 'ah' formants, vibrato, detune and reverb
  Pure      - plain sine waves
"""

from __future__ import annotations

import threading
import time

import numpy as np

from theory import INTERVALS, ROOT_HZ, chord_freqs

SAMPLE_RATE = 44100
LOOP_SECONDS = 2
N = SAMPLE_RATE * LOOP_SECONDS

INSTRUMENTS = ["Pad", "Synth", "Piano", "Keys", "Guitar", "D-Guitar", "Organ", "Bass", "808", "Choir", "Pure"]
PERCUSSIVE = {"Piano", "Keys", "Guitar", "D-Guitar", "Bass", "808"}
ROOT_ONLY = {"Bass", "808"}   # these play only the chord's root note
# General-MIDI program numbers, used when a recording is saved as a .mid file
GM_PROGRAM = {"Pad": 89, "Synth": 81, "Piano": 0, "Keys": 4, "Guitar": 25, "D-Guitar": 30,
              "Organ": 19, "Bass": 33, "808": 38, "Choir": 52, "Pure": 80}

_T = np.arange(N) / SAMPLE_RATE
_TAIL = np.minimum(1.0, (N - 1 - np.arange(N)) / (0.03 * SAMPLE_RATE))   # 30 ms fade at loop end


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _snap(freq: float) -> float:
    return round(freq * LOOP_SECONDS) / LOOP_SECONDS


def _voice(freq: float, harmonics) -> np.ndarray:
    out = np.zeros(N)
    for k, amp in harmonics:
        out += amp * np.sin(2 * np.pi * k * freq * _T)
    return out


def _impulse_response(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    length = int(1.2 * SAMPLE_RATE)
    ir = rng.standard_normal(length) * np.exp(-np.arange(length) / (0.35 * SAMPLE_RATE))
    return ir / np.sqrt(np.sum(ir ** 2))


_IR_SPECTRA = [np.fft.rfft(_impulse_response(seed), n=N) for seed in (7, 8)]


def _reverb(channels, wet_mix=0.4):
    """Circular (wrap-around) convolution: the echo wraps the loop end, so it stays seamless."""
    out = []
    for x, spectrum in zip(channels, _IR_SPECTRA):
        wet = np.fft.irfft(np.fft.rfft(x) * spectrum, n=N)
        wet *= (np.sqrt(np.mean(x ** 2)) + 1e-9) / (np.sqrt(np.mean(wet ** 2)) + 1e-9)
        out.append((1 - wet_mix) * x + wet_mix * wet)
    return out


def _pan(mono: np.ndarray, position: float):
    """Equal-power pan. position 0 = left, 1 = right."""
    angle = position * np.pi / 2
    return mono * np.cos(angle), mono * np.sin(angle)


def _finish(left, right, percussive=False, gain=1.0) -> np.ndarray:
    """Match loudness between instruments, limit peaks, convert to 16-bit stereo."""
    stereo = np.column_stack([left, right])
    if percussive:
        stereo = stereo * _TAIL[:, None]
    rms = float(np.sqrt(np.mean(stereo ** 2))) + 1e-9
    stereo = stereo * (0.14 * gain / rms)
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.9:
        stereo *= 0.9 / peak
    return np.ascontiguousarray((stereo * 32767).astype(np.int16))


# ---------------------------------------------------------------------------
# sustained instruments
# ---------------------------------------------------------------------------
_PAD_H = [(k, 1.0 / k ** 1.4) for k in range(1, 9)]
_ORGAN_H = [(1, 1.0), (2, 0.8), (3, 0.5), (4, 0.4), (6, 0.25), (8, 0.15)]
_SAW_H = [(k, 1.0 / k) for k in range(1, 13)]


def _pad(freqs):
    freqs = [_snap(f) for f in freqs]
    left = sum(_voice(f + d, _PAD_H) for f in freqs for d in (-0.5, 0.0))
    right = sum(_voice(f + d, _PAD_H) for f in freqs for d in (0.0, 0.5))
    left, right = _reverb([left, right])
    return _finish(left, right)


def _organ(freqs):
    mono = sum(_voice(_snap(f), _ORGAN_H) for f in freqs)
    mono = mono * (1.0 + 0.08 * np.sin(2 * np.pi * 4 * _T))
    left, right = _reverb([mono, mono], wet_mix=0.15)
    return _finish(left, right)


def _pure(freqs):
    mono = sum(np.sin(2 * np.pi * _snap(f) * _T) for f in freqs)
    return _finish(mono, mono)


def _synth(freqs):
    freqs = [_snap(f) for f in freqs]
    lfo = 1.0 + 0.10 * np.sin(2 * np.pi * 2 * _T)          # 2 Hz: whole cycles in the loop
    left = sum(_voice(f + d, _SAW_H) for f in freqs for d in (-1.0, 0.0)) * lfo
    right = sum(_voice(f + d, _SAW_H) for f in freqs for d in (0.0, 1.0)) * lfo
    left, right = _reverb([left, right], wet_mix=0.15)
    return _finish(left, right)


_FORMANTS = [(800.0, 150.0, 1.0), (1150.0, 200.0, 0.5), (2900.0, 300.0, 0.2)]   # vowel "ah"


def _formant_gain(freq: float) -> float:
    gain = 0.03
    for centre, width, amp in _FORMANTS:
        gain += amp * np.exp(-0.5 * ((freq - centre) / width) ** 2)
    return gain


def _choir(freqs):
    freqs = [_snap(f) for f in freqs]
    vibrato = 0.10 * np.sin(2 * np.pi * 5 * _T)             # 5 Hz vibrato: whole cycles in the loop
    left, right = np.zeros(N), np.zeros(N)
    for f in freqs:
        for d, side in ((-0.5, 0), (0.0, 2), (0.5, 1)):
            voice = np.zeros(N)
            for k in range(1, int(4500 / f) + 1):
                amp = _formant_gain(k * (f + d)) / k ** 0.8
                if amp < 0.01:
                    continue
                voice += amp * np.sin(2 * np.pi * k * (f + d) * _T + k * vibrato)
            if side in (0, 2):
                left += voice
            if side in (1, 2):
                right += voice
    left, right = _reverb([left, right], wet_mix=0.35)
    return _finish(left, right)


# ---------------------------------------------------------------------------
# percussive instruments
# ---------------------------------------------------------------------------
def _gate(delay: float):
    return np.maximum(_T - delay, 0.0), (_T >= delay).astype(float)


def _piano_note(f, delay):
    tt, gate = _gate(delay)
    beat = 0.75 + 0.25 * np.cos(2 * np.pi * 0.3 * tt)       # two strings, slightly out of tune
    sig = np.zeros(N)
    for k in range(1, 11):
        fk = k * f * np.sqrt(1 + 0.0004 * k * k)             # real strings are a little "stretched"
        if fk > 9000:
            break
        tau = 1.7 / (1 + 0.55 * (k - 1))                     # higher partials fade faster
        sig += (1.0 / k ** 1.15) * np.exp(-tt / tau) * np.sin(2 * np.pi * fk * tt)
    return sig * beat * np.minimum(tt / 0.003, 1.0) * gate


def _piano(freqs):
    left, right = np.zeros(N), np.zeros(N)
    order = sorted(freqs)
    for i, f in enumerate(order):
        l, r = _pan(_piano_note(f, i * 0.006), 0.25 + 0.5 * i / max(len(order) - 1, 1))
        left += l
        right += r
    left, right = _reverb([left, right], wet_mix=0.12)
    return _finish(left, right, percussive=True)


def _keys_note(f, delay):
    tt, gate = _gate(delay)
    sig = (np.sin(2 * np.pi * f * tt) * np.exp(-tt / 1.6)
           + 0.35 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt / 0.7)
           + 0.20 * np.sin(2 * np.pi * 7 * f * tt) * np.exp(-tt / 0.12))      # the "tine" ping
    tremolo = 1.0 + 0.12 * np.sin(2 * np.pi * 5 * _T)
    return sig * tremolo * np.minimum(tt / 0.002, 1.0) * gate


def _keys(freqs):
    left, right = np.zeros(N), np.zeros(N)
    order = sorted(freqs)
    for i, f in enumerate(order):
        l, r = _pan(_keys_note(f, i * 0.004), 0.3 + 0.4 * i / max(len(order) - 1, 1))
        left += l
        right += r
    left, right = _reverb([left, right], wet_mix=0.10)
    return _finish(left, right, percussive=True)


def _pluck_note(f, delay, tau0, tau_pow, kmax, amp_pow):
    tt, gate = _gate(delay)
    sig = np.zeros(N)
    for k in range(1, kmax + 1):
        if k * f > 10000:
            break
        sig += k ** -amp_pow * np.exp(-tt / (tau0 / k ** tau_pow)) * np.sin(2 * np.pi * k * f * tt)
    return sig * np.minimum(tt / 0.0015, 1.0) * gate


def _guitar(freqs, digital=False):
    left, right = np.zeros(N), np.zeros(N)
    order = sorted(freqs)
    strum = 0.010 if digital else 0.014                       # seconds between strings
    for i, f in enumerate(order):
        if digital:
            note = _pluck_note(f, i * strum, 3.0, 0.4, 16, 0.7)
        else:
            note = _pluck_note(f, i * strum, 1.5, 0.65, 14, 0.85)
        l, r = _pan(note, 0.2 + 0.6 * i / max(len(order) - 1, 1))
        left += l
        right += r
    if digital:                                               # distortion: soft clipping
        left, right = np.tanh(3.5 * left / (np.max(np.abs(left)) + 1e-9)), np.tanh(3.5 * right / (np.max(np.abs(right)) + 1e-9))
    left, right = _reverb([left, right], wet_mix=0.10 if not digital else 0.18)
    return _finish(left, right, percussive=True)


def _bass(root_hz):
    f = root_hz / 4
    tt, _ = _gate(0.0)
    sig = (np.sin(2 * np.pi * f * tt) + 0.45 * np.sin(2 * np.pi * 2 * f * tt) + 0.25 * np.sin(2 * np.pi * 3 * f * tt))
    sig = np.tanh(1.3 * sig) * np.exp(-tt / 1.6) * np.minimum(tt / 0.006, 1.0)
    return _finish(sig, sig, percussive=True, gain=1.0)


def _808(root_hz):
    f = root_hz / 4
    tt, _ = _gate(0.0)
    glide = 1.2 * 0.05 * (1 - np.exp(-tt / 0.05))             # pitch starts high, drops in ~0.1 s
    sig = np.sin(2 * np.pi * f * (tt + glide)) * np.exp(-tt / 1.25) * np.minimum(tt / 0.002, 1.0)
    click = 0.3 * np.exp(-tt / 0.004) * np.random.default_rng(3).standard_normal(N)
    sig = np.tanh(2.2 * (sig + click))
    return _finish(sig, sig, percussive=True, gain=1.0)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def sounding_freqs(letter: str, quality: str, instrument: str) -> list[float]:
    """The notes an instrument really plays for this chord (Bass / 808: root only, 2 octaves down)."""
    if instrument in ROOT_ONLY:
        return [ROOT_HZ[letter] / 4]
    return chord_freqs(letter, quality)


def render(letter: str, quality: str, instrument: str) -> np.ndarray:
    """int16 array (N, 2): a one-chord stereo loop for this instrument."""
    freqs = chord_freqs(letter, quality)
    if instrument == "Pad":
        return _pad(freqs)
    if instrument == "Synth":
        return _synth(freqs)
    if instrument == "Piano":
        return _piano(freqs)
    if instrument == "Keys":
        return _keys(freqs)
    if instrument == "Guitar":
        return _guitar(freqs)
    if instrument == "D-Guitar":
        return _guitar(freqs, digital=True)
    if instrument == "Organ":
        return _organ(freqs)
    if instrument == "Bass":
        return _bass(ROOT_HZ[letter])
    if instrument == "808":
        return _808(ROOT_HZ[letter])
    if instrument == "Choir":
        return _choir(freqs)
    return _pure(freqs)


# --- kept so the older handchord_2h.py still works ---------------------------
TIMBRES = ["Pad", "Organ", "Pure"]


def render_chord(freqs, timbre):
    return {"Pad": _pad, "Organ": _organ}.get(timbre, _pure)(freqs)


class SoundBank:
    """Holds every rendered loop. The first instrument is rendered right away; the others are
    rendered by a background thread while you play (Bass/808 ignore the chord type, so they
    are rendered once per letter and shared)."""

    def __init__(self):
        self._arrays: dict = {}
        self._done: set = set()
        self.progress = 0.0

    def render_instrument(self, instrument: str) -> None:
        for letter in ROOT_HZ:
            shared = None
            for quality in INTERVALS:
                if instrument in ROOT_ONLY and shared is not None:
                    self._arrays[(letter, quality, instrument)] = shared
                    continue
                arr = render(letter, quality, instrument)
                self._arrays[(letter, quality, instrument)] = arr
                shared = arr
            time.sleep(0.02)    # let the camera loop breathe
        self._done.add(instrument)

    def start_background(self, first: str = INSTRUMENTS[0], preload: bool = False) -> None:
        """Render `first` now. Other instruments are rendered only when asked for (request),
        so nothing heavy runs in the background while you play. preload=True renders all."""
        self.render_instrument(first)
        self._thread = None
        self._busy: set = set()
        if preload:
            for name in INSTRUMENTS:
                self.request(name)

    def request(self, instrument: str) -> None:
        """Start rendering one instrument in a background thread (once)."""
        if not hasattr(self, "_busy"):
            self._busy = set()
        if instrument in self._done or instrument in self._busy:
            return
        self._busy.add(instrument)

        def work():
            self.render_instrument(instrument)
            self._busy.discard(instrument)

        threading.Thread(target=work, daemon=True).start()

    @property
    def loading(self) -> str | None:
        busy = getattr(self, "_busy", set())
        return next(iter(busy)) if busy else None

    def ready(self, instrument: str) -> bool:
        return instrument in self._done

    def array(self, letter: str, quality: str, instrument: str):
        return self._arrays.get((letter, quality, instrument))
