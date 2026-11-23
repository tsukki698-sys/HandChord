"""Audio-reactive visualizers with colour themes (drawn with OpenCV on top of the camera).

How it reacts to the music
  The sound of every chord is a known buffer, and the program knows when it started playing.
  So each frame it takes a 1024-sample window from the playing buffer at the right position,
  runs an FFT, and gets 32 frequency bands + a small waveform. Louder = taller / bigger.

Modes : Bars, Wave, Rings, Burst, Off          (V key or click)
Themes: Chord (colour of the current chord), Neon, Sunset, Ocean, Forest, Mono   (C key or click)
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

THEMES = {
    "Chord": None,
    "Neon": [(0, 240, 255), (255, 0, 200), (150, 90, 255)],
    "Sunset": [(255, 94, 98), (255, 153, 102), (255, 206, 84)],
    "Ocean": [(0, 119, 182), (0, 180, 216), (144, 224, 239)],
    "Forest": [(34, 160, 60), (124, 230, 40), (255, 245, 150)],
    "Mono": [(255, 255, 255), (210, 210, 225), (160, 160, 185)],
}
THEME_NAMES = list(THEMES)
MODES = ["Bars", "Wave", "Rings", "Burst", "Off"]


def theme_color(theme: str, accent, u: float):
    """RGB colour at position u (0..1) of the theme gradient."""
    palette = THEMES[theme]
    if palette is None:   # "Chord": the chord's own colour, a bit lighter towards u = 1
        return tuple(int(c + (255 - c) * 0.45 * u) for c in accent)
    u = min(max(u, 0.0), 1.0) * (len(palette) - 1)
    i = min(int(u), len(palette) - 2)
    f = u - i
    return tuple(int(palette[i][c] * (1 - f) + palette[i + 1][c] * f) for c in range(3))


def _bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


class Analyzer:
    """Turns the audio that is playing right now into 32 bands + a waveform."""

    def __init__(self, n_bands: int = 32, sample_rate: int = 44100):
        self.n = n_bands
        self.win = np.hanning(1024).astype(np.float32)
        freqs = np.fft.rfftfreq(1024, 1 / sample_rate)
        edges = np.geomspace(70, 9000, n_bands + 1)
        self.idx = []
        for i in range(n_bands):
            a = int(np.searchsorted(freqs, edges[i]))
            b = max(int(np.searchsorted(freqs, edges[i + 1])), a + 1)
            self.idx.append((a, b))
        self.bands = np.zeros(n_bands)
        self.wave = np.zeros(256)
        self.level = 0.0
        self.onset = 0.0
        self.peak = 1e-3

    def update(self, stereo, pos: int, gain: float) -> None:
        if stereo is None or gain <= 0.001:
            self.bands *= 0.85
            self.wave *= 0.85
            self.level *= 0.85
            self.onset = 0.0
            return
        seg = stereo[(np.arange(1024) + pos) % len(stereo), 0].astype(np.float32) / 32768.0
        spec = np.abs(np.fft.rfft(seg * self.win))
        raw = np.log1p(np.array([spec[a:b].mean() for a, b in self.idx]) * 40)
        self.peak = max(self.peak * 0.998, float(raw.max()), 1e-3)
        norm = np.clip(raw / self.peak, 0, 1) * min(1.0, 0.25 + gain)
        self.bands = np.where(norm > self.bands, 0.6 * norm + 0.4 * self.bands, 0.85 * self.bands + 0.15 * norm)
        self.wave = np.clip(seg[::4][:256] * 3.0 * min(1.0, 0.3 + gain), -1, 1)
        level = float(self.bands.mean())
        self.onset = max(0.0, level - self.level)
        self.level = level


class Visualizer:
    def __init__(self):
        self.mode_i = 0
        self.theme_i = 0
        self.analyzer = Analyzer()
        self.rings: list[list] = []      # [radius, alpha, u]
        self.parts: list[list] = []      # [x, y, vx, vy, life, u]
        self._last_chord = None

    @property
    def mode(self) -> str:
        return MODES[self.mode_i]

    @property
    def theme(self) -> str:
        return THEME_NAMES[self.theme_i]

    def next_mode(self) -> None:
        self.mode_i = (self.mode_i + 1) % len(MODES)

    def next_theme(self) -> None:
        self.theme_i = (self.theme_i + 1) % len(THEME_NAMES)

    def update(self, stereo, pos: int, gain: float) -> None:
        self.analyzer.update(stereo, pos, gain)

    # ------------------------------------------------------------------
    def draw(self, frame, accent, chord: str | None, origin=None, strip: bool = False) -> None:
        """Draw the current visual on `frame` (BGR, modified in place)."""
        h, w = frame.shape[:2]
        changed = chord is not None and chord != self._last_chord
        self._last_chord = chord
        mode, theme, a = self.mode, self.theme, self.analyzer
        if mode == "Off":
            return
        col = lambda u: _bgr(theme_color(theme, accent, u))

        if mode == "Bars":
            overlay = frame.copy()
            n, gap, base, max_h = a.n, 4, h - 140, 240
            if strip:
                base, max_h = h - 10, h - 26
            bw = (w - 40 - gap * (n - 1)) / n
            for i in range(n):
                x0 = int(20 + i * (bw + gap))
                bh = int(6 + a.bands[i] * max_h)
                c = col(i / (n - 1))
                cv2.rectangle(overlay, (x0, base - bh), (int(x0 + bw), base), c, -1)
                cv2.circle(overlay, (int(x0 + bw / 2), base - bh), int(bw / 2), c, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5 if strip else 0.6, frame, 0.5 if strip else 0.4, 0, frame)

        elif mode == "Wave":
            pts = np.stack([np.linspace(0, w, 256), (h * 0.5 + a.wave * h * 0.38) if strip else (h * 0.70 + a.wave * 110)], axis=1).astype(np.int32)
            glow = frame.copy()
            cv2.polylines(glow, [pts], False, col(0.5), 11, cv2.LINE_AA)
            cv2.addWeighted(glow, 0.22, frame, 0.78, 0, frame)
            line = frame.copy()
            for i in range(255):
                cv2.line(line, tuple(pts[i]), tuple(pts[i + 1]), col(i / 255), 3, cv2.LINE_AA)
            cv2.addWeighted(line, 0.9, frame, 0.1, 0, frame)

        elif mode == "Rings":
            if changed or a.onset > 0.025:
                self.rings.append([24.0, 1.0, random.random()])
            cx, cy = w // 2, (h if strip else int(h * 0.52))
            for ring in self.rings:
                ring[0] += 6 + 14 * a.level
                ring[1] -= 0.022
                overlay = frame.copy()
                cv2.circle(overlay, (cx, cy), int(ring[0]), col(ring[2]), 4 + int(6 * ring[1]), cv2.LINE_AA)
                alpha = max(ring[1], 0) * 0.65
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            self.rings = [r for r in self.rings if r[1] > 0][-14:]

        elif mode == "Burst":
            if strip and origin:
                origin = (origin[0], h - 8)
            if changed and origin:
                for _ in range(40):
                    ang = random.uniform(-math.pi * 0.95, -math.pi * 0.05)
                    sp = random.uniform(3, 11)
                    self.parts.append([origin[0], origin[1], math.cos(ang) * sp, math.sin(ang) * sp,
                                       random.randint(26, 50), random.random()])
            for _ in range(int(a.level * 7)):       # ambient sparks rise with the music
                self.parts.append([random.uniform(30, w - 30), (h - 8) if strip else h - 130, random.uniform(-0.6, 0.6),
                                   random.uniform(-4.5, -1.5), random.randint(20, 40), random.random()])
            overlay = frame.copy()
            for p in self.parts:
                p[0] += p[2]
                p[1] += p[3]
                p[3] += 0.10
                p[4] -= 1
                cv2.circle(overlay, (int(p[0]), int(p[1])), max(1, int(2 + p[4] / 8)), col(p[5]), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            self.parts = [p for p in self.parts if p[4] > 0][-500:]