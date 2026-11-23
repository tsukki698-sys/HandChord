"""Handchord's look: card-style panels drawn on top of the camera image.

Style: warm white "cards" with a solid dark offset shadow, big rounded-feeling
type, one accent colour per chord.  Everything visual lives in this file, so
you can change colours, sizes and layout here without touching the model.

Fonts: put Poppins-Bold.ttf and Poppins-Medium.ttf in a folder called `fonts`
next to this file (free fonts from fonts.google.com). If they are missing it
falls back to Segoe UI (Windows) or a basic font.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from theory import QUALITY_LABEL, chord_note_names

HERE = Path(__file__).resolve().parent

# ---- palette (RGB for Pillow) ---------------------------------------------
INK = (28, 24, 56)          # text + shadows
CARD = (255, 251, 242)      # card background
MUTED = (110, 104, 130)     # small grey text
CHORD_INFO = {              # letter: (accent colour, notes, sign)
    "A_major": ((255, 99, 99), "A  C#  E", "1 finger"),
    "B_major": ((255, 159, 67), "B  D#  F#", "2 fingers"),
    "C_major": ((250, 204, 21), "C  E  G", "3 fingers"),
    "D_major": ((72, 199, 142), "D  F#  A", "4 fingers"),
    "E_major": ((56, 189, 248), "E  G#  B", "5 fingers"),
    "F_major": ((129, 140, 248), "F  A  C", "index + pinky"),
    "G_major": ((217, 70, 239), "G  B  D", "love you sign"),
}
SHADOW = 6

# ---- effects state (particles + last chord, kept between frames) ------------
_particles: list[list] = []     # [x, y, vx, vy, life, (r, g, b)]
_last_chord: list = [None]

_FONT_CACHE: dict = {}


def _font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    key = (weight, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    names = {
        "bold": [HERE / "fonts" / "Poppins-Bold.ttf", "C:/Windows/Fonts/segoeuib.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        "medium": [HERE / "fonts" / "Poppins-Medium.ttf", "C:/Windows/Fonts/seguisb.ttf",
                   "C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    }[weight]
    font = None
    for path in names:
        try:
            font = ImageFont.truetype(str(path), size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default(size)
    _FONT_CACHE[key] = font
    return font


def _card(d: ImageDraw.ImageDraw, box, fill=CARD, shadow=SHADOW, radius=14):
    """Rounded rectangle with a solid offset shadow and a dark outline."""
    x0, y0, x1, y1 = box
    d.rounded_rectangle((x0 + shadow, y0 + shadow, x1 + shadow, y1 + shadow), radius, fill=INK)
    d.rounded_rectangle(box, radius, fill=fill, outline=INK, width=3)


def _center(d, box, text, font, fill, dy=0):
    x0, y0, x1, y1 = box
    d.text(((x0 + x1) / 2, (y0 + y1) / 2 + dy), text, font=font, fill=fill, anchor="mm")


def draw_ui(frame_bgr, shown: str, confidence: float, volume: float, max_volume: float = 1.0,
            quality: str = "major", hint: str = "Hand sign = chord  ·  fist = mute",
            style_text: str | None = None, timbre: str | None = None):
    """Draw the interface on frame_bgr (modified in place) and return it.

    quality    : "major" / "minor" / "seventh" (shown on the big chord card)
    style_text : e.g. "Left hand: minor"; None hides the chip
    timbre     : name of the current sound, e.g. "Pad"; None hides the chip
    """
    h, w = frame_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    info = CHORD_INFO.get(shown)
    accent = info[0] if info else (170, 170, 190)

    # --- title card (top left) ---------------------------------------------
    _card(d, (20, 20, 270, 78))
    d.rectangle((20 + 3, 20 + 3, 20 + 16, 78 - 3), fill=accent)
    _center(d, (36, 20, 270, 78), "Handchord", _font("bold", 30), INK)

    # --- hint chip under the title ----------------------------------------------
    chip_w = int(d.textlength(hint, font=_font("medium", 16))) + 28
    _card(d, (20, 96, 20 + chip_w, 134), shadow=4, radius=10)
    _center(d, (20, 96, 20 + chip_w, 134), hint, _font("medium", 16), INK)
    y_chip = 150
    for text, fill in ((style_text, accent), (f"Sound: {timbre}  (press T)" if timbre else None, CARD)):
        if text:
            tw = int(d.textlength(text, font=_font("bold", 16))) + 28
            _card(d, (20, y_chip, 20 + tw, y_chip + 38), fill=fill, shadow=4, radius=10)
            _center(d, (20, y_chip, 20 + tw, y_chip + 38), text, _font("bold", 16), INK)
            y_chip += 52

    # --- big chord card (top right) -------------------------------------------
    cx0, cy0, cx1, cy1 = w - 230, 20, w - 24, 168
    _card(d, (cx0, cy0, cx1, cy1), fill=accent if info else CARD)
    if info:
        _center(d, (cx0, cy0, cx0 + 110, cy1 - 30), shown[0], _font("bold", 84), INK, dy=4)
        d.text((cx0 + 112, cy0 + 56), QUALITY_LABEL.get(quality, quality), font=_font("bold", 22), fill=INK, anchor="lm")
        d.text((cx0 + 112, cy0 + 86), "  ".join(chord_note_names(shown[0], quality)), font=_font("medium", 17), fill=INK, anchor="lm")
        d.text(((cx0 + cx1) / 2, cy1 - 20), info[2], font=_font("medium", 15), fill=INK, anchor="mm")
    elif shown == "no hand":
        _center(d, (cx0, cy0, cx1, cy1), "--", _font("bold", 60), MUTED)
    else:
        _center(d, (cx0, cy0, cx1, cy1 - 24), "Silence", _font("bold", 30), MUTED)
        _center(d, (cx0, cy0 + 40, cx1, cy1), "fist = mute", _font("medium", 16), MUTED)

    # --- vol / confidence pills (under chord card) ------------------------------
    level = float(np.clip(volume / max(max_volume, 1e-6), 0.0, 1.0))
    for i, (label, value) in enumerate((("Vol", level), ("Conf", confidence))):
        y = 196 + i * 50
        _card(d, (cx0, y, cx1, y + 38), shadow=4, radius=10)
        d.text((cx0 + 14, y + 19), label, font=_font("bold", 15), fill=INK, anchor="lm")
        bx0, bx1 = cx0 + 66, cx1 - 14
        d.rounded_rectangle((bx0, y + 12, bx1, y + 26), 6, fill=(225, 220, 235), outline=INK, width=2)
        fill_x = bx0 + max(float(np.clip(value, 0, 1)), 0.0) * (bx1 - bx0)
        if value > 0.02:
            d.rounded_rectangle((bx0, y + 12, max(fill_x, bx0 + 10), y + 26), 6, fill=accent, outline=INK, width=2)

    # --- bottom row: 7 chord cards ------------------------------------------------
    n, gap, margin = len(CHORD_INFO), 12, 24
    card_w = (w - 2 * margin - gap * (n - 1)) / n
    top, bottom = h - 120, h - 26
    burst_from = None
    for i, (key, (color, _notes, sign)) in enumerate(CHORD_INFO.items()):
        x0 = margin + i * (card_w + gap)
        x1 = x0 + card_w
        active = key == shown
        lift = 12 if active else 0
        box = (x0, top - lift, x1, bottom - lift)
        if active:
            burst_from = ((x0 + x1) / 2, top - lift)
        _card(d, box, fill=color if active else CARD, shadow=8 if active else SHADOW)
        if not active:
            d.rectangle((x0 + 3, top + 3, x1 - 3, top + 12), fill=color)  # colour tab
        _center(d, (box[0], box[1], box[2], box[3] - 22), key[0], _font("bold", 38), INK, dy=2)
        d.text(((box[0] + box[2]) / 2, box[3] - 20), sign, font=_font("medium", 13), fill=INK, anchor="mm")

    # --- effects: bouncing bars + particles that burst when the chord changes ------------
    now = time.time()
    fx = ImageDraw.Draw(img, "RGBA")
    bar_w, bar_gap, base_y = 9, 5, y_chip + 70
    for i in range(24):
        wobble = 0.25 + 0.75 * abs(math.sin(now * 3.2 + i * 0.7)) * (0.6 + 0.4 * math.sin(now * 1.3 + i))
        bar_h = 8 + 55 * level * wobble   # always taller than the corner radius
        x = 24 + i * (bar_w + bar_gap)
        fx.rounded_rectangle((x, base_y - bar_h, x + bar_w, base_y), 3, fill=accent + (235,), outline=INK + (255,), width=1)

    if info and shown != _last_chord[0] and burst_from:
        for _ in range(28):
            angle = random.uniform(-math.pi * 0.9, -math.pi * 0.1)      # upwards fan
            speed = random.uniform(3.0, 9.0)
            _particles.append([burst_from[0], burst_from[1], math.cos(angle) * speed, math.sin(angle) * speed,
                               random.randint(22, 42), accent])
    _last_chord[0] = shown if info else None
    for p in _particles:
        p[0] += p[2]
        p[1] += p[3]
        p[3] += 0.12
        p[4] -= 1
        alpha = int(255 * max(p[4], 0) / 42)
        radius = 3 + p[4] / 9
        fx.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius), fill=p[5] + (alpha,))
    _particles[:] = [p for p in _particles if p[4] > 0][:400]

    frame_bgr[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return frame_bgr