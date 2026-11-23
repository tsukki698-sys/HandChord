"""Interface of Handchord Studio (v2): a calm, dark "music studio" look.

The camera is the main canvas. Around it: a top bar, a right column (now playing, hands,
levels), a slim A-G chord row and a control row at the bottom.

render(cam_bgr, state) -> (canvas_bgr, hitboxes, extras)
  cam_bgr   : the camera picture, already CAM_W x CAM_H (the visualizer is drawn on it before)
  hitboxes  : [(name, (x0, y0, x1, y1)), ...]  first match wins; used by the mouse handler
  extras    : {"burst_x": x of the active chord, in camera coordinates}

All colours, sizes and positions are the constants right below. Change them freely.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import STYLE_HINTS
from theory import QUALITY_LABEL, chord_note_names
from ui import CHORD_INFO, _font

# ---- palette (RGB) ------------------------------------------------------------------
BG = (23, 24, 29)           # #17181D
PANEL = (34, 36, 43)        # #22242B
PANEL_HOVER = (43, 46, 55)
PANEL_PRESSED = (28, 30, 36)
BORDER = (56, 58, 68)
TEXT = (245, 243, 238)      # #F5F3EE
TEXT2 = (166, 168, 178)     # #A6A8B2
GOLD = (245, 200, 66)       # #F5C842
GREEN = (88, 201, 154)      # #58C99A
AMBER = (240, 169, 75)      # #F0A94B
TRACK = (48, 50, 59)

# ---- layout (pixels) -------------------------------------------------------------------
W, H = 1180, 760
PAD, TOP = 16, 52
CAM_X, CAM_Y = PAD, 60
CHORD_H, CTRL_H = 52, 44
CTRL_Y = H - PAD - CTRL_H                   # 700
CHORD_Y = CTRL_Y - 10 - CHORD_H             # 638
CAM_H = CHORD_Y - 12 - CAM_Y                # 566
CAM_W = CAM_H * 4 // 3                      # 754  (the webcam is 4:3: nothing is cropped)
COL_X0 = CAM_X + CAM_W + PAD                # right column
COL_X1 = W - PAD
STRIP_H = 104                               # height of the visualizer strip at the bottom of the camera

RADIUS = 14
_mask_cache: dict = {}
_glow_cache: dict = {}


# ---- small drawing helpers -------------------------------------------------------------
def _rounded_mask(size, radius):
    key = (size, radius)
    if key not in _mask_cache:
        m = Image.new("L", size, 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius, fill=255)
        _mask_cache[key] = m
    return _mask_cache[key]


def glass(img, box, radius, fill, alpha, outline=None):
    """Semi-transparent rounded panel (used on top of the camera and for floating panels)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    crop = img.crop((x0, y0, x1, y1)).convert("RGBA")
    layer = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(
        (0, 0, x1 - x0 - 1, y1 - y0 - 1), radius, fill=(*fill, int(255 * alpha)),
        outline=None if outline is None else (*outline, 255), width=1)
    img.paste(Image.alpha_composite(crop, layer).convert("RGB"), (x0, y0))


def shadow(img, box, radius=RADIUS):
    x0, y0, x1, y1 = box
    for grow, a in ((10, 0.10), (6, 0.14), (3, 0.18)):
        glass(img, (x0 - grow, y0 - grow + 6, x1 + grow, y1 + grow + 6), radius + grow, (0, 0, 0), a)


def glow_mask(size):
    if size not in _glow_cache:
        yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
        r = np.clip(1 - np.sqrt(xx ** 2 + yy ** 2), 0, 1)
        _glow_cache[size] = Image.fromarray((255 * r ** 2).astype(np.uint8), "L")
    return _glow_cache[size]


def _text_w(d, text, font):
    return int(d.textlength(text, font=font))


def _button(d, boxes, name, box, state, *, fill=PANEL, outline=BORDER):
    """Panel-style button with a hover and a pressed look. Returns (box, pressed) for the caller."""
    x0, y0, x1, y1 = box
    hover = state.get("hover") == name
    pressed = state.get("pressed") == name
    f = PANEL_PRESSED if pressed else (PANEL_HOVER if hover else fill)
    o = GOLD if (hover or pressed) and fill == PANEL else outline
    d.rounded_rectangle(box, RADIUS - 2, fill=f, outline=o, width=1)
    boxes.append((name, box))
    return 1 if pressed else 0


def _panel(d, box, title=None):
    d.rounded_rectangle(box, RADIUS, fill=PANEL, outline=BORDER, width=1)
    if title:
        d.text((box[0] + 20, box[1] + 22), title, font=_font("medium", 12), fill=TEXT2, anchor="lm")


def _bar(d, x0, x1, y, frac, color=GOLD):
    d.rounded_rectangle((x0, y, x1, y + 6), 3, fill=TRACK)
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac > 0.01:
        d.rounded_rectangle((x0, y, x0 + max(6, int((x1 - x0) * frac)), y + 6), 3, fill=color)


def _dot(d, x, y, color, r=5):
    d.ellipse((x - r, y - r, x + r, y + r), fill=color)


# ---- the main function -------------------------------------------------------------------
def render(cam_bgr, s: dict):
    img = Image.new("RGB", (W, H), BG)
    cam = Image.fromarray(cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB))
    img.paste(cam, (CAM_X, CAM_Y), _rounded_mask((CAM_W, CAM_H), RADIUS))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((CAM_X, CAM_Y, CAM_X + CAM_W - 1, CAM_Y + CAM_H - 1), RADIUS, outline=BORDER, width=1)
    boxes: list = []
    shown, quality = s["shown"], s["quality"]
    playing = shown in CHORD_INFO
    letter = shown[0] if playing else None

    # ---- top bar ---------------------------------------------------------------------
    d.ellipse((PAD, 17, PAD + 18, 35), fill=GOLD)
    d.text((PAD + 30, 26), "Handchord Studio", font=_font("medium", 20), fill=TEXT, anchor="lm")
    hb = (W - PAD - 36, 8, W - PAD, 44)
    pr = _button(d, boxes, "help", hb, s)
    d.text(((hb[0] + hb[2]) // 2, (hb[1] + hb[3]) // 2 + pr), "?", font=_font("medium", 17), fill=TEXT, anchor="mm")
    live = f"Live  ·  {s['fps']:.0f} fps" if s.get("fps") else "Live"
    lw = _text_w(d, live, _font("medium", 15)) + 44
    lb = (hb[0] - 12 - lw, 10, hb[0] - 12, 42)
    d.rounded_rectangle(lb, 16, fill=PANEL, outline=BORDER, width=1)
    _dot(d, lb[0] + 18, 26, GREEN)
    d.text((lb[0] + 32, 26), live, font=_font("medium", 15), fill=TEXT, anchor="lm")

    # ---- overlays on the camera (small, in the corners) ------------------------------------
    message = s.get("toast") or (f"Loading {s['loading']} ..." if s.get("loading") else None)
    if message:
        mw = _text_w(d, message, _font("medium", 15)) + 36
        x0 = CAM_X + (CAM_W - mw) // 2
        glass(img, (x0, CAM_Y + 12, x0 + mw, CAM_Y + 46), 17, BG, 0.78, GOLD)
        d.text((x0 + mw // 2, CAM_Y + 29), message, font=_font("medium", 15), fill=TEXT, anchor="mm")
    if s.get("attention"):
        tw = _text_w(d, s["attention"], _font("medium", 13)) + 44
        glass(img, (CAM_X + 12, CAM_Y + 12, CAM_X + 12 + tw, CAM_Y + 42), 15, BG, 0.72)
        _dot(d, CAM_X + 28, CAM_Y + 27, AMBER, 4)
        d.text((CAM_X + 40, CAM_Y + 27), s["attention"], font=_font("medium", 13), fill=TEXT, anchor="lm")

    # ---- right column ------------------------------------------------------------------------
    now_h, hands_h, level_h, gap = None, 208, 134, 14
    now_h = (H - PAD - CAM_Y) - hands_h - level_h - 2 * gap
    cx = (COL_X0 + COL_X1) // 2

    # NOW PLAYING
    box = (COL_X0, CAM_Y, COL_X1, CAM_Y + now_h)
    _panel(d, box, "NOW PLAYING")
    if playing:
        d.rounded_rectangle((COL_X0 + 1, CAM_Y + 22, COL_X0 + 5, CAM_Y + now_h - 22), 2, fill=GOLD)
        vol = float(np.clip(s["volume"] / max(s["max_volume"], 1e-6), 0.0, 1.0))
        size = 260
        mask = glow_mask(size).point(lambda v, k=0.20 + 0.30 * vol: int(v * k))
        img.paste(Image.new("RGB", (size, size), GOLD), (cx - size // 2, CAM_Y + 118 - size // 2), mask)
        d.text((cx, CAM_Y + 118), letter, font=_font("bold", 120), fill=TEXT, anchor="mm")
        d.text((cx, CAM_Y + 206), QUALITY_LABEL.get(quality, quality), font=_font("medium", 20), fill=TEXT, anchor="mm")
        d.text((cx, CAM_Y + 238), "  ·  ".join(chord_note_names(letter, quality)), font=_font("medium", 15), fill=GOLD, anchor="mm")
        d.text((cx, CAM_Y + 268), CHORD_INFO[shown][2] + "  ·  " + s["instrument"], font=_font("medium", 12), fill=TEXT2, anchor="mm")
    else:
        d.text((cx, CAM_Y + 118), "—", font=_font("bold", 96), fill=TRACK, anchor="mm")
        line = "Fist = mute" if shown == "rest" else f"Show your {s['chord_hand'].lower()} hand"
        d.text((cx, CAM_Y + 206), line, font=_font("medium", 16), fill=TEXT2, anchor="mm")
        d.text((cx, CAM_Y + 236), "to play a chord" if shown != "rest" else "open it to play", font=_font("medium", 13), fill=TEXT2, anchor="mm")

    # HANDS
    y0 = CAM_Y + now_h + gap
    _panel(d, (COL_X0, y0, COL_X1, y0 + hands_h), "HANDS")
    rows = (
        (s["chord_hand"], s["chord_seen"],
         f"{s['chord_hand']} hand  ·  " + (f"{letter} chord" if playing else ("mute" if shown == "rest" else "…")),
         f"{CHORD_INFO[shown][2]}  ·  {s['confidence']:.0%} sure" if playing else ("closed fist" if shown == "rest" else ""),
         f"Show your {s['chord_hand'].lower()} hand to play a chord"),
        (s["style_hand"], s["style_seen"],
         f"{s['style_hand']} hand  ·  {QUALITY_LABEL.get(s['style_name'], s['style_name'])}",
         f"{STYLE_HINTS.get(s['style_name'], '')}  ·  height = volume",
         f"Show your {s['style_hand'].lower()} hand to choose a style"),
    )
    for i, (_name, seen, title, detail, missing) in enumerate(rows):
        ry = y0 + 46 + i * 78
        _dot(d, COL_X0 + 26, ry + 12, GREEN if seen else AMBER)
        if seen:
            d.text((COL_X0 + 44, ry + 12), title, font=_font("medium", 15), fill=TEXT, anchor="lm")
            d.text((COL_X0 + 44, ry + 38), detail, font=_font("medium", 12), fill=TEXT2, anchor="lm")
        else:
            d.text((COL_X0 + 44, ry + 12), f"{rows[i][0]} hand  ·  not seen", font=_font("medium", 15), fill=TEXT, anchor="lm")
            d.text((COL_X0 + 44, ry + 38), missing, font=_font("medium", 12), fill=AMBER, anchor="lm")

    # LEVELS
    y1 = y0 + hands_h + gap
    _panel(d, (COL_X0, y1, COL_X1, y1 + level_h), "LEVELS")
    level = float(np.clip(s["volume"] / max(s["max_volume"], 1e-6), 0.0, 1.0))
    for i, (label, value) in enumerate((("Volume", level), ("Confidence", s["confidence"]))):
        ry = y1 + 48 + i * 40
        d.text((COL_X0 + 20, ry + 4), label, font=_font("medium", 13), fill=TEXT2, anchor="lm")
        d.text((COL_X1 - 20, ry + 4), f"{int(value * 100)}%", font=_font("medium", 13), fill=TEXT, anchor="rm")
        _bar(d, COL_X0 + 20, COL_X1 - 20, ry + 18, value)

    # ---- chord row (slim) -----------------------------------------------------------------------
    n, gap_c = len(CHORD_INFO), 8
    bw = (CAM_W - gap_c * (n - 1)) / n
    burst_x = CAM_W // 2
    for i, (key, (_color, _notes, sign)) in enumerate(CHORD_INFO.items()):
        x0 = CAM_X + i * (bw + gap_c)
        box = (int(x0), CHORD_Y, int(x0 + bw), CHORD_Y + CHORD_H)
        active = key == shown
        if active:
            burst_x = (box[0] + box[2]) // 2 - CAM_X
            d.rounded_rectangle(box, RADIUS - 2, fill=GOLD)
            lc, sc = BG, (70, 58, 20)
        else:
            d.rounded_rectangle(box, RADIUS - 2, fill=PANEL, outline=BORDER, width=1)
            lc, sc = TEXT, TEXT2
        mx = (box[0] + box[2]) // 2
        d.text((mx, CHORD_Y + 19), key[0], font=_font("bold" if active else "medium", 20), fill=lc, anchor="mm")
        d.text((mx, CHORD_Y + 39), sign, font=_font("medium", 12), fill=sc, anchor="mm")

    # ---- control row -----------------------------------------------------------------------------
    x = CAM_X
    items = (
        ("visual", "Visual", s["visual_mode"], 190, None),
        ("theme", "Theme", s["theme_name"], 230, s["theme_colors"]),
        ("instrument", "Instrument", s["instrument"], 250, "chevron"),
    )
    inst_box = None
    for name, label, value, width, extra in items:
        box = (x, CTRL_Y, x + width, CTRL_Y + CTRL_H)
        dy = _button(d, boxes, name, box, s)
        if name == "instrument":
            inst_box = box
            if s.get("menu_open"):
                d.rounded_rectangle(box, RADIUS - 2, outline=GOLD, width=1)
        ty = CTRL_Y + CTRL_H // 2 + dy
        d.text((x + 16, ty), label, font=_font("medium", 12), fill=TEXT2, anchor="lm")
        vx = x + 16 + _text_w(d, label, _font("medium", 12)) + 10
        d.text((vx, ty), value, font=_font("medium", 15), fill=TEXT, anchor="lm")
        if extra == "chevron":
            cxr, cyr = x + width - 22, ty
            d.polygon([(cxr - 5, cyr - 3), (cxr + 5, cyr - 3), (cxr, cyr + 3)] if not s.get("menu_open")
                      else [(cxr - 5, cyr + 3), (cxr + 5, cyr + 3), (cxr, cyr - 3)], fill=TEXT2)
        elif extra:
            for k, color in enumerate(extra):
                _dot(d, x + width - 24 - (2 - k) * 14, ty, color, 5)
        x += width + 8

    # ---- instrument menu (floating) -------------------------------------------------------------------
    if s.get("menu_open") and inst_box:
        names = s["instruments"]
        rows_n = (len(names) + 1) // 2
        mw, ih = 300, 34
        mh = rows_n * ih + 20
        mb = (inst_box[0], inst_box[1] - 8 - mh, inst_box[0] + mw, inst_box[1] - 8)
        shadow(img, mb)
        d.rounded_rectangle(mb, RADIUS, fill=PANEL, outline=BORDER, width=1)
        iw = (mw - 20 - 6) // 2
        for i, name in enumerate(names):
            ix = mb[0] + 10 + (i % 2) * (iw + 6)
            iy = mb[1] + 10 + (i // 2) * ih
            b = (ix, iy, ix + iw, iy + ih - 4)
            active = name == s["instrument"]
            hover = s.get("hover") == "inst:" + name
            if active:
                d.rounded_rectangle(b, 9, fill=GOLD)
            elif hover:
                d.rounded_rectangle(b, 9, fill=PANEL_HOVER)
            ready = name in s["ready"]
            color = BG if active else (TEXT if ready else TEXT2)
            d.text((b[0] + 14, (b[1] + b[3]) // 2), name, font=_font("bold" if active else "medium", 14), fill=color, anchor="lm")
            if not ready and not active:
                d.text((b[2] - 12, (b[1] + b[3]) // 2), "·", font=_font("bold", 16), fill=TEXT2, anchor="rm")
            boxes.append(("inst:" + name, b))
        boxes.append(("menu_bg", mb))          # after the items: items win

    # ---- help (modal) -----------------------------------------------------------------------------------
    if s.get("help_open"):
        veil = Image.new("RGB", (W, H), (0, 0, 0))
        img.paste(Image.blend(img, veil, 0.62))
        d = ImageDraw.Draw(img)
        pw, ph = 700, 470
        pb = ((W - pw) // 2, (H - ph) // 2, (W + pw) // 2, (H + ph) // 2)
        shadow(img, pb)
        d.rounded_rectangle(pb, RADIUS + 2, fill=PANEL, outline=BORDER, width=1)
        d.text((pb[0] + 32, pb[1] + 38), "How to play", font=_font("medium", 20), fill=TEXT, anchor="lm")
        d.text((pb[0] + 32, pb[1] + 78), f"{s['chord_hand'].upper()} HAND  ·  CHORD", font=_font("medium", 12), fill=GOLD, anchor="lm")
        for i, (key, (_c, notes, sign)) in enumerate(CHORD_INFO.items()):
            ry = pb[1] + 110 + i * 30
            d.text((pb[0] + 32, ry), key[0], font=_font("bold", 16), fill=TEXT, anchor="lm")
            d.text((pb[0] + 62, ry), sign, font=_font("medium", 14), fill=TEXT2, anchor="lm")
        d.text((pb[0] + 32, pb[1] + 110 + 7 * 30), "—", font=_font("bold", 16), fill=TEXT, anchor="lm")
        d.text((pb[0] + 62, pb[1] + 110 + 7 * 30), "closed fist = mute", font=_font("medium", 14), fill=TEXT2, anchor="lm")
        rx = pb[0] + 380
        d.text((rx, pb[1] + 78), f"{s['style_hand'].upper()} HAND  ·  STYLE", font=_font("medium", 12), fill=GOLD, anchor="lm")
        for i, (name, hint) in enumerate(STYLE_HINTS.items()):
            ry = pb[1] + 110 + i * 30
            d.text((rx, ry), QUALITY_LABEL.get(name, name), font=_font("bold", 16), fill=TEXT, anchor="lm")
            d.text((rx + 80, ry), hint, font=_font("medium", 14), fill=TEXT2, anchor="lm")
        d.text((rx, pb[1] + 110 + 3 * 30), "Height", font=_font("bold", 16), fill=TEXT, anchor="lm")
        d.text((rx + 80, pb[1] + 110 + 3 * 30), "higher hand = louder", font=_font("medium", 14), fill=TEXT2, anchor="lm")
        d.text((rx, pb[1] + 270), "KEYS", font=_font("medium", 12), fill=GOLD, anchor="lm")
        for i, line in enumerate(("T   next instrument", "V   visual style", "C   colour theme", "?   this help", "D   show FPS", "Q   quit")):
            d.text((rx, pb[1] + 300 + i * 24), line, font=_font("medium", 14), fill=TEXT2, anchor="lm")
        cb = (pb[2] - 32 - 120, pb[3] - 32 - 40, pb[2] - 32, pb[3] - 32)
        d.rounded_rectangle(cb, 12, fill=GOLD)
        d.text(((cb[0] + cb[2]) // 2, (cb[1] + cb[3]) // 2), "Got it", font=_font("bold", 15), fill=BG, anchor="mm")
        boxes[:] = [("help_close", cb), ("noop", pb), ("help_close", (0, 0, W, H))]

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), boxes, {"burst_x": burst_x}