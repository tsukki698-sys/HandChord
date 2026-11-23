"""Handchord Studio: the full instrument.

  Left hand  = chord (A-G, fist = mute)        Right hand = style (major/minor/seventh) + volume (height)

  * smooth switching : quick, stable decisions + crossfade between chords (no gaps, no clicks)
  * 11 instruments   : "Instrument" menu at the bottom, or press T
  * visualizers      : "Visual" button, or V   (Bars, Wave, Rings, Burst, Off)
  * themes           : "Theme" button, or C
  * ?  = help (gestures and keys),  D = show FPS,  Q / ESC = quit
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pygame
import tensorflow as tf

from common import (
    CHORD_HAND, CHORD_MIRROR, ROOT, STYLE_HAND, Clock, create_landmarker, detect_hands, draw_hand, open_camera, to_features,
)
from sound import INSTRUMENTS, PERCUSSIVE, SAMPLE_RATE, SoundBank
from studio_ui import CAM_H, CAM_W, STRIP_H, render
from ui import CHORD_INFO
from visual import Visualizer, theme_color

ROOT_MODEL = ROOT / "models" / "gesture_tf.keras"
ROOT_CLASSES = ROOT / "models" / "gesture_tf_classes.json"
STYLE_MODEL = ROOT / "models" / "style_tf.keras"
STYLE_CLASSES = ROOT / "models" / "style_tf_classes.json"

MIN_CONFIDENCE = 0.45     # below this a guess counts as "not sure"
HOLD_FRAMES = 3           # a new chord must be seen this many frames in a row (2 if very sure)
SURE = 0.90
MAX_VOLUME = 0.8
DEFAULT_VOLUME = 0.5
WINDOW = "Handchord Studio"

# crossfade times in milliseconds: (new sound fades in, old sound fades out)
FADE_PERCUSSIVE = (3, 140)   # piano / guitar ...: a crisp attack, the old note rings out
FADE_SUSTAINED = (60, 70)    # pad / organ ...: smooth blend

HAND_COLOR_CHORD = (66, 200, 245)    # BGR: gold
HAND_COLOR_STYLE = (238, 243, 245)   # BGR: soft white

# the dark band behind the visualizer strip (so the bars stay readable, but quiet)
_ramp = np.linspace(0.0, 0.55, STRIP_H, dtype=np.float32) ** 1.2
STRIP_KEEP = (1.0 - _ramp)[:, None, None]


def height_to_volume(palm_y: float) -> float:
    t = float(np.clip((0.95 - palm_y) / (0.95 - 0.30), 0.0, 1.0))
    return MAX_VOLUME * (0.15 + 0.85 * t)


class Stabilizer:
    """Turns noisy per-frame guesses into a steady decision, with very little delay.
    A new value is accepted after HOLD_FRAMES frames in a row (2 if the model is very sure)."""

    def __init__(self, start):
        self.value = start
        self._cand = start
        self._n = 0

    def update(self, guess, conf=1.0):
        if guess == self.value:
            self._cand, self._n = guess, 0
            return self.value
        if guess == self._cand:
            self._n += 1
        else:
            self._cand, self._n = guess, 1
        if self._n >= (2 if conf >= SURE else HOLD_FRAMES):
            self.value, self._n = guess, 0
        return self.value


def make_predictor(model):
    @tf.function(reduce_retracing=True)
    def run(x):
        return model(x, training=False)
    return lambda x: run(tf.constant(x, dtype=tf.float32)).numpy()


def predict_chord(run, classes, pts):
    """Both the normal and the mirrored hand in ONE model call; keep the more confident."""
    batch = np.stack([to_features(pts, False), to_features(pts, True)]) if CHORD_MIRROR == "auto" \
        else to_features(pts, bool(CHORD_MIRROR))[None, :]
    probs = run(batch)
    row = int(np.argmax(probs.max(axis=1)))
    best = int(np.argmax(probs[row]))
    return classes[best], float(probs[row][best])


def main() -> None:
    root_classes = json.loads(ROOT_CLASSES.read_text())
    root_run = make_predictor(tf.keras.models.load_model(ROOT_MODEL))
    style_run = style_classes = None
    if STYLE_MODEL.exists():
        style_classes = json.loads(STYLE_CLASSES.read_text())
        style_run = make_predictor(tf.keras.models.load_model(STYLE_MODEL))
    else:
        print("No style model yet: chords will be plain major.")
    root_run(np.zeros((2, 63), np.float32))                       # warm up: first call is slow
    if style_run:
        style_run(np.zeros((1, 63), np.float32))

    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(4)
    channels = [pygame.mixer.Channel(0), pygame.mixer.Channel(1)]
    print("Preparing the first sound...")
    bank = SoundBank()
    bank.start_background(INSTRUMENTS[0])                         # others render when you pick them
    pg_sounds: dict = {}

    def get_sound(key):
        if key not in pg_sounds:
            arr = bank.array(*key)
            pg_sounds[key] = pygame.sndarray.make_sound(np.ascontiguousarray(arr)) if arr is not None else None
        return pg_sounds[key]

    viz = Visualizer()
    landmarker = create_landmarker(num_hands=2)
    cap = open_camera()
    clock = Clock()
    root_stab, style_stab = Stabilizer("no hand"), Stabilizer("major")
    wanted_instrument = instrument = INSTRUMENTS[0]
    hitboxes: list = []
    clicks: list = []
    mouse = [-1, -1]
    pressed_name, pressed_until = None, 0.0
    menu_open = help_open = show_fps = False
    burst_x = CAM_W // 2
    toast, toast_until = None, 0.0
    sounding = None            # (letter, quality, instrument) currently playing
    sound_start = 0.0
    active_ch = 0
    volume = 0.0
    fps, last = 30.0, time.time()

    def on_mouse(event, x, y, _flags, _param):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    def do(action: str):
        nonlocal wanted_instrument, menu_open, help_open
        if action.startswith("inst:"):
            wanted_instrument = action[5:]
            bank.request(wanted_instrument)
            menu_open = False
        elif action == "next_inst":
            wanted_instrument = INSTRUMENTS[(INSTRUMENTS.index(wanted_instrument) + 1) % len(INSTRUMENTS)]
            bank.request(wanted_instrument)
        elif action == "instrument":
            menu_open = not menu_open
        elif action == "visual":
            viz.next_mode()
        elif action == "theme":
            viz.next_theme()
        elif action == "help":
            help_open, menu_open = not help_open, False
        elif action == "help_close":
            help_open = False

    print(f"Chord hand: {CHORD_HAND}   Style hand: {STYLE_HAND}.  Click the buttons or use the keys. ? = help, Q = quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            fps += (1.0 / max(now - last, 1e-3) - fps) * 0.1
            last = now
            frame = cv2.flip(frame, 1)
            hands = detect_hands(landmarker, frame, clock.next())
            chord_pts, style_pts = hands.get(CHORD_HAND), hands.get(STYLE_HAND)

            if bank.ready(wanted_instrument):          # an instrument becomes active when rendered
                instrument = wanted_instrument

            # ---- chord hand ------------------------------------------------------------------
            confidence = 0.0
            if chord_pts is None:
                shown = root_stab.update("no hand")
            else:
                draw_hand(frame, chord_pts, HAND_COLOR_CHORD)
                guess, confidence = predict_chord(root_run, root_classes, chord_pts)
                shown = root_stab.update(guess if confidence >= MIN_CONFIDENCE else "rest", confidence)

            # ---- style hand ------------------------------------------------------------------
            style_conf, target_volume = 0.0, DEFAULT_VOLUME * MAX_VOLUME
            if style_pts is not None:
                draw_hand(frame, style_pts, HAND_COLOR_STYLE)
                target_volume = height_to_volume(float(style_pts[9, 1]))
                if style_run is not None:
                    probs = style_run(to_features(style_pts)[None, :])[0]
                    best = int(np.argmax(probs))
                    style_conf = float(probs[best])
                    style_stab.update(style_classes[best] if style_conf >= MIN_CONFIDENCE else "major", style_conf)
            else:
                style_stab.update("major")
            quality = style_stab.value

            # ---- mouse: hover, clicks -------------------------------------------------------------
            def hit_at(x, y):
                for name, (x0, y0, x1, y1) in hitboxes:
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        return name
                return None
            while clicks:
                name = hit_at(*clicks.pop())
                if menu_open and not (name and (name.startswith("inst:") or name in ("menu_bg", "instrument"))):
                    menu_open = False
                if name and name not in ("menu_bg", "noop"):
                    do(name)
                    pressed_name, pressed_until = name, now + 0.12
            hover = hit_at(*mouse)

            # ---- sound with crossfade ------------------------------------------------------------------
            wanted = (shown[0], quality, instrument) if shown.endswith("_major") else None
            if wanted != sounding:
                fade_in, _ = FADE_PERCUSSIVE if (wanted and wanted[2] in PERCUSSIVE) else FADE_SUSTAINED
                _, fade_out = FADE_PERCUSSIVE if (sounding and sounding[2] in PERCUSSIVE) else FADE_SUSTAINED
                if sounding is not None:
                    channels[active_ch].fadeout(fade_out)
                snd = get_sound(wanted) if wanted else None
                if snd is not None:
                    active_ch = 1 - active_ch
                    channels[active_ch].set_volume(volume)
                    channels[active_ch].play(snd, loops=-1, fade_ms=fade_in)
                    sound_start = now
                    sounding = wanted
                else:
                    sounding = None
            volume += ((target_volume if sounding else 0.0) - volume) * 0.3
            channels[active_ch].set_volume(volume)

            # ---- picture: camera -> visualizer strip -> interface ---------------------------------------------
            cam = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
            accent = CHORD_INFO[shown][0] if shown in CHORD_INFO else (245, 200, 66)
            if sounding:
                arr = bank.array(*sounding)
                viz.update(arr, int((now - sound_start) * SAMPLE_RATE) % len(arr), volume / MAX_VOLUME)
            else:
                viz.update(None, 0, 0.0)
            if viz.mode != "Off":
                strip = cam[CAM_H - STRIP_H:, :]
                strip[:] = (strip * STRIP_KEEP).astype(np.uint8)
                viz.draw(strip, accent, shown[0] if sounding else None, (burst_x, STRIP_H - 8), strip=True)

            if toast and now > toast_until:
                toast = None
            if chord_pts is None:
                attention = f"Show your {CHORD_HAND.lower()} hand to play a chord"
            elif style_pts is None and style_run is not None:
                attention = f"Show your {STYLE_HAND.lower()} hand to choose a style"
            else:
                attention = None
            state = dict(
                shown=shown, confidence=confidence, volume=volume, max_volume=MAX_VOLUME, quality=quality,
                instrument=instrument, instruments=INSTRUMENTS, ready={i for i in INSTRUMENTS if bank.ready(i)},
                loading=bank.loading, toast=toast, attention=attention,
                chord_hand=CHORD_HAND, style_hand=STYLE_HAND,
                chord_seen=chord_pts is not None, style_seen=style_pts is not None,
                style_name=quality, style_conf=style_conf,
                visual_mode=viz.mode, theme_name=viz.theme,
                theme_colors=[theme_color(viz.theme, accent, u) for u in (0.0, 0.5, 1.0)],
                menu_open=menu_open, help_open=help_open, hover=hover,
                pressed=pressed_name if now < pressed_until else None,
                fps=fps if show_fps else None,
            )
            canvas, hitboxes, extras = render(cam, state)
            burst_x = extras["burst_x"]
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 27:                       # ESC closes the open panel first, then quits
                if help_open or menu_open:
                    help_open = menu_open = False
                else:
                    break
            for k, action in ((ord("t"), "next_inst"), (ord("i"), "next_inst"), (ord("v"), "visual"),
                              (ord("c"), "theme"), (ord("?"), "help"), (ord("h"), "help")):
                if key == k:
                    do(action)
            if key == ord("d"):
                show_fps = not show_fps
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        pygame.mixer.quit()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()