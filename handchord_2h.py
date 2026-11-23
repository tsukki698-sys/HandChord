"""HANDCHORD, two-hand version.

  CHORD hand  -> model 1 (gesture_tf.keras)  -> which note: A B C D E F G   (fist = silence)
  STYLE hand  -> model 2 (style_tf.keras)    -> major / minor / seventh
               and its HEIGHT in the picture -> volume
  Keys: T = change sound (Pad / Organ / Pure)    Q or ESC = quit

If the style hand is not visible, the chord plays as plain major at medium volume.
Which hand is which is set in common.py (CHORD_HAND / STYLE_HAND).
"""

from __future__ import annotations

import json
import os
from collections import Counter, deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np
import pygame
import tensorflow as tf

from common import (
    CHORD_HAND, CHORD_MIRROR, ROOT, STYLE_HAND, Clock, create_landmarker, detect_hands, draw_hand, open_camera, to_features,
)
from sound import SAMPLE_RATE, TIMBRES, render_chord
from theory import INTERVALS, ROOT_HZ, chord_freqs
from ui import draw_ui

ROOT_MODEL = ROOT / "models" / "gesture_tf.keras"
ROOT_CLASSES = ROOT / "models" / "gesture_tf_classes.json"
STYLE_MODEL = ROOT / "models" / "style_tf.keras"
STYLE_CLASSES = ROOT / "models" / "style_tf_classes.json"

MIN_CONFIDENCE = 0.45      # below this the guess is ignored
VOTE_FRAMES = 6            # majority vote over the last N frames: no flicker
MAX_VOLUME = 0.8
DEFAULT_VOLUME = 0.5       # used when the style hand is not visible
DISPLAY_SIZE = (960, 720)


def height_to_volume(palm_y: float) -> float:
    """Hand high in the picture (small y) = louder. Uses the palm, not the wrist, and
    never goes fully silent while the hand is visible (the chord hand's fist mutes)."""
    t = float(np.clip((0.95 - palm_y) / (0.95 - 0.30), 0.0, 1.0))
    return MAX_VOLUME * (0.15 + 0.85 * t)


def build_sounds() -> dict:
    """Render every chord in every timbre once. Takes a few seconds."""
    sounds = {}
    for timbre in TIMBRES:
        for letter in ROOT_HZ:
            for quality in INTERVALS:
                stereo = render_chord(chord_freqs(letter, quality), timbre)
                sounds[(letter, quality, timbre)] = pygame.sndarray.make_sound(stereo)
    return sounds


def predict(model, classes, pts, mirror=False):
    probs = model(to_features(pts, mirror)[None, :].astype(np.float32), training=False).numpy()[0]
    best = int(np.argmax(probs))
    return classes[best], float(probs[best])


def predict_chord(model, classes, pts):
    """Chord-hand guess. With CHORD_MIRROR = "auto" it tries the normal AND the mirrored shape
    and keeps the more confident one, so it works whichever hand you recorded with."""
    plain = predict(model, classes, pts, False)
    if CHORD_MIRROR == "auto":
        flipped = predict(model, classes, pts, True)
        best = flipped if flipped[1] > plain[1] else plain
        return best[0], best[1], f"normal {plain[0]} {plain[1]:.0%} | mirrored {flipped[0]} {flipped[1]:.0%}"
    guess, conf = predict(model, classes, pts, bool(CHORD_MIRROR))
    return guess, conf, f"mirror={bool(CHORD_MIRROR)}: {guess} {conf:.0%}"


def main() -> None:
    root_classes = json.loads(ROOT_CLASSES.read_text())
    root_model = tf.keras.models.load_model(ROOT_MODEL)
    style_model = style_classes = None
    if STYLE_MODEL.exists():
        style_classes = json.loads(STYLE_CLASSES.read_text())
        style_model = tf.keras.models.load_model(STYLE_MODEL)
    else:
        print("No style model yet (models/style_tf.keras): chords will be plain major.")

    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(2)
    print("Preparing sounds...")
    sounds = build_sounds()
    channel = pygame.mixer.Channel(0)
    playing_key = None
    volume = 0.0
    timbre_index = 0

    landmarker = create_landmarker(num_hands=2)
    cap = open_camera()
    clock = Clock()
    root_votes: deque[str] = deque(maxlen=VOTE_FRAMES)
    style_votes: deque[str] = deque(maxlen=VOTE_FRAMES)
    print(f"Chord hand: {CHORD_HAND}   Style hand: {STYLE_HAND}.  T = change sound, Q = quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hands = detect_hands(landmarker, frame, clock.next())
            chord_pts, style_pts = hands.get(CHORD_HAND), hands.get(STYLE_HAND)

            # ---- chord hand: which note? ----------------------------------
            shown, confidence, debug_chord = "no hand", 0.0, "no chord hand seen"
            if chord_pts is None:
                root_votes.clear()
            else:
                draw_hand(frame, chord_pts, (200, 160, 60))
                guess, confidence, debug_chord = predict_chord(root_model, root_classes, chord_pts)
                root_votes.append(guess if confidence >= MIN_CONFIDENCE else "rest")
                shown = Counter(root_votes).most_common(1)[0][0]

            # ---- style hand: major / minor / seventh, and volume ----------------
            quality, style_text, target_volume = "major", f"{STYLE_HAND} hand: not seen", DEFAULT_VOLUME * MAX_VOLUME
            if style_pts is not None:
                draw_hand(frame, style_pts, (60, 160, 255))
                target_volume = height_to_volume(float(style_pts[9, 1]))   # landmark 9 = middle knuckle (palm)
                if style_model is not None:
                    guess, conf = predict(style_model, style_classes, style_pts)
                    style_votes.append(guess if conf >= MIN_CONFIDENCE else "major")
                    quality = Counter(style_votes).most_common(1)[0][0]
                    style_text = f"{STYLE_HAND} hand: {quality}  {conf:.0%}"
                else:
                    style_text = f"{STYLE_HAND} hand: volume only"
            else:
                style_votes.clear()

            # ---- sound ----------------------------------------------------------
            timbre = TIMBRES[timbre_index]
            wanted = (shown[0], quality, timbre) if shown.endswith("_major") else None
            if wanted != playing_key:
                if wanted is None:
                    channel.fadeout(80)
                else:
                    channel.play(sounds[wanted], loops=-1, fade_ms=40)
                playing_key = wanted
            volume += ((target_volume if playing_key else 0.0) - volume) * 0.3
            channel.set_volume(volume)

            # ---- picture --------------------------------------------------------
            view = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
            draw_ui(
                view, shown, confidence, volume, MAX_VOLUME, quality=quality,
                hint=f"{CHORD_HAND} hand = chord  ·  {STYLE_HAND} hand = style",
                style_text=style_text, timbre=timbre,
            )
            # small debug lines: what the program sees right now (helps when something looks wrong)
            for i, line in enumerate((f"hands seen: {', '.join(sorted(hands)) or 'none'}", f"chord model: {debug_chord}")):
                pos = (24, view.shape[0] - 185 + i * 20)
                cv2.putText(view, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Handchord", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("t"):
                timbre_index = (timbre_index + 1) % len(TIMBRES)
                channel.stop()
                playing_key = None   # re-pick the chord in the new sound
    finally:
        pygame.mixer.quit()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()