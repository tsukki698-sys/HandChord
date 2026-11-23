"""Shared helpers: camera, hand tracking, and turning a hand into numbers.

Used by collect_data.py and live_play.py so both see the hand the same way.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ---------------------------------------------------------------------------
# YOUR GESTURES. Change these names to whatever you like.
# 1 rest=fist, 2 A=one finger, 3 B=two, 4 C=three, 5 D=four (thumb tucked),
# 6 E=five fingers, 7 F=OK sign, 8 G=thumb+pinky.
# Each name is one hand shape you will record. The model learns to tell them
# apart. Keep "rest" so the music can stay silent when you do nothing.
# ---------------------------------------------------------------------------
LABELS = ["rest", "A_major", "B_major", "C_major", "D_major", "E_major", "F_major", "G_major"]

# ---------------------------------------------------------------------------
# TWO-HAND SETTINGS (used by handchord_2h.py).  All in REAL hands, as you see them.
#
# SWAP_HANDEDNESS : MediaPipe often calls your real right hand "Left" (and the
#                   other way round). True = fix that, so "Right" really means
#                   your right hand.
# CHORD_HAND      : the hand that shows the notes A-G.
# STYLE_HAND      : the other hand: major / minor / seventh, and volume.
# CHORD_MIRROR    : "auto" is easiest. (A left hand is the mirror image of a right hand. If you
#                   RECORDED A-G with one hand but now PLAY them with the other
#                   hand, True flips the shape for the model; False does not.)
# ---------------------------------------------------------------------------
SWAP_HANDEDNESS = True
CHORD_HAND = "Left"
STYLE_HAND = "Right"
CHORD_MIRROR = "auto"   # True / False / "auto" (auto = try both, keep the more confident)
STYLE_LABELS = ["major", "minor", "seventh"]
STYLE_HINTS = {"major": "one finger", "minor": "two fingers (peace sign)", "seventh": "three fingers"}

WRIST = 0
MIDDLE_MCP = 9


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download MediaPipe's hand model once (about 8 MB)."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def create_landmarker(num_hands: int = 1):
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DirectShow works best on Windows
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Close other apps using it.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always the newest frame = less delay
    return cap


class Clock:
    """MediaPipe video mode needs timestamps that always increase."""

    def __init__(self):
        self.t = 0

    def next(self) -> int:
        self.t += 33
        return self.t


def detect(landmarker, frame_bgr, timestamp_ms):
    """Return a (21, 3) array of landmarks, or None if no hand is visible."""
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)
    if not result.hand_landmarks:
        return None
    return np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], dtype=np.float64)


def detect_hands(landmarker, frame_bgr, timestamp_ms):
    """Return {"Left": pts, "Right": pts} for the hands that are visible.

    The frame must already be mirrored (selfie view): then MediaPipe's
    Left / Right label matches your real left / right hand.
    """
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)
    found = []   # (label from MediaPipe, score, pts)
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label, score = handedness[0].category_name, handedness[0].score
        if SWAP_HANDEDNESS:
            label = "Left" if label == "Right" else "Right"
        pts = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float64)
        found.append((label, score, pts))

    if len(found) >= 2:
        # Two hands: MediaPipe sometimes gives BOTH the same label. Position is more
        # reliable: in the mirrored picture your left hand is on the left side.
        best_two = sorted(found, key=lambda item: -item[1])[:2]
        best_two.sort(key=lambda item: item[2][WRIST, 0])
        return {"Left": best_two[0][2], "Right": best_two[1][2]}
    return {label: pts for label, _score, pts in found}


def to_features(pts: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Turn 21 landmarks into 63 numbers that do not depend on where the hand is.

    1. Move the wrist to (0, 0, 0): hand position in the frame no longer matters.
    2. Divide by the palm size: distance from the camera no longer matters.
    What is left is the SHAPE of the hand, which is what we want to classify.
    """
    centered = pts - pts[WRIST]
    palm = np.linalg.norm(centered[MIDDLE_MCP, :2])
    centered = centered / max(palm, 1e-4)
    if mirror:
        centered[:, 0] *= -1   # flip left <-> right
    return centered.flatten()


def draw_hand(frame, pts, color=(200, 160, 60)) -> None:
    h, w = frame.shape[:2]
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        p1 = (int(pts[connection.start, 0] * w), int(pts[connection.start, 1] * h))
        p2 = (int(pts[connection.end, 0] * w), int(pts[connection.end, 1] * h))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for x, y, _ in pts:
        cv2.circle(frame, (int(x * w), int(y * h)), 3, (255, 230, 90), -1, cv2.LINE_AA)
