"""STEP 1: Record your own dataset.

Keys (click the camera window first):
  1..5   pick which gesture you are recording (see LABELS in common.py)
  SPACE  start / stop recording the selected gesture
  Q      quit

Each recorded frame becomes one row in data/gestures.csv:
  label, f0, f1, ... f62      (63 numbers = the hand shape)

Tips for a GOOD dataset:
  * Record 300+ samples per gesture.
  * Move the hand slowly: closer, farther, left, right, slightly tilted.
  * Record in different light, and with different people if you can.
  * Record "rest" too: relaxed hand, fist, hand halfway between gestures.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter

import cv2

from common import LABELS, ROOT, Clock, create_landmarker, detect, draw_hand, open_camera, to_features

# Normal run  : python collect_data.py        -> data/gestures.csv   (training data)
# Test session: python collect_data.py --test  -> data/test_session.csv (NEVER used for training)
TEST_MODE = "--test" in sys.argv
DATA_FILE = ROOT / "data" / ("test_session.csv" if TEST_MODE else "gestures.csv")


def load_counts() -> Counter:
    counts: Counter = Counter()
    if DATA_FILE.exists():
        with DATA_FILE.open(newline="") as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0]] += 1
    return counts


def main() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    counts = load_counts()
    landmarker = create_landmarker()
    cap = open_camera()
    clock = Clock()
    current = 0
    recording = False

    with DATA_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # mirror, like a mirror
                pts = detect(landmarker, frame, clock.next())
                label = LABELS[current]

                if pts is not None:
                    draw_hand(frame, pts)
                    if recording:
                        writer.writerow([label, *to_features(pts).round(5)])
                        counts[label] += 1

                status = "RECORDING" if recording else "paused"
                color = (60, 60, 240) if recording else (200, 200, 200)
                cv2.putText(frame, f"{status}: {label}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                for i, name in enumerate(LABELS):
                    mark = ">" if i == current else " "
                    text = f"{mark} {i + 1}  {name:10s} {counts[name]}"
                    cv2.putText(frame, text, (12, 65 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if pts is None:
                    cv2.putText(frame, "no hand", (12, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)

                cv2.imshow("Handchord - TEST session" if TEST_MODE else "Handchord - record", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    recording = not recording
                if ord("1") <= key < ord("1") + len(LABELS):
                    current = key - ord("1")
                    recording = False
        finally:
            cap.release()
            landmarker.close()
            cv2.destroyAllWindows()
    print("Saved to", DATA_FILE)
    print(dict(counts))


if __name__ == "__main__":
    main()