"""Record the STYLE hand: the signs that choose major / minor / 7th.

Same keys as collect_data.py (click the camera window first):
  1..3   pick which style sign you are recording (see STYLE_LABELS in common.py)
  SPACE  start / stop recording
  Q      quit

Show ONE hand only (the hand that will be your style hand).
The window prints which hand MediaPipe sees ("Left" or "Right").
If it does not match STYLE_HAND in common.py, change STYLE_HAND there.

Data goes to data/style.csv. Record 300+ samples per sign, moving slowly.
"""

from __future__ import annotations

import csv
from collections import Counter

import cv2

from common import (
    ROOT, STYLE_HINTS, STYLE_LABELS, Clock, create_landmarker, detect_hands, draw_hand, open_camera, to_features,
)

DATA_FILE = ROOT / "data" / "style.csv"


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
    landmarker = create_landmarker(num_hands=1)
    cap = open_camera()
    clock = Clock()
    current, recording = 0, False

    with DATA_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                hands = detect_hands(landmarker, frame, clock.next())
                label = STYLE_LABELS[current]
                seen = None
                if hands:
                    seen, pts = next(iter(hands.items()))
                    draw_hand(frame, pts, (60, 160, 255))
                    if recording:
                        writer.writerow([label, *to_features(pts).round(5)])
                        counts[label] += 1

                status = "RECORDING" if recording else "paused"
                color = (60, 60, 240) if recording else (200, 200, 200)
                cv2.putText(frame, f"{status}: {label}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"hint: {STYLE_HINTS.get(label, '')}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 120), 1)
                for i, name in enumerate(STYLE_LABELS):
                    mark = ">" if i == current else " "
                    cv2.putText(frame, f"{mark} {i + 1}  {name:10s} {counts[name]}", (12, 92 + i * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                note = f"MediaPipe sees: {seen} hand" if seen else "no hand"
                cv2.putText(frame, note, (12, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2)

                cv2.imshow("Handchord - record STYLE hand", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    recording = not recording
                if ord("1") <= key < ord("1") + len(STYLE_LABELS):
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
