"""Honest test: score the trained model on a session it has NEVER seen.

1. Record a test session (different light / time / angle):
       python collect_data.py --test
2. Score it:
       python evaluate_test.py          (TensorFlow model)
       python evaluate_test.py sklearn  (scikit-learn model)
"""

from __future__ import annotations

import csv
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from common import ROOT

TEST_FILE = ROOT / "data" / "test_session.csv"
OUT = ROOT / "models" / "confusion_matrix_test_session.png"


def main() -> None:
    labels, rows = [], []
    with TEST_FILE.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) == 64:
                labels.append(row[0])
                rows.append([float(v) for v in row[1:]])
    X, y_true = np.array(rows, dtype=np.float32), np.array(labels)

    if len(sys.argv) > 1 and sys.argv[1] == "sklearn":
        import joblib
        y_pred = joblib.load(ROOT / "models" / "gesture_model.joblib").predict(X)
        name = "scikit-learn"
    else:
        import tensorflow as tf
        classes = json.loads((ROOT / "models" / "gesture_tf_classes.json").read_text())
        model = tf.keras.models.load_model(ROOT / "models" / "gesture_tf.keras")
        y_pred = np.array(classes)[np.argmax(model.predict(X, verbose=0), axis=1)]
        name = "TensorFlow"

    acc = accuracy_score(y_true, y_pred)
    print(f"{name} model on the NEW test session: {acc:.1%}  ({len(y_true)} samples)\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, xticks_rotation=45, cmap="Blues")
    ax.set_title(f"{name}, new session: {acc:.1%}")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("Saved picture:", OUT)


if __name__ == "__main__":
    main()
