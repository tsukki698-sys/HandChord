"""Train the SECOND model: style hand -> major / minor / seventh.

Run:  python train_style.py

Same method as train_tf.py (63 hand numbers -> small neural network),
but on data/style.csv with only 3 classes.
"""

from __future__ import annotations

import csv
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from common import ROOT

DATA_FILE = ROOT / "data" / "style.csv"
MODEL_FILE = ROOT / "models" / "style_tf.keras"
CLASSES_FILE = ROOT / "models" / "style_tf_classes.json"
MATRIX_FILE = ROOT / "models" / "confusion_matrix_style.png"


def load_data():
    labels, rows = [], []
    with DATA_FILE.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) == 64:
                labels.append(row[0])
                rows.append([float(v) for v in row[1:]])
    return np.array(rows, dtype=np.float32), np.array(labels)


def main() -> None:
    tf.keras.utils.set_random_seed(42)
    X, y_names = load_data()
    classes = sorted(set(y_names))
    if len(classes) < 2:
        raise SystemExit("Record at least 2 style signs first (python collect_style.py).")
    y = np.array([classes.index(n) for n in y_names])
    print("Samples per sign:", {c: int((y_names == c).sum()) for c in classes})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.1%}")
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, predictions, target_names=classes))

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    CLASSES_FILE.write_text(json.dumps(classes))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=classes, ax=ax, cmap="Oranges")
    ax.set_title(f"Style model: {acc:.1%}")
    fig.tight_layout()
    fig.savefig(MATRIX_FILE, dpi=150)
    print("Saved:", MODEL_FILE, "and", MATRIX_FILE)


if __name__ == "__main__":
    main()
