"""Train the gesture model with TensorFlow / Keras.

Run:  python train_tf.py

Same data as before (data/gestures.csv). A small neural network:
  63 hand numbers -> 128 -> 64 -> one score per sign (softmax)
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

DATA_FILE = ROOT / "data" / "gestures.csv"
MODEL_FILE = ROOT / "models" / "gesture_tf.keras"
CLASSES_FILE = ROOT / "models" / "gesture_tf_classes.json"
MATRIX_FILE = ROOT / "models" / "confusion_matrix_tf.png"
CURVE_FILE = ROOT / "models" / "training_curve_tf.png"


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
    y = np.array([classes.index(n) for n in y_names])
    print("Samples per sign:", {c: int((y_names == c).sum()) for c in classes})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),   # randomly ignore 30% of neurons: reduces over-fitting
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=2,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.1%}")
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, predictions, target_names=classes))

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    CLASSES_FILE.write_text(json.dumps(classes))

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=classes, ax=ax, xticks_rotation=45, cmap="Blues"
    )
    ax.set_title(f"TensorFlow model: {acc:.1%}")
    fig.tight_layout()
    fig.savefig(MATRIX_FILE, dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy per epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss per epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(CURVE_FILE, dpi=150)
    print("Saved:", MODEL_FILE)
    print("Saved pictures:", MATRIX_FILE, "and", CURVE_FILE)


if __name__ == "__main__":
    main()
