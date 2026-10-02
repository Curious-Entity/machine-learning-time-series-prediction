"""Report accuracy for the saved predictions or a supplied Keras model."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate next-app predictions on the held-out test split.")
    parser.add_argument("--model", type=Path, help="Optional .keras model to evaluate instead of y_hat.npy")
    args = parser.parse_args()

    truth = np.load(ROOT / "y_te.npy")
    if args.model:
        features = np.load(ROOT / "X_te.npy")
        model = tf.keras.models.load_model(ROOT / args.model, compile=False)
        predicted = np.argmax(model.predict(features, batch_size=256, verbose=1), axis=1)
    else:
        predicted = np.load(ROOT / "y_hat.npy")

    accuracy = np.mean(predicted == truth)
    print(f"Test examples: {truth.size:,}")
    print(f"Top-1 accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
