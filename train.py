"""Train the next-app prediction model from the prepared NumPy arrays."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).parent


def build_model(vocab_size: int, n_steps: int, params: dict) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(n_steps,), dtype="int32", name="app_sequence")
    x = tf.keras.layers.Embedding(vocab_size, params["emb"])(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(params["lstm"], dropout=params["drop"])
    )(x)
    for _ in range(2):
        x = tf.keras.layers.Dense(params["dense"])(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)
        x = tf.keras.layers.Dropout(params["drop"])(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", name="next_app")(x)
    return tf.keras.Model(inputs, outputs, name="app_predictor")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the bidirectional LSTM app predictor.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/app_predictor_model.keras"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    config = json.loads((ROOT / "model_config.json").read_text())
    params = json.loads((ROOT / "best_hyperparams.json").read_text())
    x_train = np.load(ROOT / "X_train.npy", mmap_mode="r")
    y_train = np.load(ROOT / "y_train.npy", mmap_mode="r")
    x_val = np.load(ROOT / "X_val.npy", mmap_mode="r")
    y_val = np.load(ROOT / "y_val.npy", mmap_mode="r")

    model = build_model(config["vocab_size"], config["n_steps"], params)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        params["lr"], decay_steps=10_000, alpha=0.1
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(schedule, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True),
    ]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=params["bs"],
        epochs=args.epochs,
        callbacks=callbacks,
    )
    print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()
