"""Predict the next app from a comma-separated sequence of 30 app names."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict likely next apps from a 30-app sequence.")
    parser.add_argument("apps", help="Thirty comma-separated app names, oldest to newest.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", type=Path, default=Path("app_predictor_model (1).keras"))
    args = parser.parse_args()

    sequence = [app.strip().lower() for app in args.apps.split(",") if app.strip()]
    config = json.loads((ROOT / "model_config.json").read_text())
    if len(sequence) != config["n_steps"]:
        parser.error(f"Expected exactly {config['n_steps']} app names; received {len(sequence)}.")

    # This project-owned pickle contains the Keras tokenizer used to create the arrays.
    with (ROOT / "tokenizer.pkl").open("rb") as artifact:
        tokenizer = pickle.load(artifact)
    encoded = tokenizer.texts_to_sequences([" ".join(sequence)])
    model = tf.keras.models.load_model(ROOT / args.model, compile=False)
    probabilities = model.predict(np.asarray(encoded, dtype="int32"), verbose=0)[0]
    index_to_app = {int(key): value for key, value in json.loads((ROOT / "index_to_app.json").read_text()).items()}

    top_k = min(args.top_k, len(probabilities))
    for rank, index in enumerate(np.argsort(probabilities)[-top_k:][::-1], start=1):
        print(f"{rank}. {index_to_app.get(int(index), '<padding>')}: {probabilities[index]:.2%}")


if __name__ == "__main__":
    main()
