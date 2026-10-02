"""Inspect the binary artifacts in this project from the terminal.

Examples:
    python view.py X_val.npy
    python view.py "app_predictor_model (1).keras"
    python view.py tokenizer.pkl --trusted-pickle
"""

import argparse
import pickle
import sys
from pathlib import Path


def inspect_npy(path: Path) -> None:
    import numpy as np

    values = np.load(path, allow_pickle=False, mmap_mode="r")
    print(f"File: {path.name}")
    print(f"Shape: {values.shape}")
    print(f"Data type: {values.dtype}")
    print("First values:")
    print(values[:5])


def inspect_keras(path: Path) -> None:
    import tensorflow as tf

    model = tf.keras.models.load_model(path, compile=False)
    print(f"File: {path.name}")
    print(f"Model: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    model.summary()


def inspect_pickle(path: Path, trusted: bool) -> None:
    if not trusted:
        print("Pickle files can run code while loading. This viewer will not open one")
        print("unless you explicitly mark it trusted.")
        print(f"For this project file, run: python view.py '{path.name}' --trusted-pickle")
        return

    with path.open("rb") as artifact:
        value = pickle.load(artifact)

    print(f"File: {path.name}")
    print(f"Type: {type(value).__module__}.{type(value).__name__}")
    if hasattr(value, "word_index"):
        words = value.word_index
        print(f"Vocabulary size: {len(words)}")
        print("First words:")
        print(list(words.items())[:20])
    else:
        print(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect NumPy, Keras, and trusted pickle artifacts.")
    parser.add_argument("file", type=Path, help="A .npy, .keras, or .pkl file")
    parser.add_argument(
        "--trusted-pickle",
        action="store_true",
        help="Allow loading a pickle you trust. Pickles from unknown sources are unsafe.",
    )
    args = parser.parse_args()
    path = args.file

    if not path.is_file():
        parser.error(f"File not found: {path}")

    try:
        if path.suffix == ".npy":
            inspect_npy(path)
        elif path.suffix == ".keras":
            inspect_keras(path)
        elif path.suffix in {".pkl", ".pickle"}:
            inspect_pickle(path, args.trusted_pickle)
        else:
            parser.error("Supported formats: .npy, .keras, .pkl, .pickle")
    except ModuleNotFoundError as error:
        print(f"Missing dependency: {error.name}", file=sys.stderr)
        print("Create the project environment and install requirements first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
