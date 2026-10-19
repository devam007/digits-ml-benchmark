"""Predict an 8x8 CSV digit with a model produced by the training command."""

import argparse
from pathlib import Path

import joblib
import numpy as np


def load_pixels(path):
    pixels = np.loadtxt(path, delimiter=",")
    if pixels.shape != (8, 8):
        raise ValueError("Input must be an 8x8 CSV matrix.")
    if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 16)):
        raise ValueError("Pixel intensities must be finite numbers in [0, 16].")
    return pixels.reshape(1, 64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pixels", type=Path)
    parser.add_argument("--model", type=Path, default=Path("reports/model.joblib"))
    args = parser.parse_args()
    try:
        features = load_pixels(args.pixels)
        model = joblib.load(args.model)
        print(f"Predicted digit: {int(model.predict(features)[0])}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
