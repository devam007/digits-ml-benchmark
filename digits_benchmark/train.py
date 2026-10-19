"""Select a classifier on validation data, then evaluate once on held-out data."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_indices(labels, seed=42):
    """Return disjoint stratified 60/20/20 train, validation and test indices."""
    development, test = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=labels[development], random_state=seed
    )
    return train, validation, test


def candidates(seed=42):
    """All preprocessing is fitted inside each pipeline on training data only."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1),
    }


def metrics(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
    }


def run(output: Path, seed: int = 42):
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_digits()
    features, labels = dataset.data, dataset.target
    train, validation, test = split_indices(labels, seed)
    models = candidates(seed)
    scores = {}
    for name, model in models.items():
        model.fit(features[train], labels[train])
        scores[name] = metrics(labels[validation], model.predict(features[validation]))
    # The test set is never used to choose the model or preprocessing.
    selected = max(scores, key=lambda name: scores[name]["macro_f1"])
    development = np.concatenate([train, validation])
    model = clone(models[selected]).fit(features[development], labels[development])
    predicted = model.predict(features[test])
    baseline = DummyClassifier(strategy="most_frequent").fit(
        features[development], labels[development]
    )
    report = {
        "dataset": "sklearn.datasets.load_digits: 1797 samples, 64 pixels, classes 0-9",
        "dataset_sha256": hashlib.sha256(features.tobytes() + labels.tobytes()).hexdigest(),
        "seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "validation_scores": scores,
        "selection_metric": "validation macro F1",
        "selected_model": selected,
        "test_scores": metrics(labels[test], predicted),
        "test_baseline": metrics(labels[test], baseline.predict(features[test])),
        "classification_report": classification_report(
            labels[test], predicted, output_dict=True, zero_division=0
        ),
        "test_errors": int(np.sum(predicted != labels[test])),
        "limitations": [
            "Small educational dataset; no claim about arbitrary handwriting images.",
            "Random split is not a writer-independent evaluation.",
            "One fixed split; scores have sampling uncertainty.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "split_indices.json").write_text(
        json.dumps(
            {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset_index", "actual", "predicted", "correct"])
        writer.writerows(
            (int(index), int(actual), int(prediction), bool(actual == prediction))
            for index, actual, prediction in zip(test, labels[test], predicted)
        )
    matrix = confusion_matrix(labels[test], predicted, labels=np.arange(10))
    display = ConfusionMatrixDisplay(matrix, display_labels=np.arange(10))
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title(f"Held-out test set: {selected}")
    display.figure_.tight_layout()
    display.figure_.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)
    errors = np.flatnonzero(predicted != labels[test])[:12]
    figure, axes = plt.subplots(3, 4, figsize=(9, 7))
    for axis in axes.flat:
        axis.axis("off")
    for axis, error in zip(axes.flat, errors):
        axis.imshow(dataset.images[test[error]], cmap="gray_r", vmin=0, vmax=16)
        axis.set_title(f"True {labels[test[error]]} / predicted {predicted[error]}")
    figure.suptitle("First held-out errors in saved test order")
    figure.tight_layout()
    figure.savefig(output / "error_examples.png", dpi=150)
    plt.close(figure)
    joblib.dump(model, output / "model.joblib")
    # A held-out sample for a reproducible command-line prediction demonstration.
    np.savetxt(output / "demo_digit.csv", dataset.images[test[0]], delimiter=",", fmt="%d")
    (output / "demo_label.txt").write_text(str(labels[test[0]]) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run(args.output, args.seed)
    print(
        json.dumps(
            {key: report[key] for key in ["selected_model", "test_scores", "test_errors"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
