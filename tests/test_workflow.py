import numpy as np
import pytest
from sklearn.datasets import load_digits

from digits_benchmark.predict import load_pixels
from digits_benchmark.train import candidates, split_indices


def test_split_is_disjoint_complete_and_reproducible():
    labels = load_digits().target
    splits = split_indices(labels)
    assert sorted(np.concatenate(splits).tolist()) == list(range(len(labels)))
    assert not (set(splits[0]) & set(splits[1]))
    assert not (set(splits[0]) & set(splits[2]))
    assert not (set(splits[1]) & set(splits[2]))
    for first, second in zip(splits, split_indices(labels)):
        np.testing.assert_array_equal(first, second)
        assert set(labels[first]) == set(range(10))


def test_scaler_uses_only_training_rows():
    data = load_digits()
    train, validation, _ = split_indices(data.target)
    pipeline = candidates()["logistic_regression"]
    pipeline.fit(data.data[train], data.target[train])
    scaler = pipeline.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, data.data[train].mean(axis=0))
    before = scaler.mean_.copy()
    pipeline.predict(data.data[validation] + 1000)
    np.testing.assert_array_equal(scaler.mean_, before)


def test_pixel_validation(tmp_path):
    path = tmp_path / "digit.csv"
    np.savetxt(path, np.zeros((8, 8)), delimiter=",")
    assert load_pixels(path).shape == (1, 64)
    for pixels in [np.zeros((7, 8)), np.full((8, 8), 17), np.full((8, 8), np.nan)]:
        np.savetxt(path, pixels, delimiter=",")
        with pytest.raises(ValueError):
            load_pixels(path)
