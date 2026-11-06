import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import apply_label, process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "census.csv")


@pytest.fixture(scope="module")
def data():
    """Load a reproducible sample of the census data to keep tests fast."""
    return pd.read_csv(DATA_PATH).sample(n=3000, random_state=0)


@pytest.fixture(scope="module")
def processed(data):
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test, encoder, lb, test


def test_train_model_uses_expected_algorithm(processed):
    """train_model returns a fitted RandomForestClassifier."""
    X_train, y_train, *_ = processed
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "feature_importances_")  # fitted


def test_inference_returns_binary_predictions_of_expected_size(processed):
    """inference returns a numpy array with one 0/1 prediction per row."""
    X_train, y_train, X_test, y_test, *_ = processed
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0] == y_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_expected_values():
    """Metrics match hand-computed precision, recall and F1."""
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)


def test_train_test_split_sizes_and_columns(data, processed):
    """The 80/20 split has the expected sizes, and the data has the expected columns."""
    X_train, y_train, X_test, y_test, *_ = processed
    assert X_train.shape[0] == y_train.shape[0] == int(len(data) * 0.8)
    assert X_test.shape[0] == y_test.shape[0] == len(data) - int(len(data) * 0.8)
    assert X_train.shape[1] == X_test.shape[1]
    assert set(CAT_FEATURES + ["salary"]).issubset(data.columns)


def test_save_and_load_model_roundtrip(processed, tmp_path):
    """A saved model loads back and gives identical predictions."""
    X_train, y_train, X_test, *_ = processed
    model = train_model(X_train, y_train)
    path = tmp_path / "model.pkl"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert np.array_equal(inference(model, X_test), inference(loaded, X_test))


def test_performance_on_categorical_slice(processed):
    """Slice metrics are floats in [0, 1]."""
    X_train, y_train, _, _, encoder, lb, test = processed
    model = train_model(X_train, y_train)
    value = test["sex"].iloc[0]
    metrics = performance_on_categorical_slice(
        test, "sex", value, CAT_FEATURES, "salary", encoder, lb, model
    )
    assert len(metrics) == 3
    assert all(0.0 <= m <= 1.0 for m in metrics)


def test_apply_label():
    """apply_label maps 1 -> '>50K' and 0 -> '<=50K'."""
    assert apply_label(np.array([1])) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"
