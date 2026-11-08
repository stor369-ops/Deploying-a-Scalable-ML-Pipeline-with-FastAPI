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


class ThresholdModel:
    """Stub model: predicts 1 when the first (continuous) feature, age, is >= 50.

    This lets a test control the predictions exactly through the age column.
    """

    def predict(self, X):
        return (X[:, 0] >= 50).astype(int)


def test_performance_on_categorical_slice_expected_values():
    """Slice metrics match hand-computed values on a small deterministic dataset.

    Male slice: labels [1, 1, 1, 1, 0, 0], predictions [1, 1, 0, 0, 1, 0]
        TP = 2, FP = 1, FN = 2 -> precision 2/3, recall 1/2, F1 4/7
    Female slice: labels [1, 0], predictions [0, 1]
        TP = 0, FP = 1, FN = 1 -> precision 0, recall 0, F1 0
    """
    df = pd.DataFrame(
        {
            "age": [60, 60, 20, 20, 60, 20, 20, 60],
            "sex": ["Male"] * 6 + ["Female"] * 2,
            "salary": [
                ">50K", ">50K", ">50K", ">50K", "<=50K", "<=50K",
                ">50K", "<=50K",
            ],
        }
    )
    _, _, encoder, lb = process_data(
        df, categorical_features=["sex"], label="salary", training=True
    )

    male = performance_on_categorical_slice(
        df, "sex", "Male", ["sex"], "salary", encoder, lb, ThresholdModel()
    )
    assert male == (
        pytest.approx(2 / 3),
        pytest.approx(1 / 2),
        pytest.approx(4 / 7),
    )

    female = performance_on_categorical_slice(
        df, "sex", "Female", ["sex"], "salary", encoder, lb, ThresholdModel()
    )
    assert female == (
        pytest.approx(0.0),
        pytest.approx(0.0),
        pytest.approx(0.0),
    )


def test_apply_label():
    """apply_label maps 1 -> '>50K' and 0 -> '<=50K'."""
    assert apply_label(np.array([1])) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"
