"""Tests for src/features/preprocess.py on small synthetic tables. Run with: pytest -q"""
import json

import numpy as np
import pandas as pd

from src.features.preprocess import NOT_FEATURES, fit_preprocessor, transform


def train_and_eval_table() -> pd.DataFrame:
    """Six training rows (block 1) and three evaluation rows (block 2),
    with every column that must never be a feature."""
    return pd.DataFrame({
        "TransactionID": range(9),
        "TransactionDT": range(9),
        "day": [0, 0, 0, 1, 1, 1, 9, 9, 9],
        "block": [1] * 6 + [2] * 3,
        "isFraud": [0, 1, 0, 0, 0, 0, 1, 0, 0],
        "uid": pd.Series(list("AABBCCAAD"), dtype="category"),
        "TransactionAmt": np.arange(9, dtype="float32"),
        "num": np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9], dtype="float32"),
        "color": pd.Series(["red", "red", "blue", None, "red", "blue", "green", None, "red"],
                           dtype="category"),
    })


def test_drops_only_columns_more_than_threshold_missing():
    train = pd.DataFrame({
        "at_threshold": [1.0] * 5 + [np.nan] * 95,   # exactly 95% missing: kept
        "above": [1.0] * 4 + [np.nan] * 96,          # 96% missing: dropped
        "full": np.arange(100, dtype="float32"),
    })
    state = fit_preprocessor(train, max_missing=0.95)
    assert state["drop"] == ["above"]
    assert state["features"] == ["at_threshold", "full"]
    assert "above" not in transform(train, state).columns


def test_frequency_encoding_uses_training_shares():
    train = pd.DataFrame({"color": pd.Series(["red", "red", "blue", None], dtype="category")})
    state = fit_preprocessor(train, max_missing=0.95)
    assert state["freq"]["color"] == {"red": 0.5, "blue": 0.25, "<missing>": 0.25}
    assert state["features"] == ["color_freq"]
    # Blue is 3/5 of these rows but still gets its training share 0.25.
    evaluation = pd.DataFrame({"color": pd.Series(["blue", "blue", "blue", None, "green"],
                                                  dtype="category")})
    out = transform(evaluation, state)
    assert out["color_freq"].tolist() == [0.25, 0.25, 0.25, 0.25, 0.0]  # unseen "green" -> 0
    assert out["color_freq"].dtype == "float32"
    pd.testing.assert_series_equal(out["color"], evaluation["color"])  # kept for step 6.1


def test_fitted_state_ignores_evaluation_rows_and_labels():
    df = train_and_eval_table()
    state = fit_preprocessor(df[df["block"] == 1], max_missing=0.5)
    # Change every evaluation row and flip every label: the state fitted on the
    # training rows stays the same.
    changed = df.copy()
    is_eval = changed["block"] == 2
    changed.loc[is_eval, "num"] = np.nan
    changed["color"] = changed["color"].cat.add_categories("purple")
    changed.loc[is_eval, "color"] = "purple"
    changed["isFraud"] = 1 - changed["isFraud"]
    assert fit_preprocessor(changed[changed["block"] == 1], max_missing=0.5) == state
    # Fitting on all rows WOULD change it, so the check above can fail.
    assert fit_preprocessor(changed, max_missing=0.5) != state


def test_ids_time_label_uid_and_raw_amount_are_never_features():
    df = train_and_eval_table()
    state = fit_preprocessor(df, max_missing=0.95)
    assert state["features"] == ["num", "color_freq"]
    assert not set(NOT_FEATURES) & set(state["features"])
    # They stay in the table: evaluation needs the IDs, days, labels and amounts.
    assert set(NOT_FEATURES) <= set(transform(df, state).columns)


def test_state_survives_json_round_trip():
    # Stage 2 (step 4.5) saves the state as JSON.
    state = fit_preprocessor(train_and_eval_table(), max_missing=0.95)
    assert json.loads(json.dumps(state)) == state
