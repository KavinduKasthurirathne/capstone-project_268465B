"""Tests for src/features/engineer.py on small synthetic tables. Run with: pytest -q

The tables have no isFraud column, which shows the features never use labels.
"""
import numpy as np
import pandas as pd

from src.data.split import SECONDS_PER_DAY
from src.features.engineer import add_time_amount_features

START = 86_400  # the real data also starts at TransactionDT = 86400
HOUR = 3_600


def random_table(n_rows: int, n_days: int, seed: int = 0) -> pd.DataFrame:
    """n_rows transactions at random seconds over n_days, with long-tailed amounts."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(n_rows),
        "TransactionDT": rng.integers(START, START + n_days * SECONDS_PER_DAY, n_rows),
        "TransactionAmt": rng.lognormal(4, 1.5, n_rows).astype("float32"),
    })


def test_values_worked_by_hand():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [
            START,                                # first second of a day
            START + HOUR - 1,                     # last second of the same hour
            START + 5 * HOUR,                     # start of hour 5
            START + SECONDS_PER_DAY - 1,          # last second of the same day
            START + 6 * SECONDS_PER_DAY,          # seven days after day 1: weekday wraps to 0
        ],
        "TransactionAmt": np.array([0.0, 1.0, 9.0, 99.0, 31937.39], dtype="float32"),
    })
    out = add_time_amount_features(df)
    assert out["hour"].tolist() == [0, 0, 5, 23, 0]
    assert out["weekday"].tolist() == [1, 1, 1, 1, 0]
    np.testing.assert_allclose(out["log_amount"], np.log1p(df["TransactionAmt"]), rtol=1e-6)
    assert out["log_amount"].iloc[0] == 0  # log(1 + 0) = 0
    # The raw amount is kept for savings and expected loss.
    pd.testing.assert_series_equal(out["TransactionAmt"], df["TransactionAmt"])
    # Small types save memory on ~590k rows.
    assert out["log_amount"].dtype == "float32"
    assert out["hour"].dtype == "int8" and out["weekday"].dtype == "int8"


def test_values_in_range():
    out = add_time_amount_features(random_table(n_rows=2_000, n_days=30))
    assert set(out["hour"]) == set(range(24))
    assert set(out["weekday"]) == set(range(7))
    assert np.isfinite(out["log_amount"]).all()
    assert (out["log_amount"] >= 0).all()


def test_row_values_do_not_depend_on_other_rows():
    df = random_table(n_rows=200, n_days=30)
    cols = ["log_amount", "hour", "weekday"]
    expected = add_time_amount_features(df).set_index("TransactionID")[cols]
    # Shuffling the rows does not change any row's values.
    shuffled = add_time_amount_features(df.sample(frac=1, random_state=0)).set_index("TransactionID")
    pd.testing.assert_frame_equal(shuffled[cols].sort_index(), expected)
    # Neither does dropping all rows after the first 50.
    first_rows = add_time_amount_features(df.head(50)).set_index("TransactionID")
    pd.testing.assert_frame_equal(first_rows[cols], expected.head(50))
