"""Tests for src/features/engineer.py on small synthetic tables. Run with: pytest -q

The tables have no isFraud column, which shows the features never use labels.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.split import SECONDS_PER_DAY
from src.features.engineer import add_card_history, add_time_amount_features, add_uid

START = 86_400  # the real data also starts at TransactionDT = 86400
HOUR = 3_600


def random_table(n_rows: int, n_days: int, seed: int = 0) -> pd.DataFrame:
    """n_rows transactions in time order over n_days, as in the real data, with
    long-tailed amounts, a few cards and some missing addresses.

    D1 is set so every card started on day 0 or day 1. That gives 5 x 3 x 2 = 30
    possible uids, so each card comes back several times and has a history.
    """
    rng = np.random.default_rng(seed)
    seconds = np.sort(rng.integers(START, START + n_days * SECONDS_PER_DAY, n_rows))
    card_start_day = rng.integers(0, 2, n_rows)
    return pd.DataFrame({
        "TransactionID": np.arange(n_rows),  # rises in time order, as in the real data
        "TransactionDT": seconds,
        "TransactionAmt": rng.lognormal(4, 1.5, n_rows).astype("float32"),
        "card1": rng.integers(1000, 1005, n_rows),
        "addr1": rng.choice(np.array([100, 200, np.nan], dtype="float32"), n_rows),
        "D1": (seconds // SECONDS_PER_DAY - card_start_day).astype("float32"),
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


def test_uid_worked_by_hand():
    day = SECONDS_PER_DAY
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6, 7],
        # TransactionDT // 86400: day 1 or day 11
        "TransactionDT": [START, START + 10 * day, START + 10 * day, START,
                          START, START + 10 * day, START],
        "card1": [1000, 1000, 1000, 2000, 1000, 1000, 1000],
        "addr1": np.array([100, 100, 100, 100, np.nan, np.nan, 100], dtype="float32"),
        "D1": np.array([0, 10, 0, 0, 0, 10, np.nan], dtype="float32"),
    })
    uid = add_uid(df)["uid"].astype(str).tolist()
    assert uid[0] == "1000_100_1"      # card started on day 1 - 0 = 1
    assert uid[1] == uid[0]            # same card 10 days later: D1 has grown by 10
    assert uid[2] == "1000_100_11"     # same card1 and addr1, but a card that started on day 11
    assert uid[3] == "2000_100_1"      # different card1
    assert uid[4] == "1000_<NA>_1"     # missing address is kept as a value ...
    assert uid[5] == uid[4]            # ... so the same card without an address still matches
    assert uid[6] == "1000_100_<NA>"   # missing D1
    assert add_uid(df)["uid"].dtype == "category"  # each distinct text stored once


def test_card_history_worked_by_hand():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "TransactionDT": [START, START + 100, START + 100, START + 400, START + 1000, START + 1000],
        "TransactionAmt": np.array([10, 50, 20, 30, 60, 5], dtype="float32"),
        "uid": pd.Series(["A", "B", "A", "B", "A", "A"], dtype="category"),
    })
    # Feed the rows in reverse order: the function must sort them by time itself.
    out = add_card_history(df.iloc[::-1]).sort_values("TransactionID")
    # Card A: 10 at t=0, 20 at t=100, 60 and 5 both at t=1000 (ID 5 counts as before ID 6).
    # Card B: 50 at t=100, 30 at t=400.
    assert out["card_prev_count"].tolist() == [0, 0, 1, 1, 2, 3]
    np.testing.assert_allclose(out["card_prev_mean_amount"],
                               [np.nan, np.nan, 10, 50, 15, 30])  # 15 = (10+20)/2, 30 = (10+20+60)/3
    np.testing.assert_allclose(out["card_secs_since_prev"],
                               [np.nan, np.nan, 100, 300, 900, 0])
    assert out["card_prev_count"].dtype == "int32"
    assert out["card_prev_mean_amount"].dtype == "float32"
    assert out["card_secs_since_prev"].dtype == "float32"


@pytest.mark.parametrize("add, cols", [
    (add_time_amount_features, ["log_amount", "hour", "weekday"]),
    (add_uid, ["uid"]),
    (add_card_history, ["card_prev_count", "card_prev_mean_amount", "card_secs_since_prev"]),
])
def test_values_do_not_depend_on_row_order_or_later_rows(add, cols):
    df = add_uid(random_table(n_rows=200, n_days=30))  # card history needs uid
    expected = add(df).set_index("TransactionID")[cols]
    # check_categorical=False compares uid values; the list of categories itself
    # depends on which rows are in the table.
    # Shuffling the rows does not change any row's values.
    shuffled = add(df.sample(frac=1, random_state=0)).set_index("TransactionID")
    pd.testing.assert_frame_equal(shuffled[cols].sort_index(), expected, check_categorical=False)
    # Prefix test: the table is in time order, so its first 50 rows are the 50
    # earliest. Dropping every later row does not change their values.
    first_rows = add(df.head(50)).set_index("TransactionID")
    pd.testing.assert_frame_equal(first_rows[cols], expected.head(50), check_categorical=False)
