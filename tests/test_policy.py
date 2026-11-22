"""Tests for src/evaluation/policy.py on small synthetic tables. Run with: pytest -q

The tables have no isFraud column, which shows the policy never uses labels.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.policy import top_k_per_day
from src.utils.config import load_config


def scored_table(rows_per_day: list[int], seed: int = 0) -> pd.DataFrame:
    """rows_per_day[d] transactions on day d with random scores, IDs in time order."""
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(len(rows_per_day)), rows_per_day)
    return pd.DataFrame({
        "TransactionID": np.arange(len(day)),
        "day": day,
        "score": rng.random(len(day)).astype("float32"),
    })


@pytest.mark.parametrize("k", load_config()["evaluation"]["budgets"])
def test_alerts_per_day_are_k_or_the_whole_day(k):
    rows_per_day = [200, 100, 50]
    df = scored_table(rows_per_day)
    alerts_per_day = top_k_per_day(df, k).groupby(df["day"]).sum()
    assert alerts_per_day.tolist() == [min(k, n) for n in rows_per_day]


def test_highest_scores_are_picked():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "day": [0, 0, 0, 1, 1, 1],
        "score": [0.2, 0.9, 0.5, 0.1, 0.3, 0.8],
    })
    # Day 0: 0.9 (ID 2) and 0.5 (ID 3). Day 1: 0.8 (ID 6) and 0.3 (ID 5).
    alerts = top_k_per_day(df, k=2)
    assert df.loc[alerts, "TransactionID"].tolist() == [2, 3, 5, 6]


def test_tie_at_cut_off_goes_to_lower_transaction_id():
    # Three rows tie at 0.4 for the second place. ID 5 comes first in the table,
    # but ID 3 is lower, so ID 3 gets the alert.
    df = pd.DataFrame({
        "TransactionID": [7, 5, 9, 3],
        "day": [0, 0, 0, 0],
        "score": [0.9, 0.4, 0.4, 0.4],
    })
    alerts = top_k_per_day(df, k=2)
    assert df.loc[alerts, "TransactionID"].tolist() == [7, 3]


def test_row_order_does_not_change_alerts():
    df = scored_table([30, 20, 25])
    df["score"] = df["score"].round(1)  # many ties, so the tie-break is used
    in_order = df.loc[top_k_per_day(df, k=8), "TransactionID"]
    shuffled = df.sample(frac=1, random_state=0)
    from_shuffled = shuffled.loc[top_k_per_day(shuffled, k=8), "TransactionID"]
    assert sorted(from_shuffled) == sorted(in_order)


def test_other_days_do_not_change_a_days_alerts():
    """Adding a later day with much higher scores leaves day 0's alerts unchanged,
    so a day's queue never depends on another day's transactions."""
    day0 = scored_table([30])
    later = scored_table([30], seed=1).assign(
        day=1, TransactionID=lambda t: t["TransactionID"] + 100, score=lambda t: t["score"] + 10)
    together = pd.concat([day0, later], ignore_index=True)
    assert top_k_per_day(together, k=5).iloc[:30].tolist() == top_k_per_day(day0, k=5).tolist()


def test_missing_score_raises():
    df = scored_table([5])
    df.loc[2, "score"] = np.nan
    with pytest.raises(ValueError, match="Missing scores"):
        top_k_per_day(df, k=2)
