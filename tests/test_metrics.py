"""Tests for src/evaluation/metrics.py on small hand-made tables. Run with: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import daily_metrics, summarise_days
from src.evaluation.policy import top_k_per_day
from src.utils.config import load_config


def toy() -> tuple[pd.DataFrame, pd.Series]:
    """Two days, two alerts a day (k = 2), worked by hand in the test below."""
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6, 7],
        "day": [0, 0, 0, 0, 1, 1, 1],
        "isFraud": [1, 0, 1, 0, 1, 1, 0],
        "TransactionAmt": np.array([100, 50, 70, 20, 30, 40, 60], dtype="float32"),
    })
    alerts = pd.Series([True, True, False, False, True, True, False])
    return df, alerts


def test_toy_example_worked_by_hand():
    # Day 0: alerts on IDs 1 and 2; ID 1 is fraud. 2 frauds that day (IDs 1, 3).
    #   precision 1/2, recall 1/2, amount caught 100, savings 100 - 5 x 2 = 90
    # Day 1: alerts on IDs 5 and 6; both fraud. 2 frauds that day.
    #   precision 2/2, recall 2/2, amount caught 30 + 40 = 70, savings 70 - 5 x 2 = 60
    df, alerts = toy()
    daily = daily_metrics(df, alerts, review_cost=5)
    assert daily["alerts"].tolist() == [2, 2]
    assert daily["frauds"].tolist() == [2, 2]
    assert daily["caught"].tolist() == [1, 2]
    assert daily["precision"].tolist() == [0.5, 1.0]
    assert daily["recall"].tolist() == [0.5, 1.0]
    assert daily["amount_caught"].tolist() == [100.0, 70.0]
    assert daily["savings"].tolist() == [90.0, 60.0]
    # Means over the two days: (0.5 + 1) / 2 = 0.75; totals 4 alerts, 3 caught of 4, 90 + 60.
    assert summarise_days(daily) == {"precision_at_k": 0.75, "recall_at_k": 0.75, "alerts": 4,
                                     "frauds": 4, "caught": 3, "savings": 150.0}


def test_perfect_scores():
    # Day 0 has 5 frauds and day 1 has 2. Every fraud scores above every genuine row.
    fraud = [1] * 5 + [0] * 15 + [1] * 2 + [0] * 18
    df = pd.DataFrame({
        "TransactionID": range(40),
        "day": [0] * 20 + [1] * 20,
        "isFraud": fraud,
        "TransactionAmt": np.float32(10),
        "score": fraud,
    })
    daily = daily_metrics(df, top_k_per_day(df, k=3), review_cost=5)
    assert daily["precision"].tolist() == pytest.approx([1.0, 2 / 3])  # k <= frauds: every alert is fraud
    assert daily["recall"].tolist() == pytest.approx([3 / 5, 1.0])     # k >= frauds: every fraud is caught


def test_day_without_fraud_is_left_out_of_mean_recall():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "day": [0, 0, 1, 1],
        "isFraud": [1, 0, 0, 0],  # day 1 has no fraud
        "TransactionAmt": np.array([80, 10, 10, 10], dtype="float32"),
    })
    daily = daily_metrics(df, pd.Series([True, False, True, False]), review_cost=5)
    assert np.isnan(daily.loc[1, "recall"])      # nothing to catch, so recall is undefined
    assert daily.loc[1, "precision"] == 0.0     # the alert was still sent, and it was wrong
    assert daily.loc[1, "savings"] == -5.0      # one wasted review
    summary = summarise_days(daily)
    assert summary["recall_at_k"] == 1.0         # day 0 alone, not (1 + 0) / 2
    assert summary["precision_at_k"] == 0.5


def test_savings_fall_by_review_cost_per_alert():
    df, alerts = toy()
    cost = load_config()["evaluation"]["review_cost"]
    free = daily_metrics(df, alerts, review_cost=0)["savings"]
    paid = daily_metrics(df, alerts, review_cost=cost)["savings"]
    assert (free - paid).tolist() == [cost * 2, cost * 2]  # 2 alerts a day


def test_amounts_are_added_up_in_float64():
    # float32 has 24 bits for the digits, so 16,777,216 + 1 rounds back to
    # 16,777,216 (2 to the power 24). Added up in float64, the extra dollar is kept.
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "day": [0, 0],
        "isFraud": [1, 1],
        "TransactionAmt": np.array([16_777_216, 1], dtype="float32"),
    })
    daily = daily_metrics(df, pd.Series([True, True]), review_cost=0)
    assert daily["amount_caught"].dtype == "float64"
    # float() first: NumPy compares a plain number with a float32 in float32,
    # where 16,777,217 also rounds to 16,777,216, so the check would always pass.
    assert float(daily.loc[0, "amount_caught"]) == 16_777_217
