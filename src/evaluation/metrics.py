"""Metrics that score each day's alerts the way the fraud team would see them.

The team cares about two things each day: how many of the alerts it reviewed
were fraud (precision@k), and how much of that day's fraud it caught
(recall@k), plus what that was worth in money (savings). Everything here is at
transaction level; card-level metrics are Milestone 3.

The daily functions take the alerts as a True/False column instead of a budget
k, so they score any policy's alerts: Policy A now, Policies B and C in
Milestone 3. PR-AUC and ROC-AUC score the whole ranking instead, ignoring days
and budgets; RQ1 asks whether the two kinds of metric agree.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.policy import DAY, SCORE, top_k_per_day
from src.features.engineer import AMOUNT
from src.features.preprocess import LABEL


def daily_metrics(df: pd.DataFrame, alerts: pd.Series, review_cost: float) -> pd.DataFrame:
    """One row per day: alerts sent, frauds, frauds caught, amount caught,
    precision, recall and savings.

    precision = caught / alerts sent, not / k, so a day with fewer than k
    transactions is not marked down for alerts it could not send (D9).
    recall = caught / that day's frauds.
    savings = fraud amount caught - review_cost x alerts sent: every alert
    costs one review, and a caught fraud is assumed to save its whole amount.

    Amounts are added up in float64: the table stores them as float32, which
    keeps only about 7 significant digits, too few for a month's total to the cent.
    """
    fraud = df[LABEL] == 1
    caught = alerts & fraud
    per_row = pd.DataFrame({
        DAY: df[DAY],
        "alerts": alerts,
        "frauds": fraud,
        "caught": caught,
        "amount_caught": df[AMOUNT].astype("float64").where(caught, 0.0),
    })
    daily = per_row.groupby(DAY).sum()
    daily["precision"] = daily["caught"] / daily["alerts"]
    daily["recall"] = daily["caught"] / daily["frauds"]  # 0 / 0 gives NaN on a day with no fraud
    daily["savings"] = daily["amount_caught"] - review_cost * daily["alerts"]
    return daily


def summarise_days(daily: pd.DataFrame) -> dict:
    """Mean daily precision@k and recall@k (D9), and totals over the window.

    Each day counts equally, as one analyst shift. pandas' mean skips NaN, so a
    day with no fraud is left out of the recall mean instead of counting as 0.
    """
    return {
        "precision_at_k": float(daily["precision"].mean()),
        "recall_at_k": float(daily["recall"].mean()),
        "alerts": int(daily["alerts"].sum()),
        "frauds": int(daily["frauds"].sum()),
        "caught": int(daily["caught"].sum()),
        "savings": float(daily["savings"].sum()),
    }


def ranking_metrics(df: pd.DataFrame) -> dict:
    """PR-AUC and ROC-AUC of the scores over all rows in the window.

    ROC-AUC is the chance that a random fraud scores above a random genuine
    transaction. With about 4% fraud, most of those comparisons are with genuine
    rows far down the list that no analyst will ever see, so it can stay high
    while the top of the queue is poor. PR-AUC (average precision, D10) follows
    precision as the frauds are found from the top down, so it is closer to
    what the analysts experience. A random score gives ROC-AUC 0.5 but PR-AUC
    only about the fraud rate.

    Both are undefined unless the labels hold both classes, so that raises an error.
    """
    if df[LABEL].nunique() < 2:
        raise ValueError("Ranking metrics need both fraud and genuine transactions")
    return {
        "pr_auc": float(average_precision_score(df[LABEL], df[SCORE])),
        "roc_auc": float(roc_auc_score(df[LABEL], df[SCORE])),
    }


def budget_results(df: pd.DataFrame, budgets: list[int], review_cost: float) -> pd.DataFrame:
    """Policy A at each daily budget k: one row per k, with the summary above.

    This is the table steps 6.2 and 7.1 write to their results CSVs.
    """
    rows = []
    for k in budgets:
        daily = daily_metrics(df, top_k_per_day(df, k), review_cost)
        rows.append({"k": k, **summarise_days(daily)})
    return pd.DataFrame(rows)
