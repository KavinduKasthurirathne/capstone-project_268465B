"""Alert policies: turn model scores into each day's queue of alerts.

The analyst team can review only k cases a day, so what matters is which k
transactions reach the top of each day's queue. Each day is ranked on its own:
the queue is built from that day's transactions for the analysts to work
through, so a day's alerts depend only on that day's scores. Labels are never
read here; they are only used later, to score the alerts.

Policy A is here. Policies B and C (expected loss) are Milestone 3.
"""
from __future__ import annotations

import pandas as pd

from src.data.load import KEY

# Column names in a scored table, not settings.
DAY = "day"
SCORE = "score"


def top_k_per_day(df: pd.DataFrame, k: int) -> pd.Series:
    """Policy A: mark the k highest-scoring transactions of each day.

    A day with fewer than k transactions has all of them marked. When scores
    tie at the cut-off, the lower TransactionID wins, so the same scores always
    give the same alerts, whatever the order of the rows. Returns a True/False
    column (mask) lined up with df, so the caller can count the alerts or pick
    out those rows without copying the table.
    """
    if df[SCORE].isna().any():
        raise ValueError("Missing scores: every transaction needs a score to be ranked")
    ranked = df.sort_values([DAY, SCORE, KEY], ascending=[True, False, True])
    position = ranked.groupby(DAY).cumcount()  # 0 = the day's highest score
    return (position < k).reindex(df.index).rename("alert")
