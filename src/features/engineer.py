"""Engineered features for each transaction.

The features here use only the transaction's own values, so a row's features
do not depend on which other rows are in the table or on their order. That
rules out leakage from future rows by construction. isFraud is never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.split import SECONDS_PER_DAY, TIME

# Fixed facts about the dataset and the calendar, not settings.
AMOUNT = "TransactionAmt"
SECONDS_PER_HOUR = 3_600
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7


def add_time_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_amount, hour and weekday columns.

    log_amount = log(1 + amount) shrinks the long right tail of the amount found
    in the EDA, so a few very large amounts do not dominate the logistic
    regression. The raw amount is kept, because savings and expected loss need
    it in dollars.

    hour and weekday come straight from TransactionDT. Its reference point is
    hidden, so hour 0 is not necessarily midnight and weekday 0 is not
    necessarily Monday. They are positions in the daily and weekly cycle, which
    is what a model needs to pick up a time-of-day or day-of-week pattern.
    """
    seconds = df[TIME]
    return df.assign(
        log_amount=np.log1p(df[AMOUNT]).astype("float32"),
        hour=((seconds // SECONDS_PER_HOUR) % HOURS_PER_DAY).astype("int8"),
        weekday=((seconds // SECONDS_PER_DAY) % DAYS_PER_WEEK).astype("int8"),
    )
