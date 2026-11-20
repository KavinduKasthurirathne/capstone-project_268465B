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
CARD = "card1"
ADDRESS = "addr1"
CARD_AGE = "D1"  # days since the card was first used
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


def add_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Add uid, a proxy card ID made of card1, addr1 and D1n (decision D3).

    The data has no card number. card1 and addr1 alone put many different cards
    into one group. D1 counts the days since the card was first used, so
    D1n = day - D1 is the day the card started. It stays the same on every
    transaction of one card and separates cards that share card1 and addr1.

    The day is taken from TransactionDT, not from the `day` column, so the uid
    uses only the row's own values. It is one more than `day` on every row, and
    shifting every row by the same amount does not change which rows share a uid.

    The uid is text, such as "13926_315_-3", so the same three values always give
    the same uid, whatever other rows are in the table. A missing addr1 or D1
    shows as "<NA>": missing counts as one more value, so those rows still get a
    uid. Converting to whole numbers first raises an error if a value has a
    fraction, instead of quietly rounding it.
    """
    day = df[TIME] // SECONDS_PER_DAY
    card_start = day - df[CARD_AGE]
    card, address, start = (s.astype("Int64").astype(str)
                            for s in (df[CARD], df[ADDRESS], card_start))
    uid = card + "_" + address + "_" + start
    return df.assign(uid=uid.astype("category"))
