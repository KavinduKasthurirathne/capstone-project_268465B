"""Engineered features for each transaction.

Most features here use only the transaction's own values. The card history
features also use the same card's EARLIER transactions, never later ones. So no
feature can see the future, and none depends on the order of the rows in the
table. isFraud is never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import KEY
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


def add_card_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add card_prev_count, card_prev_mean_amount and card_secs_since_prev.

    Each value describes the same card's (uid's) transactions BEFORE this one,
    because a model in use only knows the past. "Before" means an earlier
    TransactionDT, or the same second and a smaller TransactionID (IDs rise in
    time order). A row never counts in its own history. On a card's first
    transaction the count is 0 and the other two are NaN: there is no earlier
    amount or time to use, and "card not seen before" is itself a signal.

    Run this on all the rows up to the evaluation window together, before the
    split, so evaluation rows still see their card's earlier transactions. That
    is not leakage: a transaction is known when it happens, and only its label
    is delayed.

    Only the four columns needed are sorted, not the whole table. assign puts
    the results back by row label, so the input can be in any order.
    """
    # float64 so the running total over a long card history stays exact enough.
    rows = df[[TIME, KEY, "uid", AMOUNT]].astype({AMOUNT: "float64"}).sort_values([TIME, KEY])
    card = rows.groupby("uid", observed=True)
    count = card.cumcount()  # 0, 1, 2, ... = earlier rows of the same card
    # Running total up to and including this row, minus this row = total of the earlier rows.
    earlier_total = card[AMOUNT].cumsum() - rows[AMOUNT]
    mean_earlier = (earlier_total / count).where(count > 0)
    secs_since_prev = rows[TIME] - card[TIME].shift()
    return df.assign(
        card_prev_count=count.astype("int32"),
        card_prev_mean_amount=mean_earlier.astype("float32"),
        card_secs_since_prev=secs_since_prev.astype("float32"),
    )
