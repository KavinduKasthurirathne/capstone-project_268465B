"""Split the data by time into 30-day blocks, with a gap before each evaluation window.

A fraud model is always used on transactions that happen after the ones it
learned from, so it must be tested the same way. Rows are never split at random.
The gap reflects label delay: fraud is confirmed days after a transaction, so
the newest labels would not yet be known when the model is trained.
"""
from __future__ import annotations

import pandas as pd

# Fixed facts about the dataset, not settings.
TIME = "TransactionDT"  # seconds from a hidden reference point
SECONDS_PER_DAY = 86_400


def add_blocks(df: pd.DataFrame, block_days: int) -> pd.DataFrame:
    """Add `day` and `block` columns, and drop the incomplete block at the end.

    Day 0 is the day of the first transaction. Block 1 holds days
    0 .. block_days-1, block 2 the next block_days days, and so on. The last
    block is dropped if it is shorter than block_days, so every block covers the
    same length of time and blocks can be compared fairly.
    """
    day = (df[TIME] - df[TIME].min()) // SECONDS_PER_DAY
    block = day // block_days + 1
    full_blocks = (day.max() + 1) // block_days
    keep = block <= full_blocks
    return df[keep].assign(day=day[keep], block=block[keep])


def time_split(df: pd.DataFrame, train_blocks: list[int], eval_blocks: list[int],
               gap_days: int) -> tuple[pd.Series, pd.Series]:
    """Mark the training rows and the evaluation rows, with a gap between them.

    Evaluation rows are all rows in eval_blocks. Training rows are rows in
    train_blocks that fall at least gap_days before the first evaluation day;
    rows inside the gap are used for neither. Two True/False columns (masks) are
    returned instead of two tables, so the large table is not copied just to
    decide which rows go where.
    """
    eval_rows = df["block"].isin(eval_blocks)
    first_eval_day = df.loc[eval_rows, "day"].min()
    train_rows = df["block"].isin(train_blocks) & (df["day"] < first_eval_day - gap_days)
    return train_rows, eval_rows


def check_no_leak(train_days: pd.Series, eval_days: pd.Series, gap_days: int) -> None:
    """Raise an error if any training row is inside the gap or after it.

    This checks the finished split, not the code that made it, so it can also be
    run on tables built later (for example the stage 2 feature tables). It
    raises an error instead of using assert, because Python skips asserts when
    run with -O.
    """
    last_train_day = train_days.max()
    first_eval_day = eval_days.min()
    if last_train_day >= first_eval_day - gap_days:
        raise ValueError(
            f"Leakage: last training day {last_train_day} is not at least "
            f"{gap_days} days before the first evaluation day {first_eval_day}"
        )
