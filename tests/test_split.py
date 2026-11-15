"""Tests for src/data/split.py on small synthetic tables. Run with: pytest -q

The tables have no isFraud column, which shows the split never uses labels.
"""
import pandas as pd
import pytest

from src.data.split import SECONDS_PER_DAY, add_blocks, check_no_leak, time_split

START = 86_400  # the real data also starts at TransactionDT = 86400


def one_row_per_day(n_days: int) -> pd.DataFrame:
    """One transaction at the same time of day on each of n_days days."""
    return pd.DataFrame({
        "TransactionID": range(n_days),
        "TransactionDT": [START + d * SECONDS_PER_DAY for d in range(n_days)],
    })


def test_day_counts_whole_days_from_first_transaction():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        # first second, last second of day 0, first second of day 1, day 5
        "TransactionDT": [START, START + SECONDS_PER_DAY - 1,
                          START + SECONDS_PER_DAY, START + 5 * SECONDS_PER_DAY],
    })
    out = add_blocks(df, block_days=3)
    assert out["day"].tolist() == [0, 0, 1, 5]
    assert out["block"].tolist() == [1, 1, 1, 2]


def test_block_spans_and_incomplete_last_block():
    out = add_blocks(one_row_per_day(65), block_days=30)
    assert out.loc[out["block"] == 1, "day"].tolist() == list(range(0, 30))
    assert out.loc[out["block"] == 2, "day"].tolist() == list(range(30, 60))
    # Days 60-64 would form a 5-day third block, so they are dropped.
    assert out["block"].max() == 2
    assert out["day"].max() == 59
    # When the days fill whole blocks exactly, nothing is dropped.
    assert len(add_blocks(one_row_per_day(60), block_days=30)) == 60


def test_gap_left_between_train_and_eval():
    df = add_blocks(one_row_per_day(150), block_days=30)  # 5 blocks
    train_rows, eval_rows = time_split(df, train_blocks=[1, 2, 3], eval_blocks=[4], gap_days=7)
    assert df.loc[train_rows, "day"].tolist() == list(range(0, 83))
    assert df.loc[eval_rows, "day"].tolist() == list(range(90, 120))
    # Gap days 83-89 and the later block 5 are in neither set, and no row is in both.
    assert not (train_rows & eval_rows).any()


def test_leak_check_passes_clean_split():
    df = add_blocks(one_row_per_day(120), block_days=30)
    train_rows, eval_rows = time_split(df, [1, 2, 3], [4], gap_days=7)
    check_no_leak(df.loc[train_rows, "day"], df.loc[eval_rows, "day"], gap_days=7)


@pytest.mark.parametrize("future_day", [83, 95])  # first day of the gap; a day inside block 4
def test_leak_check_fires_on_future_row(future_day):
    df = add_blocks(one_row_per_day(120), block_days=30)
    train_rows, eval_rows = time_split(df, [1, 2, 3], [4], gap_days=7)
    leaky_train_days = pd.concat([df.loc[train_rows, "day"], pd.Series([future_day])])
    with pytest.raises(ValueError, match="Leakage"):
        check_no_leak(leaky_train_days, df.loc[eval_rows, "day"], gap_days=7)


def test_row_order_does_not_change_day_or_block():
    df = one_row_per_day(65)
    in_order = add_blocks(df, 30).set_index("TransactionID")[["day", "block"]]
    shuffled = add_blocks(df.sample(frac=1, random_state=0), 30).set_index("TransactionID")
    pd.testing.assert_frame_equal(shuffled[["day", "block"]].sort_index(), in_order)
