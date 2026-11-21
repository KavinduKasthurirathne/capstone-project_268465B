"""Tests for src/features/build_features.py on a tiny synthetic blocked table. Run with: pytest -q"""
import json

import numpy as np
import pandas as pd

from src.data.split import SECONDS_PER_DAY, add_blocks, check_no_leak
from src.features.build_features import build_features

START = 86_400  # the real data also starts at TransactionDT = 86400


def make_cfg(tmp_path) -> dict:
    """Write a tiny blocked parquet and return a config dict pointing at temporary folders.

    Two transactions a day (cards 1000 and 2000) on days 0-11. With 3-day blocks:
    block 1 = days 0-2, block 2 = days 3-5, block 3 = days 6-8 (tune), and block 4
    = days 9-11 stands in for the final test blocks. Train on blocks 1-2 with a
    1-day gap: days 0-4. Block 4 is the only place the colour "secret" appears.
    """
    days = np.repeat(np.arange(12), 2)
    n = len(days)
    raw = pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "isFraud": np.tile([0, 1], 12),
        "TransactionDT": START + days * SECONDS_PER_DAY + np.tile([0, 3_600], 12),
        "TransactionAmt": np.linspace(5, 120, n).astype("float32"),
        "card1": np.tile([1000, 2000], 12),
        "addr1": np.full(n, 100, dtype="float32"),
        "D1": (days + 1).astype("float32"),  # both cards started on the same day, so 2 uids
        "color": pd.Series(np.where(days >= 9, "secret", np.tile(["red", "blue"], 12)),
                           dtype="category"),
        "empty": np.full(n, np.nan, dtype="float32"),  # 100% missing: must be dropped
    })
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    add_blocks(raw, block_days=3).to_parquet(interim_dir / "blocked.parquet", index=False)
    return {
        "paths": {"interim_dir": interim_dir, "processed_dir": tmp_path / "processed"},
        "files": {"blocked": "blocked.parquet", "train_features": "train.parquet",
                  "tune_features": "tune.parquet", "preprocess_state": "state.json"},
        "split": {"block_days": 3, "gap_days": 1, "train_blocks": [1, 2], "tune_blocks": [3]},
        "preprocess": {"max_missing": 0.95},
    }


def run(tmp_path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_path, tune_path, state_path = build_features(make_cfg(tmp_path))
    assert train_path == tmp_path / "processed" / "train.parquet"
    assert tune_path == tmp_path / "processed" / "tune.parquet"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return pd.read_parquet(train_path), pd.read_parquet(tune_path), state


def test_tables_keep_the_gap_and_pass_the_leakage_check(tmp_path):
    train, tune, _ = run(tmp_path)
    assert sorted(set(train["day"])) == [0, 1, 2, 3, 4]  # day 5 is the gap: in neither table
    assert sorted(set(tune["day"])) == [6, 7, 8]
    check_no_leak(train["day"], tune["day"], gap_days=1)  # run again on the saved tables


def test_final_test_block_never_reaches_the_outputs(tmp_path):
    train, tune, state = run(tmp_path)
    for table in (train, tune):
        assert table["block"].max() <= 3
        assert "secret" not in table["color"].cat.categories  # not even in the category list
    assert "secret" not in state["freq"]["color"]


def test_features_fitted_on_training_rows_and_history_carried_over(tmp_path):
    train, tune, state = run(tmp_path)
    assert state["drop"] == ["empty"]
    for table in (train, tune):
        assert set(state["features"]) <= set(table.columns)
        assert "empty" not in table.columns
    assert "isFraud" in tune.columns and "isFraud" not in state["features"]
    # Red and blue are half each in the training rows (days 0-4).
    assert state["freq"]["color"] == {"red": 0.5, "blue": 0.5}
    # Decision D7: the first block-3 row (card 1000, day 6) still sees that card's
    # 6 earlier rows on days 0-5, including gap day 5, which is in neither table.
    assert tune.sort_values("TransactionID")["card_prev_count"].iloc[0] == 6
