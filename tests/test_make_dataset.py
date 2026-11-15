"""Tests for src/data/make_dataset.py on tiny synthetic CSVs. Run with: pytest -q"""
import pandas as pd

from src.data.make_dataset import make_dataset

DAY = 86_400


def make_cfg(tmp_path) -> dict:
    """Write two tiny CSVs and return a config dict pointing at temporary folders.

    The five transactions fall on days 0-4. With 2-day blocks, days 0-1 are
    block 1, days 2-3 are block 2, and day 4 alone is an incomplete block 3.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "isFraud": [0, 1, 0, 0, 0],
        "TransactionDT": [DAY, 2 * DAY, 3 * DAY, 4 * DAY, 5 * DAY],
        "TransactionAmt": [10.5, 99.0, 5.25, 20.0, 7.5],
        "card4": ["visa", None, "visa", "mastercard", "visa"],
    }).to_csv(raw_dir / "trans.csv", index=False)
    pd.DataFrame({
        "TransactionID": [2],
        "DeviceType": ["mobile"],
    }).to_csv(raw_dir / "ident.csv", index=False)
    return {
        "paths": {"raw_dir": raw_dir, "interim_dir": tmp_path / "interim"},
        "files": {"transaction": "trans.csv", "identity": "ident.csv",
                  "merged": "merged.parquet", "blocked": "blocked.parquet"},
        "split": {"block_days": 2, "gap_days": 1, "train_blocks": [1], "tune_blocks": [2]},
    }


def test_parquet_files_written_to_interim_dir(tmp_path):
    merged_path, blocked_path = make_dataset(make_cfg(tmp_path))
    assert merged_path == tmp_path / "interim" / "merged.parquet"
    assert blocked_path == tmp_path / "interim" / "blocked.parquet"
    assert merged_path.exists()
    assert blocked_path.exists()


def test_merged_parquet_keeps_rows_and_types(tmp_path):
    merged_path, _ = make_dataset(make_cfg(tmp_path))
    out = pd.read_parquet(merged_path)
    # The merged file keeps every row, including the incomplete last block.
    assert out["TransactionID"].tolist() == [1, 2, 3, 4, 5]
    assert out["DeviceType"].isna().tolist() == [True, False, True, True, True]
    # The smaller types chosen in load.py must survive the save and reload.
    assert out["TransactionAmt"].dtype == "float32"
    assert out["card4"].dtype == "category"
    assert out["DeviceType"].dtype == "category"
    assert out["isFraud"].dtype == "int64"


def test_blocked_parquet_has_blocks_and_drops_incomplete_block(tmp_path):
    _, blocked_path = make_dataset(make_cfg(tmp_path))
    out = pd.read_parquet(blocked_path)
    assert out["TransactionID"].tolist() == [1, 2, 3, 4]  # day 4 (ID 5) dropped
    assert out["day"].tolist() == [0, 1, 2, 3]
    assert out["block"].tolist() == [1, 1, 2, 2]
    assert out["TransactionAmt"].dtype == "float32"
    assert out["card4"].dtype == "category"
