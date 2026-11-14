"""Tests for src/data/make_dataset.py on tiny synthetic CSVs. Run with: pytest -q"""
import pandas as pd

from src.data.make_dataset import make_dataset


def make_cfg(tmp_path) -> dict:
    """Write two tiny CSVs and return a config dict pointing at temporary folders."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 1, 0],
        "TransactionAmt": [10.5, 99.0, 5.25],
        "card4": ["visa", None, "visa"],
    }).to_csv(raw_dir / "trans.csv", index=False)
    pd.DataFrame({
        "TransactionID": [2],
        "DeviceType": ["mobile"],
    }).to_csv(raw_dir / "ident.csv", index=False)
    return {
        "paths": {"raw_dir": raw_dir, "interim_dir": tmp_path / "interim"},
        "files": {"transaction": "trans.csv", "identity": "ident.csv", "merged": "merged.parquet"},
    }


def test_parquet_written_to_interim_dir(tmp_path):
    cfg = make_cfg(tmp_path)
    out_path = make_dataset(cfg)
    assert out_path == tmp_path / "interim" / "merged.parquet"
    assert out_path.exists()


def test_parquet_keeps_rows_and_types(tmp_path):
    out = pd.read_parquet(make_dataset(make_cfg(tmp_path)))
    assert out["TransactionID"].tolist() == [1, 2, 3]
    assert out["DeviceType"].isna().tolist() == [True, False, True]
    # The smaller types chosen in load.py must survive the save and reload.
    assert out["TransactionAmt"].dtype == "float32"
    assert out["card4"].dtype == "category"
    assert out["DeviceType"].dtype == "category"
    assert out["isFraud"].dtype == "int64"
