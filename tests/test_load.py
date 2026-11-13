"""Tests for src/data/load.py on tiny synthetic tables. Run with: pytest -q"""
import pandas as pd
import pytest

from src.data.load import downcast, load_raw, merge_tables


def make_transactions() -> pd.DataFrame:
    # 4 transactions; only IDs 1 and 3 will have an identity row.
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "isFraud": [0, 1, 0, 0],
        "TransactionDT": [86400, 86500, 90000, 200000],
        "TransactionAmt": [10.5, 99.0, 5.25, 300.0],
        "card4": ["visa", "mastercard", None, "visa"],
    })


def make_identity() -> pd.DataFrame:
    return pd.DataFrame({
        "TransactionID": [1, 3],
        "id_01": [-5.0, 0.0],
        "DeviceType": ["mobile", "desktop"],
    })


def write_csvs(folder) -> dict:
    """Write both tables as CSVs and return a config dict pointing at them."""
    make_transactions().to_csv(folder / "trans.csv", index=False)
    make_identity().to_csv(folder / "ident.csv", index=False)
    return {
        "paths": {"raw_dir": folder},
        "files": {"transaction": "trans.csv", "identity": "ident.csv"},
    }


def test_downcast_changes_floats_and_text_only():
    out = downcast(make_transactions())
    assert out["TransactionAmt"].dtype == "float32"
    assert out["card4"].dtype == "category"
    # Integers are untouched so IDs, time and labels stay exact.
    assert out["TransactionID"].dtype == "int64"
    assert out["TransactionDT"].dtype == "int64"
    assert out["isFraud"].dtype == "int64"
    # Values survive the change of type (missing text stays missing).
    assert out["TransactionAmt"].tolist() == [10.5, 99.0, 5.25, 300.0]
    assert out["card4"].isna().tolist() == [False, False, True, False]


def test_every_transaction_row_kept():
    merged = merge_tables(downcast(make_transactions()), downcast(make_identity()))
    assert len(merged) == 4
    assert merged["TransactionID"].tolist() == [1, 2, 3, 4]
    # Transactions without an identity row get missing identity columns.
    assert merged["DeviceType"].isna().tolist() == [False, True, False, True]
    assert merged["id_01"].isna().tolist() == [False, True, False, True]


def test_duplicate_identity_ids_rejected():
    identity = pd.concat([make_identity(), make_identity().iloc[[0]]])  # ID 1 twice
    with pytest.raises(pd.errors.MergeError):
        merge_tables(make_transactions(), identity)


def test_no_float64_after_merge():
    # The join fills unmatched rows with NaN; that must not undo the downcast.
    merged = merge_tables(downcast(make_transactions()), downcast(make_identity()))
    assert "float64" not in merged.dtypes.astype(str).tolist()
    assert merged["id_01"].dtype == "float32"
    assert merged["DeviceType"].dtype == "category"


def test_load_raw_reads_csvs_named_in_config(tmp_path):
    cfg = write_csvs(tmp_path)
    merged = load_raw(cfg)
    assert merged.shape == (4, 7)  # 5 transaction columns + 2 identity columns
    assert merged["TransactionAmt"].dtype == "float32"
    assert merged["card4"].dtype == "category"
    assert merged["DeviceType"].dtype == "category"
