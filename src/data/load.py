"""Read the two training CSVs, shrink their memory use, and join them.

Only train_transaction.csv and train_identity.csv are read. The competition
test files have no labels and are never used.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Column shared by both tables. It is a fixed name in the dataset, not a setting.
KEY = "TransactionID"


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Turn float64 columns into float32 and text columns into category.

    float32 halves the memory of the ~400 numeric columns. Category stores each
    distinct text value once instead of once per row. Integer columns (IDs,
    TransactionDT, isFraud) are left alone so their values stay exact.
    """
    new_types = {col: "float32" for col in df.select_dtypes(include="float64").columns}
    new_types.update({col: "category" for col in df.select_dtypes(include="object").columns})
    return df.astype(new_types)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read one CSV and downcast it straight away, before any join."""
    return downcast(pd.read_csv(path))


def merge_tables(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Left-join identity onto transactions by TransactionID.

    A left join keeps every transaction, including the many with no identity
    row. validate="one_to_one" makes pandas raise an error if either table has
    a repeated ID, so a bad file cannot silently duplicate transactions.
    """
    return transactions.merge(identity, on=KEY, how="left", validate="one_to_one")


def load_raw(cfg: dict) -> pd.DataFrame:
    """Read both training CSVs named in the config and return the merged table."""
    raw_dir = cfg["paths"]["raw_dir"]
    transactions = read_table(raw_dir / cfg["files"]["transaction"])
    identity = read_table(raw_dir / cfg["files"]["identity"])
    return merge_tables(transactions, identity)
