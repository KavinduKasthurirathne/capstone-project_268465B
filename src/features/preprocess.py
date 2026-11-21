"""Train-only column pruning and frequency encoding.

Both are FITTED: they learn something from data (which columns are nearly
empty, how common each category is). They learn it from training rows only
(decision P1), because both can change over time, and evaluation rows must not
shape anything the model is built from. The fitted result is a plain
dictionary, so it can be saved as JSON and checked by eye.
"""
from __future__ import annotations

import pandas as pd

from src.data.load import KEY
from src.data.split import TIME
from src.features.engineer import AMOUNT

LABEL = "isFraud"
# Never model inputs: IDs, time, the label and the card ID. The raw amount is
# kept in the table for savings, but the model gets log_amount instead.
NOT_FEATURES = [KEY, TIME, "day", "block", LABEL, "uid", AMOUNT]
MISSING = "<missing>"  # a missing text value counts as a value of its own (P6)


def with_missing(values: pd.Series) -> pd.Series:
    """Text values with every missing entry replaced by MISSING."""
    return values.astype("object").fillna(MISSING)


def fit_preprocessor(train: pd.DataFrame, max_missing: float) -> dict:
    """Learn from the training rows which columns to drop and how to encode text.

    A column is dropped when MORE than max_missing of its training values are
    missing (P3). Each remaining text column is encoded by the share of training
    rows having each value (frequency encoding): one number per column however
    many values it has, so it suits text columns with hundreds of values. It
    never looks at isFraud, so it cannot leak the label.

    Returns {"drop": columns to drop, "freq": {column: {value: share}},
    "features": the model's input columns}.
    """
    missing_share = train.isna().mean()
    drop = [col for col in train.columns
            if col not in NOT_FEATURES and missing_share[col] > max_missing]
    kept = [col for col in train.columns if col not in NOT_FEATURES and col not in drop]
    text = [col for col in kept if not pd.api.types.is_numeric_dtype(train[col])]
    freq = {col: with_missing(train[col]).value_counts(normalize=True).to_dict() for col in text}
    numeric = [col for col in kept if col not in text]
    return {"drop": drop, "freq": freq, "features": numeric + [f"{col}_freq" for col in text]}


def transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Drop the columns chosen in training and add a <column>_freq per text column.

    Each value gets its share in the TRAINING rows, whatever its share in df. A
    value never seen in training gets 0, as it was that rare there. The original
    text columns stay in the table for the logistic regression's one-hot
    encoding (step 6.1).
    """
    out = df.drop(columns=state["drop"])
    encoded = {f"{col}_freq": with_missing(out[col]).map(shares).fillna(0).astype("float32")
               for col, shares in state["freq"].items()}
    return out.assign(**encoded)
