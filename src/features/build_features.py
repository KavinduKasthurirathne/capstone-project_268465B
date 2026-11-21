"""Stage 2: turn the blocked table into a train table and a block-4 tune table.

Run with: python -m src.features.build_features

Only blocks 1-4 are read from the blocked parquet, so rows from the final test
blocks 5-6 are never loaded in Milestone 2. Then:
1. the engineered features are added to blocks 1-4 together, because card
   history needs each card's earlier rows (decision D7);
2. the rows are split by time with the 7-day gap, and the split is checked;
3. column pruning and frequency encoding are fitted on the training rows only
   (P1) and applied to both tables;
4. the two tables and the fitted state are saved to processed_dir.
Only row and column counts are logged, never label statistics.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from src.data.load import megabytes
from src.data.make_dataset import save_parquet
from src.data.split import check_no_leak, time_split
from src.features.engineer import add_card_history, add_time_amount_features, add_uid
from src.features.preprocess import fit_preprocessor, transform
from src.utils.config import load_config, set_seed

logger = logging.getLogger(__name__)


def drop_unused_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Keep, in each text column's list of categories, only the values in this table.

    The blocked parquet stores each column's category list for all 182 days, so
    even a table of blocks 1-4 lists values first seen in blocks 5-6. Trimming
    the lists keeps those values out of the saved tables altogether.
    """
    text = df.select_dtypes("category").columns
    return df.assign(**{col: df[col].cat.remove_unused_categories() for col in text})


def build_features(cfg: dict) -> tuple[Path, Path, Path]:
    """Build and save the train and tune feature tables and the fitted state.

    Returns the paths of the train table, the tune table and the state file.
    """
    split = cfg["split"]
    blocks = split["train_blocks"] + split["tune_blocks"]
    df = pd.read_parquet(cfg["paths"]["interim_dir"] / cfg["files"]["blocked"],
                         filters=[("block", "in", blocks)])
    logger.info("read blocks %s: %d rows x %d columns, %.0f MB",
                blocks, len(df), df.shape[1], megabytes(df))

    df = add_card_history(add_uid(add_time_amount_features(df)))
    logger.info("added engineered features: %d columns", df.shape[1])

    train_rows, tune_rows = time_split(df, split["train_blocks"],
                                       split["tune_blocks"], split["gap_days"])
    train, tune = df[train_rows], df[tune_rows]
    logger.info("train: %d rows, days %d-%d | tune: %d rows, days %d-%d | %d gap rows in neither",
                len(train), train["day"].min(), train["day"].max(),
                len(tune), tune["day"].min(), tune["day"].max(), len(df) - len(train) - len(tune))
    del df  # the two tables hold everything still needed
    check_no_leak(train["day"], tune["day"], split["gap_days"])
    logger.info("leakage check passed")

    max_missing = cfg["preprocess"]["max_missing"]
    state = fit_preprocessor(train, max_missing)
    logger.info("dropped %d columns more than %.0f%% missing in training rows: %s",
                len(state["drop"]), 100 * max_missing, ", ".join(state["drop"]))
    train = drop_unused_categories(transform(train, state))
    tune = drop_unused_categories(transform(tune, state))
    logger.info("features: %d (%d frequency-encoded text columns) | train %d x %d | tune %d x %d",
                len(state["features"]), len(state["freq"]), *train.shape, *tune.shape)

    out_dir = cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / cfg["files"]["train_features"]
    tune_path = out_dir / cfg["files"]["tune_features"]
    state_path = out_dir / cfg["files"]["preprocess_state"]
    save_parquet(train, train_path)
    save_parquet(tune, tune_path)
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    logger.info("saved %s", state_path)
    return train_path, tune_path, state_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(cfg["seed"])
    start = time.perf_counter()
    build_features(cfg)
    logger.info("done in %.0f s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
