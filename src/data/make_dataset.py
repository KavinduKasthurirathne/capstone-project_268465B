"""Stage 1: turn the raw CSVs into two parquet files.

Run with: python -m src.data.make_dataset

This is the only script that reads the raw CSVs. It saves:
- the merged table, all 182 days, exactly as loaded;
- the blocked table: the same rows with `day` and `block` columns added and the
  incomplete last block dropped. Later stages read this file.

Parquet loads in seconds and keeps the float32/category types. Only row counts
are logged, never label statistics: the files include the final test blocks
5-6, which must stay unseen in Milestone 2.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from src.data.load import load_raw, megabytes
from src.data.split import add_blocks, check_no_leak, time_split
from src.utils.config import load_config, set_seed

logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a table as parquet without the index and log its size on disk."""
    df.to_parquet(path, index=False)
    logger.info("saved %s (%.0f MB on disk)", path, path.stat().st_size / 1024**2)


def make_dataset(cfg: dict) -> tuple[Path, Path]:
    """Load and merge the training tables, add time blocks, save both tables.

    The train/tune split is also run here, on the real data, so every run of
    stage 1 proves the gap holds. Returns the merged and blocked file paths.
    """
    interim_dir = cfg["paths"]["interim_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)
    split = cfg["split"]

    df = load_raw(cfg)
    logger.info("merged: %d rows x %d columns, %.0f MB", len(df), df.shape[1], megabytes(df))
    merged_path = interim_dir / cfg["files"]["merged"]
    save_parquet(df, merged_path)

    blocked = add_blocks(df, split["block_days"])
    logger.info("blocked: %d rows kept, %d dropped in the incomplete last block",
                len(blocked), len(df) - len(blocked))
    del df  # free the merged table; the blocked table holds everything still needed
    counts = blocked["block"].value_counts().sort_index()
    logger.info("rows per block: %s", ", ".join(f"{b}: {n}" for b, n in counts.items()))

    train_rows, tune_rows = time_split(blocked, split["train_blocks"],
                                       split["tune_blocks"], split["gap_days"])
    train_days = blocked.loc[train_rows, "day"]
    tune_days = blocked.loc[tune_rows, "day"]
    check_no_leak(train_days, tune_days, split["gap_days"])
    logger.info("train: %d rows, days %d-%d | tune: %d rows, days %d-%d | leakage check passed",
                len(train_days), train_days.min(), train_days.max(),
                len(tune_days), tune_days.min(), tune_days.max())

    blocked_path = interim_dir / cfg["files"]["blocked"]
    save_parquet(blocked, blocked_path)
    return merged_path, blocked_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(cfg["seed"])
    start = time.perf_counter()
    make_dataset(cfg)
    logger.info("done in %.0f s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
