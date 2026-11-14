"""Stage 1: turn the raw CSVs into one merged parquet file.

Run with: python -m src.data.make_dataset

This is the only script that reads the raw CSVs. Everything later reads the
parquet file, which loads in seconds and keeps the float32/category types.
No label statistics are logged: the file covers all 182 days, including the
final test blocks 5-6, which must stay unseen in Milestone 2.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from src.data.load import load_raw, megabytes
from src.utils.config import load_config, set_seed

logger = logging.getLogger(__name__)


def make_dataset(cfg: dict) -> Path:
    """Load and merge the training tables, save them as parquet, return the path."""
    df = load_raw(cfg)
    logger.info("merged: %d rows x %d columns, %.0f MB", len(df), df.shape[1], megabytes(df))

    out_path = cfg["paths"]["interim_dir"] / cfg["files"]["merged"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("saved %s (%.0f MB on disk)", out_path, out_path.stat().st_size / 1024**2)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(cfg["seed"])
    start = time.perf_counter()
    make_dataset(cfg)
    logger.info("done in %.0f s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
