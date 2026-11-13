"""Load the project config and fix random seeds.

All settings live in configs/config.yaml so that no module hard-codes a number
or a path, and a run can be repeated exactly.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

# src/utils/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read the YAML config and make every entry under `paths` absolute.

    Paths are stored relative to the project root so the repo works wherever it
    is cloned. Resolving them here means scripts and notebooks find the same
    files no matter which folder they are run from.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {name: PROJECT_ROOT / p for name, p in cfg["paths"].items()}
    return cfg


def set_seed(seed: int) -> None:
    """Seed Python's and NumPy's global random number generators.

    This does not cover everything: models such as scikit-learn and LightGBM are
    also given the seed directly when they are created.
    """
    random.seed(seed)
    np.random.seed(seed)
