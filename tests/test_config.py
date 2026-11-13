"""Tests for config loading and seeding. Run with: pytest -q"""
import random

import numpy as np

from src.utils.config import PROJECT_ROOT, load_config, set_seed


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg["seed"], int)
    assert cfg["files"]["transaction"] == "train_transaction.csv"
    assert cfg["files"]["identity"] == "train_identity.csv"


def test_paths_are_absolute_and_inside_project():
    cfg = load_config()
    for path in cfg["paths"].values():
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents


def test_custom_config_file(tmp_path):
    custom = tmp_path / "config.yaml"
    custom.write_text("seed: 7\npaths:\n  raw_dir: somewhere/raw\n", encoding="utf-8")
    cfg = load_config(custom)
    assert cfg["seed"] == 7
    assert cfg["paths"]["raw_dir"] == PROJECT_ROOT / "somewhere" / "raw"


def test_same_seed_gives_same_draws():
    set_seed(42)
    first = (random.random(), np.random.rand(3).tolist())
    set_seed(42)
    second = (random.random(), np.random.rand(3).tolist())
    assert first == second
