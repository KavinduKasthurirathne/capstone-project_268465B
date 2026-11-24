"""Tests for src/models/train.py on tiny synthetic stage 2 tables. Run with: pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import budget_results, ranking_metrics
from src.features.preprocess import fit_preprocessor, transform
from src.models.train import MODELS, train_and_evaluate
from src.utils.config import load_config

BUDGETS = [2, 5]


def make_cfg(tmp_path) -> dict:
    """Write tiny train and tune tables plus their fitted state, the way stage 2
    saves them, and return a config dict pointing at temporary folders.

    30 transactions a day. Training rows: days 0-9. A 2-day gap (days 10-11).
    Tune rows ("block 4"): days 12-14. Fraud is more likely when x is high.
    LightGBM gets a 3-day early-stopping holdout (days 7-9), so its fit part
    is days 0-4, 2 days before it.
    """
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(15), 30)
    n = len(day)
    x = rng.normal(size=n)
    fraud = rng.random(n) < 1 / (1 + np.exp(-(2 * x - 2)))
    x[rng.random(n) < 0.2] = np.nan
    df = pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "day": day,
        "isFraud": fraud.astype("int8"),
        "TransactionAmt": rng.uniform(5, 500, size=n).astype("float32"),
        "x": x.astype("float32"),
        "color": pd.Series(rng.choice(["red", "blue", None], size=n), dtype="category"),
    })
    train, tune = df[df["day"] <= 9], df[df["day"] >= 12]
    state = fit_preprocessor(train, max_missing=0.95)
    processed = tmp_path / "processed"
    processed.mkdir()
    transform(train, state).to_parquet(processed / "train.parquet", index=False)
    transform(tune, state).to_parquet(processed / "tune.parquet", index=False)
    (processed / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return {
        "seed": 42,
        "paths": {"processed_dir": processed, "reports_dir": tmp_path / "reports"},
        "files": {"train_features": "train.parquet", "tune_features": "tune.parquet",
                  "preprocess_state": "state.json", "results": "results_{model}.csv",
                  "scores": "scores_{model}.parquet"},
        "split": {"gap_days": 2, "tune_blocks": [4]},
        "baseline": load_config()["baseline"],
        "lgbm": {**load_config()["lgbm"], "holdout_days": 3},
        "evaluation": {"budgets": BUDGETS, "review_cost": 5},
    }


def rewrite_tune(cfg: dict, **changes) -> None:
    """Overwrite columns of the saved tune table, e.g. to flip its labels."""
    path = cfg["paths"]["processed_dir"] / "tune.parquet"
    pd.read_parquet(path).assign(**changes).to_parquet(path, index=False)


@pytest.mark.parametrize("model", MODELS)
def test_results_csv_matches_calling_the_metrics_directly(tmp_path, model):
    cfg = make_cfg(tmp_path)
    returned = train_and_evaluate(cfg, model)
    saved = pd.read_csv(tmp_path / "reports" / f"results_{model}.csv")
    pd.testing.assert_frame_equal(saved, returned, check_dtype=False)
    assert saved["k"].tolist() == BUDGETS
    assert (saved["model"] == model).all()
    assert (saved["window"] == "validation (block 4)").all()
    # One score per tune row, all probabilities.
    scores = pd.read_parquet(tmp_path / "processed" / f"scores_{model}.parquet")
    assert len(scores) == 90 and scores["day"].min() == 12
    assert scores["score"].between(0, 1).all()
    # The same numbers as calling steps 5.1-5.3 directly on the saved scores.
    expected = budget_results(scores, BUDGETS, review_cost=5).assign(**ranking_metrics(scores))
    pd.testing.assert_frame_equal(saved[expected.columns], expected, check_dtype=False)


@pytest.mark.parametrize("model", MODELS)
def test_block4_labels_never_reach_the_model(tmp_path, model):
    cfg = make_cfg(tmp_path)
    before = train_and_evaluate(cfg, model)
    scores = pd.read_parquet(tmp_path / "processed" / f"scores_{model}.parquet")["score"]
    # Flip every block-4 label: the scores stay the same, only the metrics change.
    labels = pd.read_parquet(tmp_path / "processed" / "tune.parquet")["isFraud"]
    rewrite_tune(cfg, isFraud=(1 - labels).astype("int8"))
    after = train_and_evaluate(cfg, model)
    pd.testing.assert_series_equal(
        pd.read_parquet(tmp_path / "processed" / f"scores_{model}.parquet")["score"], scores)
    assert not before["precision_at_k"].equals(after["precision_at_k"])


def test_leakage_check_stops_the_run(tmp_path):
    cfg = make_cfg(tmp_path)
    rewrite_tune(cfg, day=np.full(90, 10))  # day 10 is inside the gap after training day 9
    with pytest.raises(ValueError, match="Leakage"):
        train_and_evaluate(cfg, "baseline")
    assert not (tmp_path / "reports").exists()  # stopped before writing any results


def test_unknown_model_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown model"):
        train_and_evaluate(make_cfg(tmp_path), "random_forest")
