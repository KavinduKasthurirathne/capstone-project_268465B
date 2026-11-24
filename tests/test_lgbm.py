"""Tests for src/models/lgbm.py on small synthetic tables. Run with: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.models.lgbm import early_stopping_split, find_best_iteration, fit_lgbm, make_classifier
from src.utils.config import load_config

FEATURES = ["x", "n", "noise"]
GAP = 2
# The real settings (which also checks config has every key the model reads),
# with a holdout short enough for a 30-day synthetic training window.
PARAMS = {**load_config()["lgbm"], "holdout_days": 5}


def table(days: int = 30, per_day: int = 60, seed: int = 0) -> pd.DataFrame:
    """A synthetic training window. Fraud is more likely when x is high."""
    rng = np.random.default_rng(seed)
    n = days * per_day
    x = rng.normal(size=n)
    fraud = rng.random(n) < 1 / (1 + np.exp(-(2 * x - 2)))
    x[rng.random(n) < 0.2] = np.nan  # gaps, left as they are: trees handle them
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "day": np.repeat(np.arange(days), per_day),
        "isFraud": fraud.astype("int8"),
        "TransactionAmt": rng.uniform(5, 500, size=n).astype("float32"),
        "x": x.astype("float32"),
        "n": rng.integers(0, 50, size=n),
        "noise": rng.normal(size=n).astype("float32"),
    })


def best_iteration(train: pd.DataFrame) -> int:
    return find_best_iteration(train, FEATURES, PARAMS, GAP, seed=42)


def test_early_stopping_split_worked_by_hand():
    # Days 0-29, 5-day holdout, 2-day gap: holdout 25-29, gap 23-24, fit part 0-22.
    days = pd.Series(range(30))
    fit_part, holdout = early_stopping_split(days, holdout_days=5, gap_days=2)
    assert days[holdout].tolist() == [25, 26, 27, 28, 29]
    assert days[fit_part].tolist() == list(range(23))
    assert days[~fit_part & ~holdout].tolist() == [23, 24]  # used by neither


def test_window_too_short_is_rejected():
    # Days 0-5: the holdout starts on day 1, so no day is 2 days before it.
    with pytest.raises(ValueError, match="too short"):
        early_stopping_split(pd.Series(range(6)), holdout_days=5, gap_days=2)


def test_gap_rows_do_not_affect_early_stopping():
    train = table()
    best = best_iteration(train)
    # Flip the labels of the gap days (23-24): the fit part and holdout are unchanged.
    in_gap = train["day"].between(23, 24)
    flipped_gap = train.assign(isFraud=np.where(in_gap, 1 - train["isFraud"], train["isFraud"]))
    assert best_iteration(flipped_gap) == best
    # Flipping the holdout labels (days 25-29) DOES change it, so the check above can fail.
    in_holdout = train["day"] >= 25
    flipped_holdout = train.assign(isFraud=np.where(in_holdout, 1 - train["isFraud"], train["isFraud"]))
    assert best_iteration(flipped_holdout) != best


def test_final_model_is_refit_on_all_training_rows_with_the_best_tree_count():
    train = table()
    model = fit_lgbm(train, FEATURES, PARAMS, GAP, seed=42)
    best = best_iteration(train)
    assert model.n_estimators == best and model.booster_.num_trees() == best
    # Only the feature columns are inputs: no ID, day, label or raw amount.
    assert model.feature_name_ == FEATURES
    # The same as refitting by hand on every training row, gap days included.
    by_hand = make_classifier(PARAMS, best, seed=42).fit(train[FEATURES], train["isFraud"])
    np.testing.assert_array_equal(model.predict_proba(train[FEATURES]),
                                  by_hand.predict_proba(train[FEATURES]))


def test_same_seed_gives_same_predictions():
    train, evaluation = table(seed=0), table(seed=1)
    first = fit_lgbm(train, FEATURES, PARAMS, GAP, seed=42)
    second = fit_lgbm(train, FEATURES, PARAMS, GAP, seed=42)
    np.testing.assert_array_equal(first.predict_proba(evaluation[FEATURES]),
                                  second.predict_proba(evaluation[FEATURES]))


def test_new_gaps_still_give_one_probability_per_row():
    model = fit_lgbm(table(seed=0), FEATURES, PARAMS, GAP, seed=42)
    evaluation = table(seed=1)
    evaluation["n"] = evaluation["n"].astype("float64")
    evaluation.loc[evaluation.index[:50], "n"] = np.nan  # n had no gaps in training
    scores = model.predict_proba(evaluation[FEATURES])[:, 1]
    assert scores.shape == (len(evaluation),)
    assert not np.isnan(scores).any()
    assert ((scores >= 0) & (scores <= 1)).all()
