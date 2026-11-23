"""Tests for src/models/baseline.py on small synthetic tables. Run with: pytest -q"""
import numpy as np
import pandas as pd

from src.models.baseline import fit_baseline, one_hot_columns
from src.utils.config import load_config

FEATURES = ["x", "n", "color_freq"]
ONE_HOT = ["color"]
PARAMS = load_config()["baseline"]  # also checks the config has the keys the model needs


def table(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows: block 1 = training, block 2 = evaluation. Fraud is more
    likely when x is high or color is "red", so the model has something to learn."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    color = rng.choice(["red", "blue", "green"], size=n)
    fraud = rng.random(n) < 1 / (1 + np.exp(-(2 * x + (color == "red") - 3)))
    x[rng.random(n) < 0.2] = np.nan  # gaps in x, so it gets a missing flag
    color[rng.random(n) < 0.1] = None
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "block": np.where(np.arange(n) < 2 * n // 3, 1, 2),
        "isFraud": fraud.astype("int8"),
        "x": x.astype("float32"),
        "n": rng.integers(0, 50, size=n),  # no gaps
        "color": pd.Series(color, dtype="category"),
    })
    df["color_freq"] = df["color"].map(df["color"].value_counts(normalize=True)).astype("float32")
    return df


def fit(train: pd.DataFrame):
    return fit_baseline(train, FEATURES, ONE_HOT, PARAMS, seed=42)


def test_one_hot_threshold_is_inclusive_and_counts_missing():
    train = pd.DataFrame({
        "three": pd.Series(["a", "b", "c", "a"], dtype="category"),
        "two_and_missing": pd.Series(["a", "b", None, "a"], dtype="category"),
        "four": pd.Series(["a", "b", "c", "d"], dtype="category"),
    })
    assert one_hot_columns(train, list(train.columns), max_levels=3) == ["three", "two_and_missing"]
    assert one_hot_columns(train, list(train.columns), max_levels=2) == []


def test_fitted_statistics_come_from_training_rows_only():
    df = table()
    train = df[df["block"] == 1]
    model = fit(train)
    imputer, scaler = model.named_steps["prepare"].named_transformers_["numeric"]
    # Medians and means worked out directly from the training rows.
    medians = train[FEATURES].median()
    np.testing.assert_allclose(imputer.statistics_, medians, rtol=1e-6)
    np.testing.assert_allclose(scaler.mean_[:len(FEATURES)],
                               train[FEATURES].fillna(medians).mean(), rtol=1e-5)
    # Change every evaluation row: the model fitted on the training rows stays the same.
    changed = df.copy()
    is_eval = changed["block"] == 2
    changed.loc[is_eval, ["x", "n", "color_freq"]] = 1000.0
    changed.loc[is_eval, "isFraud"] = 1
    coef = model.named_steps["classifier"].coef_
    np.testing.assert_array_equal(fit(changed[~is_eval]).named_steps["classifier"].coef_, coef)
    # Fitting on all rows WOULD change it, so the check above can fail.
    assert not np.allclose(fit(changed).named_steps["classifier"].coef_, coef)


def test_inputs_are_features_missing_flags_and_dummies_only():
    df = table()
    model = fit(df[df["block"] == 1])
    # 3 scaled features + 1 missing flag (only x has gaps) + 4 dummies (red, blue,
    # green, missing). No ID, block or label column becomes an input.
    assert model.named_steps["classifier"].coef_.shape == (1, 3 + 1 + 4)


def test_fit_converges_within_the_iteration_limit():
    # scikit-learn only warns when the solver stops early, so check it here:
    # stopping at the limit would leave the baseline under-trained.
    df = table()
    clf = fit(df[df["block"] == 1]).named_steps["classifier"]
    assert clf.n_iter_[0] < PARAMS["max_iter"]


def test_unseen_category_and_new_gaps_still_give_scores():
    df = table()
    model = fit(df[df["block"] == 1])
    evaluation = df[df["block"] == 2].copy()
    evaluation["color"] = evaluation["color"].cat.add_categories("purple")
    evaluation.loc[evaluation.index[:5], "color"] = "purple"  # never seen in training
    evaluation["n"] = evaluation["n"].astype("float64")
    evaluation.loc[evaluation.index[5:10], "n"] = np.nan    # n had no gaps in training
    scores = model.predict_proba(evaluation)[:, 1]
    assert not np.isnan(scores).any()


def test_scores_are_probabilities_one_per_row():
    df = table()
    model = fit(df[df["block"] == 1])
    evaluation = df[df["block"] == 2]
    scores = model.predict_proba(evaluation)[:, 1]
    assert scores.shape == (len(evaluation),)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_scoring_needs_no_label_or_ids():
    df = table()
    model = fit(df[df["block"] == 1])
    evaluation = df[df["block"] == 2]
    stripped = evaluation.drop(columns=["isFraud", "TransactionID", "block"])
    np.testing.assert_array_equal(model.predict_proba(stripped), model.predict_proba(evaluation))


def test_same_seed_gives_same_scores():
    df = table()
    train, evaluation = df[df["block"] == 1], df[df["block"] == 2]
    np.testing.assert_array_equal(fit(train).predict_proba(evaluation),
                                  fit(train).predict_proba(evaluation))
