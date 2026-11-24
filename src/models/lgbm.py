"""LightGBM, the improved model for RQ1.

Gradient boosting adds small trees one at a time, each one correcting the
errors of the trees before it. Too few trees miss patterns; too many start
learning noise in the training rows. So the number of trees is chosen by
early stopping: keep adding trees while the score on held-out rows improves.

The held-out rows must come AFTER the rows the trees learn from, with the same
7-day gap as the main split, and they are the end of the training window, not
block 4, so block 4 stays unseen by both models (D2). Once the number of trees
is known, the model is refitted on the whole training window, so it also
learns from the newest training days.

Trees need no imputation or scaling: a split only asks whether a value is
above a threshold, so the scale does not matter, and each split learns which
side missing values should go to. Text columns enter through their _freq
column (D8), the same features the baseline gets.
"""
from __future__ import annotations

import logging

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from src.data.split import check_no_leak
from src.evaluation.policy import DAY
from src.features.preprocess import LABEL

logger = logging.getLogger(__name__)


def early_stopping_split(days: pd.Series, holdout_days: int,
                         gap_days: int) -> tuple[pd.Series, pd.Series]:
    """Split the training window into a fit part and a later holdout, with a gap.

    The holdout is the last holdout_days days of the window. The fit part ends
    gap_days before the holdout starts, the same label delay as time_split,
    and the days in between are used by neither. Returns two True/False masks.
    """
    first_holdout_day = days.max() - holdout_days + 1
    holdout = days >= first_holdout_day
    fit_part = days < first_holdout_day - gap_days
    if not fit_part.any():
        raise ValueError(f"Training window too short for a {holdout_days}-day holdout "
                         f"and a {gap_days}-day gap")
    return fit_part, holdout


def make_classifier(params: dict, n_trees: int, seed: int) -> LGBMClassifier:
    """An unfitted LightGBM classifier with n_trees trees and the config settings.

    class_weight="balanced" weights the classes the same way as the baseline
    (P2), so RQ1 compares the models, not the weighting. Early stopping watches
    average precision, the same measure as PR-AUC (D10): it judges the ranking,
    as the alert queue does, and the class weights do not distort it.
    deterministic and force_row_wise make the same data, seed and number of
    threads always give the same trees; row-wise is the mode LightGBM picks by
    itself on the real training rows, and the fastest one there.
    """
    return LGBMClassifier(
        n_estimators=n_trees,
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        n_jobs=params["n_jobs"],
        class_weight="balanced",
        metric="average_precision",
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )


def find_best_iteration(train: pd.DataFrame, features: list[str], params: dict,
                        gap_days: int, seed: int) -> int:
    """Number of trees with the best average precision on the holdout.

    Trees are fitted on the fit part only. Training stops once
    early_stopping_rounds trees in a row bring no improvement, or at max_rounds.
    """
    fit_part, holdout = early_stopping_split(train[DAY], params["holdout_days"], gap_days)
    check_no_leak(train.loc[fit_part, DAY], train.loc[holdout, DAY], gap_days)
    logger.info("early stopping: fit part %d rows, days %d-%d | holdout %d rows, days %d-%d | "
                "%d gap rows in neither", fit_part.sum(), train.loc[fit_part, DAY].min(),
                train.loc[fit_part, DAY].max(), holdout.sum(), train.loc[holdout, DAY].min(),
                train.loc[holdout, DAY].max(), len(train) - fit_part.sum() - holdout.sum())

    model = make_classifier(params, params["max_rounds"], seed)
    model.fit(train.loc[fit_part, features], train.loc[fit_part, LABEL],
              eval_set=[(train.loc[holdout, features], train.loc[holdout, LABEL])],
              callbacks=[early_stopping(params["early_stopping_rounds"], verbose=False)])
    best = model.best_iteration_
    logger.info("best number of trees: %d of at most %d (holdout average precision %.4f)",
                best, params["max_rounds"], model.best_score_["valid_0"]["average_precision"])
    return best


def fit_lgbm(train: pd.DataFrame, features: list[str], params: dict,
             gap_days: int, seed: int) -> LGBMClassifier:
    """Find the number of trees by early stopping, then refit on all training rows.

    Only the features columns are given to LightGBM, so IDs, time, the label
    and the raw amount can never be inputs. Returns the fitted classifier:
    predict_proba(df[features])[:, 1] gives fraud scores.
    """
    best = find_best_iteration(train, features, params, gap_days, seed)
    return make_classifier(params, best, seed).fit(train[features], train[LABEL])
