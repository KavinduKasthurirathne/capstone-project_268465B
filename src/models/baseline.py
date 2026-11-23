"""Logistic regression baseline (RQ1): the simple model LightGBM has to beat.

A linear model needs more preparation than a tree model: it cannot take missing
values, columns on large scales pull it around, and it can use text only as
numbers. So every column goes through steps that are FITTED on the training
rows only (P1): median imputation with a missing flag (P4), scaling, and
one-hot encoding for text columns with few values (P6). Fraud rows are given
more weight instead of being resampled (P2).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.preprocess import LABEL


def one_hot_columns(train: pd.DataFrame, text: list[str], max_levels: int) -> list[str]:
    """Text columns with at most max_levels distinct values in the training rows.

    Missing counts as a value of its own (P6). One dummy column per value suits
    a column like ProductCD (5 values) but not DeviceInfo (over 1,300), so
    columns with many values are left to their _freq column alone.
    """
    return [col for col in text if train[col].nunique(dropna=False) <= max_levels]


def fit_baseline(train: pd.DataFrame, features: list[str], one_hot: list[str],
                 params: dict, seed: int) -> Pipeline:
    """Fit the logistic regression and all its preparation on the training rows.

    features are the numeric model inputs (the same list LightGBM gets). Each
    is imputed with its training median, and a 0/1 flag is added for every
    column that had gaps in training, because being missing is itself a fraud
    signal (EDA finding 5). Then each column is scaled to mean 0 and standard
    deviation 1, so the L2 penalty treats all columns alike. A column with no
    gaps in training but gaps later just gets its training median.

    The one_hot text columns get one 0/1 column per training value, missing
    included. A value never seen in training gets all zeros instead of an error.

    class_weight="balanced" weights each class by n_rows / (2 x its row count),
    so fraud and genuine rows count equally in total. Every other column in
    train (IDs, time, label, raw amount) is dropped, so it can never be an input.

    The solver comes from config. Many inputs are near-copies of each other
    (missing flags of columns that go missing together, correlated V columns),
    which makes lbfgs, a gradient method, crawl: on the real training rows it
    had not converged after 1000 iterations. newton-cholesky uses the full
    curvature of the loss, which is cheap with many rows but under a thousand
    inputs, and converged in 8.

    Returns the fitted pipeline: predict_proba(df)[:, 1] gives fraud scores.
    """
    prepare = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), features),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), one_hot),
    ])
    model = Pipeline([
        ("prepare", prepare),
        ("classifier", LogisticRegression(class_weight="balanced", C=params["C"],
                                          solver=params["solver"], max_iter=params["max_iter"],
                                          random_state=seed)),
    ])
    return model.fit(train, train[LABEL])
