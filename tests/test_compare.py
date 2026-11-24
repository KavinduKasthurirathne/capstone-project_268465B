"""Tests for src/evaluation/compare.py on tiny synthetic scores. Run with: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.compare import (comparison_table, compare, daily_precision, format_table,
                                    markdown_table)
from src.evaluation.metrics import budget_results, ranking_metrics

BUDGETS = [2, 5]
FIGURES = {"precision_at_k": "m2_precision_at_k.png", "daily_precision": "m2_daily_precision.png",
           "gaps": "m2_gaps.png"}


def make_cfg(tmp_path) -> dict:
    """Save scores and results tables for both models the way stage 3 does, and
    return a config dict pointing at temporary folders.

    3 days of 10 transactions. LightGBM's scores follow the label more closely
    than the baseline's, so it should come out ahead.
    """
    rng = np.random.default_rng(0)
    n = 30
    fraud = (rng.random(n) < 0.3).astype("int8")
    rows = pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "day": np.repeat([0, 1, 2], 10),
        "isFraud": fraud,
        "TransactionAmt": rng.uniform(5, 500, size=n).astype("float32"),
    })
    processed, reports = tmp_path / "processed", tmp_path / "reports"
    processed.mkdir()
    reports.mkdir()
    for model, noise in [("baseline", 1.0), ("lgbm", 0.3)]:
        scored = rows.assign(score=fraud + noise * rng.normal(size=n))
        scored.to_parquet(processed / f"scores_{model}.parquet", index=False)
        results = budget_results(scored, BUDGETS, review_cost=5).assign(**ranking_metrics(scored))
        results.insert(0, "window", "validation (block 4)")
        results.insert(0, "model", model)
        results.to_csv(reports / f"results_{model}.csv", index=False)
    return {
        "seed": 42,
        "paths": {"processed_dir": processed, "reports_dir": reports, "figures_dir": reports / "figures"},
        "files": {"results": "results_{model}.csv", "scores": "scores_{model}.parquet",
                  "comparison": "comparison.md"},
        "evaluation": {"budgets": BUDGETS, "review_cost": 5},
        "compare": {"daily_k": 2, "figures": FIGURES},
    }


def hand_results() -> dict:
    """Two tiny results tables, budgets k = 1 and 2, for the worked example below."""
    base = pd.DataFrame({"k": [1, 2], "precision_at_k": [0.5, 0.25], "recall_at_k": [0.2, 0.4],
                         "savings": [10.0, 20.0], "pr_auc": 0.4, "roc_auc": 0.8})
    new = pd.DataFrame({"k": [1, 2], "precision_at_k": [0.75, 0.5], "recall_at_k": [0.3, 0.4],
                        "savings": [15.0, 10.0], "pr_auc": 0.5, "roc_auc": 0.9})
    return {"baseline": base, "lgbm": new}


def test_comparison_table_worked_by_hand():
    table = comparison_table(hand_results())
    # ROC-AUC and PR-AUC once each (no k), then each budget metric once per k.
    assert table["metric"].tolist() == ["roc_auc", "pr_auc"] + ["precision_at_k"] * 2 \
        + ["recall_at_k"] * 2 + ["savings"] * 2
    assert table["k"].isna().tolist() == [True, True] + [False] * 6
    assert table["k"].dropna().tolist() == [1, 2, 1, 2, 1, 2]
    # difference = LightGBM - baseline; relative change = difference / baseline.
    # ROC-AUC: 0.9 - 0.8 = 0.1, 0.1 / 0.8 = 0.125. Precision@2: 0.5 - 0.25 = 0.25, 0.25 / 0.25 = 1.
    # Savings at k = 2: 10 - 20 = -10, -10 / 20 = -0.5 (LightGBM worse there).
    assert table["difference"].tolist() == pytest.approx([0.1, 0.1, 0.25, 0.25, 0.1, 0.0, 5.0, -10.0])
    assert table["relative_change"].tolist() == pytest.approx([0.125, 0.25, 0.5, 1.0, 0.5, 0.0, 0.5, -0.5])


def test_format_table_and_markdown():
    text = format_table(comparison_table(hand_results()))
    assert text.iloc[0].tolist() == ["ROC-AUC", "all", "0.8000", "0.9000", "+0.1000", "+12.5%"]
    assert text.iloc[-1].tolist() == ["Savings ($)", "2", "20", "10", "-10", "-50.0%"]
    small = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    assert markdown_table(small) == "| a | b |\n| --- | --- |\n| 1 | x |\n| 2 | y |"


def test_daily_precision_worked_by_hand():
    # k = 1. Day 0: IDs 1-3, only ID 1 is fraud. The baseline's top score is ID 2
    # (genuine): precision 0. LightGBM's is ID 1: precision 1. A perfect model: 1.
    # Day 1: no fraud at all, so every model, even a perfect one, gets 0.
    rows = pd.DataFrame({"TransactionID": [1, 2, 3, 4, 5, 6], "day": [0, 0, 0, 1, 1, 1],
                         "isFraud": [1, 0, 0, 0, 0, 0], "TransactionAmt": np.float32(10.0)})
    scores = {"baseline": rows.assign(score=[0.1, 0.9, 0.5, 0.3, 0.2, 0.1]),
              "lgbm": rows.assign(score=[0.9, 0.1, 0.5, 0.3, 0.2, 0.1])}
    daily = daily_precision(scores, k=1, review_cost=5)
    assert daily["baseline"].tolist() == [0.0, 0.0]
    assert daily["lgbm"].tolist() == [1.0, 0.0]
    assert daily["perfect"].tolist() == [1.0, 0.0]


def test_compare_writes_table_and_figures(tmp_path):
    cfg = make_cfg(tmp_path)
    table = compare(cfg)
    # The table's numbers are the ones in the saved results tables.
    for model in ["baseline", "lgbm"]:
        saved = pd.read_csv(tmp_path / "reports" / f"results_{model}.csv")
        assert table.loc[table["metric"] == "precision_at_k", model].tolist() == \
            pytest.approx(saved["precision_at_k"].tolist())
    text = (tmp_path / "reports" / "comparison.md").read_text(encoding="utf-8")
    assert "validation (block 4)" in text and "| Metric | k |" in text
    assert "LightGBM is higher on" in text and "of 3 days" in text
    for file in FIGURES.values():
        assert f"(figures/{file})" in text  # linked relative to the markdown file
        assert (tmp_path / "reports" / "figures" / file).read_bytes()[:4] == b"\x89PNG"


@pytest.mark.parametrize("change", ["another transaction", "another label"])
def test_score_files_with_different_rows_rejected(tmp_path, change):
    cfg = make_cfg(tmp_path)
    path = tmp_path / "processed" / "scores_lgbm.parquet"
    scored = pd.read_parquet(path)
    if change == "another transaction":
        scored.loc[0, "TransactionID"] = 999
    else:
        scored.loc[0, "isFraud"] = 1 - scored.loc[0, "isFraud"]
    scored.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="different rows"):
        compare(cfg)


def test_scores_from_another_run_rejected(tmp_path):
    # New LightGBM scores (as if stage 3 were rerun with other settings) but the old results table.
    cfg = make_cfg(tmp_path)
    path = tmp_path / "processed" / "scores_lgbm.parquet"
    scored = pd.read_parquet(path)
    scored.assign(score=-scored["score"]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="do not reproduce"):
        compare(cfg)
