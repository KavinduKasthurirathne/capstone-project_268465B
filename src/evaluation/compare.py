"""Compare the logistic baseline and LightGBM on block 4 (RQ1).

Run with: python -m src.evaluation.compare
after:    python -m src.models.train --model baseline
          python -m src.models.train --model lgbm

RQ1 asks how much LightGBM gains over the baseline under a fixed daily alert
budget, and whether that gain matches the gain in ROC-AUC. This reads what
stage 3 saved for each model (its results table and its block-4 scores) and
writes one markdown table and three figures. Nothing is refitted and nothing
outside block 4 is read, so every number is a Milestone 2 *validation* result.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.data.load import KEY
from src.evaluation.metrics import budget_results, daily_metrics, ranking_metrics
from src.evaluation.policy import DAY, SCORE, top_k_per_day
from src.features.engineer import AMOUNT
from src.features.preprocess import LABEL
from src.utils.config import load_config, set_seed

logger = logging.getLogger(__name__)

BASELINE, IMPROVED = "baseline", "lgbm"  # the --model names in src/models/train.py
MODELS = [BASELINE, IMPROVED]
NAMES = {BASELINE: "Logistic regression", IMPROVED: "LightGBM"}
# Blue and orange stay easy to tell apart for colour-blind readers too.
COLORS = {BASELINE: "#2a78d6", IMPROVED: "#eb6834"}
GREY = "#52514e"  # reference lines

RANKING_METRICS = ["roc_auc", "pr_auc"]  # one value per model, over the whole window
BUDGET_METRICS = ["precision_at_k", "recall_at_k", "savings"]  # one value per model and budget k
LABELS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "precision_at_k": "Precision@k",
          "recall_at_k": "Recall@k", "savings": "Savings ($)"}
DECIMALS = {"roc_auc": ".4f", "pr_auc": ".4f", "precision_at_k": ".3f", "recall_at_k": ".3f",
            "savings": ",.0f"}


def load_inputs(cfg: dict) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Read each model's results table and block-4 scores, and check they belong together.

    Two checks stop the comparison rather than let it mix runs:
    - both score files must hold the same block-4 rows with the same labels and
      amounts, so the two models are judged on identical transactions;
    - each model's scores must reproduce its results table. If they do not, one
      of the two files is left over from an older run.
    """
    reports, processed = cfg["paths"]["reports_dir"], cfg["paths"]["processed_dir"]
    files, evaluation = cfg["files"], cfg["evaluation"]
    results = {m: pd.read_csv(reports / files["results"].format(model=m)) for m in MODELS}
    scores = {m: pd.read_parquet(processed / files["scores"].format(model=m)) for m in MODELS}

    rows = [KEY, DAY, LABEL, AMOUNT]
    if not scores[BASELINE][rows].equals(scores[IMPROVED][rows]):
        raise ValueError("The two score files hold different rows; rerun stage 3 for both models")

    for m in MODELS:
        recomputed = budget_results(scores[m], evaluation["budgets"], evaluation["review_cost"])
        recomputed = recomputed.assign(**ranking_metrics(scores[m]))
        saved = results[m][recomputed.columns]
        if saved.shape != recomputed.shape or not np.allclose(saved, recomputed):
            raise ValueError(f"{m}: the saved scores do not reproduce the results table; "
                             f"rerun python -m src.models.train --model {m}")
    return results, scores


def comparison_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per metric (and budget k): both models, the difference and the relative change.

    difference = LightGBM - baseline, and relative change = difference / baseline.
    ROC-AUC and PR-AUC score the whole ranking, so they get one row each, with
    k left empty (the results tables repeat them on every budget row).
    """
    base, new = results[BASELINE].set_index("k"), results[IMPROVED].set_index("k")
    rows = [{"metric": m, "k": pd.NA, BASELINE: base[m].iloc[0], IMPROVED: new[m].iloc[0]}
            for m in RANKING_METRICS]
    rows += [{"metric": m, "k": k, BASELINE: base.at[k, m], IMPROVED: new.at[k, m]}
             for m in BUDGET_METRICS for k in base.index]
    table = pd.DataFrame(rows).astype({"k": "Int64"})
    table["difference"] = table[IMPROVED] - table[BASELINE]
    table["relative_change"] = table["difference"] / table[BASELINE]
    return table


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """The comparison table as text: each metric with its own decimals, differences with a sign."""
    rows = []
    for row in table.to_dict("records"):
        decimals = DECIMALS[row["metric"]]
        rows.append({
            "Metric": LABELS[row["metric"]],
            "k": "all" if pd.isna(row["k"]) else str(row["k"]),
            NAMES[BASELINE]: format(row[BASELINE], decimals),
            NAMES[IMPROVED]: format(row[IMPROVED], decimals),
            "Difference": format(row["difference"], "+" + decimals),
            "Relative change": format(row["relative_change"], "+.1%"),
        })
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    """A markdown table of a table of text.

    Written by hand because pandas' to_markdown needs an extra library (tabulate).
    """
    lines = ["| " + " | ".join(df.columns) + " |", "|" + " --- |" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def perfect_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """The same rows scored by a perfect model: score = label, so every fraud
    ranks above every genuine transaction. No real model can do better, so this
    is the ceiling for precision@k: on a day with fewer than k frauds even a
    perfect model must send some genuine alerts."""
    return scored.assign(**{SCORE: scored[LABEL]})


def daily_precision(scores: dict[str, pd.DataFrame], k: int, review_cost: float) -> pd.DataFrame:
    """Precision@k on each day: one column per model, plus the perfect-model ceiling."""
    def precision(scored: pd.DataFrame) -> pd.Series:
        return daily_metrics(scored, top_k_per_day(scored, k), review_cost)["precision"]

    daily = pd.DataFrame({m: precision(scores[m]) for m in MODELS})
    daily["perfect"] = precision(perfect_scores(scores[BASELINE]))  # both files hold the same labels
    return daily


def _tidy(ax) -> None:
    """Light gridlines behind the data and no box around the plot, so the lines stand out."""
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_precision_at_k(results: dict[str, pd.DataFrame], perfect: pd.DataFrame, path: Path) -> None:
    """Figure 1: mean daily precision@k against the budget k, both models and the ceiling."""
    fig = Figure(figsize=(6.5, 4))
    ax = fig.subplots()
    for m in MODELS:
        ax.plot(results[m]["k"], results[m]["precision_at_k"], color=COLORS[m], linewidth=2,
                marker="o", markersize=6, label=NAMES[m])
    ax.plot(perfect["k"], perfect["precision_at_k"], color=GREY, linewidth=1.5, linestyle="--",
            label="Perfect ranking (ceiling)")
    ax.set(xticks=perfect["k"].tolist(), ylim=(0, 1.05), xlabel="Daily alert budget k",
           ylabel="Mean daily precision@k", title="Precision@k on block 4 (validation)")
    ax.legend(frameon=False, loc="lower left")
    _tidy(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_daily_precision(daily: pd.DataFrame, k: int, path: Path) -> None:
    """Figure 2: precision@k on each day, to show whether one model leads on most days
    or only on average."""
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    for m in MODELS:
        ax.plot(daily.index, daily[m], color=COLORS[m], linewidth=2, label=NAMES[m])
    ax.plot(daily.index, daily["perfect"], color=GREY, linewidth=1.5, linestyle="--",
            label="Perfect ranking (ceiling)")
    ax.set(ylim=(0, 1.05), xlabel="Day", ylabel=f"Precision@{k}",
           title=f"Daily precision@{k} on block 4 (validation)")
    ax.legend(frameon=False, loc="lower left", ncols=3)
    _tidy(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_gaps(table: pd.DataFrame, path: Path) -> None:
    """Figure 3: LightGBM's gain over the baseline on ROC-AUC, PR-AUC and precision@k.

    Left: the absolute difference. Right: the difference as a share of the
    baseline's value. Both are shown because a relative change looks larger
    when the baseline's value is small (precision at large k) and smaller when
    it is already high (ROC-AUC).
    """
    rows = table[table["metric"].isin(RANKING_METRICS + ["precision_at_k"])]
    labels = [LABELS[m] if pd.isna(k) else f"Precision@{k}" for m, k in zip(rows["metric"], rows["k"])]
    fig = Figure(figsize=(9, 3.5))
    left, right = fig.subplots(1, 2, sharey=True)
    panels = [(left, rows["difference"], "{:+.3f}", "Difference (LightGBM - baseline)"),
              (right, rows["relative_change"] * 100, "{:+.1f}%", "Relative change (% of baseline)")]
    for ax, values, label_format, xlabel in panels:
        bars = ax.barh(labels, values, height=0.5, color=COLORS[IMPROVED])
        ax.bar_label(bars, fmt=label_format, padding=3)
        ax.axvline(0, color=GREY, linewidth=0.8)
        ax.margins(x=0.25)  # room for the value labels
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
    left.invert_yaxis()  # ROC-AUC on top; the y axis is shared, so this flips both panels
    fig.suptitle("LightGBM gain over the logistic baseline on block 4 (validation)")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def report_markdown(table: pd.DataFrame, perfect: pd.DataFrame, daily: pd.DataFrame, k: int,
                    scored: pd.DataFrame, window: str, review_cost: float,
                    figures: dict[str, str]) -> str:
    """The text of the comparison file: what was compared, the table, and the figures."""
    gain = daily[IMPROVED] - daily[BASELINE]
    ceiling = " / ".join(format(p, ".3f") for p in perfect["precision_at_k"])
    budgets = " / ".join(str(b) for b in perfect["k"])
    lines = [
        f"# Logistic regression vs LightGBM: {window}",
        "",
        "Generated by `python -m src.evaluation.compare`. Do not edit by hand: a rerun overwrites this file.",
        "",
        f"Validation results on days {scored[DAY].min()}-{scored[DAY].max()}: {len(scored):,} transactions, "
        f"{int(scored[LABEL].sum()):,} frauds. Not the final test, which is blocks 5-6, scored once in "
        f"Milestone 3. Policy A (each day's top k transactions by fraud score), review cost ${review_cost} "
        "per alert.",
        "",
        markdown_table(format_table(table)),
        "",
        "Difference = LightGBM - logistic regression. Relative change = difference / logistic regression. "
        "Precision@k and recall@k are means over days (D9); savings are totals over the window.",
        "",
        f"A perfect ranking (every fraud above every genuine transaction) would reach precision@k "
        f"{ceiling} for k = {budgets}: on some days there are fewer frauds than k.",
        "",
        f"Daily precision@{k}: LightGBM is higher on {(gain > 0).sum()} of {len(gain)} days, "
        f"lower on {(gain < 0).sum()} and equal on {(gain == 0).sum()}.",
        "",
    ]
    lines += [f"![{title}]({link})\n" for title, link in figures.items()]
    return "\n".join(lines)


def compare(cfg: dict) -> pd.DataFrame:
    """Read both models' block-4 outputs, then write the comparison file and the three figures.

    Returns the comparison table (numbers, not text).
    """
    results, scores = load_inputs(cfg)
    review_cost, k = cfg["evaluation"]["review_cost"], cfg["compare"]["daily_k"]
    table = comparison_table(results)
    perfect = budget_results(perfect_scores(scores[BASELINE]), cfg["evaluation"]["budgets"], review_cost)
    daily = daily_precision(scores, k, review_cost)

    reports_dir, figures_dir = cfg["paths"]["reports_dir"], cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: figures_dir / file for name, file in cfg["compare"]["figures"].items()}
    plot_precision_at_k(results, perfect, paths["precision_at_k"])
    plot_daily_precision(daily, k, paths["daily_precision"])
    plot_gaps(table, paths["gaps"])
    for path in paths.values():
        logger.info("saved %s", path)

    # Links relative to the markdown file, so they work on GitHub and in a local clone.
    titles = {"precision_at_k": "Precision@k against k", "daily_precision": f"Daily precision@{k}",
              "gaps": "LightGBM gain over the baseline"}
    links = {titles[name]: Path(os.path.relpath(path, reports_dir)).as_posix() for name, path in paths.items()}
    window = results[BASELINE]["window"].iloc[0]
    text = report_markdown(table, perfect, daily, k, scores[BASELINE], window, review_cost, links)
    out = reports_dir / cfg["files"]["comparison"]
    out.write_text(text, encoding="utf-8")
    logger.info("saved %s", out)
    return table


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(cfg["seed"])  # nothing here is random; kept so every entry point starts the same way
    table = compare(cfg)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        logger.info("comparison (validation, block 4):\n%s", format_table(table).to_string(index=False))


if __name__ == "__main__":
    main()
