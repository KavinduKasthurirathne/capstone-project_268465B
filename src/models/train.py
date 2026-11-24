"""Stage 3: fit a model on the training rows, score block 4 and evaluate it.

Run with: python -m src.models.train --model baseline

It reads the stage 2 tables, which hold blocks 1-4 only, so the final test
blocks 5-6 are never touched. The model is fitted on the training rows
(blocks 1-3 minus the gap) and then scores block 4. Block-4 labels are read
only to evaluate the alerts, never to fit anything, so every number here is a
Milestone 2 *validation* result. Step 7.1 adds LightGBM to the same script.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

from src.data.load import KEY
from src.data.make_dataset import save_parquet
from src.data.split import check_no_leak
from src.evaluation.metrics import budget_results, ranking_metrics
from src.evaluation.policy import DAY, SCORE
from src.features.engineer import AMOUNT
from src.features.preprocess import LABEL
from src.models.baseline import fit_baseline, one_hot_columns
from src.utils.config import load_config, set_seed

logger = logging.getLogger(__name__)

MODELS = ["baseline"]
# Besides the score, the policy and metrics need the ID (tie-break), day, label and raw amount.
EVAL_COLUMNS = [KEY, DAY, LABEL, AMOUNT]


def fit_model(name: str, train: pd.DataFrame, state: dict, cfg: dict):
    """Fit the named model on the training rows and return it."""
    if name == "baseline":
        one_hot = one_hot_columns(train, list(state["freq"]), cfg["baseline"]["max_onehot_levels"])
        model = fit_baseline(train, state["features"], one_hot, cfg["baseline"], cfg["seed"])
        logger.info("baseline: %d one-hot columns, converged in %d iterations",
                    len(one_hot), model.named_steps["classifier"].n_iter_[0])
        return model
    raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")


def train_and_evaluate(cfg: dict, name: str) -> pd.DataFrame:
    """Fit, score block 4, evaluate Policy A at every budget, and save the results.

    Writes the results table (one row per budget k, with PR-AUC and ROC-AUC
    repeated on each row) to reports_dir, and the block-4 scores to
    processed_dir for the comparison in step 8.1. Returns the results table.
    """
    processed, files = cfg["paths"]["processed_dir"], cfg["files"]
    state = json.loads((processed / files["preprocess_state"]).read_text(encoding="utf-8"))
    inputs = state["features"] + list(state["freq"])  # model features + the text columns for one-hot

    # Read only the columns needed: the label and day for training, plus what evaluation needs.
    train = pd.read_parquet(processed / files["train_features"], columns=[DAY, LABEL] + inputs)
    tune = pd.read_parquet(processed / files["tune_features"], columns=EVAL_COLUMNS + inputs)
    check_no_leak(train[DAY], tune[DAY], cfg["split"]["gap_days"])
    logger.info("train: %d rows, days %d-%d | block 4: %d rows, days %d-%d | leakage check passed",
                len(train), train[DAY].min(), train[DAY].max(),
                len(tune), tune[DAY].min(), tune[DAY].max())

    start = time.perf_counter()
    model = fit_model(name, train, state, cfg)
    logger.info("fitted %s in %.0f s", name, time.perf_counter() - start)
    del train  # the fitted model holds everything still needed from it

    scored = tune[EVAL_COLUMNS].assign(**{SCORE: model.predict_proba(tune)[:, 1]})
    del tune
    evaluation = cfg["evaluation"]
    results = budget_results(scored, evaluation["budgets"], evaluation["review_cost"])
    results = results.assign(**ranking_metrics(scored))
    tune_blocks = ", ".join(str(b) for b in cfg["split"]["tune_blocks"])
    results.insert(0, "window", f"validation (block {tune_blocks})")
    results.insert(0, "model", name)

    reports_dir = cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / files["results"].format(model=name)
    results.to_csv(results_path, index=False)
    logger.info("saved %s", results_path)
    save_parquet(scored, processed / files["scores"].format(model=name))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a model on the training rows and evaluate it on block 4.")
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(cfg["seed"])
    start = time.perf_counter()
    results = train_and_evaluate(cfg, args.model)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        logger.info("validation results:\n%s", results.to_string(index=False))
    logger.info("done in %.0f s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
