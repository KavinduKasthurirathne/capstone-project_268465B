# Milestone 2 Plan: Technical Checkpoint (Week 8, 25%)

Deliverables: data preprocessing, EDA, baseline + improved method, system architecture diagram,
public GitHub repo, demo video (max 6 minutes).

**Rules**
- Everything is built from scratch in this repo. One step = one small change = one commit, made by the author.
- Each step: proposal → "go" → code → `pytest -q` and run → walkthrough → viva questions → commit message
  → `git add` list. The `git add` list includes this file, with the step ticked.
- Keep each piece as simple as possible. No extra abstractions, classes or options unless a step needs them.
- M2 results are on block 4 and labelled *validation results*. Blocks 5–6 are not touched.
- Steps marked [author] are written by the author, not the assistant.
- A new library is pinned with `==` in requirements.txt in the step that first needs it, and committed with
  that step. Step 0.2 cuts requirements.txt back to the libraries needed so far, keeping the already-installed versions.

## Decisions
| ID | Decision |
|---|---|
| D2 | LightGBM early stopping uses the last 14 days of the training window, after a 7-day gap. The model is then refit on the full training window using the best iteration count. |
| D3 | Proxy card ID `uid` = card1 + addr1 + D1n, where D1n = day − D1 (Deotte, 2019). Logged as a PDD deviation. |
| D5 | No label statistics for blocks 5–6 anywhere in M2. |
| D6 | Review cost = 5 and friction cost = 0, both set in config. |

## Phase 0: Setup (1 session)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 0.1 | Relabel the scope row to "LLM used inside the system: None", add a "Use of AI tools" section, update status | README.md, docs/M2_PLAN.md | README renders; scope row and new section present | `docs: add AI-use statement, update status and add M2 plan` | [x] |
| 0.2 | Folder structure, config.yaml, config loader and seed utility | configs/config.yaml, src/\_\_init\_\_.py, src/utils/\_\_init\_\_.py, src/utils/config.py, pytest.ini, tests/test_config.py, requirements.txt (numpy, pyyaml, pytest only) | `pytest -q`: paths resolve under the project root; same seed gives the same numpy draws | `feat(config): add project skeleton, config and seed utility` | [x] |

## Phase 1: Data loading (1–2 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 1.1 | Read both CSVs, downcast each table to float32/category BEFORE merging, then left-join on TransactionID | src/data/\_\_init\_\_.py, src/data/load.py, tests/test_load.py, requirements.txt (pandas) | Tiny synthetic CSVs: every transaction row kept; duplicate identity IDs rejected; float64 → float32; text → category | `feat(data): load, downcast and left-join the training tables` | [x] |
| 1.2 | Entry point that turns the raw CSVs into data/interim/train_merged.parquet | src/data/make_dataset.py, src/data/load.py (memory log), configs/config.yaml (output file name), tests/test_make_dataset.py, requirements.txt (pyarrow) | `python -m src.data.make_dataset` logs rows, columns and memory before/after; `pytest -q` | `feat(data): add stage 1 script saving the merged table to parquet` | [x] |

## Phase 2: Time split (1 session)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 2.1 | 30-day blocks with the trailing partial block dropped; train on blocks 1–3 and tune on block 4 with a 7-day gap; leakage check; stage 1 also saves the blocked parquet | src/data/split.py, src/data/make_dataset.py, tests/test_split.py | Synthetic tests: block spans, gap respected, leak detector fires on a future row; the real-data split passes | `feat(data): add 30-day blocks and gap-aware split with leakage test` | [ ] |

## Phase 3: EDA (1–2 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 3.1 | One notebook on training blocks 1–3, using the blocks from `src/data/split.py`: class balance, daily volume and fraud, missingness, transaction amount, a few key categoricals | notebooks/01_eda.ipynb, reports/figures/eda_*.png, requirements.txt (matplotlib, jupyter) | Notebook executes top to bottom; outputs are aggregates only; no rows from blocks 4–6 | `analysis: add EDA on training blocks 1-3` | [ ] |
| 3.2 [author] | Write the findings and the preprocessing decisions, including the missingness threshold | notebooks/01_eda.ipynb | Quoted numbers match the outputs | `docs(eda): add EDA findings and preprocessing decisions` | [ ] |

## Phase 4: Preprocessing (2–3 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 4.1 | Log amount, hour, weekday | src/features/\_\_init\_\_.py, src/features/engineer.py, tests/test_features.py | Values in range; shuffling rows does not change any row's values | `feat(features): add log amount, hour and weekday` | [ ] |
| 4.2 | Proxy card ID: `uid` = card1 + addr1 + D1n (D3) | src/features/engineer.py, tests/test_features.py, docs/M2_PLAN.md | Same fields give the same uid; isFraud not used; deviation logged | `feat(features): add proxy card ID (card1 + addr1 + D1n)` | [ ] |
| 4.3 | Simple past-only card aggregates: number of earlier transactions, mean earlier amount, time since the previous transaction | src/features/engineer.py, tests/test_features.py | Prefix test: values for the first N rows do not change when later rows are added; isFraud not used | `feat(features): add past-only card aggregates` | [ ] |
| 4.4 | Drop near-empty columns (threshold from EDA) and frequency-encode categoricals, both fitted on training rows only | src/features/preprocess.py, configs/config.yaml, tests/test_preprocess.py | Fitted state ignores evaluation rows; unseen category → 0; ID and time columns are never features | `feat(features): add train-only column pruning and frequency encoding` | [ ] |
| 4.5 | Entry point that turns the blocked parquet into a train table (blocks 1–3 minus the gap) and a tune table (block 4) | src/features/build_features.py | Script logs shapes and dropped columns; leakage check on the outputs; `pytest -q` | `feat(features): add stage 2 script building train and block-4 tables` | [ ] |

## Phase 5: Evaluation (1–2 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 5.1 | Policy A: top k transactions per day by score, k = 30/60/100/150 | src/evaluation/\_\_init\_\_.py, src/evaluation/policy.py, configs/config.yaml, tests/test_policy.py | Exactly min(k, day size) alerts per day; highest scores picked; ties broken by TransactionID | `feat(eval): add Policy A daily top-k alerts` | [ ] |
| 5.2 | PR-AUC, ROC-AUC, daily transaction-level precision@k and recall@k, and savings with review cost from config | src/evaluation/metrics.py, configs/config.yaml, tests/test_metrics.py | Toy example worked by hand; perfect scores give precision 1 when k ≤ daily frauds; savings summed in float64 | `feat(eval): add ranking, precision@k, recall@k and savings metrics` | [ ] |

## Phase 6: Baseline (1–2 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 6.1 | Class-weighted logistic regression: median imputation, missing flags and scaling for numeric columns; one-hot for low-cardinality categoricals | src/models/\_\_init\_\_.py, src/models/baseline.py, tests/test_baseline.py, requirements.txt (scikit-learn) | Fitted statistics come from training rows only; an unseen category does not crash; outputs lie in [0, 1] | `feat(models): add class-weighted logistic regression baseline` | [ ] |
| 6.2 ★ | First full end-to-end result: train on blocks 1–3, score block 4, then Policy A + metrics | src/models/train.py, reports/results_baseline.csv | `python -m src.models.train --model baseline` runs end to end; results CSV written; `pytest -q` | `feat(models): run baseline end to end on block 4` | [ ] |

## Phase 7: Improved model (1–2 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 7.1 | LightGBM with class weights and early stopping as in D2, scored on block 4 | src/models/lgbm.py, src/models/train.py, configs/config.yaml, tests/test_lgbm.py, reports/results_lgbm.csv, requirements.txt (lightgbm) | Early-stopping holdout is later than its training part, with a 7-day gap; same seed gives the same predictions; script runs | `feat(models): add LightGBM with time-ordered early stopping` | [ ] |

## Phase 8: Comparison (1 session)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 8.1 | Results table and 2–3 figures comparing baseline and LightGBM on block 4: precision@k vs k, daily precision@100, ROC-AUC gap vs precision@k gap | src/evaluation/compare.py, reports/m2_results.md, reports/figures/m2_*.png | `python -m src.evaluation.compare` writes the table and the figures | `analysis: compare baseline and LightGBM on block 4` | [ ] |
| 8.2 [author] | Short RQ1 note: does the precision@k gap match the ROC-AUC gap? | reports/m2_results.md | Quoted numbers match the table | `docs: add preliminary RQ1 note` | [ ] |

## Phase 9: Packaging (1–2 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 9.1 | Mermaid architecture diagram showing what is done and what is planned for M3 | docs/architecture.md | Renders on GitHub | `docs: add system architecture diagram` | [ ] |
| 9.2 | README results and reproduction commands | README.md | Every command runs once from a fresh clone | `docs: add M2 results and reproduction steps to README` | [ ] |
| 9.3 | Demo video outline (6 minutes or less) with timings and commands | docs/demo_outline.md | Timings add up to 6:00 or less | `docs: add demo video outline` | [ ] |

The author creates the `m2` tag after 9.3.

## Deferred to M3
PSI stability screen, card-level precision/recall, Isolation Forest hybrid, Policies B and C,
drift study, calibration, SHAP, bootstrap confidence intervals, Streamlit dashboard.

## Deviations log
| Date | Item | Deviation | Reason |
|---|---|---|---|
| 2026-11-13 | D3, step 4.2 | `uid` adds D1n to the PDD's "card and address fields" | Card and address fields alone merge many cards into one; D1n stays constant per card (Deotte, 2019) |
| 2026-11-13 | D2, step 7.1 | Early stopping uses the end of the training window, not block 4 | Keeps block 4 unseen by both models, so the RQ1 gap is not biased towards LightGBM |
| 2026-11-13 | Scope | PSI screen and card-level metrics moved to M3 | Keeps M2 simple |

## Run log
Filled in from real runs: row counts, memory, per-block counts, timings, best iteration.

| Date | Step | Result |
|---|---|---|
| 2026-11-14 | 1.2 | train_transaction: 590,540 × 394, 2,101 MB → 873 MB after downcast. train_identity: 144,233 × 41, 158 MB → 17 MB. Merged: 590,540 × 434, 936 MB in memory, 76 MB parquet on disk; 399 float32, 31 category, 4 int64 columns. All 144,233 identity rows matched a transaction. Stage 1 took 62 s; the parquet reloads in 0.5 s. |
