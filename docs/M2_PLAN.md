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
| D6 | Review cost = 5, set in config (step 5.2). Friction cost is left out of M2: the M2 savings formula has no friction term, and at 0 it would change nothing. It is added in M3 with the cost sweep. |
| D7 | Card history (step 4.3) is computed on blocks 1–4 together, before the train/tune split (step 4.5), so block-4 rows keep their card's earlier transactions, including those in the gap days. Features never use labels, and a transaction is known when it happens; the 7-day gap is for label delay only. |
| D8 | Never model features: TransactionID, TransactionDT, day, block, isFraud, uid and the raw TransactionAmt. They stay in the tables for evaluation. uid is an ID, and its training frequency would count the card's later transactions. log_amount replaces the raw amount as a feature, and the raw amount is kept for savings. Each text column gets a `<column>_freq` column next to it, and the original stays for the one-hot encoding in step 6.1. |
| D9 | Daily precision@k and recall@k are averaged over days, so each day (one analyst shift) counts equally. Precision divides by the alerts actually sent, min(k, rows that day). A day with no fraud has no recall and is left out of the recall mean. Every block-4 day has at least 2,287 rows, so every day sends k alerts. |
| D10 | PR-AUC is average precision (scikit-learn `average_precision_score`), not the trapezoid area under the PR curve, which overstates it (Davis & Goadrich, 2006). |

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
| 2.1 | 30-day blocks with the trailing partial block dropped; train on blocks 1–3 and tune on block 4 with a 7-day gap; leakage check; stage 1 also saves the blocked parquet | src/data/split.py, src/data/make_dataset.py, configs/config.yaml, tests/test_split.py, tests/test_make_dataset.py | Synthetic tests: block spans, gap respected, leak detector fires on a future row; the real-data split passes | `feat(data): add 30-day blocks and gap-aware split with leakage test` | [x] |

## Phase 3: EDA (1–2 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 3.1 | One notebook on training blocks 1–3, using the blocks from `src/data/split.py`: class balance, daily volume and fraud, missingness, transaction amount, a few key categoricals | notebooks/01_eda.ipynb, reports/figures/eda_*.png, requirements.txt (matplotlib, jupyter) | Notebook executes top to bottom; outputs are aggregates only; no rows from blocks 4–6 | `analysis: add EDA on training blocks 1-3` | [x] |
| 3.2 [author] | Write the findings and the preprocessing decisions, including the missingness threshold | notebooks/01_eda.ipynb | Quoted numbers match the outputs | `docs(eda): add EDA findings and preprocessing decisions` | [x] |

## Phase 4: Preprocessing (2–3 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 4.1 | Log amount, hour, weekday | src/features/\_\_init\_\_.py, src/features/engineer.py, tests/test_features.py | Values in range; shuffling rows does not change any row's values | `feat(features): add log amount, hour and weekday` | [x] |
| 4.2 | Proxy card ID: `uid` = card1 + addr1 + D1n (D3) | src/features/engineer.py, tests/test_features.py, docs/M2_PLAN.md | Same fields give the same uid; isFraud not used; deviation logged | `feat(features): add proxy card ID (card1 + addr1 + D1n)` | [x] |
| 4.3 | Simple past-only card aggregates: number of earlier transactions, mean earlier amount, time since the previous transaction | src/features/engineer.py, tests/test_features.py | Prefix test: values for the first N rows do not change when later rows are added; isFraud not used | `feat(features): add past-only card aggregates` | [x] |
| 4.4 | Drop near-empty columns (threshold from EDA) and frequency-encode categoricals, both fitted on training rows only | src/features/preprocess.py, configs/config.yaml, tests/test_preprocess.py | Fitted state ignores evaluation rows; unseen category → 0; ID and time columns are never features | `feat(features): add train-only column pruning and frequency encoding` | [x] |
| 4.5 | Entry point that turns the blocked parquet into a train table (blocks 1–3 minus the gap) and a tune table (block 4) | src/features/build_features.py | Script logs shapes and dropped columns; leakage check on the outputs; `pytest -q` | `feat(features): add stage 2 script building train and block-4 tables` | [x] |

## Phase 5: Evaluation (1–2 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 5.1 | Policy A: top k transactions per day by score, k = 30/60/100/150 | src/evaluation/\_\_init\_\_.py, src/evaluation/policy.py, configs/config.yaml, tests/test_policy.py | Exactly min(k, day size) alerts per day; highest scores picked; ties broken by TransactionID; row order and other days do not change a day's alerts | `feat(eval): add Policy A daily top-k alerts` | [x] |
| 5.2 | Daily transaction-level precision@k and recall@k (D9), and savings with review cost from config | src/evaluation/metrics.py, configs/config.yaml, tests/test_metrics.py | Toy example worked by hand; perfect scores give precision 1 when k ≤ daily frauds; a day with no fraud is left out of mean recall; savings summed in float64 | `feat(eval): add daily precision@k, recall@k and savings` | [ ] |
| 5.3 | PR-AUC and ROC-AUC (D10), and a results table with one row per budget | src/evaluation/metrics.py, tests/test_metrics.py, requirements.txt (scikit-learn) | Hand-worked AP and ROC-AUC; perfect ranking gives 1; labels with one class rejected; the table matches calling 5.1 and 5.2 directly | `feat(eval): add ranking metrics and per-budget results table` | [ ] |

## Phase 6: Baseline (1–2 sessions)
| Step | Goal | Files | Checks | Commit | Done |
|---|---|---|---|---|---|
| 6.1 | Class-weighted logistic regression: median imputation, missing flags and scaling for numeric columns; one-hot for low-cardinality categoricals | src/models/\_\_init\_\_.py, src/models/baseline.py, tests/test_baseline.py | Fitted statistics come from training rows only; an unseen category does not crash; outputs lie in [0, 1] | `feat(models): add class-weighted logistic regression baseline` | [ ] |
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
| 2026-11-14 | Step 2.1 | Also edits configs/config.yaml and tests/test_make_dataset.py | Split settings belong in config (no magic numbers); stage 1 now needs TransactionDT, which the old synthetic test CSV did not have |
| 2026-11-15 | Step 3.1 | EDA uses the `time_split` training rows (days 0–82), not all of blocks 1–3 (the gap days 83–89 are left out) | These are the exact rows the models and the train-only column pruning (step 4.4) are fitted on, so the threshold chosen in 3.2 is based on the same data |
| 2026-11-15 | Step 3.1 | requirements.txt pins `ipykernel` and `nbconvert` instead of `jupyter` | The `jupyter` metapackage is not installed; these are the two parts the step uses (running the notebook and executing it from the command line) |
| 2026-11-19 | Step 3.2 | Drafted by the AI assistant instead of the author; reviewed by the author before commit | Author's time constraint, at the author's request |
| 2026-11-19 | Step 3.2 | Also adds two notebook outputs: a frauds-per-day column in the daily summary and a table of the 30 most-missing columns | So every number quoted in the findings appears in an output |
| 2026-11-20 | D3, step 4.2 | D1n uses TransactionDT // 86400 − D1 instead of the `day` column. A missing addr1 or D1 becomes "<NA>" in the uid text | The uid then uses only the row's own values, not the table's first transaction. On the real data this day is `day` + 1 on every row, a constant shift that does not change which rows share a uid. Keeping missing as a value gives those rows (11.55% in blocks 1–4) a uid for the card aggregates in step 4.3 |
| 2026-11-21 | Step 4.5 | Also adds tests/test_build_features.py and three output file names in configs/config.yaml | CLAUDE.md: every new module gets tests, and no hard-coded paths |
| 2026-11-21 | Step 4.5 | Before saving, each table's text columns keep only the categories present in that table | The blocked parquet stores category lists for all 182 days. Read back as blocks 1–4, id_31, DeviceInfo and id_33 still listed 22, 240 and 77 values that never occur in those blocks, so the saved tables would have carried value names from blocks 5–6. The fitted frequency tables were never affected, because they count only values present in the training rows |
| 2026-11-21 | Phase 5 | Step 5.2 split into 5.2 (daily precision@k, recall@k, savings) and 5.3 (PR-AUC, ROC-AUC, per-budget results table) | Keeps each commit to one small change; 5.3 is the only one that needs a new library |
| 2026-11-21 | Steps 5.3, 6.1 | scikit-learn is pinned in 5.3 instead of 6.1 | 5.3 is the first step that needs it (PR-AUC and ROC-AUC) |
| 2026-11-21 | D6 | Friction cost is not put in config in M2 | The M2 savings formula (fraud amount caught − review cost × alerts) has no friction term, so the key would be unused; at 0 it changes no M2 number |

## Run log
Filled in from real runs: row counts, memory, per-block counts, timings, best iteration.

| Date | Step | Result |
|---|---|---|
| 2026-11-14 | 1.2 | train_transaction: 590,540 × 394, 2,101 MB → 873 MB after downcast. train_identity: 144,233 × 41, 158 MB → 17 MB. Merged: 590,540 × 434, 936 MB in memory, 76 MB parquet on disk; 399 float32, 31 category, 4 int64 columns. All 144,233 identity rows matched a transaction. Stage 1 took 62 s; the parquet reloads in 0.5 s. |
| 2026-11-14 | Phase 1 check | `pytest -q`: 11 passed in 0.9 s (.venv, Python 3.11.4). Stage 1 rerun from the raw CSVs gave the same shapes and memory as step 1.2, in 61 s. Parquet: 590,540 × 434, TransactionID unique, TransactionDT already ascending, 182 distinct days (day 0–181); reloads in 0.4 s. |
| 2026-11-14 | 2.1 | 585,047 rows kept; the incomplete block 7 (days 180–181, 5,493 rows) dropped. Rows per block: 1: 134,339, 2: 89,399, 3: 92,189, 4: 98,615, 5: 83,571, 6: 86,934. Train: 294,193 rows, days 0–82 (21,734 gap rows on days 83–89 unused). Tune: 98,615 rows, days 90–119. Leakage check passed. train_blocked.parquet is 75 MB. Stage 1 now takes 59–75 s. Peak memory 7.0 GB, reached while reading the CSV; the blocking step does not raise it. |
| 2026-11-15 | 3.1 | EDA on 294,193 training rows (days 0–82): 9,854 frauds (3.35%); by block 2.53% / 4.00% / 4.08%. 371 of 434 columns have missing values; columns above 50/75/90/95/99% missing: 230/77/12/9/0. Notebook executes top to bottom in ~11 s; 4 figures in reports/figures/. |
| 2026-11-19 | 3.2 | Missingness threshold: drop columns more than 95% missing in training rows, which drops 9 columns (id_07, id_08, id_21 to id_27, all 98.9–99.0% missing; the next column, D7, is 93.6%). Training period: 118.72 frauds a day on average (median 117, range 51–234). Preprocessing decisions P1–P6 are recorded in the notebook. |
| 2026-11-19 | 4.1 | Checked on blocks 1–4 (414,542 rows; labels not read): log_amount 0.224–10.372, no NaN; hour 0–23 and weekday 0–6 all present. Clear daily cycle: busiest hour 19 (30,242 rows), quietest hour 9 (1,580), so hour 0 is not local midnight. Weekday shares 11.8–17.7%. Computing the three columns takes 0.01 s. `pytest -q`: 22 passed in 0.84 s. |
| 2026-11-20 | 4.2 | Checked on blocks 1–4 (414,542 rows; labels not read): 164,985 uids (median 1 row, largest 321 rows), against 34,787 card1 + addr1 groups (median 2, largest 7,187) and 12,251 card1 values (median 4, largest 10,242). Among groups with 2+ rows, the share whose card2–card6 values all agree is 95.0% for uid, 89.3% for card1 + addr1 and 89.5% for card1, so uid mixes fewer cards. 11.55% of rows have "<NA>" in their uid. add_uid takes 0.55 s; the uid column uses 16.5 MB. `pytest -q`: 24 passed in 1.58 s. |
| 2026-11-21 | 4.3 | Checked on blocks 1–4 (414,542 rows; labels not read): 39.8% of rows are a card's first transaction (count 0; the other two columns are NaN on exactly these rows). The first-seen share falls from 52.3% in block 1 to 36.5%, 33.1% and 32.2% in blocks 2–4, because block 1 starts with no history; so the training rows hold more first-seen cards than block 4. Earlier transactions per row: median 1, 99th percentile 45, max 320. Days since the previous transaction: median 1.04, max 119.0; 68 rows have 0 s (same card, same second). Prefix check on real data: blocks 1–3 computed alone match the blocks 1–4 run on all 315,927 block 1–3 rows. A deliberately leaky version (each card's mean over all its rows) fails the prefix test. add_card_history takes 0.19 s; the three columns use 4.7 MB. `pytest -q`: 26 passed in 0.86 s. |
| 2026-11-21 | 4.4 | Fitted on the real training rows (294,193 × 443 after the 4.1–4.3 features; labels not used): drops exactly the 9 columns found in the EDA (id_07, id_08, id_21 to id_27). 29 text columns are frequency-encoded, giving 427 features (398 numeric + 29 `_freq`). The state is 83 KB as JSON. In block 4, values never seen in training appear in 3 columns: id_31 in 3.11% of rows, DeviceInfo 0.39%, id_33 0.05%. Fit 2.5 s; transforming train and block 4 takes 3.9 s. `pytest -q`: 31 passed in 0.92 s. |
| 2026-11-21 | 4.5 | `python -m src.features.build_features` in 18 s. Read blocks 1–4 only: 414,542 × 436 (664 MB); 443 columns after the 4.1–4.3 features. Train: 294,193 rows, days 0–82, blocks 1–3. Tune: 98,615 rows, days 90–119, block 4. 21,734 gap rows in neither. Leakage check passed in the script and again on the reloaded files. 9 columns dropped (as in 4.4); 427 features, all numeric; both tables 463 columns. train_features.parquet 45 MB on disk (513 MB in memory), tune_features.parquet 15 MB (173 MB); both reload in 0.3 s. TransactionID unique across the two tables; no unused categories left. With the category trimming switched off, the "secret" test fails as it should. `pytest -q`: 34 passed in 0.97 s. |
| 2026-11-21 | 5.1 | No model scores yet, so checked on block 4's 98,615 rows (30 days, 2,287–6,252 rows a day) with random scores; only TransactionID and day read, labels not read. Every day gets exactly k alerts: 900 / 1,800 / 3,000 / 4,500 alerts for k = 30 / 60 / 100 / 150. top_k_per_day takes 0.03 s per budget. `pytest -q`: 43 passed in 0.95 s. |
