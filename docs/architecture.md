# System Architecture

The pipeline runs as four stages, each a script run with `python -m`. Each stage reads the files saved by
the stage before it, so a later stage can be rerun without repeating the slow earlier ones. Solid boxes are
built and run in Milestone 2. Dashed boxes are planned for Milestone 3. Boxes with a thick border are where
the time split is protected from leakage.

```mermaid
flowchart TD
    CFG["configs/config.yaml + seed<br>read by every stage"]

    subgraph S1["Stage 1: src.data.make_dataset"]
        RAW[("data/raw<br>train_transaction.csv<br>train_identity.csv")]
        LOAD["Downcast each table,<br>left-join on TransactionID"]
        MERGED[("data/interim<br>train_merged.parquet")]
        BLOCK["Add day and 30-day<br>blocks 1-6, drop the<br>partial block 7"]
        BLOCKED[("data/interim<br>train_blocked.parquet")]
        RAW --> LOAD --> MERGED
        LOAD --> BLOCK --> BLOCKED
    end

    EDA["notebooks/01_eda.ipynb<br>EDA on training rows"]

    subgraph S2["Stage 2: src.features.build_features"]
        READ4["Read blocks 1-4 only<br>(blocks 5-6 stay unread)"]
        FEAT["Features from the current<br>or earlier rows only:<br>log amount, hour, weekday,<br>card ID uid, card history"]
        SPLIT["Time split + leakage check<br>train: days 0-82<br>gap: days 83-89, unused<br>tune: block 4, days 90-119"]
        PREP["Fitted on train rows only:<br>drop columns more than<br>95% missing, and<br>frequency-encode text"]
        TRAINF[("data/processed<br>train_features.parquet")]
        TUNEF[("data/processed<br>tune_features.parquet")]
        STATE[("data/processed<br>preprocess_state.json")]
        READ4 --> FEAT --> SPLIT --> PREP
        PREP --> TRAINF
        PREP --> STATE
        PREP --> TUNEF
    end

    subgraph S3["Stage 3: src.models.train --model baseline / lgbm"]
        FIT["Fit on train rows,<br>class-weighted:<br>logistic regression or<br>LightGBM, time-ordered<br>early stopping (D2)"]
        SCORE["Score block 4"]
        POLA["Policy A: top k per day<br>by score<br>k = 30, 60, 100, 150"]
        MET["precision@k, recall@k,<br>savings, PR-AUC, ROC-AUC"]
        RES[("reports<br>results_baseline.csv<br>results_lgbm.csv")]
        SC[("data/processed<br>scores_baseline.parquet<br>scores_lgbm.parquet")]
        FIT --> SCORE --> POLA --> MET --> RES
        SCORE --> SC
    end

    subgraph S4["Stage 4: src.evaluation.compare"]
        CMP["Baseline vs LightGBM<br>on block 4 (validation)"]
        CMPOUT[("reports/m2_comparison.md<br>reports/figures/m2_*.png")]
        NOTE["RQ1 note by the author<br>reports/m2_results.md"]
        CMP --> CMPOUT --> NOTE
    end

    subgraph M3["Planned for Milestone 3"]
        direction TB
        IF["Isolation Forest score,<br>with and without (RQ2)"]
        POLBC["Policies B and C,<br>review-cost sweep (RQ3)"]
        TEST["Final test on blocks 5-6,<br>scored once"]
        DRIFT["Monthly decay, PSI,<br>retraining (RQ4)"]
        EXTRA["Card-level metrics,<br>calibration, SHAP,<br>bootstrap CIs"]
        DASH["Streamlit dashboard"]
        TEST -.-> DRIFT
    end

    BLOCKED --> EDA
    BLOCKED --> READ4
    TRAINF --> FIT
    STATE --> FIT
    TUNEF --> SCORE
    RES --> CMP
    SC --> CMP
    NOTE -.->|next| M3

    classDef planned stroke-dasharray: 6 4
    classDef guard stroke-width: 3px
    class IF,POLBC,TEST,DRIFT,EXTRA,DASH planned
    class READ4,FEAT,SPLIT,PREP,FIT guard
```

## Stages

| Stage | Command | Reads | Writes |
|---|---|---|---|
| 1 | `python -m src.data.make_dataset` | the two raw training CSVs | `data/interim/train_merged.parquet`, `data/interim/train_blocked.parquet` |
| 2 | `python -m src.features.build_features` | blocks 1-4 of `train_blocked.parquet` | `data/processed/train_features.parquet`, `tune_features.parquet`, `preprocess_state.json` |
| 3 | `python -m src.models.train --model baseline` (or `lgbm`) | the stage 2 files | `reports/results_{model}.csv`, `data/processed/scores_{model}.parquet` |
| 4 | `python -m src.evaluation.compare` | the stage 3 files of both models | `reports/m2_comparison.md`, `reports/figures/m2_*.png` |

Stage 1 is the only code that reads the raw CSVs. The competition test files have no labels and are never read.

## Leakage guards (thick borders)

- **Read blocks 1-4 only.** Stage 2 reads only blocks 1-4, so the final test blocks 5-6 are never loaded in
  Milestone 2.
- **Time split.** Training rows end 7 days before block 4 starts, for label delay. `check_no_leak` checks this
  in stages 1, 2 and 3.
- **Fitted on train rows only.** Column pruning, frequency encoding, the baseline's imputation, scaling and
  one-hot encoding, and LightGBM's early stopping use training rows only. LightGBM's early-stopping holdout is the last 14 days
  of the training window, after its own 7-day gap, so block 4 stays unseen by both models (decision D2).
- **Current or earlier rows only.** Each feature uses only the row itself or the card's earlier transactions,
  and no feature uses `isFraud`. Features are added to blocks 1-4 before the split, so block-4 rows keep their
  card's earlier transactions (decision D7). This is safe because a transaction is known when it happens; the
  7-day gap is for label delay, and labels are never used.

There is no LLM anywhere in the pipeline.
