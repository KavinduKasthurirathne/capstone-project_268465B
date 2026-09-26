# Fraud Detection Under Analyst Capacity Limits

Evaluating fraud detection models by daily alert budget, monetary savings and performance drift.

**CS5998 Capstone Project** | MSc Data Science & Artificial Intelligence
**Author:** K.K.I. Kasthurirathne (268465B)
**Supervisor:** Dr. Sandareka Wickramanayake
**Status:** Milestone 1 (Project Definition)

---

## Project Scope

| Dimension | Choice |
|---|---|
| Problem type | Detection (binary classification under a fixed review budget) |
| Data type | Tabular, time-stamped transactions |
| Technique category | Supervised ML + unsupervised anomaly detection |
| System context | End-to-end pipeline, comparative analysis, decision-support dashboard |
| External LLM used | **None** |

---

## Overview

Card fraud is not stopped by a model alone. It is stopped by a small team of analysts working through a queue of alerts the model has ranked, and that team can only review a limited number of cases per day. What matters in practice is not overall accuracy but how many of the cases reviewed tomorrow are genuinely fraud, and how much money they represent.

Most public work on the IEEE-CIS dataset ranks models by ROC-AUC on a random split. That measures ordering across the whole dataset and says nothing about the top of a short daily queue, nothing about money saved, and nothing about how fast a model goes stale. A random split also lets future transactions influence a model tested on earlier ones, which cannot happen in deployment.

This project builds a complete fraud detection pipeline and evaluates it the way a fraud operations team would experience it: under a fixed daily alert budget, scored in both detection and financial terms, and tracked month by month.

### Research questions

- **RQ1** Under a fixed daily alert budget, how much does LightGBM improve on a logistic regression baseline, and does that gap match what ROC-AUC suggests?
- **RQ2** Does adding an unsupervised anomaly score produce measurable lift under the same budget?
- **RQ3** Is it better to rank alerts by fraud probability or by expected monetary loss?
- **RQ4** How fast does a frozen model decay month over month, and how much does retraining recover?

---

## Dataset

**Primary:** IEEE-CIS Fraud Detection (Vesta Corporation, hosted on Kaggle)

- 590,540 labelled transactions, 20,663 fraudulent (~3.5%)
- Two tables joined on `TransactionID`; around 430 columns combined
- Time field spans roughly 182 days, giving six 30-day blocks
- Most column names are masked by the provider
- Only the labelled training files are used, as the competition test file has no public labels

**Raw data is not committed to this repository.** Accept the competition rules on Kaggle, then run the download script:

**Fallback:** Bank Account Fraud suite (Feedzai, NeurIPS 2022), used only if access to the primary dataset fails.

---

## Method

**Evaluation design.** Strict time-based split into six 30-day blocks. Train on blocks 1 to 3, tune on block 4, test once on blocks 5 and 6. A 7-day gap sits before each evaluation window to reflect real label delay. No random splitting at any stage.

**Models.**
1. Class-weighted logistic regression (baseline)
2. LightGBM on time-safe engineered features (improved)
3. LightGBM + Isolation Forest anomaly score, run with and without it as an ablation (hybrid)

**Alert policies.** Three ways of turning scores into daily work, each tested at budgets of 30, 60, 100 and 150 alerts per day:
- **A** Top-k by fraud probability
- **B** Top-k by expected loss (probability × amount)
- **C** No cap, flag whenever expected loss exceeds review cost

**Metrics.** PR-AUC and ROC-AUC for ranking; daily precision@k and recall@k at transaction and card level for operations; savings (fraud prevented minus review costs) for financial impact; monthly scores and PSI for drift. Bootstrap confidence intervals and paired tests confirm differences are real.

---

## Milestones

- [ ] **Milestone 1** (Week 4, 25%) Project Definition
- [ ] **Milestone 2** (Week 8, 25%) Technical Checkpoint: preprocessing, EDA, baseline + improved model, architecture diagram, demo video
- [ ] **Milestone 3** (Week 12, 50%) Final report, code, reproducibility checklist

---

## References

- Dal Pozzolo et al. (2018). *Credit card fraud detection: a realistic modeling and a novel learning strategy.* IEEE TNNLS.
- Carcillo et al. (2021). *Combining unsupervised and supervised learning in credit card fraud detection.* Information Sciences.
- Bahnsen et al. (2015). *Example-dependent cost-sensitive decision trees.* Expert Systems with Applications.
- Le Borgne et al. (2022). *Reproducible Machine Learning for Credit Card Fraud Detection: Practical Handbook.*

---
