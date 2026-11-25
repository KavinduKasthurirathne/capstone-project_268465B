# Milestone 2 results: preliminary RQ1 note

**Validation results, block 4 only** (days 90-119: 98,615 transactions, 3,898 frauds). These are not the
final results: the test blocks 5-6 are scored once, in Milestone 3. Every number below is taken from
[m2_comparison.md](m2_comparison.md), which `python -m src.evaluation.compare` generates, or is worked out
from it as shown. Alerts follow Policy A (each day's top k transactions by fraud score), and each review
costs $5.

## RQ1: does the gain at a fixed daily budget match the ROC-AUC gain?

**LightGBM beats the logistic baseline at every budget.** Precision@k rises by +0.101 / +0.106 / +0.101 /
+0.069 at k = 30 / 60 / 100 / 150. At k = 100 that is about 10 more frauds in each day's 100 reviews
(0.101 x 100; every day sends exactly 100 alerts), and +$57,132 in savings over the 30 days (+42.0%).
A few good days do not explain the gain: LightGBM has the higher precision@100 on 28 of the 30 days.

**The ROC-AUC gap does not match it.** ROC-AUC rises by only +0.0430 (+5.1%). The precision@k gains are
1.6-2.5 times as large in absolute terms and 2.6-4.1 times as large in relative terms (+13.2% to +20.9%).
Taken alone, the ROC-AUC gap suggests a small improvement, while the analysts would see about 10 more
frauds a day. A likely reason is that ROC-AUC averages over every pair of one fraud and one genuine
transaction, and most of those genuine transactions sit far below the daily cut-off, where no analyst
looks. Precision@k depends only on each day's top k. PR-AUC, which follows precision from the top of the
list down, rises by +0.1101 (+26.2%), much closer to the precision@k gains.

**How the gap is measured matters, though.** ROC-AUC starts at 0.8469, so it has little room left to
grow. Measured as the share of the room to a perfect ranking that LightGBM closes, the picture changes:

| Metric | Difference | Relative change | Share of the room to perfect closed |
| --- | --- | --- | --- |
| ROC-AUC | +0.0430 | +5.1% | 28% |
| PR-AUC | +0.1101 | +26.2% | 19% |
| Precision@30 | +0.101 | +13.2% | 44% |
| Precision@60 | +0.106 | +16.7% | 29% |
| Precision@100 | +0.101 | +20.9% | 20% |
| Precision@150 | +0.069 | +18.0% | 15% |

Room to perfect = perfect value - baseline value. A perfect ranking scores 1 on ROC-AUC and PR-AUC; for
precision@k its value is the ceiling given in m2_comparison.md (1.000 / 1.000 / 0.978 / 0.837 for
k = 30 / 60 / 100 / 150). For example, ROC-AUC: 0.0430 / (1 - 0.8469) = 28%; precision@100:
0.101 / (0.978 - 0.484) = 20%.

On this scale ROC-AUC sits in the middle of the precision@k range, and the precision@k share falls
steadily as k grows, from 44% to 15%: LightGBM's advantage is largest at the very top of the queue. So
even after rescaling, the ROC-AUC gap cannot stand in for the gain at a budget. It is a single number,
while the gain depends on k.

**Preliminary answer.** LightGBM improves on the baseline at every daily budget tested, by about 0.10 in
precision@k for k = 30-100 and 0.069 at k = 150. The ROC-AUC gap points the same way but does not show
how large the gain is at a given budget. In absolute and relative terms it understates the gain; measured
against the room to perfect, it is larger than the gain at some budgets and smaller at others. The gain
therefore has to be measured at the budget the team actually works with.

## Limits of this result

- One validation month (block 4). Blocks 5-6 are not scored until Milestone 3.
- No confidence intervals. "28 of 30 days" is a count, not a significance test; bootstrap intervals are
  Milestone 3.
- Both models use fixed settings, not tuned on block 4 (D11; the baseline keeps scikit-learn's default C),
  so tuning either model could change the gap.
- Transaction-level metrics and Policy A only. Card-level metrics and Policies B and C are Milestone 3.
