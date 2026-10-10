# IV-only three-family sensitivity v6: frozen execution plan

This simulation study evaluates sensitivity to sample size, fold count, and
learner regularization. Fitting, software tests, and scientific analysis use
an isolated Linux research environment. Prior runs are unchanged.

## Fixed design before new results

Fresh seed 2026100861; disjoint mechanics seed 2026100761. R=200 per cell,
n={300,1000,3000}, p=20, theta=.5, original noise/correlation. Final score is
structural IV-type ONLY. Both original and moderate v5 DGPs are evaluated with
Lasso/RF/XGB in both nuisance channels. The training-only joint Lasso pilot stays
fixed, alpha=.001. No oracle or causal truth enters fitting. Treatment candidates
are [x0,x2,x4,x6] for BOTH levels, including two irrelevant candidates at baseline;
the positive/monotone treatment priors of v5 are retained and disclosed. This is
not unrestricted variable discovery or a universal robustness experiment.

Sample-size/fold grid: all six n x K combinations with baseline model settings.
Parameter grid: baseline plus four ONE-ROLE-AT-A-TIME alternatives, at n=1000,K=2.
No full factorial interaction between n,K and hyperparameter alternatives is
claimed. This yields 10 distinct scenarios, each with 4 design/DGP pairs, two
difficulty levels and 3 learners: 8000 job receipts, 120000 estimates, 600
summary cells. Replications are paired within factors, not independent cells.

Capacity/penalty alternatives are frozen in configs/sensitivity_v6.json:
- RF g leaf=1/2/5; RF ell leaf=1/1/5. The weak-outcome Figure2 RF setting is an
  EXACT baseline alias because the existing leaf=1 is already maximal. Retain
  and identify it, never count it as independent evidence or a distinct model.
- XGB g depth=4/3/2 and ell depth=7/6/4 (weak/base/strong outcome restriction).
- Lasso outcome penalties scaled .5/1/2, with its fixed raw/sigmoid dictionary.
- Treatment alternatives change only RF leaf=5/10/20, XGB L2=2.5/5/10, or
  Lasso alpha=.0005/.001/.002. Outcome and pilot parameters stay unchanged.

## Common randomness and genuine equal-training-size comparison

For each design/DGP/replication, generate one n=3000 X/U/V draw and take prefixes
for each n. Within an n, levels, fold counts and presets share these arrays;
each K uses the same seeded permutation, partitioned into K equal folds.
Model seeds omit K and preset, to avoid changing random state with the factor.

Bank model j trains on ALL observations outside fold j. Cross-fit predicts
fold j with model j. Same-bank reuse predicts fold j with model (j+1) mod K,
which contains those observations in its training set. Both arms use one
prediction per row, the exact same bank, same per-model training size, pilot,
target and fitted parameters. No averaging of K-1 models on the reuse side.
For K=2 this is the previous paired construction with bank indices relabelled.
K=5 trains on .8n instead of .5n: fold contrasts DO NOT isolate fold count from
training fraction. Ordinary full No-split trains on n. Its repeated K labels
are exact aliases, not additional independent experiments.

Figure1 compares genuine naive raw-D ratio and orthogonal IV ratio using exactly
the SAME cross-fit structural g. Figure2 retains full No-split, same-bank reuse,
and cross-fit. All levels/factors get identical nuisance rules within comparisons.

## Endpoints, inference and boundaries

Report every cell's mean bias, individual Student-t MC95 interval, MC SD,
mean SE, MAE, RMSE, nominal95 coverage with Wilson interval, g/m diagnostic RMSE,
and exact weighted IV error decomposition. Practical centering requires the
ENTIRE bias interval in [-.01,.01]; excluding zero detects mean bias and is NOT
the opposite of practical centering. No simulation proves exact unbiasedness.

Primary comparison family: all 360 method contrasts (naive->DML; full/reuse->CF)
with paired one-sided MAE and MSE tests; Holm correction separately per metric.
Factor diagnostics compare K5-K2, n1000-n300, n3000-n1000 and each alternative-base.
Use paired two-sided tests, separate Holm family per factor and metric; bias,
MAE, MSE, coverage and nuisance RMSE changes are reported. No universal direction
is assumed for n or K. The moderate-baseline changes are descriptive paired
diagnostics. All point estimates and uncertainty remain visible, including failures.

No new bias percentages are mixed with historical scores or sample sizes.
Frozen sources, single launch lock, per-job hashes, independent numerical verifier,
DoubleML external-prediction score/SE parity, server-only exact software replay,
all-cell plots and selected original-style histograms are required. Software
failures block final scientific reporting; unfavorable results do not block it.
