# IV-Type DML: Sensitivity Protocol

This study evaluates sample-size, fold-count, and learner-regularization
sensitivity in an isolated Linux research environment. Earlier runs are
preserved separately. The historical protocol identifier is
`iv-only-three-family-sensitivity-v6`.

## Design and replication budget

Formal seed 2026100861 and mechanics seed 2026100761 are disjoint. Each cell uses
200 replications, p=20, true effect=0.5, and the original noise/correlation
structure. The final estimator uses the structural IV-type score.

The original and moderate v5 DGPs are evaluated with Lasso, RF, and XGBoost in
both nuisance channels. The training-only joint Lasso pilot is fixed at
alpha=.001. Neither oracle nuisances nor causal truth enter fitting. Treatment
candidates are [x0,x2,x4,x6] at both difficulty levels, including two irrelevant
baseline candidates. The positive/monotone treatment priors are retained.
This is not unrestricted variable discovery or a universal robustness test.

The sample-size/fold grid crosses n={300,1000,3000} with K={2,5}, using baseline
model settings. A separate parameter grid evaluates baseline and four
one-role-at-a-time alternatives at n=1000,K=2. It is not a full factorial of
sample size, fold count, and hyperparameters.

The resulting 10 distinct scenarios cover four design/DGP pairs, two difficulty
levels, and three learners: 8,000 job receipts, 120,000 estimates, and 600
summary cells. Replications are paired within factors; summary cells are not
independent samples.

## Prespecified capacity and penalty alternatives

Settings are recorded in `configs/sensitivity.json`:

- RF g minimum leaf sizes 1/2/5; RF ell leaf sizes 1/1/5. The weak-outcome
  Figure 2 RF arm is an exact baseline alias because leaf=1 is already maximal
  capacity. It remains identified as an alias, not independent evidence.
- XGBoost g depths 4/3/2 and ell depths 7/6/4 correspond to weak/base/strong
  outcome restriction.
- Lasso outcome penalties are scaled .5/1/2 with the fixed raw/sigmoid dictionary.
- Treatment alternatives change only RF leaf sizes 5/10/20, XGBoost L2 penalties
  2.5/5/10, or Lasso alpha .0005/.001/.002. Outcome and pilot settings are fixed.

## Common randomness and matched model banks

Each design/DGP/replication generates one n=3,000 X/U/V draw; smaller sample
sizes use prefixes. Within each n, difficulty levels, fold counts, and presets
share these arrays. Each K uses the same seeded permutation partitioned into
K equal folds. Model seeds omit K and preset.

Bank model j trains outside fold j. Cross-fitting predicts fold j with model j;
same-bank reuse uses model (j+1) mod K, whose training set contains fold j.
Both arms use one prediction per observation and the same fitted bank,
per-model training size, pilot, target, and parameters. Reuse does not average
K-1 models. At K=2 this is the earlier paired construction with relabelled bank
indices.

K=5 trains on .8n rather than .5n, so fold-count contrasts also change the
training fraction. Full-sample no-split estimation trains on n. Its repeated
K labels are exact aliases, not additional independent experiments.

Figure 1 compares the non-orthogonal raw-D ratio with the orthogonal IV ratio
using identical cross-fit structural-g predictions. Figure 2 retains full-sample
no-split, same-bank reuse, and cross-fitting. Within each comparison, nuisance
rules are identical across levels and factors.

## Endpoints, inference, and interpretation

Each cell reports mean bias, its Student-t Monte Carlo 95% interval, Monte Carlo
standard deviation, mean standard error, MAE, RMSE, nominal 95% coverage with a
Wilson interval, g/m prediction RMSE, and the exact weighted IV error
decomposition. Practical centering requires the entire bias interval within
[-.01,.01]. Excluding zero detects mean bias and is not the opposite of practical
centering. Finite simulations do not prove exact unbiasedness.

The primary family contains 360 method contrasts: non-orthogonal ML versus DML,
and full-sample/same-bank reuse versus cross-fitting. Paired one-sided MAE and
MSE tests use separate Holm correction per metric across this complete family.

Factor diagnostics compare K5-K2, n1000-n300, n3000-n1000, and each alternative
against baseline. Paired two-sided tests use separate Holm families per factor
and metric. Bias, MAE, MSE, coverage, and nuisance-RMSE changes are reported.
No uniform direction is assumed for n or K. Moderate-versus-baseline changes
are descriptive paired diagnostics. All estimates and uncertainty, including
adverse findings and failures, are retained.

## Verification and execution boundaries

New bias percentages are not combined with historical scores or sample sizes.
Frozen source hashes, a single launch lock, per-job hashes, independent numerical
verification, DoubleML external-prediction coefficient/standard-error parity,
server-only exact software replay, all-cell plots, and selected original-style
histograms are required.

Software failures prevent final scientific reporting; unfavorable results do
not. Historical source verification requires the original frozen execution
archive rather than the reorganized presentation source.
