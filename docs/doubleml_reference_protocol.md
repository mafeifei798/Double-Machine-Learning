# DoubleML Reference Implementation: Study Protocol

This study applies the official DoubleML IV-type procedure to the original
undergraduate project's data-generating processes and learners. It is not an
exact reproduction of an unidentified historical GitHub revision or an
independent preregistration of the research question. Configuration and source
were frozen before accessing this run's formal estimates; earlier archives are
preserved separately.

## Design

- Four cells: Figure 1/2 crossed with linear/nonlinear DGPs, using the original
  functions and coefficients recorded in the archived source snapshot.
- n=1,000, p=20, true effect=0.5, Toeplitz correlation rho=0.7, and independent
  standard-normal outcome and treatment errors. The historical studies used
  n=500/300, so this is not a sample-size-matched historical rerun.
- Figure 1 uses 300 replications per DGP; Figure 2 uses 200.
- Original RF, XGBoost, and Lasso settings, including design-specific linear
  Lasso penalties 0.02/0.05. The same family learns ell, m, and g.
- A fixed two-fold partition is shared across learners and scores within each
  replication.
- Main arms use learned nuisances without Series OLS treatment substitution,
  oracle training, own-outcome residual injection, or outcome-based selection.

## Estimation

Reference implementation: `DoubleML/doubleml-for-py`, commit
`6884b5f9816f180324d854a4d81b22e9d334f5e0`,
`doubleml/plm/plr.py`, functions `_nuisance_est` and `_score_elements`.
The notebook repository reference is commit
`61900b8a8fac8a624af532db32b4be6d504566f0`.

1. Learn ell=E[Y|X] and m=E[D|X] using in-sample or fold-out predictions.
2. Compute the preliminary scalar
   pilot=mean((D-m_hat)*(Y-ell_hat))/mean((D-m_hat)^2).
3. Learn g from Y-pilot*D using the corresponding training scheme.
4. Estimate theta=mean((D-m_hat)*(Y-g_hat))/mean((D-m_hat)*D).

The residual-ratio pilot is an internal step; the final reported estimator is
IV-type, not partialling-out. The official cross-fit pilot pools fold-out
predictions and outcomes. It is therefore not strictly independent of each
held-out outcome: g-model fitting is training-fold-only, but its generated target
depends on the pooled pilot. This is not a fully nested, training-only pilot.
The pooled step is retained for reference-implementation parity, rather than
replaced by the earlier v2 fixed-point/Series pilot.

## Comparisons and pairing

All three methods use the same generated data:

- `nonorthogonal_crossfit`: raw-D score with exactly the same structural-g
  predictions as `iv_crossfit`, isolating the score change.
- `iv_no_split`: full-sample learned nuisances and the official preliminary step.
- `iv_crossfit`: fold-out learned nuisances and the official preliminary step.

The primary Figure 1 comparison is non-orthogonal ML versus cross-fitted IV-type
DML; Figure 2 compares full-sample no-split estimation with cross-fitting. Both
comparisons are reported in every cell. Learner seeds are paired by role across
training schemes, whose training sizes necessarily differ. Neither nuisance
truth nor the causal effect is used to fit a learner or compute the pilot.

## Reporting and execution

Each learner/DGP cell reports mean bias and its Monte Carlo t interval, Monte
Carlo standard deviation, MAE, RMSE, mean sandwich standard error, and nominal
95% coverage with a Wilson interval. Paired absolute-error, squared-error, and
signed-bias contrasts include Monte Carlo intervals. Mean bias is distinct from
MAE; pooled or rounded means do not establish exact unbiasedness, universal
robustness, or uniform cross-fitting superiority.

Finite and nonfinite estimates, explicit failures, and every replication are
retained. Artifacts are committed atomically and bound to configuration and
source hashes. Resumption requires the same identity. Missing or failed estimates
prevent complete-comparison claims; they are neither replaced by zero nor
excluded.

Development uses a separate seed. Formal results do not trigger new seeds,
learner settings, or sample sizes. Worker counts may change without changing
scientific identity.
