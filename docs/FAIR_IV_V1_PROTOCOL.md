# Fair IV-type rerun v1 (2026-10-02)

This is a new adaptation of the official DoubleML IV-type procedure to the
original thesis DGPs and learners. It is NOT an exact reproduction of an
unidentified historical GitHub revision, nor an independent preregistration of
the research question. The configuration and source are locked before accessing
this run's formal estimates. Old archives are not overwritten.

## Fixed design

- Four original cells: Figure 1/2, linear/nonlinear; same functions and
  coefficients as `legacy_snapshot/` (including different Figure 1/2 DGPs).
- n=1,000, p=20, true effect=0.5, Toeplitz rho=0.7, independent standard-normal
  outcome/treatment errors. The sample size differs from historical n=500/300,
  so this is not an exact historical rerun.
- Original Monte Carlo counts: Figure 1 R=300, Figure 2 R=200 per DGP.
- Original RF/XGBoost/Lasso settings, including the original design-specific
  linear Lasso penalties 0.02/0.05. The SAME learner family fits ell, m and g.
- Fixed two-fold partition shared across all learners and scores per replication.
- No Series OLS treatment substitution, no oracle nuisance in a main arm, no
  artificial own-outcome residual injection, no outcome-based tuning or filtering.

## Procedure and comparison

Pinned reference: `DoubleML/doubleml-for-py`, commit
`6884b5f9816f180324d854a4d81b22e9d334f5e0`, `doubleml/plm/plr.py`,
`_nuisance_est` and `_score_elements`; notebook repository commit
`61900b8a8fac8a624af532db32b4be6d504566f0`.

1. Learn ell=E[Y|X] and m=E[D|X], with same-sample or fold-out predictions.
2. Compute the official preliminary scalar:
   pilot = mean((D-m_hat)*(Y-ell_hat))/mean((D-m_hat)^2).
3. Learn g from Y-pilot*D on the same training scheme as ell and m.
4. Final IV-type estimate is mean((D-m_hat)*(Y-g_hat))/mean((D-m_hat)*D).
   No partialling-out estimator is reported as a final result. The residual-ratio
   pilot is a disclosed internal step of the official IV-type implementation.

The official cross-fit pilot is pooled over all fold-out predictions and
outcomes. Consequently, the final scalar pilot is NOT strictly independent of
each held-out outcome. The g model fit uses only its training fold, but its
generated target uses that pooled pilot. Do not describe this as a fully nested,
training-only pilot. This is retained to match the official implementation,
instead of silently replacing it with the prior v2 fixed-point/Series pilot.

Three reported methods on the exact same data:

- nonorthogonal_crossfit: raw D score, using the EXACT same structural g
  predictions as iv_crossfit (not ell); this isolates score replacement.
- iv_no_split: ordinary all-sample learned nuisances and official initial step.
- iv_crossfit: ordinary fold-out nuisances and official initial step.

Primary Figure 1 contrast is nonorthogonal_crossfit versus iv_crossfit;
primary Figure 2 contrast is iv_no_split versus iv_crossfit. Both contrasts are
also reported in every cell. Random learner seeds are paired by role across
training schemes; training sample size necessarily differs. No truth is used
to train a learner or compute the pilot.

## Reporting and failures

Report every learner/DGP cell, mean bias and its Monte Carlo t interval, MC SD,
MAE, RMSE, average sandwich SE and nominal 95% coverage with Wilson intervals.
Also report paired absolute-error, squared-error and signed-bias contrasts with
Monte Carlo intervals. Mean bias is not MAE. Do not claim exact unbiasedness,
universal robustness, or cross-fitting superiority from rounded pooled means.

Every finite/nonfinite estimate and explicit failure is retained. Replication
artifacts are atomically committed, bound to config/source hashes, and resumed
only under the same identity. Any missing or failed estimates block a complete
numerical comparison rather than being converted to zero or dropped.

Development uses a separate seed. No new formal seed, learner setting, or
sample-size change is allowed in response to formal bias. Computational worker
count can change without changing scientific identity.
