# Fair IV-type v1: server results and claim boundaries

## Execution and evidence

Completed on `rna-server` in `/data2/lzx/dhy/dml_fair_iv_v1_20261002`.
The server preflight passed 14 correctness tests, followed by 1,000 paired
Monte Carlo replications and 9,000 estimates. No estimates failed and no
warnings were recorded. Simulation runtime was approximately 372 seconds.
An independent read-only verifier recomputed all reported cell metrics and
intervals, checked the paired prediction identities and source/result ledgers,
and verified all four PNGs. All four figures were visually inspected.

The frozen design uses n=1,000, p=20, true effect 0.5, two folds, the original
linear/nonlinear DGPs and RF/XGBoost/Lasso hyperparameters. Figure 1 has 300
replications per DGP; Figure 2 has 200. It uses learned outcome and treatment
nuisances, without artificial residual pressure or a Series OLS replacement.
Only the final IV-type score is evaluated as DML. Its structural outcome
nuisance is trained on `Y - pilot_theta * D`; the pooled initial scalar follows
the pinned official algorithm and is not a strictly training-only nested pilot.
See `FAIR_IV_V1_PROTOCOL.md` for the full specification and upstream commits.

## Main findings

Each aggregate below is the equal-weight average of absolute Monte Carlo mean
bias across six learner-by-DGP cells within one design. It is **not** mean
absolute estimation error (MAE).

| Design | Nonorthogonal cross-fit ML | IV-type cross-fit | Bias reduction, paired bootstrap 95% interval |
|---|---:|---:|---|
| Figure 1 | 0.036011 | 0.013449 | 62.65% [58.25%, 66.50%] |
| Figure 2 | 0.052306 | 0.022793 | 56.42% [53.07%, 58.81%] |

Orthogonalization improves bias and RMSE relative to the matched nonorthogonal
cross-fitted comparator, but does not remove all finite-sample bias.

The ordinary sample-splitting comparison is unfavorable on aggregate:

| Design | IV-type No-split | IV-type cross-fit | Increase in absolute mean bias |
|---|---:|---:|---:|
| Figure 1 | 0.009128 | 0.013449 | 47.34% |
| Figure 2 | 0.015604 | 0.022793 | 46.07% |

RF and XGBoost cross-fit estimates retain positive bias; their Monte Carlo
95% bias intervals exclude zero in every reported cell. Lasso is nearly
centered in the linear DGP, but retains small positive bias in the nonlinear
DGP. Nominal 95% coverage is not uniformly calibrated: RF cross-fit coverage
ranges from 33.0% to 81.0%. Cross-fitting also has larger RMSE than No-split
in these fixed cells. The comparison changes both sample reuse and nuisance
training sample size; it does not isolate which mechanism causes the adverse
finite-sample differences.

Therefore the experiment supports **bias reduction from a correctly targeted
orthogonal score**, not the stronger claims that DML is exactly unbiased,
No-split must be biased, or cross-fitting must improve every finite-sample
learner. No hyperparameters, seeds or cells were selected after inspecting
formal results to recover those desired conclusions.

## Relation to earlier experiments

The earlier controlled Series OLS/Decision Tree experiments remain separate
mechanism evidence. The v2 contrast near 0.500 versus 0.411 used a prescribed
own-observation residual stress on No-split and must not be presented as an
ordinary RF/XGBoost/Lasso comparison. Earlier percentage improvements from
different protocols are not interchangeable with the percentages above.
This run adapts the preserved thesis code and the pinned official IV-type
algorithm; it is not an exact historical GitHub reproduction, and n=1,000
differs from the original scripts' sample sizes.

## Complete results and artifacts

- Full cell means, bias intervals, RMSE, MAE and coverage:
  [RESULTS.md](../output/fair_iv_v1_server_20261002/output/fair_iv_v1_server_formal/analysis/RESULTS.md).
- Every paired bias/MAE/MSE contrast:
  [paired_contrasts.json](../output/fair_iv_v1_server_20261002/output/fair_iv_v1_server_formal/analysis/paired_contrasts.json).
- Original-style histograms:
  [Figure 1 linear](../output/fair_iv_v1_server_20261002/output/fair_iv_v1_server_formal/analysis/figure1_linear.png),
  [Figure 1 nonlinear](../output/fair_iv_v1_server_20261002/output/fair_iv_v1_server_formal/analysis/figure1_nonlinear.png),
  [Figure 2 linear](../output/fair_iv_v1_server_20261002/output/fair_iv_v1_server_formal/analysis/figure2_linear.png),
  [Figure 2 nonlinear](../output/fair_iv_v1_server_20261002/output/fair_iv_v1_server_formal/analysis/figure2_nonlinear.png).
- Downloaded immutable server evidence: `output/fair_iv_v1_server_20261002/evidence.tgz`.
  SHA-256: `a77ca857e294421d3baff61efb4dcf171bae99c6cb7c82c777971e4d43f49cd9`.
- Frozen run identity: `9edb0e358d136225e6c1ffad4bd28395fbda97c53a6fd9308e046302c6df2c52`.

Local development and the interrupted local formal attempt are excluded.
Do not resume them or silently pool their estimates with the server results.
