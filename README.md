# Double Machine Learning for Causal Effect Estimation

University of Macau undergraduate project | Feb-Jun 2026

IV-type Double Machine Learning and cross-fitting with Lasso, Random Forest,
and XGBoost, plus a separate Series OLS / decision-tree study of training-data
reuse. The saved results include subsequent repair, extension, and verification
runs from October 2026; original execution identifiers and dates are retained.

## Code and saved results

Core estimators: [src/dml_original](src/dml_original).
Settings: [configs](configs). Runners and checks: [scripts](scripts).
The six [runtime protocols](docs) are retained because the runners read them
as frozen experiment inputs. Standalone narrative reports are not bundled.

| Study | Saved JSON summaries and figures |
| --- | --- |
| Main three-learner study and sensitivity analysis | [600-cell summary](output/sensitivity_v6_server_20261002/evidence/output/formal/analysis/SUMMARY.json), [figures](output/sensitivity_v6_server_20261002/evidence/presentation) |
| Series OLS / decision-tree reuse study | [summary and figures](output/iv_repair_v3_server_20261002/output/formal/analysis) |
| Supplementary three-learner study, R=1,000 | [summary and figures](output/two_claims_v4_server_20261002/output/formal/analysis) |
| Supplementary difficulty extension, R=200 | [summary and figures](output/harder_v5_server_20261002/output/exploratory/analysis) |
| Earlier ordinary-learner IV comparison | [summary and figures](output/fair_iv_v1_server_20261002/output/fair_iv_v1_server_formal/analysis) |
| Earlier equal-training diagnostic | [summary and figures](output/equal_training_iv_v1_server_20261002/output/equal_training_iv_v1_server_formal/analysis) |

## Selected measurements

The main comparison uses the prespecified moderate-DGP anchor: n=1,000, p=20,
true effect 0.5, two folds, baseline hyperparameters, and 200 paired Monte Carlo
repetitions per cell. Each comparison weights six learner-by-DGP cells equally.

| Comparison | Comparator absolute MC mean bias | DML / cross-fit | Reduction |
| --- | ---: | ---: | ---: |
| Non-orthogonal ML vs IV-type DML | 0.0365543 | 0.00387464 | 89.4% |
| Full-sample no-split vs cross-fitting | 0.255777 | 0.00648017 | 97.5% |

The metric is the six-cell average of abs(mean(theta_hat) - 0.5), not MAE or
an average of cellwise percentage improvements. R=1,000 supplementary results
are a separate protocol and are not pooled into these percentages.

The correctly specified Series OLS treatment learner / decision-tree outcome
learner study spans two DGPs and four leaf sizes. Its eight cross-fit settings
have maximum absolute MC mean bias 0.00457467; this is not an exact-unbiasedness
guarantee or a bound on individual estimates.

## Evaluation scope and limitations

- The three-learner studies learn both nuisance channels. Known low-dimensional
  treatment candidates and monotone DGP structure are disclosed design controls.
- Matched reuse and cross-fit arms share fitted model banks, training sizes,
  targets, pilots, and random states; prediction membership changes.
- Sensitivity covers n=300/1,000/3,000, two/five folds, and four one-role-at-a-time
  hyperparameter alternatives at n=1,000 with two folds, not a full factorial.
- Retained summaries include paired Holm-adjusted MAE/MSE tests, error
  decompositions, and adverse findings. Linear RF retains residual mean bias
  around +0.01275 at the moderate anchor; linear Lasso MAE rises from about
  0.02800 to 0.03051 in one no-split/cross-fit comparison. Minimum cross-fit
  coverage in the sensitivity grid is 87.5%.
- Bias reduction does not imply uniformly lower total error, universal
  unbiasedness, or uniformly valid confidence intervals.

## Install and verify saved results

Python 3.10 or newer is required to install the package:

```bash
python -m pip install -e '.[test]'
```

The following standard-library-only check needs no package installation and
fits no models:

```bash
python -B scripts/check_saved_results.py
```

Formal fitting and fitting-based tests remain server-only. Runners enforce
their original authorized Linux roots; inspect the retained protocols before
any separately authorized reproduction. No simulations were rerun for this
report-only cleanup.

This is a selected code-and-results snapshot, not a turnkey full replication
archive. Per-replication archives and server logs are retained separately;
historical raw-data verifiers may require them. The original sensitivity
archive has SHA256 `3bdde95dfc1d016e0d8b98c151db432475c2dc6c1d7c0582c44ac01b0bee774a`.
Source identities and the current file inventory are in
[publication_manifest.json](publication_manifest.json).

No license is assigned by this upload.
