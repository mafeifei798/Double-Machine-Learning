# Double Machine Learning for Causal Effect Estimation

Simulation studies of IV-type Double Machine Learning with Lasso, Random Forest,
and XGBoost, together with a Series OLS / decision-tree analysis of training-data
reuse. The studies examine orthogonalization, cross-fitting, and sensitivity to
sample size, fold count, and learner regularization.

The original University of Macau undergraduate project ran from February to
June 2026. The retained evidence also includes subsequent methodological
corrections, extensions, and verification from October 2026. These are distinct
stages of the work; original execution identifiers and dates are retained.

## Code and saved results

Estimators and study definitions: [src/dml_studies](src/dml_studies).
Configurations: [configs](configs). Execution and validation: [scripts](scripts).
The [study protocols](docs) specify the designs and interpretation boundaries.

| Study | Saved JSON summaries and figures |
| --- | --- |
| Main three-learner study and sensitivity analysis | [600-cell summary](results/sensitivity/summary.json), [figures](results/sensitivity/figures) |
| Nuisance-learner comparison, including Series OLS / decision-tree reuse | [summary and figures](results/nuisance_comparison) |

Sensitivity plots are in [figures](results/sensitivity/figures), and six
moderate-DGP histograms are in [distributions](results/sensitivity/distributions).
Earlier estimator modules remain only as dependencies of the retained studies.

## Selected measurements

The main comparison uses the prespecified moderate-DGP anchor: n=1,000, p=20,
true effect 0.5, two folds, baseline hyperparameters, and 200 paired Monte Carlo
repetitions per cell. Each comparison weights six learner-by-DGP cells equally.

| Comparison | Comparator absolute MC mean bias | DML / cross-fit | Reduction |
| --- | ---: | ---: | ---: |
| Non-orthogonal ML vs IV-type DML | 0.0365543 | 0.00387464 | 89.4% |
| Full-sample no-split vs cross-fitting | 0.255777 | 0.00648017 | 97.5% |

The metric is the six-cell average of abs(mean(theta_hat) - 0.5), not MAE or
an average of cellwise percentage improvements or a full-grid aggregate.

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

Experiment runners and fitting-based tests target the original Linux research
environment; the runners retain its guarded execution paths. They are not
portable launch commands for this checkout. See the study protocols for the
execution conditions.

This repository contains selected code and saved results, not a turnkey
replication archive. Full historical verification also requires the original
per-replication archives and server logs, which are not included.

Current filenames are organized by research function. Historical protocol and
source paths embedded in saved results refer to the original execution archives.
Imports, path references, comments, and protocols have been updated for
readability. Saved result files are byte-identical to the retained evidence;
the numerical procedures, protocol identifiers, random seeds, and historical
source-hash checks are unchanged. Historical byte-level verification requires
the original frozen source at
[this commit](https://github.com/mafeifei798/Double-Machine-Learning/tree/d8aea4bebe87399232be710464a0afc62784b1df).
The reorganized source is not a replacement for that frozen execution archive.

No license has been assigned.
