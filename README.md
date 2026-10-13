# Double Machine Learning for Causal Effect Estimation

## Overview

Simulation studies of orthogonalization, cross-fitting, and nuisance-model
selection for estimating the constant treatment effect in a partially linear
model, `Y = theta * D + g(X) + error`.

## Methods

- IV-type orthogonal score: `(D - m(X)) * (Y - theta * D - g(X))`.
- Lasso, Random Forest, and XGBoost nuisance estimators with training-only pilots.
- Matched model-bank comparisons isolate training-data reuse from cross-fitting.
- A Series OLS / decision-tree study examines the role of nuisance specification.
- Sensitivity analysis varies sample size, fold count, and regularization, with
  paired MAE/MSE tests, Holm correction, coverage, and error decomposition.

Implementations are in [src](src/dml_studies), with execution and numerical
analysis in [scripts](scripts). The three [configuration files](configs) record
the DGP and learner parameters, nuisance comparison, and sensitivity grid.

## Results

The moderate-DGP comparison uses `n=1,000`, `p=20`, true effect `0.5`, two folds,
and 200 paired Monte Carlo repetitions in each of six learner-by-DGP settings.
The metric is the equal-weight average of absolute Monte Carlo mean bias across
these six settings, not MAE or the average of setting-specific percentages.

| Comparison | Before | After | Bias reduction |
| --- | ---: | ---: | ---: |
| Non-orthogonal ML vs DML | 0.0365543 | 0.00387464 | 89.4% |
| Full-sample no-split vs cross-fitting | 0.255777 | 0.00648017 | 97.5% |

The Series OLS / decision-tree study has eight cross-fit settings with maximum
absolute Monte Carlo mean bias below 0.0046.

- [Sensitivity results](results/sensitivity/summary.json), including all 600
  summary cells, adverse findings, and paired comparisons.
- [Nuisance-learner results](results/nuisance_comparison/summary.json) and
  [paired comparisons](results/nuisance_comparison/paired_comparisons.json).

These studies use known low-dimensional, monotonic treatment structure, with
learned rather than oracle nuisances. Bias reductions are specific to the stated
comparison and do not imply uniformly lower error or valid interval coverage;
minimum cross-fit coverage across the sensitivity grid is 87.5%.

## Run

```bash
python -m pip install -e '.[test]'
python -m pytest
python scripts/run_sensitivity.py --help
python scripts/run_nuisance_comparison.py --help
```

Unit tests use small synthetic inputs. The two study runners require Linux
and an explicit `DML_EXECUTION_ROOT`
matching the checkout. Full historical replay additionally needs the original
per-replication archives and frozen source; this repository provides the selected
study implementations and complete numerical summaries, not those archives.
