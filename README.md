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
analysis in [scripts](scripts). The [configuration files](configs) record
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

Key figures:

- Non-orthogonal ML vs DML: [linear](results/sensitivity/distributions/moderate_figure1_linear_naive_ml_hist.png)
  and [nonlinear](results/sensitivity/distributions/moderate_figure1_nonlinear_naive_ml_hist.png).
- Full-sample no-split vs cross-fitting: [linear](results/sensitivity/distributions/moderate_figure2_linear_full_no_split_hist.png)
  and [nonlinear](results/sensitivity/distributions/moderate_figure2_nonlinear_full_no_split_hist.png).
- Series OLS / decision-tree comparison: [linear](results/nuisance_comparison/controlled_figure2_linear.png)
  and [nonlinear](results/nuisance_comparison/controlled_figure2_nonlinear.png).
- Sensitivity: [sample size and folds — bias](results/sensitivity/figures/moderate_figure2_sample_fold_bias.png)
  and [regularization — interval coverage](results/sensitivity/figures/moderate_figure2_parameters_coverage.png).

Histograms show studentized estimation errors; panel annotations report mean
bias on the original scale. The plot label `naive_ml` denotes non-orthogonal ML.

- [Sensitivity results](results/sensitivity/summary.json), including all 600
  summary cells, adverse findings, and paired comparisons.
- [Nuisance-learner results](results/nuisance_comparison/summary.json) and
  [paired comparisons](results/nuisance_comparison/paired_comparisons.json).

These studies use known low-dimensional, monotonic treatment structure, with
learned rather than oracle nuisances. Bias reductions are specific to the stated
comparison and do not imply uniformly lower error or valid interval coverage;
minimum cross-fit coverage across the sensitivity grid is 87.5%.

## Complete moderate-DGP evidence: linear and nonlinear

- [Complete 32-page report](output/pdf/DML_CV_Evidence.pdf)
- All 41 charts: [six main figures plus an overview](results/cv_evidence/figures)
  and [34 extension figures](results/cv_evidence/extensions)
- [All 11 table images](results/cv_evidence/tables), including the
  [headline comparison](results/cv_evidence/tables/table_0_headline.png)
- [Offline gallery, grouped by Linear / Nonlinear DGP](results/cv_evidence/index.html)
- [Numerical results](results/cv_evidence/summary.json),
  [selected saved draws](results/cv_evidence/primary_draws.csv), and
  [verification](results/cv_evidence/verification.json)
- [Saved-record reproduction instructions](results/cv_evidence/REPRODUCE.txt)

Both headline comparisons use the moderate `n=1,000`, `K=2`, `R=200` anchor
described above. The 89.4% and 97.5% reductions are unchanged. This report does
not pool those estimates with the separate baseline completion chapter below.

| Comparison | Linear DGP | Nonlinear DGP |
| --- | --- | --- |
| Non-orthogonal ML vs DML | [Table 1](results/cv_evidence/tables/table_1.png), [Figure 1](results/cv_evidence/figures/figure_1_figure1_linear.png) | [Table 2](results/cv_evidence/tables/table_2.png), [Figure 2](results/cv_evidence/figures/figure_2_figure1_nonlinear.png) |
| Learner choice | [Table 3](results/cv_evidence/tables/table_3.png), [Figure 3](results/cv_evidence/figures/figure_3_learner_sensitivity_linear.png) | [Table 4](results/cv_evidence/tables/table_4.png), [Figure 4](results/cv_evidence/figures/figure_4_learner_sensitivity_nonlinear.png) |
| Full no-split vs cross-fitting | [Table 5](results/cv_evidence/tables/table_5.png), [Figure 5](results/cv_evidence/figures/figure_5_figure2_linear.png), [raw-error companion](results/cv_evidence/extensions/raw_crossfitting_linear.png) | [Table 6](results/cv_evidence/tables/table_6.png), [Figure 6](results/cv_evidence/figures/figure_6_figure2_nonlinear.png), [raw-error companion](results/cv_evidence/extensions/raw_crossfitting_nonlinear.png) |
| Equal-training shared-bank control | [Table 7](results/cv_evidence/tables/table_7.png), [histogram](results/cv_evidence/extensions/moderate_figure2_linear_equal_train_reuse_hist.png) | [Table 8](results/cv_evidence/tables/table_8.png), [histogram](results/cv_evidence/extensions/moderate_figure2_nonlinear_equal_train_reuse_hist.png) |
| Series OLS / decision-tree control | [Linear figure](results/cv_evidence/extensions/controlled_figure2_linear.png) | [Nonlinear figure](results/cv_evidence/extensions/controlled_figure2_nonlinear.png) |

All 24 sensitivity figures are retained: baseline/moderate DGP levels, both
mechanisms, sample-size/fold-count and regularization factors, and bias/RMSE/
coverage. Each sensitivity image contains linear panels in its upper row and
nonlinear panels in its lower row. The full 600-cell sensitivity and 106-cell
nuisance summaries remain included, with adverse and inconclusive outcomes.
Original and restyled views of the same saved data are retained; 41 images
does not mean 41 independent experiments.

Figures 1-2 show raw errors; Figures 5-6 show studentized errors. Figures 3-4
use mean +/-1.96 MC SD as a sampling spread, not a confidence interval for the
Monte Carlo mean. Residual bias, variance tradeoffs and coverage shortfalls are
not removed or relabelled as universal unbiasedness.

Download the repository and open `results/cv_evidence/index.html` for the
complete offline gallery; GitHub itself displays HTML source, while PNGs and
the PDF are directly viewable. Independently recount all 6,000 selected saved
estimates without installing or running ML learners:

```bash
python3 -B scripts/verify_cv_simulation_evidence.py
```

The cached CSV is included; the 800 original receipt hashes are recorded,
but the full 800 per-job JSON files are not bundled in this attachment.
Creation-time flags such as `github_modified: false` in the report evidence
describe the earlier build, not this subsequent publication. No experiment
was rerun, and no earlier result or extension file was replaced or deleted.

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
per-replication archives and frozen source. This repository provides complete
numerical summaries and the explicitly selected raw records described below,
not the full historical v6 archive.

## Completed legacy-style simulation chapter

- [Corrected simulation chapter (PDF)](output/pdf/Simulation_Completed.pdf)
- [All restored tables, figures, formulas, and extension gallery](results/legacy_report/index.html)
- [Saved numerical results](results/legacy_report/summary.json)
- [Hash, pairing, and source verification](results/legacy_report/verification.json)
- [All six primary figures](results/legacy_report/figures),
  [eight report-style tables](results/legacy_report/tables), and
  [32 retained extension figures](results/legacy_report/extensions)
- [Archived records and source hashes](evidence/completed_simulation_manifest.json)
- [Saved-record reproduction instructions](REPRODUCE.txt)

GitHub displays the PNGs and PDF directly. Download the repository and open
`results/legacy_report/index.html` for the offline gallery with formulas,
native tables, and the complete sensitivity and nuisance-study summaries.

| Restored comparison | Linear | Nonlinear | Protocol |
| --- | --- | --- | --- |
| Non-orthogonal ML / DML histograms | [Figure 1](results/legacy_report/figures/figure_1_orthogonalization_linear.png) | [Figure 2](results/legacy_report/figures/figure_2_orthogonalization_nonlinear.png) | Raw errors; n=500, K=2, R=300 |
| Learner sensitivity | [Figure 3](results/legacy_report/figures/figure_3_learner_sensitivity_linear.png) | [Figure 4](results/legacy_report/figures/figure_4_learner_sensitivity_nonlinear.png) | Mean +/- 1.96 MC SD; sampling spread, not a mean CI |
| Full no-split / cross-fitting histograms | [Figure 5](results/legacy_report/figures/figure_5_crossfitting_linear.png) | [Figure 6](results/legacy_report/figures/figure_6_crossfitting_nonlinear.png) | Studentized errors; n=300, K=2, R=200 |

The restored Tables 1-4 / Figures 1-4 use a fixed server-only `n=500`, `K=2`,
`R=300` bridge with the repaired structural IV target. Tables 5-6 / Figures 5-6
reuse the existing baseline `n=300`, `K=2`, `R=200` receipts without fitting.
Tables 7-8 retain equal-training-size shared-bank comparisons. The report
separates the two baseline DGPs, corrects structural `g` versus outcome `ell`,
and labels MC sampling spread separately from confidence intervals.

The bridge is complete: do not relaunch it. Local report building and the
stdlib-only saved-record verifier perform no ML fitting. The evidence archives
include all 600 bridge jobs and the 400 selected v6 jobs, with frozen source and
hash ledgers; no historical raw record was edited. Extract and audit them using
`REPRODUCE.txt` without installing or running any ML learner.

All pre-existing results remain at their original paths. The moderate-DGP
headline numbers above are unchanged and are not pooled with the new n=500
bridge. Adverse findings, residual finite-sample bias, and coverage shortfalls
remain visible in the full summaries and extension figures. The n=500 linear
Lasso case reduces absolute mean bias but has slightly higher MAE and RMSE;
the chapter does not claim uniform finite-sample unbiasedness or dominance.

The earlier [curated evidence snapshot](https://github.com/mafeifei798/Double-Machine-Learning/tree/ae6c92b40eb1556e36aa59cc406595aef1105e4f)
is also retained in repository history. Its historical studies remain separate
from this completed chapter and the current headline comparison.
