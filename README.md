# Double Machine Learning for Causal Effect Estimation

Simulation studies of orthogonalization, cross-fitting, and nuisance-model
selection in the partially linear model `Y = theta * D + g(X) + error`.

## Methods

- IV-type orthogonal score with training-only pilot and nuisance fitting.
- Lasso, Random Forest, and XGBoost with matched model-bank comparisons.
- Series OLS and decision-tree controls for nuisance specification.
- Sensitivity analyses of sample size, fold count, and regularization.

Algorithm implementations are in [src](src/dml_studies), experiment settings in
[configs](src/dml_studies/configs), and execution and analysis tools in [command-line tools](src/dml_studies/cli).

## Results

The main comparison uses the moderate DGP, `n=1,000`, `p=20`, true effect
`0.5`, two folds, and 200 paired Monte Carlo repetitions. The metric is the
equal-weight average of absolute Monte Carlo mean bias over six
learner-by-DGP settings, not MAE or the mean of setting-specific percentages.

| Comparison | Before | After | Bias reduction |
| --- | ---: | ---: | ---: |
| Non-orthogonal ML vs DML | 0.0365543 | 0.00387464 | 89.4% |
| Full-sample no-split vs cross-fitting | 0.255777 | 0.00648017 | 97.5% |

- [Main numerical results](results/primary/summary.json) and [6,000 saved estimates](results/primary/estimates.csv).
- [Bias comparison](results/primary/figures/bias_comparison.png).
- Non-orthogonal ML vs DML: [linear](results/primary/figures/orthogonalization_linear.png), [nonlinear](results/primary/figures/orthogonalization_nonlinear.png).
- Full no-split vs cross-fitting: [linear](results/primary/figures/cross_fitting_linear.png), [nonlinear](results/primary/figures/cross_fitting_nonlinear.png).
- Matched-bank training reuse: [linear](results/primary/figures/training_reuse_linear.png), [nonlinear](results/primary/figures/training_reuse_nonlinear.png).
- Series OLS / decision-tree controls: [linear](results/nuisance_comparison/figures/nuisance_models_linear.png), [nonlinear](results/nuisance_comparison/figures/nuisance_models_nonlinear.png).
- Sensitivity: [sample-size/fold-count bias](results/sensitivity/figures/sample_size_and_fold_bias.png), [regularization coverage](results/sensitivity/figures/regularization_coverage.png).
- [All 600 sensitivity cells and paired tests](results/sensitivity/summary.json).
- [All 106 nuisance-study cells](results/nuisance_comparison/summary.json) and [paired comparisons](results/nuisance_comparison/paired_comparisons.json).
- [Additional baseline simulation results](results/finite_sample/summary.json): `n=500, R=300` for orthogonalization and `n=300, R=200` for cross-fitting. These are not pooled with the main comparison.

The separate Series OLS / decision-tree study has maximum absolute Monte Carlo
mean bias below 0.0046 across eight cross-fitting settings. Treatment models use
known low-dimensional, monotonic structure; this is not unrestricted variable
discovery. Bias reductions do not imply uniformly lower error or valid interval
coverage. The sensitivity grid includes residual bias, bias-variance tradeoffs,
and minimum cross-fit coverage of 87.5%; all adverse and inconclusive numerical
results are retained. Plot labels `naive_ml` denote non-orthogonal ML.

## Verify saved results

The [derived saved-job export](results/simulation_records.jsonl.gz) retains 1,000 jobs
and all 10,800 saved estimates, including nested model-bank records, seeds,
warnings and elapsed times. Of these, 7,200 support the additional baseline
comparison and 3,600 are its retained moderate-DGP counterparts.
[Provenance and checksums](results/data_manifest.json) bind the exports to the
original source commit, archive hashes and individual receipt hashes.

These are curated exports, not replacement original execution receipts.
Original administrative records and frozen source archives remain available
in Git history at the recorded source commit. The full historical sensitivity
raw archive is not bundled; all 600 numerical summary cells remain included.
The main CSV additionally retains 6,000 estimates and the provenance records
its 800 original receipt hashes.

Recompute saved statistics and validate pairing, error decomposition and file
hashes with the Python standard library only. No models are fitted:

```bash
PYTHONPATH=src python3 -B -m dml_studies.cli.verify_results
```

To validate every derived job against the original records, supply the original
evidence directory recovered separately from the recorded Git commit:

```bash
PYTHONPATH=src python3 -B -m dml_studies.cli.verify_results --input-directory /path/to/original/evidence
```

The export helper also accepts an original evidence directory; use
`PYTHONPATH=src python3 -m dml_studies.cli.export_results --help` for its explicit input and
source-binding arguments.

## Code checks and new experiments

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m dml_studies.cli.run_sensitivity --help
python -m dml_studies.cli.run_nuisance_comparison --help
python -m dml_studies.cli.run_finite_sample --help
```

Unit tests use small synthetic inputs. Full experiment runners require Linux
and an explicitly set absolute `DML_EXECUTION_ROOT` matching the checkout.
Single-launch locks and existing-result checks prevent accidental replacement.
Running a new experiment creates new execution records and does not reproduce
the original frozen source identity merely by matching its settings.
