# Double Machine Learning for Causal Effect Estimation

Undergraduate simulation project · University of Macau · **Feb–Jun 2026**

Controlled simulations of Neyman-orthogonal IV-type scores and cross-fitting,
with Lasso, Random Forest, and XGBoost nuisance learners. A separate Series OLS
and decision-tree study isolates training-data reuse. This snapshot contains
the selected code, complete selected-stage summaries, figures, and limitations.

The project period above is a CV display period, not a claim about when every
archived run was executed. Original execution identifiers and provenance are
retained; presentation copies do not backdate experiments.

## Results corresponding to the research CV

The headline comparison uses the prespecified moderate-DGP anchor:
n=1,000, p=20, true effect 0.5, 2-fold cross-fitting, baseline hyperparameters,
and 200 paired Monte Carlo repetitions per cell. Each comparison gives equal
weight to six learner-by-DGP cells: three learners and two DGPs.

| Comparison | Average absolute MC mean bias: comparator | DML / cross-fit | Reduction |
| --- | ---: | ---: | ---: |
| Non-orthogonal ML vs IV-type DML | 0.0365543 | 0.00387464 | **89.4%** |
| Full-sample no-split vs cross-fitting | 0.255777 | 0.00648017 | **97.5%** |

Here absolute MC mean bias means abs(mean(theta_hat) − 0.5), averaged across
the six cells. It is **not MAE**, the mean of cellwise percentage improvements,
or an aggregate across the entire sensitivity grid.

A separate, correctly specified **Series OLS treatment learner + decision-tree
outcome learner** study uses two DGPs and four minimum-leaf sizes (2, 5, 10, 20).
Across its eight cross-fit settings, the maximum absolute MC mean bias is
**0.00457467**, below 0.0046. This is a point-bias bound, not a guarantee of
exact finite-sample unbiasedness or a bound on every confidence-interval endpoint.

Sensitivity analysis spans n=300/1,000/3,000 and 2/5 folds, plus four
one-nuisance-role-at-a-time hyperparameter alternatives at n=1,000, K=2.
The full 600-cell summaries retain paired Holm-adjusted MAE/MSE tests,
error decompositions, residual bias, bias-variance tradeoffs, and undercoverage.
This is not a full hyperparameter factorial.

[Exact CV-to-evidence mapping](CV_DML_EVIDENCE_MAP.json) ·
[Read-only numerical checker](scripts/check_cv_dml_claims.py)

## Design and fairness

- Final mainline estimates use the structural IV-type score, not a
  partialling-out score with a mismatched nuisance target.
- The three-family experiments learn both nuisance channels using actual
  Lasso/RF/XGBoost. Series OLS, decision trees, and oracle functions are not
  substituted into those comparisons.
- Known low-dimensional treatment candidates and monotone DGP structure
  are disclosed controls, not unrestricted variable discovery.
- Cross-fit and equal-training-size reuse share fitted model banks, training
  sizes, targets, pilots, and random states; prediction membership changes.
  Ordinary full-sample no-split remains a separate comparison.
- No extra own-response pressure is added only to no-split in the selected
  mainline. Earlier adverse ordinary-learner findings are retained.

## Read the reports and view the figures

| Research question | Main evidence and complete figures |
| --- | --- |
| Three-family bias reduction and sensitivity | [Results presentation](docs/SENSITIVITY_V6_RESULTS_PRESENTATION.md), [all 600 cells](docs/SENSITIVITY_V6_ALL_CELLS.md), [30-figure gallery](docs/SENSITIVITY_V6_FIGURES.md), [protocol](docs/SENSITIVITY_V6_PROTOCOL.md) |
| Series OLS + decision-tree reuse mechanism | [Results and figures](docs/IV_REPAIR_V3_RESULTS.md), [protocol](docs/IV_REPAIR_V3_PROTOCOL.md) |
| Supplementary formal three-family experiment, R=1,000 | [Results and figures](docs/TWO_CLAIMS_V4_RESULTS.md), [protocol](docs/TWO_CLAIMS_V4_PROTOCOL.md) |
| Supplementary paired difficulty extension, R=200 | [Results and figures](docs/HARDER_V5_RESULTS.md), [protocol](docs/HARDER_V5_PROTOCOL.md) |
| Earlier ordinary-learner adverse findings | [Ordinary IV comparison](docs/FAIR_IV_V1_RESULTS.md), [equal-training diagnostic](docs/EQUAL_TRAINING_IV_V1_RESULTS.md) |

The R=1,000 supplementary experiment and R=200 studies are separate protocols;
their percentage improvements are not pooled. The CV headline values above
come only from the specified moderate sensitivity-study anchor.

## Limits, including negative results

Bias reduction is not universal unbiasedness, uniformly lower total error,
or uniformly valid inference. For example:

- Moderate Figure 1 linear RF at n=1,000, K=2 retains mean bias about +0.01275.
- In the moderate Figure 2 linear Lasso anchor, bias decreases but MAE rises
  from about 0.02800 to 0.03051; RMSE also rises.
- The lowest cross-fit coverage in the tested sensitivity grid is **87.5%**.
- Five folds are not uniformly better than two. Earlier ordinary all-input
  RF/XGBoost comparisons can have worse cross-fit bias than no-split.

All favorable, adverse, and inconclusive selected-stage cells remain available.

## Code and checking this snapshot

Core implementations are under [src/dml_original](src/dml_original);
experiment settings are under [configs](configs); runners, analysis, and
verification utilities are under [scripts](scripts).

The following check only reads saved JSON summaries, uses the Python standard
library, and performs **no model fitting**:

```bash
python -B scripts/check_cv_dml_claims.py
```

Formal simulations and fitting-based tests are server-only. The archived
runners enforce the original authorized Linux execution root; this snapshot
does not silently remove that safeguard or launch any job on a local laptop.
Inspect the relevant protocol before any future authorized reproduction.

This is a selected evidence snapshot, **not a turnkey full replication archive**.
Raw per-replication records and server logs are not bundled; some historical
verifiers require those offline files. See [raw-evidence scope](docs/RAW_EVIDENCE.md)
and [file provenance](publication_manifest.json).

Source/result bytes and original provenance JSON are preserved. Copied Markdown
may contain explicitly labelled workspace-link normalization; this curated
README is a presentation derivative. No third-party paper PDF, CV PDF,
credentials, SSH files, caches, or unrelated research project is included.
No license is assigned by this upload.
