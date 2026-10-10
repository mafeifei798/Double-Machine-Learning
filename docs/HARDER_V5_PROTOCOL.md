# Slightly harder paired DGP extension v5: frozen exploratory protocol

This is a separate exploratory extension, not a post-hoc replacement of v4.
Earlier results are unchanged. Execution uses the isolated v3 research
environment, with at most 24 single-threaded CPU workers and no GPU.

## Fixed modest increase in complexity

Retain n=1000, p=20, theta=.5, Toeplitz X correlation .7, independent
unit Gaussian U,V, two folds, and the two original design coefficient sets.
Each design has its original linear/nonlinear baseline and a moderate level.
For moderate linear DGPs, add .20*x4+.15*x6 to m and .15*x4+.15*x6 to g.
For moderate nonlinear DGPs, add .20*x4+.20*sigmoid(x6) to m and
.25*sigmoid(x4)+.20*x6+.30*tanh(x0)*tanh(x4) to g. The interaction is
bounded in magnitude by .30, belongs ONLY to the outcome function, and can
be non-monotone. Treatment functions remain additive monotone. This raises
effective dimension from two to four without deliberately breaking the
monotonicity imposed on the treatment learners. Signal variance also changes;
this is a combined difficulty stress test, not isolated attribution of one
component or a fixed-SNR experiment.

## Matched learning rules

BOTH levels use the SAME treatment candidate inputs x0,x2,x4,x6, including
two irrelevant candidates at baseline. Nonlinear Lasso m additionally uses
their four sigmoid features. Thus input dimensions do NOT differ by level.
Both levels have the same structural prior and all positive treatment
directions; this remains a controlled known-candidate-input experiment,
not discovery among all 20 inputs. Outcome RF/XGB use all 20 raw inputs;
Lasso outcome features remain raw X in linear Figure 1, and raw+sigmoid X
in nonlinear Figure 1 / all Figure 2. The bounded nonlinear interaction is
NOT added to the Lasso or pilot dictionary; this is disclosed outcome
misspecification, while the Lasso treatment channel remains representable.

Preserve v4 learner hyperparameters, pilot alpha=.001 and target rules:
Figure 1 fits g to training Y-pilot*D and compares genuine naive ML versus
IV DML using exactly the same held-out g predictions. Figure 2 learns ell,
constructs g=ell-pilot*m and compares full No-split, same-500-row-bank
own-sample reuse, and opposite-bank Cross-fitting. All final DML scores are
IV-type. No oracle fits, true-pilot inputs, artificial response pressure,
separate No-split penalties, or final-bias-dependent parameter selection.

## Paired baseline and difficulty experiment

Fresh seed 2026100741 is disjoint from earlier runs; preflight uses
2026100641. Draw X,U,V ONCE per design/dgp/replication, then construct BOTH
levels from those same random variables. Shared fold indices and model
random states across levels. Store and verify common X/U/V hashes.
Both levels' full and cross-fit arms also satisfy within-level matching.
The new baseline is not numerically identical to v4 because it has four
common treatment candidates and a new seed. Difficulty contrasts must use
the new paired baseline, not claim differences from old runs as isolated
DGP effects.

200 replications per design/dgp, four such pairs: 800 random draws, 1600
level-replications, 12000 estimates, 60 summary cells, 36 within-level
method contrasts, and 30 matched moderate-minus-baseline estimator contrasts.
Parameters/source freeze BEFORE software preflight. No statistical selection,
no additional replications based on observed bias, no hidden retries.

Report bias, t-MC95 intervals, RMSE, MAE, coverage with Wilson intervals,
nuisance errors and exact oracle/first-order/second-order decomposition.
Use individual practical-zero +/- .01 intervals only descriptively; this
200-rep exploratory test is not an assertion of exact unbiasedness.

For paired MAE and MSE changes (Cross-fit minus baseline method), retain
individual two-sided t-MC95 intervals and one-sided paired-t improvement
p-values. Apply Holm adjustment separately to all 36 MAE and all 36 MSE
comparisons across both levels. Do not confuse these metric families with
a simultaneous guarantee across every reported endpoint. Also report paired
2000-draw percentile-bootstrap intervals for changes in ABSOLUTE mean bias,
using a fixed analysis seed and joint resampling of the paired estimates.
These bootstrap intervals are exploratory, especially near zero bias.

All cells and negative outcomes must remain visible. Actual DoubleML 0.10.1
external-prediction IV score/SE checks cover all 24 level/design/dgp/family
combinations, checking SCORE parity, not default pipeline equivalence.
Independent statistics/hash/universe/bank pairing verification must pass.
Original-style three-learner by two-method histograms include all tails and
use the same x range and bins within paired panels. Display annotations above
the axes, avoiding overlap with the density curves. Preserve earlier plots.
