# Two claims, three actual learner families: frozen controlled v4

The requested comparisons are (1) genuinely non-orthogonal ML versus DML,
and (2) No-split versus Cross-fitting, EACH using Lasso, Random Forest, and
XGBoost in BOTH learned nuisance channels. No Series OLS or standalone
Decision Tree is substituted. No simulation or model fit runs locally.

## Scope and scientific boundary

Retain all four original Figure 1/2 linear/nonlinear DGPs: n=1000, p=20,
Toeplitz correlation 0.7, theta=0.5, unit Gaussian treatment/outcome noise.
This is an explicitly CONTROLLED known-support experiment, not the previous
ordinary all-20-input learner experiment. The treatment learner is told its
two relevant input columns and monotonic directions, but NOT coefficients,
true nuisance values, noise, or the causal truth. RF/XGB enforce monotonicity;
Lasso uses a positive weakly penalized sparse channel. This encodes original
DGP structure to improve nuisance accuracy rather than changing the DGP or n.
The outcome learner still sees all 20 columns (Lasso also a declared sigmoid
dictionary). Claims must carry these assumptions. Ordinary negative results
remain valid and are not overwritten.

## Experiment 1: regularization / orthogonalization

On each outer training half, fit a joint Lasso pilot of Y on D and the fixed
raw+sigmoid X dictionary, with training-only scaling and alpha=.001. Recover
the D coefficient in raw units. It is an INTERNAL common Lasso pilot, not
the final estimator. Fit the designated family's structural g learner to
Y-pilot*D, and its m learner to D. Never substitute ell=E[Y|X] for g.
Generate held-out g,m predictions. Both estimators reuse EXACTLY those g
predictions and data: naive=sum D(Y-ghat)/sum D^2; DML solves the structural
IV-type score (D-mhat)(Y-theta D-ghat). g has the original RF/XGB penalties,
and fixed Lasso alpha .02/.06. m is the declared controlled fast channel.

## Experiment 2: genuine sample reuse

Fit the designated family's high-capacity outcome regression ell to Y;
construct structural g=ell-pilot*m using ONLY the corresponding training
sample. RF is deep bootstrap RF; XGB is a deep long boosted ensemble;
Lasso uses a fixed 40-column raw+sigmoid dictionary with weak alpha=.0001.
There is NO injected own-outcome term, artificial pressure, true-pilot
substitution, or changed target rule for the No-split arm. Every final
estimate uses the SAME structural IV score.

Report ordinary full No-split (1000 training rows) and two-fold Cross-fitting
(500 per fit). Additionally, train a single bank of two 500-row fits and
assemble own-half reuse versus opposite-half Cross-fitting from the EXACT
same bank, seeds, targets, pilots, and models. This extra comparison isolates
sample reuse without unequal training sizes. Full No-split uses the identical
algorithm and hyperparameters, only its permitted training size differs.

## One-shot confirmation and inference

JSON parameters and all source hashes freeze before preflight. Software
preflight uses a disjoint seed solely to check identities and isolation, not
to select causal biases. The formal universe is 1000 replications per each
of four cells, 4000 total, 30000 estimates. Parameters are NOT retuned after
observing final results. Use <=24 CPU workers with one numerical/model
thread each; GPU disabled; existing isolated environment is read-only.

Actual DoubleML 0.10.1 external-prediction IV-score coefficient and standard
error parity is checked for all three families and four DGPs. This checks the
FINAL SCORE, not equivalence to DoubleML's default nuisance-fitting pipeline.

All cells, warnings, numerical failures, paired contrasts, coverage, RMSE,
nuisance errors and exact IV bias decomposition are retained. A t-based MC95
interval wholly within +/- .01 supports practical-zero mean bias at n=1000;
a bias interval excluding zero detects a nonzero mean. These can BOTH hold
for a sufficiently small detectable bias. An interval merely containing zero
does not prove unbiasedness. These are individual-cell intervals, NOT joint
95% simultaneous inference; all-cell attainment and inconclusive cells are
listed explicitly. Exact finite-sample unbiasedness is NOT promised.

Original-style three learner rows by two method columns show standardized
estimation-error histograms and N(0,1) reference. Never re-center by the
observed Monte Carlo mean, crop out tails, or select favorable seeds.

RF constraints: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
XGB constraints: https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html
