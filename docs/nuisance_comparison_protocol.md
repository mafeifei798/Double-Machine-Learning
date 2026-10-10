# Nuisance-Learner Comparison: Study Protocol

This simulation study uses n=1,000 and the original four design/DGP functions.
It is a separate development-and-confirmation experiment, not a retroactive
replacement of earlier estimates. The historical protocol identifier is
`iv_repair_v3_20261002`.

## Study questions

### Controlled nuisance specification

The treatment learner is correctly specified Series OLS. The outcome regression
ell uses Series OLS for Figure 1 or Decision Trees with minimum leaf sizes
2/5/10/20 for Figure 2. A mixed RF-outcome / Series-OLS-treatment sensitivity
analysis is reported separately.

Structural g is constructed as ell-pilot*m, and the final estimator solves
(D-m)*(Y-theta*D-g)=0. This is an IV-type estimator, not the partialling-out
estimator; ell is not inserted unchanged as g. Models receive neither nuisance
truth nor theta0. The fixed five-column basis
[1,x0,x2,sigmoid(x0),sigmoid(x2)] is knowingly correctly specified. This
structural assumption limits the interpretation of the controlled study.

### Learned nuisance specification

RF, XGBoost, and Lasso learn both m and structural g, using Y-pilot*D as the
structural-outcome target. The pilot is the D coefficient from joint OLS of Y on
[D,1,all 20 X,all 20 sigmoid(X)], fitted within each training set. This richer
series pilot differs from a same-family pilot and is a disclosed methodological
change. The original Lasso feature mapping is retained. Fitting uses no
truth-based feature selection, oracle training, or response-residual injection.

## Pairing and training-data reuse

For each learner, a shared two-model bank uses 500 observations per model.
Same-bank reuse predicts each observation with its own half's model;
cross-fitting uses the other half's identical model, target, and pilot. Both
estimators use 1,000 observations.

Full-sample no-split estimation fits a separate model on 1,000 observations.
It is reported separately from equal-training reuse. Controlled no-split arms
likewise include both full-sample and equal-training versions. All pilots,
transformations, and targets use training observations only, without a pooled
outer pilot. The construction g=ell-pilot*m is a valid alternative nuisance
construction, but is not asserted to equal DoubleML's direct-target fit.

## Prediction-based development selection

Four fresh development data sets per cell use seed 2026100421. Selection uses
only the first predetermined 500-observation training half from each data set,
split into two inner folds. Validation squared prediction losses are pooled
across the four data sets and both folds.

The m target is observed D. The g target is Y-pilot_inner_train*D, with the pilot
refitted on each inner training subset. Minimum pooled prediction loss selects
each cell/family/role combination; candidate order breaks ties. All losses are
recorded. Causal estimates, theta0, g0, m0, bias, and coverage are not selection
criteria.

Candidate sets use 300 trees where applicable:

- RF: sqrt/depth8/leaf2; all-features depth5/leaf2; and all-features unbounded
  depth with minimum leaf sizes 2/5/10/20, giving six candidates.
- XGBoost: lr.01/depth3; and lr.05 or .1 crossed with depth2 or3, giving five
  candidates. Row/column subsampling .9, lambda1, and child weight2 are fixed.
- Lasso: alpha .005/.02/.06/.1, with the original linear/nonlinear feature
  mapping and training-only scaling.

Selected parameters and all selection receipts are frozen before confirmation.
There is no post-result selection round. Controlled learners are fixed rather
than selected by prediction or causal-effect performance.

## Confirmation and validation

Formal seed 2026100521 is disjoint from historical and development seeds.
Figure 1 uses 300 replications per DGP; Figure 2 uses 200. Every seed and cell
is retained.

Preflight checks cover held-out-outcome perturbation isolation, shared-bank
identity, structural conversion, score algebra, training-only pilots, and all
methods. DoubleML 0.10.1 checks numerical nuisance/coefficient/standard-error
parity with the historical official pooled-pilot implementation. Separate
checks supply the study's external predictions to DoubleML IV-type estimation.
These validate distinct procedures, not identical training algorithms.

Source, configuration, and environment hashes, durable per-replication receipts,
failures, warnings, and the complete row ledger are recorded. Independent checks
verify the algebra and reported statistics. Failures prevent complete-case
success claims and ordinary complete-result plots; no failures are discarded.

## Endpoints and interpretation

Reports include mean bias with Monte Carlo 95% t intervals, RMSE, MAE, nominal
95% coverage with Wilson intervals, paired mean/MSE differences, and exact
per-replication first-/second-order weighted error contributions. Bias is not
MAE.

The predeclared practical-centering margin is ±0.01, or 2% of the true effect
0.5. Support requires the entire Monte Carlo 95% bias interval within this
margin. Merely containing zero is inconclusive, and this finite-simulation
criterion is not a theorem of exact unbiasedness.

Studentized-error histograms use paired axes/bins, all observations, and an
N(0,1) reference. Ordinary learners, controlled trees, Series OLS, and mixed-RF
sensitivity remain separately identified, with Monte Carlo bias-interval plots.
Mixed RF / Series OLS is not presented as RF learning both nuisances. Neither
no-split bias nor cross-fitting superiority is assumed in every finite-sample
setting; adverse findings are retained.

## Historical execution conditions

Execution used the isolated directory
`/data2/lzx/dhy/dml_iv_repair_v3_20261002`, 12 CPU worker processes, one thread
per worker, no GPU, and no local fitting or simulation. The budget allowed one
development selection and one formal replication universe, without automated
retries, sample-size expansion, or formal retuning. Faulty tests could be
corrected, but selected and formal scientific procedures were fixed after launch.
