# IV-only repair v3: frozen protocol before development results

Scope: simulation only, server only, n=1000, original four design/DGP functions.
No historical result is overwritten. This is a new development-and-confirmation
experiment, not a retroactive correction of historical estimates.

## Two distinct questions

1. Controlled mechanism: keep the original correctly specified Series OLS m
   channel; learn ell with Series OLS (Figure 1) or genuine Decision Trees with
   leaf sizes 2/5/10/20 (Figure 2). An additional RF outcome / Series OLS m
   sensitivity is labelled separately. Construct structural g=ell-pilot*m.
   Final estimator ALWAYS solves (D-m)*(Y-theta*D-g)=0. This is not the
   reported partialling-out estimator and g is NOT ell inserted unchanged.
   Models receive neither nuisance truth nor theta0. The fixed five-column
   series basis [1,x0,x2,sigmoid(x0),sigmoid(x2)] is knowingly correctly
   specified, as in controlled v3; this assumption must remain visible.
2. Ordinary learner repair: RF/XGB/Lasso learn BOTH m and structural g,
   with target Y-pilot*D. The pilot is the D coefficient in joint OLS of Y
   on [D,1,all 20 X,all 20 sigmoid(X)] INSIDE each training set. This rich
   series pilot is a disclosed change; it is not a pure same-family pilot.
   The original Lasso feature mapping is preserved. No features selected
   using truth, no oracle training and no response-residual injection.

## Pairing, nuisance construction, and pilots

For each learner, fit a shared A/B bank (500 observations per model). Reuse
predicts a row with its OWN half's model; Cross-fitting predicts it with the
OTHER half's exact same model, target and pilot. Both estimate on 1000 rows.
Separately fit ordinary full-sample No-split (1000 training rows), explicitly
labelled; do not conflate it with equal-training reuse. Both comparisons are
reported, irrespective of direction. Controlled No-split likewise has both
full-sample and equal-training versions. All pilots, transformations, and
targets depend on training observations only. There is no pooled outer pilot
in the repaired algorithm. g=ell-pilot*m is a valid alternative nuisance
construction, not numerically asserted to equal DoubleML's direct-target fit.

## Prediction-only development selection

Use 4 fresh development data sets per cell, seed 2026100421. Use only the
first predetermined training half (500 rows) of each data set. Split it
into two inner folds; aggregate validation squared prediction errors across
all four development cases and both folds. For m the target is observed D.
For g the target is Y-pilot_inner_train*D; the pilot is re-estimated only
on the inner training subset. Record ALL losses and select the minimum
pooled error separately by cell, family and role, ties by candidate order.
No development theta estimates, theta0, g0, m0, bias or coverage enter selection.

Frozen candidates, all 300 trees:
- RF: original sqrt/depth8/leaf2; all-features depth5/leaf2; all-features
  unbounded depth with leaf2,5,10,20 (six candidates).
- XGB: original lr.01/depth3; lr.05 or .1 with depth2 or3 (five candidates);
  original .9 row/column subsampling, lambda1, child weight2 unchanged.
- Lasso: alpha .005,.02,.06,.1 (four candidates), original linear/nonlinear
  feature mapping and training-only scaling unchanged.

No second selection round is allowed after seeing formal estimates. Selected
parameters and ALL selection receipts must be frozen before confirmation.
Controlled learners are fixed, not chosen by prediction or theta performance.

## Verification and independent confirmation

- Fresh formal seed 2026100521, disjoint from historical and development seeds.
  Figure1 R=300 per DGP; Figure2 R=200 per DGP; no omitted seeds/cells.
- Preflight: holdout-outcome perturbation isolation, shared-bank identity,
  structural conversion, score algebra, training-only pilot, all methods.
- Actual installed DoubleML 0.10.1: numerical nuisance/coef/SE parity with
  the historical official pooled-pilot implementation; additionally feed
  repaired external predictions to DoubleML IV-type for coefficient/SE checks.
  These are two different checks, not claimed identical training algorithms.
- Record source/config/environment hashes, per-replication durable receipts,
  failures and warnings, all-row ledger, and independent algebra/statistics
  verification. Preserve all failures; any failures prevent complete-case
  success claims and normal complete-result plots.
- Report mean bias with Monte Carlo 95% t intervals, RMSE, MAE, coverage with
  Wilson intervals, paired mean/MSE differences, and exact per-replica first/
  second-order weighted contributions. Bias is not MAE.
- Predeclared practical-zero margin ±0.01 (2% of true .5). Support requires
  the ENTIRE MC95 bias interval within this margin. An interval merely
  containing zero is inconclusive, not proof of unbiasedness. This operational
  finite-simulation criterion is not a theorem of exact unbiasedness.

## Figures and interpretation

Original-style blue histograms plus red N(0,1) on studentized error, paired
axes/bins and all samples retained. Separate ordinary learners, controlled
trees, Series OLS and mixed RF sensitivity; also include bias MC95 plots.
Do not disguise mixed RF+Series OLS as RF learning both nuisances. There is
no scientific success gate requiring four expected conclusions. No-split
need not be biased for every learner; Cross-fitting need not dominate every
finite-sample configuration. Retain and explain adverse results.

Execution cap: one isolated directory /data2/lzx/dhy/dml_iv_repair_v3_20261002,
12 CPU worker processes, one thread each, no GPU, no local fits or simulations,
one development selection and one formal universe only. No automated retries,
expansion of sample size or formal retuning. Tests may be repaired if faulty,
but no selected/formal scientific procedure is changed after formal launch.
