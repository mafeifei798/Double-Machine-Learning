# Equal-training IV-type diagnostic v1

## Purpose and chronology

The user requested an equal-training-size diagnostic after seeing the adverse
ordinary two-fold comparison. This is a follow-up on the same simulated data,
not independent confirmation, preregistration of the original question, or a
guarantee of the desired bias pattern. Source/config/analysis are frozen before
this diagnostic's estimates are inspected. Historical results stay immutable.
All fitting and simulation, including tests, run on rna-server only.

## Fixed background

Use the four preserved thesis design/DGP cells, n=1000, p=20, theta0=0.5,
Toeplitz rho=0.7, original RF/XGBoost/Lasso hyperparameters, and the same data,
fold and model-role seed namespace as fair_iv_v1. Thus this is a paired
follow-up using the earlier data, not a newly selected favorable seed. Figure 1
has R=300 per DGP and Figure 2 R=200. There is no stress injection, Series OLS
substitution, oracle training, adaptive hyperparameter selection or row dropping.

## One shared model bank; two prediction assignments

Partition each dataset into A and B, exactly 500 each. For each learner and each
training half S, fit ell_S on Y_S and m_S on D_S. Use their IN-SAMPLE predictions
within S to compute the preliminary residual-ratio scalar theta_S. Then fit
g_S on Y_S-theta_S*D_S. Save g_S and m_S predictions for BOTH halves. There are
exactly six fits per learner (ell,m,g times two halves); no refit per comparison
arm. Every pilot is training-half-only, and every model has 500 training rows.

- equal_train_reuse: A-trained models predict A; B-trained models predict B.
- equal_train_crossfit: B-trained models predict A; A-trained models predict B.

Both assemble 1000 predictions and use the SAME final IV-type ratio
mean((D-mhat)*(Y-ghat))/mean((D-mhat)*D). Both use the same trained bank,
hyperparameters, pilots and training targets. Only which bank model supplies
each observation's prediction changes. Record bank/prediction/membership hashes,
training/evaluation sizes, pilots and numerical failures.

The reuse arm is NOT ordinary full-sample No-split. The crossfit arm uses a
training-half pilot, NOT the official pooled OOF pilot of fair_iv_v1. These
changes are explicit; results cannot be silently spliced into the former run.
In particular, generated-target errors inherited from a training-half pilot
remain part of the nuisance learner's behavior. A zero bias result is not
assumed. Sharing a bank does not make its in-sample and out-of-sample prediction
quality identical; that difference is precisely part of the reuse diagnostic.

## Analysis fixed before estimates

Report all 24 learner/DGP/design/arm cells: signed Monte Carlo bias with t95
interval, empirical SD, MAE, RMSE, mean sandwich SE and nominal 95% interval
coverage with Wilson interval. Nominal sandwich coverage is descriptive;
validity is NOT assumed for the reused-sample arm or all finite-sample learners.
Report paired crossfit-minus-reuse bias/MAE/MSE differences with t95 intervals.
Report equal-weight absolute mean bias across six learner/DGP cells per design,
and its paired block-bootstrap difference (2000 draws, fixed seed2026100501,
joint learner blocks within each DGP). The absolute difference, not an unstable
percentage relative to possibly near-zero reuse bias, is primary aggregate.

Record the exact IV score decomposition at true theta using known simulation
truth for DIAGNOSTICS ONLY: mean(V*U), mean(-V*a-b*U), mean(b*a), where
a=ghat-g0 and b=mhat-m0. Truth is never a training input. This is a descriptive
decomposition, not a proof that a particular component caused an improvement.

Four original-style 3x2 learner-by-arm histograms share bins and x limits within
each paired row, with no clipping or outlier removal. Also provide one compact
bias comparison figure with MC intervals. Keep all warnings/failures. Any
failed estimate or missing replication blocks a complete scientific report.
No success gate depends on recovering the four thesis conclusions; no post-hoc
seed/setting changes are allowed.
