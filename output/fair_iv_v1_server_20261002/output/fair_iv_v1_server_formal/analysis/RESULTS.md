# Fair IV-type comparison: complete results

FORMAL: fixed configuration; all replication IDs and result hashes verified.

No artificial residual pressure; learned RF/XGB/Lasso for both nuisance channels.
Original DGPs and hyperparameters; n=1,000; two folds. The official pooled initial scalar is disclosed in the protocol.

Bias is signed Monte Carlo mean error, not MAE. Intervals below are Monte Carlo uncertainty for the mean bias, not individual-estimate confidence intervals.

| Design | DGP | Learner | Method | R | Mean | Bias [MC 95% CI] | RMSE | MAE | Coverage [Wilson 95% CI] |
|---|---|---|---|---:|---:|---|---:|---:|---|
| figure1 | linear | lasso | iv_crossfit | 300 | 0.500496 | +0.000496 [-0.003295, +0.004286] | 0.033310 | 0.026970 | 0.920 [0.884, 0.946] |
| figure1 | linear | lasso | iv_no_split | 300 | 0.500495 | +0.000495 [-0.003165, +0.004156] | 0.032165 | 0.025582 | 0.933 [0.899, 0.956] |
| figure1 | linear | lasso | nonorthogonal_crossfit | 300 | 0.510573 | +0.010573 [+0.006825, +0.014321] | 0.034586 | 0.027180 | 0.780 [0.730, 0.823] |
| figure1 | linear | rf | iv_crossfit | 300 | 0.531252 | +0.031252 [+0.027516, +0.034989] | 0.045327 | 0.036815 | 0.727 [0.674, 0.774] |
| figure1 | linear | rf | iv_no_split | 300 | 0.521318 | +0.021318 [+0.017671, +0.024966] | 0.038492 | 0.030113 | 0.687 [0.632, 0.737] |
| figure1 | linear | rf | nonorthogonal_crossfit | 300 | 0.558248 | +0.058248 [+0.054834, +0.061662] | 0.065520 | 0.058975 | 0.333 [0.282, 0.388] |
| figure1 | linear | xgb | iv_crossfit | 300 | 0.514756 | +0.014756 [+0.010763, +0.018748] | 0.038060 | 0.030283 | 0.880 [0.838, 0.912] |
| figure1 | linear | xgb | iv_no_split | 300 | 0.509519 | +0.009519 [+0.005793, +0.013245] | 0.034096 | 0.026959 | 0.857 [0.812, 0.892] |
| figure1 | linear | xgb | nonorthogonal_crossfit | 300 | 0.540399 | +0.040399 [+0.036682, +0.044115] | 0.051947 | 0.043930 | 0.540 [0.483, 0.596] |
| figure1 | nonlinear | lasso | iv_crossfit | 300 | 0.504859 | +0.004859 [+0.000934, +0.008784] | 0.034829 | 0.027729 | 0.903 [0.865, 0.932] |
| figure1 | nonlinear | lasso | iv_no_split | 300 | 0.505130 | +0.005130 [+0.001311, +0.008949] | 0.033947 | 0.026643 | 0.893 [0.853, 0.923] |
| figure1 | nonlinear | lasso | nonorthogonal_crossfit | 300 | 0.532335 | +0.032335 [+0.028430, +0.036240] | 0.047145 | 0.038666 | 0.637 [0.581, 0.689] |
| figure1 | nonlinear | rf | iv_crossfit | 300 | 0.520426 | +0.020426 [+0.016676, +0.024175] | 0.038766 | 0.031092 | 0.810 [0.762, 0.850] |
| figure1 | nonlinear | rf | iv_no_split | 300 | 0.512569 | +0.012569 [+0.008837, +0.016302] | 0.035123 | 0.027794 | 0.750 [0.698, 0.796] |
| figure1 | nonlinear | rf | nonorthogonal_crossfit | 300 | 0.543719 | +0.043719 [+0.040277, +0.047161] | 0.053159 | 0.045979 | 0.563 [0.507, 0.618] |
| figure1 | nonlinear | xgb | iv_crossfit | 300 | 0.508906 | +0.008906 [+0.004829, +0.012984] | 0.036919 | 0.029475 | 0.887 [0.846, 0.918] |
| figure1 | nonlinear | xgb | iv_no_split | 300 | 0.505736 | +0.005736 [+0.001870, +0.009602] | 0.034450 | 0.027045 | 0.843 [0.798, 0.880] |
| figure1 | nonlinear | xgb | nonorthogonal_crossfit | 300 | 0.530795 | +0.030795 [+0.026967, +0.034622] | 0.045600 | 0.037415 | 0.677 [0.622, 0.727] |
| figure2 | linear | lasso | iv_crossfit | 200 | 0.500461 | +0.000461 [-0.004245, +0.005168] | 0.033671 | 0.027176 | 0.930 [0.886, 0.958] |
| figure2 | linear | lasso | iv_no_split | 200 | 0.500178 | +0.000178 [-0.004431, +0.004787] | 0.032969 | 0.026468 | 0.925 [0.880, 0.954] |
| figure2 | linear | lasso | nonorthogonal_crossfit | 200 | 0.524001 | +0.024001 [+0.019311, +0.028690] | 0.041251 | 0.033515 | 0.580 [0.511, 0.646] |
| figure2 | linear | rf | iv_crossfit | 200 | 0.560942 | +0.060942 [+0.056466, +0.065417] | 0.068839 | 0.061814 | 0.330 [0.269, 0.398] |
| figure2 | linear | rf | iv_no_split | 200 | 0.541959 | +0.041959 [+0.037492, +0.046426] | 0.052744 | 0.045225 | 0.385 [0.320, 0.454] |
| figure2 | linear | rf | nonorthogonal_crossfit | 200 | 0.599480 | +0.099480 [+0.095381, +0.103579] | 0.103711 | 0.099480 | 0.020 [0.008, 0.050] |
| figure2 | linear | xgb | iv_crossfit | 200 | 0.522165 | +0.022165 [+0.017115, +0.027216] | 0.042387 | 0.033805 | 0.815 [0.755, 0.863] |
| figure2 | linear | xgb | iv_no_split | 200 | 0.513900 | +0.013900 [+0.009245, +0.018555] | 0.036087 | 0.028573 | 0.780 [0.718, 0.832] |
| figure2 | linear | xgb | nonorthogonal_crossfit | 200 | 0.555466 | +0.055466 [+0.050743, +0.060189] | 0.064947 | 0.057021 | 0.285 [0.227, 0.351] |
| figure2 | nonlinear | lasso | iv_crossfit | 200 | 0.507401 | +0.007401 [+0.002820, +0.011983] | 0.033600 | 0.027035 | 0.920 [0.874, 0.950] |
| figure2 | nonlinear | lasso | iv_no_split | 200 | 0.507910 | +0.007910 [+0.003435, +0.012385] | 0.032975 | 0.026447 | 0.920 [0.874, 0.950] |
| figure2 | nonlinear | lasso | nonorthogonal_crossfit | 200 | 0.534423 | +0.034423 [+0.029885, +0.038962] | 0.047318 | 0.038495 | 0.540 [0.471, 0.608] |
| figure2 | nonlinear | rf | iv_crossfit | 200 | 0.531436 | +0.031436 [+0.027268, +0.035604] | 0.043328 | 0.034885 | 0.725 [0.659, 0.782] |
| figure2 | nonlinear | rf | iv_no_split | 200 | 0.519574 | +0.019574 [+0.015443, +0.023704] | 0.035443 | 0.027976 | 0.670 [0.602, 0.731] |
| figure2 | nonlinear | rf | nonorthogonal_crossfit | 200 | 0.559722 | +0.059722 [+0.055904, +0.063540] | 0.065670 | 0.059766 | 0.265 [0.209, 0.330] |
| figure2 | nonlinear | xgb | iv_crossfit | 200 | 0.514351 | +0.014351 [+0.009808, +0.018893] | 0.035522 | 0.028259 | 0.895 [0.845, 0.930] |
| figure2 | nonlinear | xgb | iv_no_split | 200 | 0.510102 | +0.010102 [+0.005763, +0.014440] | 0.032639 | 0.025937 | 0.875 [0.822, 0.914] |
| figure2 | nonlinear | xgb | nonorthogonal_crossfit | 200 | 0.540741 | +0.040741 [+0.036516, +0.044967] | 0.050732 | 0.042933 | 0.480 [0.412, 0.549] |

## Equal-weight absolute-mean-bias contrasts

A negative reduction means the cross-fit estimator had larger absolute mean bias. Do not select favorable cells or reinterpret this as universal robustness.

- figure1, nonorthogonal_crossfit vs IV cross-fit: 0.036011 -> 0.013449; reduction 62.65% (paired block-bootstrap 95% interval 58.25% to 66.50%).
- figure1, iv_no_split vs IV cross-fit: 0.009128 -> 0.013449; reduction -47.34% (paired block-bootstrap 95% interval -63.13% to -34.82%).
- figure2, nonorthogonal_crossfit vs IV cross-fit: 0.052306 -> 0.022793; reduction 56.42% (paired block-bootstrap 95% interval 53.07% to 58.81%).
- figure2, iv_no_split vs IV cross-fit: 0.015604 -> 0.022793; reduction -46.07% (paired block-bootstrap 95% interval -55.40% to -36.71%).

Every per-cell paired MAE/MSE/bias contrast, including adverse results, is in paired_contrasts.json.
The bootstrap interval is descriptive finite-Monte-Carlo uncertainty, not a generalization guarantee. No scientific success gate is based on obtaining four desired conclusions.

Rows: 9000; failed estimates: 0; recorded warnings: 0.
