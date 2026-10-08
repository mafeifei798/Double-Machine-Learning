# Equal-training IV-type: complete paired results

Formal follow-up; all planned replication IDs retained and hashes verified.

Same data and same trained A/B model bank for both arms. Each model trains on 500 observations; both arms estimate on 1000.
Training-half-only pilots are shared by both arms. No artificial pressure, oracle fitting or Series OLS replacement.
Reuse is an equal-training mechanism comparator, NOT ordinary full-sample No-split. This crossfit pilot differs from the earlier official pooled pilot.
Bias intervals describe Monte Carlo uncertainty. Nominal sandwich coverage is descriptive, not assumed valid for reuse.

| Design | DGP | Learner | Arm | R | Mean | Bias [MC95] | RMSE | MAE | Nominal coverage |
|---|---|---|---|---:|---:|---|---:|---:|---:|
| figure1 | linear | lasso | equal_train_crossfit | 300 | 0.500216 | +0.000216 [-0.003565, +0.003997] | 0.033223 | 0.026977 | 0.927 |
| figure1 | linear | lasso | equal_train_reuse | 300 | 0.500810 | +0.000810 [-0.002864, +0.004483] | 0.032288 | 0.025575 | 0.940 |
| figure1 | linear | rf | equal_train_crossfit | 300 | 0.530343 | +0.030343 [+0.026638, +0.034047] | 0.044498 | 0.035949 | 0.750 |
| figure1 | linear | rf | equal_train_reuse | 300 | 0.524633 | +0.024633 [+0.021004, +0.028262] | 0.040295 | 0.031670 | 0.617 |
| figure1 | linear | xgb | equal_train_crossfit | 300 | 0.514383 | +0.014383 [+0.010426, +0.018340] | 0.037625 | 0.029918 | 0.877 |
| figure1 | linear | xgb | equal_train_reuse | 300 | 0.511371 | +0.011371 [+0.007536, +0.015207] | 0.035566 | 0.027778 | 0.787 |
| figure1 | nonlinear | lasso | equal_train_crossfit | 300 | 0.504686 | +0.004686 [+0.000756, +0.008615] | 0.034844 | 0.027766 | 0.897 |
| figure1 | nonlinear | lasso | equal_train_reuse | 300 | 0.504932 | +0.004932 [+0.001081, +0.008784] | 0.034201 | 0.026728 | 0.900 |
| figure1 | nonlinear | rf | equal_train_crossfit | 300 | 0.519617 | +0.019617 [+0.015897, +0.023338] | 0.038125 | 0.030549 | 0.803 |
| figure1 | nonlinear | rf | equal_train_reuse | 300 | 0.515458 | +0.015458 [+0.011749, +0.019168] | 0.036075 | 0.028647 | 0.663 |
| figure1 | nonlinear | xgb | equal_train_crossfit | 300 | 0.508510 | +0.008510 [+0.004458, +0.012562] | 0.036608 | 0.029242 | 0.887 |
| figure1 | nonlinear | xgb | equal_train_reuse | 300 | 0.506456 | +0.006456 [+0.002565, +0.010346] | 0.034786 | 0.027020 | 0.803 |
| figure2 | linear | lasso | equal_train_crossfit | 200 | 0.500106 | +0.000106 [-0.004605, +0.004817] | 0.033701 | 0.027158 | 0.920 |
| figure2 | linear | lasso | equal_train_reuse | 200 | 0.500472 | +0.000472 [-0.004122, +0.005067] | 0.032872 | 0.026479 | 0.915 |
| figure2 | linear | rf | equal_train_crossfit | 200 | 0.558781 | +0.058781 [+0.054407, +0.063155] | 0.066591 | 0.059708 | 0.365 |
| figure2 | linear | rf | equal_train_reuse | 200 | 0.551103 | +0.051103 [+0.046722, +0.055484] | 0.059947 | 0.053019 | 0.235 |
| figure2 | linear | xgb | equal_train_crossfit | 200 | 0.521518 | +0.021518 [+0.016566, +0.026470] | 0.041450 | 0.033093 | 0.820 |
| figure2 | linear | xgb | equal_train_reuse | 200 | 0.519496 | +0.019496 [+0.014799, +0.024193] | 0.038847 | 0.031429 | 0.730 |
| figure2 | nonlinear | lasso | equal_train_crossfit | 200 | 0.507588 | +0.007588 [+0.002987, +0.012188] | 0.033773 | 0.027110 | 0.915 |
| figure2 | nonlinear | lasso | equal_train_reuse | 200 | 0.508036 | +0.008036 [+0.003543, +0.012529] | 0.033131 | 0.026728 | 0.920 |
| figure2 | nonlinear | rf | equal_train_crossfit | 200 | 0.529950 | +0.029950 [+0.025816, +0.034084] | 0.042088 | 0.033722 | 0.745 |
| figure2 | nonlinear | rf | equal_train_reuse | 200 | 0.523200 | +0.023200 [+0.019035, +0.027365] | 0.037764 | 0.030050 | 0.580 |
| figure2 | nonlinear | xgb | equal_train_crossfit | 200 | 0.514012 | +0.014012 [+0.009508, +0.018516] | 0.035137 | 0.027986 | 0.905 |
| figure2 | nonlinear | xgb | equal_train_reuse | 200 | 0.511257 | +0.011257 [+0.006668, +0.015846] | 0.034705 | 0.027529 | 0.835 |

## Equal-weight absolute Monte Carlo mean bias

Average of six learner/DGP absolute mean biases per design; NOT MAE. Negative differences favor cross-fitting.

- figure1: reuse 0.010610; crossfit 0.012959; difference +0.002349, paired bootstrap95 [0.0015310920135296854, 0.0032788312522820527].
- figure2: reuse 0.018928; crossfit 0.021992; difference +0.003065, paired bootstrap95 [0.002037677857875478, 0.004138950293052322].

## Paired contrasts

| Design | DGP | Learner | CF-minus-reuse bias [MC95] | MAE difference [MC95] | MSE difference [MC95] |
|---|---|---|---|---|---|
| figure1 | linear | rf | +0.005709 [+0.004104, +0.007315] | +0.004279 [+0.002736, +0.005821] | +0.000356 [+0.000218, +0.000494] |
| figure1 | linear | xgb | +0.003012 [+0.001006, +0.005018] | +0.002140 [+0.000310, +0.003970] | +0.000151 [+0.000011, +0.000291] |
| figure1 | linear | lasso | -0.000593 [-0.001843, +0.000656] | +0.001402 [+0.000252, +0.002553] | +0.000061 [-0.000012, +0.000135] |
| figure1 | nonlinear | rf | +0.004159 [+0.002623, +0.005695] | +0.001902 [+0.000442, +0.003362] | +0.000152 [+0.000035, +0.000269] |
| figure1 | nonlinear | xgb | +0.002054 [+0.000239, +0.003870] | +0.002222 [+0.000601, +0.003843] | +0.000130 [-0.000006, +0.000266] |
| figure1 | nonlinear | lasso | -0.000247 [-0.001209, +0.000716] | +0.001038 [+0.000137, +0.001938] | +0.000044 [-0.000020, +0.000109] |
| figure2 | linear | rf | +0.007678 [+0.005849, +0.009506] | +0.006689 [+0.004847, +0.008531] | +0.000841 [+0.000588, +0.001094] |
| figure2 | linear | xgb | +0.002022 [-0.000245, +0.004289] | +0.001664 [-0.000355, +0.003683] | +0.000209 [+0.000016, +0.000401] |
| figure2 | linear | lasso | -0.000367 [-0.001400, +0.000667] | +0.000679 [-0.000273, +0.001630] | +0.000055 [-0.000019, +0.000129] |
| figure2 | nonlinear | rf | +0.006750 [+0.004858, +0.008642] | +0.003673 [+0.001788, +0.005557] | +0.000345 [+0.000169, +0.000521] |
| figure2 | nonlinear | xgb | +0.002755 [+0.000279, +0.005230] | +0.000457 [-0.001736, +0.002650] | +0.000030 [-0.000141, +0.000201] |
| figure2 | nonlinear | lasso | -0.000449 [-0.001718, +0.000821] | +0.000382 [-0.000761, +0.001525] | +0.000043 [-0.000044, +0.000130] |

All nuisance RMSE and score-decomposition means/intervals are in summary.json. They are descriptive diagnostics only.
No cells, seeds or hyperparameters are selected based on the results; no exact-unbiasedness or universal robustness claim is implied.
Rows: 6000; failures: 0; warnings: 0.
