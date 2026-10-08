# IV-only repair v3: independent confirmation

Original four DGPs; n=1000; two folds; all replications and all methods retained.
Controlled Series OLS treatment nuisance is correctly specified. Mixed RF outcome / Series OLS m is NOT pure RF.
Ordinary RF/XGB/Lasso fit both nuisances using prediction-selected parameters and a training-only rich-series pilot.
All final DML estimates use structural IV-type score. No artificial pressure or oracle training.

Practical-zero support means the entire MC95 mean-bias interval is inside the prespecified ±0.01 margin; not exact unbiasedness.
Bias detection (interval excludes zero) and practical equivalence answer different questions; both may be true.
Coverage uses usual asymptotic IV-score standard errors; No-split intervals are descriptive, not guaranteed valid.

| Design | DGP | Track | Learner | Method | R | Mean | Bias [MC95] | RMSE | Coverage | Practical zero |
|---|---|---|---|---|---:|---:|---|---:|---:|---|
| figure1 | linear | controlled | series_ols | crossfit | 300 | 0.501876 | +0.001876 [-0.001484,+0.005236] | 0.029581 | 0.963 | True |
| figure1 | linear | controlled | series_ols | equal_train_reuse | 300 | 0.502696 | +0.002696 [-0.000592,+0.005984] | 0.029017 | 0.967 | True |
| figure1 | linear | controlled | series_ols | full_no_split | 300 | 0.502508 | +0.002508 [-0.000768,+0.005783] | 0.028893 | 0.970 | True |
| figure1 | linear | controlled | series_ols | nonorthogonal_crossfit | 300 | 0.502437 | +0.002437 [-0.000864,+0.005738] | 0.029108 | 0.883 | True |
| figure1 | linear | controlled | series_ols | outcome_mismatch_crossfit | 300 | 0.248269 | -0.251731 [-0.253982,-0.249479] | 0.252507 | 0.000 | False |
| figure1 | linear | ordinary | lasso | crossfit | 300 | 0.507490 | +0.007490 [+0.004139,+0.010841] | 0.030383 | 0.947 | False |
| figure1 | linear | ordinary | lasso | equal_train_reuse | 300 | 0.507621 | +0.007621 [+0.004330,+0.010912] | 0.029907 | 0.960 | False |
| figure1 | linear | ordinary | lasso | full_no_split | 300 | 0.507263 | +0.007263 [+0.003978,+0.010548] | 0.029765 | 0.953 | False |
| figure1 | linear | ordinary | lasso | nonorthogonal_crossfit | 300 | 0.535727 | +0.035727 [+0.032390,+0.039064] | 0.046219 | 0.573 | False |
| figure1 | linear | ordinary | rf | crossfit | 300 | 0.510797 | +0.010797 [+0.007210,+0.014383] | 0.033312 | 0.950 | False |
| figure1 | linear | ordinary | rf | equal_train_reuse | 300 | 0.509051 | +0.009051 [+0.005507,+0.012594] | 0.032424 | 0.913 | False |
| figure1 | linear | ordinary | rf | full_no_split | 300 | 0.506051 | +0.006051 [+0.002622,+0.009480] | 0.030734 | 0.923 | True |
| figure1 | linear | ordinary | rf | nonorthogonal_crossfit | 300 | 0.528968 | +0.028968 [+0.025632,+0.032303] | 0.041207 | 0.690 | False |
| figure1 | linear | ordinary | xgb | crossfit | 300 | 0.515744 | +0.015744 [+0.012185,+0.019303] | 0.035014 | 0.917 | False |
| figure1 | linear | ordinary | xgb | equal_train_reuse | 300 | 0.513305 | +0.013305 [+0.009826,+0.016784] | 0.033339 | 0.847 | False |
| figure1 | linear | ordinary | xgb | full_no_split | 300 | 0.511027 | +0.011027 [+0.007634,+0.014420] | 0.031788 | 0.903 | False |
| figure1 | linear | ordinary | xgb | nonorthogonal_crossfit | 300 | 0.539025 | +0.039025 [+0.035797,+0.042253] | 0.048244 | 0.560 | False |
| figure1 | nonlinear | controlled | series_ols | crossfit | 300 | 0.502618 | +0.002618 [-0.001261,+0.006497] | 0.034186 | 0.937 | True |
| figure1 | nonlinear | controlled | series_ols | equal_train_reuse | 300 | 0.503060 | +0.003060 [-0.000588,+0.006708] | 0.032198 | 0.940 | True |
| figure1 | nonlinear | controlled | series_ols | full_no_split | 300 | 0.502938 | +0.002938 [-0.000733,+0.006609] | 0.032388 | 0.937 | True |
| figure1 | nonlinear | controlled | series_ols | nonorthogonal_crossfit | 300 | 0.502951 | +0.002951 [-0.000783,+0.006684] | 0.032935 | 0.827 | True |
| figure1 | nonlinear | controlled | series_ols | outcome_mismatch_crossfit | 300 | 0.271405 | -0.228595 [-0.231057,-0.226133] | 0.229616 | 0.000 | False |
| figure1 | nonlinear | ordinary | lasso | crossfit | 300 | 0.509322 | +0.009322 [+0.005540,+0.013104] | 0.034515 | 0.910 | False |
| figure1 | nonlinear | ordinary | lasso | equal_train_reuse | 300 | 0.509268 | +0.009268 [+0.005619,+0.012918] | 0.033380 | 0.903 | False |
| figure1 | nonlinear | ordinary | lasso | full_no_split | 300 | 0.509627 | +0.009627 [+0.005961,+0.013293] | 0.033622 | 0.900 | False |
| figure1 | nonlinear | ordinary | lasso | nonorthogonal_crossfit | 300 | 0.553401 | +0.053401 [+0.049689,+0.057112] | 0.062571 | 0.413 | False |
| figure1 | nonlinear | ordinary | rf | crossfit | 300 | 0.506280 | +0.006280 [+0.002207,+0.010354] | 0.036341 | 0.920 | False |
| figure1 | nonlinear | ordinary | rf | equal_train_reuse | 300 | 0.505563 | +0.005563 [+0.001695,+0.009431] | 0.034438 | 0.883 | True |
| figure1 | nonlinear | ordinary | rf | full_no_split | 300 | 0.504288 | +0.004288 [+0.000467,+0.008109] | 0.033848 | 0.883 | True |
| figure1 | nonlinear | ordinary | rf | nonorthogonal_crossfit | 300 | 0.523118 | +0.023118 [+0.019434,+0.026803] | 0.039780 | 0.733 | False |
| figure1 | nonlinear | ordinary | xgb | crossfit | 300 | 0.510625 | +0.010625 [+0.006708,+0.014542] | 0.036023 | 0.907 | False |
| figure1 | nonlinear | ordinary | xgb | equal_train_reuse | 300 | 0.509154 | +0.009154 [+0.005380,+0.012928] | 0.034398 | 0.807 | False |
| figure1 | nonlinear | ordinary | xgb | full_no_split | 300 | 0.507607 | +0.007607 [+0.003891,+0.011323] | 0.033525 | 0.860 | False |
| figure1 | nonlinear | ordinary | xgb | nonorthogonal_crossfit | 300 | 0.532353 | +0.032353 [+0.028773,+0.035932] | 0.045120 | 0.677 | False |
| figure2 | linear | controlled | rf_outcome_series_m | crossfit | 200 | 0.500699 | +0.000699 [-0.004369,+0.005767] | 0.036262 | 0.945 | True |
| figure2 | linear | controlled | rf_outcome_series_m | equal_train_reuse | 200 | 0.211212 | -0.288788 [-0.290877,-0.286698] | 0.289174 | 0.000 | False |
| figure2 | linear | controlled | rf_outcome_series_m | full_no_split | 200 | 0.210110 | -0.289890 [-0.291975,-0.287805] | 0.290273 | 0.000 | False |
| figure2 | linear | controlled | rf_outcome_series_m | nonorthogonal_crossfit | 200 | 0.537745 | +0.037745 [+0.033087,+0.042404] | 0.050351 | 0.520 | False |
| figure2 | linear | controlled | series_ols | crossfit | 200 | 0.501484 | +0.001484 [-0.003169,+0.006137] | 0.033318 | 0.940 | True |
| figure2 | linear | controlled | series_ols | equal_train_reuse | 200 | 0.501103 | +0.001103 [-0.003348,+0.005554] | 0.031861 | 0.950 | True |
| figure2 | linear | controlled | series_ols | full_no_split | 200 | 0.501192 | +0.001192 [-0.003258,+0.005642] | 0.031857 | 0.960 | True |
| figure2 | linear | controlled | series_ols | nonorthogonal_crossfit | 200 | 0.500721 | +0.000721 [-0.003716,+0.005158] | 0.031748 | 0.765 | True |
| figure2 | linear | controlled | tree_leaf_10 | crossfit | 200 | 0.501265 | +0.001265 [-0.004554,+0.007083] | 0.041644 | 0.950 | True |
| figure2 | linear | controlled | tree_leaf_10 | equal_train_reuse | 200 | 0.301018 | -0.198982 [-0.202948,-0.195016] | 0.200994 | 0.000 | False |
| figure2 | linear | controlled | tree_leaf_10 | full_no_split | 200 | 0.295149 | -0.204851 [-0.208750,-0.200952] | 0.206741 | 0.000 | False |
| figure2 | linear | controlled | tree_leaf_10 | nonorthogonal_crossfit | 200 | 0.528925 | +0.028925 [+0.023867,+0.033983] | 0.046325 | 0.640 | False |
| figure2 | linear | controlled | tree_leaf_2 | crossfit | 200 | 0.498431 | -0.001569 [-0.008793,+0.005654] | 0.051696 | 0.955 | True |
| figure2 | linear | controlled | tree_leaf_2 | equal_train_reuse | 200 | 0.033242 | -0.466758 [-0.467947,-0.465570] | 0.466836 | 0.000 | False |
| figure2 | linear | controlled | tree_leaf_2 | full_no_split | 200 | 0.032016 | -0.467984 [-0.469263,-0.466705] | 0.468073 | 0.000 | False |
| figure2 | linear | controlled | tree_leaf_2 | nonorthogonal_crossfit | 200 | 0.525254 | +0.025254 [+0.019724,+0.030783] | 0.046930 | 0.750 | False |
| figure2 | linear | controlled | tree_leaf_20 | crossfit | 200 | 0.500677 | +0.000677 [-0.005084,+0.006439] | 0.041221 | 0.925 | True |
| figure2 | linear | controlled | tree_leaf_20 | equal_train_reuse | 200 | 0.397526 | -0.102474 [-0.106854,-0.098094] | 0.107158 | 0.095 | False |
| figure2 | linear | controlled | tree_leaf_20 | full_no_split | 200 | 0.390884 | -0.109116 [-0.113592,-0.104641] | 0.113717 | 0.065 | False |
| figure2 | linear | controlled | tree_leaf_20 | nonorthogonal_crossfit | 200 | 0.538377 | +0.038377 [+0.033517,+0.043237] | 0.051783 | 0.525 | False |
| figure2 | linear | controlled | tree_leaf_5 | crossfit | 200 | 0.499599 | -0.000401 [-0.007055,+0.006253] | 0.047601 | 0.935 | True |
| figure2 | linear | controlled | tree_leaf_5 | equal_train_reuse | 200 | 0.166462 | -0.333538 [-0.336515,-0.330562] | 0.334217 | 0.000 | False |
| figure2 | linear | controlled | tree_leaf_5 | full_no_split | 200 | 0.165853 | -0.334147 [-0.336934,-0.331359] | 0.334741 | 0.000 | False |
| figure2 | linear | controlled | tree_leaf_5 | nonorthogonal_crossfit | 200 | 0.525911 | +0.025911 [+0.020539,+0.031283] | 0.046350 | 0.700 | False |
| figure2 | linear | ordinary | lasso | crossfit | 200 | 0.506301 | +0.006301 [+0.001691,+0.010911] | 0.033574 | 0.910 | False |
| figure2 | linear | ordinary | lasso | equal_train_reuse | 200 | 0.505602 | +0.005602 [+0.001127,+0.010077] | 0.032499 | 0.915 | False |
| figure2 | linear | ordinary | lasso | full_no_split | 200 | 0.505607 | +0.005607 [+0.001152,+0.010063] | 0.032363 | 0.925 | False |
| figure2 | linear | ordinary | lasso | nonorthogonal_crossfit | 200 | 0.532095 | +0.032095 [+0.027489,+0.036702] | 0.046000 | 0.530 | False |
| figure2 | linear | ordinary | rf | crossfit | 200 | 0.523174 | +0.023174 [+0.018217,+0.028131] | 0.042362 | 0.830 | False |
| figure2 | linear | ordinary | rf | equal_train_reuse | 200 | 0.511853 | +0.011853 [+0.007167,+0.016540] | 0.035557 | 0.730 | False |
| figure2 | linear | ordinary | rf | full_no_split | 200 | 0.509681 | +0.009681 [+0.005027,+0.014335] | 0.034673 | 0.765 | False |
| figure2 | linear | ordinary | rf | nonorthogonal_crossfit | 200 | 0.571172 | +0.071172 [+0.067157,+0.075187] | 0.076750 | 0.115 | False |
| figure2 | linear | ordinary | xgb | crossfit | 200 | 0.524746 | +0.024746 [+0.019901,+0.029592] | 0.042590 | 0.780 | False |
| figure2 | linear | ordinary | xgb | equal_train_reuse | 200 | 0.518943 | +0.018943 [+0.014338,+0.023548] | 0.038001 | 0.755 | False |
| figure2 | linear | ordinary | xgb | full_no_split | 200 | 0.516039 | +0.016039 [+0.011519,+0.020558] | 0.036090 | 0.815 | False |
| figure2 | linear | ordinary | xgb | nonorthogonal_crossfit | 200 | 0.548567 | +0.048567 [+0.044170,+0.052964] | 0.057863 | 0.415 | False |
| figure2 | nonlinear | controlled | rf_outcome_series_m | crossfit | 200 | 0.502051 | +0.002051 [-0.002810,+0.006912] | 0.034835 | 0.945 | True |
| figure2 | nonlinear | controlled | rf_outcome_series_m | equal_train_reuse | 200 | 0.211860 | -0.288140 [-0.290203,-0.286078] | 0.288518 | 0.000 | False |
| figure2 | nonlinear | controlled | rf_outcome_series_m | full_no_split | 200 | 0.211046 | -0.288954 [-0.290983,-0.286925] | 0.289318 | 0.000 | False |
| figure2 | nonlinear | controlled | rf_outcome_series_m | nonorthogonal_crossfit | 200 | 0.529511 | +0.029511 [+0.024733,+0.034290] | 0.045160 | 0.645 | False |
| figure2 | nonlinear | controlled | series_ols | crossfit | 200 | 0.503238 | +0.003238 [-0.001423,+0.007898] | 0.033496 | 0.950 | True |
| figure2 | nonlinear | controlled | series_ols | equal_train_reuse | 200 | 0.502718 | +0.002718 [-0.001895,+0.007331] | 0.033111 | 0.945 | True |
| figure2 | nonlinear | controlled | series_ols | full_no_split | 200 | 0.502814 | +0.002814 [-0.001766,+0.007395] | 0.032890 | 0.945 | True |
| figure2 | nonlinear | controlled | series_ols | nonorthogonal_crossfit | 200 | 0.502829 | +0.002829 [-0.001821,+0.007480] | 0.033391 | 0.750 | True |
| figure2 | nonlinear | controlled | tree_leaf_10 | crossfit | 200 | 0.499105 | -0.000895 [-0.006550,+0.004759] | 0.040458 | 0.955 | True |
| figure2 | nonlinear | controlled | tree_leaf_10 | equal_train_reuse | 200 | 0.299530 | -0.200470 [-0.204034,-0.196906] | 0.202085 | 0.000 | False |
| figure2 | nonlinear | controlled | tree_leaf_10 | full_no_split | 200 | 0.293763 | -0.206237 [-0.209822,-0.202652] | 0.207825 | 0.000 | False |
| figure2 | nonlinear | controlled | tree_leaf_10 | nonorthogonal_crossfit | 200 | 0.520888 | +0.020888 [+0.015514,+0.026262] | 0.043753 | 0.750 | False |
| figure2 | nonlinear | controlled | tree_leaf_2 | crossfit | 200 | 0.497168 | -0.002832 [-0.009329,+0.003665] | 0.046562 | 0.955 | True |
| figure2 | nonlinear | controlled | tree_leaf_2 | equal_train_reuse | 200 | 0.032556 | -0.467444 [-0.468713,-0.466176] | 0.467533 | 0.000 | False |
| figure2 | nonlinear | controlled | tree_leaf_2 | full_no_split | 200 | 0.031545 | -0.468455 [-0.469713,-0.467196] | 0.468541 | 0.000 | False |
| figure2 | nonlinear | controlled | tree_leaf_2 | nonorthogonal_crossfit | 200 | 0.518875 | +0.018875 [+0.013423,+0.024327] | 0.043330 | 0.815 | False |
| figure2 | nonlinear | controlled | tree_leaf_20 | crossfit | 200 | 0.501387 | +0.001387 [-0.003845,+0.006619] | 0.037454 | 0.935 | True |
| figure2 | nonlinear | controlled | tree_leaf_20 | equal_train_reuse | 200 | 0.395209 | -0.104791 [-0.108984,-0.100597] | 0.108999 | 0.080 | False |
| figure2 | nonlinear | controlled | tree_leaf_20 | full_no_split | 200 | 0.389006 | -0.110994 [-0.115041,-0.106947] | 0.114708 | 0.045 | False |
| figure2 | nonlinear | controlled | tree_leaf_20 | nonorthogonal_crossfit | 200 | 0.525835 | +0.025835 [+0.020754,+0.030916] | 0.044593 | 0.730 | False |
| figure2 | nonlinear | controlled | tree_leaf_5 | crossfit | 200 | 0.504575 | +0.004575 [-0.001557,+0.010706] | 0.044101 | 0.930 | False |
| figure2 | nonlinear | controlled | tree_leaf_5 | equal_train_reuse | 200 | 0.166813 | -0.333187 [-0.335996,-0.330377] | 0.333792 | 0.000 | False |
| figure2 | nonlinear | controlled | tree_leaf_5 | full_no_split | 200 | 0.162471 | -0.337529 [-0.340088,-0.334971] | 0.338025 | 0.000 | False |
| figure2 | nonlinear | controlled | tree_leaf_5 | nonorthogonal_crossfit | 200 | 0.523172 | +0.023172 [+0.017733,+0.028612] | 0.045287 | 0.800 | False |
| figure2 | nonlinear | ordinary | lasso | crossfit | 200 | 0.510369 | +0.010369 [+0.005714,+0.015024] | 0.034879 | 0.875 | False |
| figure2 | nonlinear | ordinary | lasso | equal_train_reuse | 200 | 0.509749 | +0.009749 [+0.005111,+0.014386] | 0.034575 | 0.860 | False |
| figure2 | nonlinear | ordinary | lasso | full_no_split | 200 | 0.509943 | +0.009943 [+0.005335,+0.014551] | 0.034431 | 0.875 | False |
| figure2 | nonlinear | ordinary | lasso | nonorthogonal_crossfit | 200 | 0.533297 | +0.033297 [+0.028614,+0.037979] | 0.047230 | 0.580 | False |
| figure2 | nonlinear | ordinary | rf | crossfit | 200 | 0.509273 | +0.009273 [+0.004530,+0.014016] | 0.035174 | 0.900 | False |
| figure2 | nonlinear | ordinary | rf | equal_train_reuse | 200 | 0.508686 | +0.008686 [+0.004124,+0.013248] | 0.033769 | 0.885 | False |
| figure2 | nonlinear | ordinary | rf | full_no_split | 200 | 0.505761 | +0.005761 [+0.001169,+0.010353] | 0.033350 | 0.890 | False |
| figure2 | nonlinear | ordinary | rf | nonorthogonal_crossfit | 200 | 0.530188 | +0.030188 [+0.025645,+0.034731] | 0.044355 | 0.595 | False |
| figure2 | nonlinear | ordinary | xgb | crossfit | 200 | 0.513470 | +0.013470 [+0.008812,+0.018128] | 0.035943 | 0.865 | False |
| figure2 | nonlinear | ordinary | xgb | equal_train_reuse | 200 | 0.512906 | +0.012906 [+0.008337,+0.017476] | 0.035145 | 0.780 | False |
| figure2 | nonlinear | ordinary | xgb | full_no_split | 200 | 0.510570 | +0.010570 [+0.006105,+0.015034] | 0.033642 | 0.820 | False |
| figure2 | nonlinear | ordinary | xgb | nonorthogonal_crossfit | 200 | 0.537773 | +0.037773 [+0.033465,+0.042081] | 0.048750 | 0.525 | False |

Replications: 1000; estimates: 24600; failures: 0; warnings: 0.
All paired contrasts and weighted decompositions remain in JSON, including adverse outcomes.
No formal outcome is used for learner/parameter selection. Historical runs are unchanged.
