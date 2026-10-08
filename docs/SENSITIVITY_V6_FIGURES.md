> Repository presentation copy: workspace links are normalized; offline raw-evidence links point to RAW_EVIDENCE.md. Original reports remain preserved separately. Scientific results and original execution identifiers are not edited.

# v6 全部图片

24 张敏感性图使用独立排版修正版，6 张直方图为原正式输出。两组都经 SHA256 核验和逐图目视检查；排版脚本另外核对了 540 条系列、1980 个绘图点。原始 30 张图完整保留在证据包中。所有图包含三个 learner 和两个 DGP，不删除尾部、不把分布重新居中。

[结果解释](SENSITIVITY_V6_RESULTS.md) · [全部 600 个统计单元](SENSITIVITY_V6_ALL_CELLS.md)

## baseline / figure1 / sample_fold

n=300/1000/3000，K=2/5；K=5 同时增加单个 nuisance 模型的训练比例。Full No-split 的两条 K 线是精确重复。

### BIAS

![baseline figure1 sample_fold bias](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure1_sample_fold_bias.png)

### RMSE

![baseline figure1 sample_fold rmse](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure1_sample_fold_rmse.png)

### COVERAGE

![baseline figure1 sample_fold coverage](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure1_sample_fold_coverage.png)

## baseline / figure1 / parameters

n=1000，K=2；out weak/strong 为结果模型的较弱/较强约束，m weak/strong 为处理模型的较弱/较强约束。Figure 2 RF out weak 与 base 精确相同。

### BIAS

![baseline figure1 parameters bias](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure1_parameters_bias.png)

### RMSE

![baseline figure1 parameters rmse](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure1_parameters_rmse.png)

### COVERAGE

![baseline figure1 parameters coverage](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure1_parameters_coverage.png)

## baseline / figure2 / sample_fold

n=300/1000/3000，K=2/5；K=5 同时增加单个 nuisance 模型的训练比例。Full No-split 的两条 K 线是精确重复。

### BIAS

![baseline figure2 sample_fold bias](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure2_sample_fold_bias.png)

### RMSE

![baseline figure2 sample_fold rmse](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure2_sample_fold_rmse.png)

### COVERAGE

![baseline figure2 sample_fold coverage](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure2_sample_fold_coverage.png)

## baseline / figure2 / parameters

n=1000，K=2；out weak/strong 为结果模型的较弱/较强约束，m weak/strong 为处理模型的较弱/较强约束。Figure 2 RF out weak 与 base 精确相同。

### BIAS

![baseline figure2 parameters bias](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure2_parameters_bias.png)

### RMSE

![baseline figure2 parameters rmse](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure2_parameters_rmse.png)

### COVERAGE

![baseline figure2 parameters coverage](../output/sensitivity_v6_server_20261002/evidence/presentation/baseline_figure2_parameters_coverage.png)

## moderate / figure1 / sample_fold

n=300/1000/3000，K=2/5；K=5 同时增加单个 nuisance 模型的训练比例。Full No-split 的两条 K 线是精确重复。

### BIAS

![moderate figure1 sample_fold bias](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure1_sample_fold_bias.png)

### RMSE

![moderate figure1 sample_fold rmse](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure1_sample_fold_rmse.png)

### COVERAGE

![moderate figure1 sample_fold coverage](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure1_sample_fold_coverage.png)

## moderate / figure1 / parameters

n=1000，K=2；out weak/strong 为结果模型的较弱/较强约束，m weak/strong 为处理模型的较弱/较强约束。Figure 2 RF out weak 与 base 精确相同。

### BIAS

![moderate figure1 parameters bias](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure1_parameters_bias.png)

### RMSE

![moderate figure1 parameters rmse](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure1_parameters_rmse.png)

### COVERAGE

![moderate figure1 parameters coverage](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure1_parameters_coverage.png)

## moderate / figure2 / sample_fold

n=300/1000/3000，K=2/5；K=5 同时增加单个 nuisance 模型的训练比例。Full No-split 的两条 K 线是精确重复。

### BIAS

![moderate figure2 sample_fold bias](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure2_sample_fold_bias.png)

### RMSE

![moderate figure2 sample_fold rmse](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure2_sample_fold_rmse.png)

### COVERAGE

![moderate figure2 sample_fold coverage](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure2_sample_fold_coverage.png)

## moderate / figure2 / parameters

n=1000，K=2；out weak/strong 为结果模型的较弱/较强约束，m weak/strong 为处理模型的较弱/较强约束。Figure 2 RF out weak 与 base 精确相同。

### BIAS

![moderate figure2 parameters bias](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure2_parameters_bias.png)

### RMSE

![moderate figure2 parameters rmse](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure2_parameters_rmse.png)

### COVERAGE

![moderate figure2 parameters coverage](../output/sensitivity_v6_server_20261002/evidence/presentation/moderate_figure2_parameters_coverage.png)

## 较难 DGP 的直方图对照

固定 n=1000、K=2、base、R=200；三行是 Lasso/RF/XGBoost。横轴是 (estimate−0.5)/estimated SE，红线为 N(0,1)，不是原始估计量密度。每行左右共享横轴和分箱，保留全尾部；RF/XGB 的巨大 No-split 偏移会压缩右侧的细节，请结合标题中的偏差区间和 Coverage 表判断。

### figure1 / linear / naive_ml

![figure1 linear naive_ml](../output/sensitivity_v6_server_20261002/evidence/output/formal/analysis/moderate_figure1_linear_naive_ml_hist.png)

### figure1 / nonlinear / naive_ml

![figure1 nonlinear naive_ml](../output/sensitivity_v6_server_20261002/evidence/output/formal/analysis/moderate_figure1_nonlinear_naive_ml_hist.png)

### figure2 / linear / full_no_split

![figure2 linear full_no_split](../output/sensitivity_v6_server_20261002/evidence/output/formal/analysis/moderate_figure2_linear_full_no_split_hist.png)

### figure2 / linear / equal_train_reuse

![figure2 linear equal_train_reuse](../output/sensitivity_v6_server_20261002/evidence/output/formal/analysis/moderate_figure2_linear_equal_train_reuse_hist.png)

### figure2 / nonlinear / full_no_split

![figure2 nonlinear full_no_split](../output/sensitivity_v6_server_20261002/evidence/output/formal/analysis/moderate_figure2_nonlinear_full_no_split_hist.png)

### figure2 / nonlinear / equal_train_reuse

![figure2 nonlinear equal_train_reuse](../output/sensitivity_v6_server_20261002/evidence/output/formal/analysis/moderate_figure2_nonlinear_equal_train_reuse_hist.png)


