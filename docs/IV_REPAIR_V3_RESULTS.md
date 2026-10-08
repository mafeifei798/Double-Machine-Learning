> Repository presentation copy: an offline raw-archive link points to RAW_EVIDENCE.md. Original reports and execution identifiers are preserved separately; scientific results are unchanged.

# IV-only repair v3：修复后的独立结果

## 结论

修复和独立服务器重跑已经完成。正确指定的 Series OLS treatment channel
加真实回归树，恢复了“样本复用产生明显过拟合偏差、Cross-fitting 接近真值”
的受控机制；没有人为注入 outcome residual，没有 oracle 参与训练。
普通 RF/XGBoost/Lasso 两条 nuisance 都由 ML 学习的结果仍有残余偏差，
不能写成“所有 DML / RF / Cross-fitting 均无偏”。

四个原结论应有条件地表述：错配估计有偏；正确结构 nuisance 的 IV-type
DML 近似居中；真实过拟合学习器的 No-split 有偏；相同学习规则的
Cross-fitting 在正确指定 treatment channel 下近似居中。不能把这些
受控结论推广到任意普通 nuisance learner 或任意有限样本。

## 实际修改

- n=1000、p=20、theta0=0.5、两折、原四个 DGP 均保留。
- 受控组用训练内部 joint OLS pilot；学习 ell、m 后正确构造
  g=ell_hat-pilot*m_hat。最终报告的 score 始终是
  (D-m_hat)*(Y-theta*D-g_hat)，不是把 ell 原封不动放入 g。
- 普通组直接学习 Y-pilot*D 的结构目标及 D 的条件均值，使用
  training-only 的全 20 X 与 sigmoid(X) rich-series joint OLS pilot。
  这是明确披露的新 pilot，并非同一 RF/XGB/Lasso family 的 pilot。
- 普通超参数仅按开发数据的 observed prediction loss 选择，一轮后冻结。
  开发阶段不记录 theta/bias/coverage，不使用真实 nuisance 或真实 theta。
- 同时报告 full-sample No-split 和 equal-training reuse。后者与 Cross-fitting
  使用完全相同的 A/B 模型银行、目标和 pilot，各模型训练 500 行，估计都用
  1000 行。没有把这两个 No-split 对照混为一谈。

## Figure 1：错配与正确 IV-type DML

Series OLS 是明确正确指定的受控模型，不是 ordinary RF 的结果。

| DGP | 错配估计均值 | IV-type Cross-fitting 均值 | IV 平均偏差的 MC95 区间 | IV 覆盖率 |
|---|---:|---:|---|---:|
| linear | 0.248269 | 0.501876 | [-0.001484, +0.005236] | 0.963 |
| nonlinear | 0.271405 | 0.502618 | [-0.001261, +0.006497] | 0.937 |

错配是 target/score 不一致的诊断，不能直接称为“正确 naive ML 的
regularization bias”。正确 nonorthogonal structural-g 对照亦完整保留。

## Figure 2：Series OLS m + 真实 Decision Tree outcome

下面是 full-sample No-split 与 Cross-fitting 的 Monte Carlo 均值；真值 .5。

| min_samples_leaf | linear No-split | linear Cross-fit | nonlinear No-split | nonlinear Cross-fit |
|---:|---:|---:|---:|---:|
| 2 | 0.032016 | 0.498431 | 0.031545 | 0.497168 |
| 5 | 0.165853 | 0.499599 | 0.162471 | 0.504575 |
| 10 | 0.295149 | 0.501265 | 0.293763 | 0.499105 |
| 20 | 0.390884 | 0.500677 | 0.389006 | 0.501387 |

八个 No-split 单元都明显向下偏。八个 Cross-fit 单元平均偏差在
[-0.002832,+0.004575]，覆盖率在 .925–.955。
等训练量复用均值亦在 .032556–.397526，不能将该受控优势仅归因于
两组训练样本数不同。所有配对对比及区间均在完整结果中。

预先固定 practical-zero margin 为 ±0.01，要求整个 MC95 mean-bias
区间落在该范围内。八个树单元中七个支持此标准；nonlinear leaf5 的区间
[-0.001557,+0.010706] 略超上界，因此标为 **未充分确认**，不是“精确无偏”。
没有为它扩展 replications、换种子、改 margin 或重选参数。

包括四个 Series OLS 与两个混合 RF 单元后，14 个受控 Cross-fit 单元中
13 个支持上述 practical-zero 标准。Series OLS 的 No-split 也近似居中，
所以不能写成“所有 No-split 必定有偏”。

## RF outcome + Series OLS m：单独标注的混合敏感性

| DGP | full No-split | equal-training reuse | Cross-fit | Cross-fit bias MC95 | Cross-fit coverage |
|---|---:|---:|---:|---|---:|
| linear | 0.210110 | 0.211212 | 0.500699 | [-0.004369,+0.005767] | 0.945 |
| nonlinear | 0.211046 | 0.211860 | 0.502051 | [-0.002810,+0.006912] | 0.945 |

这是真实 RF outcome 学习器，没有 artificial pressure，但 treatment nuisance
是正确指定的 Series OLS。不得简写为“RF 学 g、m 两条 channel 都无偏”。

## 普通 RF/XGB/Lasso 双通道：仍然不满足零偏差标准

全部参数由预测验证选择，全部单元保留。以下是 Cross-fit 的平均偏差。

| Design / DGP | RF | XGBoost | Lasso |
|---|---:|---:|---:|
| Figure1 / linear | +0.010797 | +0.015744 | +0.007490 |
| Figure1 / nonlinear | +0.006280 | +0.010625 | +0.009322 |
| Figure2 / linear | +0.023174 | +0.024746 | +0.006301 |
| Figure2 / nonlinear | +0.009273 | +0.013470 | +0.010369 |

这 12 个单元的 MC95 mean-bias 区间均未完全落入 ±0.01，且都排除零。
不能称为“无偏”。例如 Figure2 linear RF：full No-split 均值 .509681，
equal-training reuse .511853，Cross-fit .523174；不能隐瞒 Cross-fit
在该背景下偏差更大的结果。

普通与受控是两个不同问题：预测选参不保证二阶 nuisance 误差乘积足够小；
正确指定的快收敛 m channel 可以将机制分离出来。受控优势不替代普通结果。

## 运行和核验

只在 `rna-server` 的 `/data2/lzx/dhy/dml_iv_repair_v3_20261002` 运行。
共 1000 次数据复制、24600 个估计、106 个 summary cells，formal 数值失败
0、formal warnings 0。Figure1 每个 DGP 300 次，Figure2 每个 DGP 200 次。
固定开发选择 96 个任务，正式样本与开发/历史 seed 分离。

- 27 项服务器单元测试通过，包括 held-out outcome 扰动隔离与配对。
- 实际 DoubleML 0.10.1 的 20 项数值核对通过，容差 1e-12：
  旧 RF/XGB pooled-pilot 拟合算法与新外部预测 IV score 是两类不同核对。
  包兼容性检查另记录 48 条 sklearn 参数重命名 FutureWarnings，非数值失败。
- 独立核验确认所有 raw receipts、源文件/结果 hash、预测参数选择、模型银行
  配对、106 个均值/MC 区间、RMSE/MAE/coverage、80 个 paired contrasts，
  以及逐复制加权 score 分解。
- 14 张图片已视觉检查；额外只读布局检查全部通过，原图 hash 不变。
- 历史 formal 文件和环境未修改。没有启动第二轮正式调参或重复模拟。

Formal identity SHA-256：
`72afd3f29c4795b6ce4727fe613c03f164aff8cda35888b4bb32425525affb3a`。
下载证据包 SHA-256：
`949f9271e3f67acc058b08347e8808717535487bed277b157d568b58ff090d83`。

## 文件和全部图片

- [冻结协议](IV_REPAIR_V3_PROTOCOL.md)
- [全部 106 单元结果](../output/iv_repair_v3_server_20261002/output/formal/analysis/RESULTS.md)
- [原始 summary / 完整区间与分解](../output/iv_repair_v3_server_20261002/output/formal/analysis/summary.json)
- [全部配对对比](../output/iv_repair_v3_server_20261002/output/formal/analysis/paired_contrasts.json)
- [独立核验](../output/iv_repair_v3_server_20261002/INDEPENDENT_VERIFICATION.json)
- [独立图片布局核验](../output/iv_repair_v3_server_20261002/FIGURE_QA.json)
- [完整冻结证据包](RAW_EVIDENCE.md)

| 图片类别 | Linear | Nonlinear |
|---|---|---|
| 受控 Figure1 | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/controlled_figure1_linear.png) | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/controlled_figure1_nonlinear.png) |
| 受控 Figure2 四种树 | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/controlled_figure2_linear.png) | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/controlled_figure2_nonlinear.png) |
| 混合 RF outcome / Series m | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/mixed_rf_series_linear.png) | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/mixed_rf_series_nonlinear.png) |
| 普通 Figure1 | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/ordinary_figure1_linear.png) | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/ordinary_figure1_nonlinear.png) |
| 普通 Figure2 | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/ordinary_figure2_linear.png) | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/ordinary_figure2_nonlinear.png) |
| 普通等训练量对照 | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/ordinary_equal_training_linear.png) | [图](../output/iv_repair_v3_server_20261002/output/formal/analysis/ordinary_equal_training_nonlinear.png) |

[受控全部 bias 区间图](../output/iv_repair_v3_server_20261002/output/formal/analysis/bias_controlled.png)；
[普通全部 bias 区间图](../output/iv_repair_v3_server_20261002/output/formal/analysis/bias_ordinary.png)。
