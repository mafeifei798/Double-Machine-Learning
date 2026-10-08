# 等训练量、同一模型的 IV-type 对照结果

## 完成状态与公平性

2026-10-02（美国西海岸日期）在用户指定的 `rna-server` 完成；模拟和拟合
没有在本地运行。新旧实现的 25 项服务器测试通过，独立开发种子的端到端
预检通过后，固定方案正式运行。1,000 次重复产生 6,000 个估计，零失败、
零警告；模拟耗时约 215 秒，独立核验通过，五张图片均已检查。

两组使用同一批 n=1,000 数据、同一 A/B 划分、原来的 RF/XGBoost/Lasso
超参数和 IV-type 最终 score。每个模型只用 500 个观测训练；每组最终都用
1,000 个观测估计。每个 learner 只有一套训练好的 A/B 模型银行，两组直接
复用它，连训练半样本内的 pilot 和生成的 g 训练目标都相同：

- 等训练量复用组：A 模型预测 A，B 模型预测 B。
- 等训练量 Cross-fitting：A 模型预测 B，B 模型预测 A。

无单边或双边人工加压，无 Series OLS 替换，无 oracle 参与训练。
核验重新构造了预测归属和训练成员索引，并确认数据种子、fold 哈希与
上一轮正式数据一致。这是看到前一轮结果后的配对机制诊断，不是独立确认。

复用组不是普通全样本 No-split。本轮两组共用 training-half-only pilot；
它与上一轮标准方法比较的 official pooled pilot 不同。因此不能把本轮
Cross-fitting 结果偷偷替换到上一轮报告里。科学方案见
[EQUAL_TRAINING_IV_V1_PROTOCOL.md](EQUAL_TRAINING_IV_V1_PROTOCOL.md)。

## 主要结果

下表是每个 design 内六个 learner/DGP 单元的等权绝对 Monte Carlo 平均偏差，
不是 MAE。最后一列为 Cross-fitting 减复用组的配对 block-bootstrap 95% 区间。

| 背景 | 等训练量复用组 | 等训练量 Cross-fitting | 偏差差值 [95%区间] |
|---|---:|---:|---:|
| Figure 1 的 DGP 设置 | 0.010610 | 0.012959 | +0.002349 [0.001531, 0.003279] |
| Figure 2 的 DGP 设置 | 0.018928 | 0.021992 | +0.003065 [0.002038, 0.004139] |

RF 和 XGBoost 在本轮各背景下的 Cross-fitting 平均偏差仍大于复用组。
RF 的配对偏差差值区间均在零以上；XGBoost Figure 2 linear 的差值区间
包含零，不能把全部 XGBoost 差异都说成显著。Lasso 的平均偏差点估计均
略微改善，但配对偏差差值区间均包含零，改善证据不明确。
Cross-fitting 的 RMSE 点估计在本轮各单元均更大；每个配对 MSE 区间都
完整报告，不将所有点估计恶化都说成显著。

例如 Figure 2 linear 的 RF（真值 0.5）：

| 指标 | 复用组 | Cross-fitting |
|---|---:|---:|
| 平均估计 | 0.551103 | 0.558781 |
| 平均偏差 | +0.051103 | +0.058781 |
| RMSE | 0.059947 | 0.066591 |
| 名义 95% 区间覆盖率 | 23.5% | 36.5% |

Cross-fitting 的偏差差值为 +0.007678，配对 MC95 区间
[+0.005849,+0.009506]。覆盖率有所改善，但 36.5% 仍远低于 95%，不能
称为可靠推断。RF/XGBoost 的覆盖率在本轮均有所改善，这也说明不能仅
根据平均偏差把所有性能维度混成一句“完全更差”。Lasso 覆盖率变化混合。
复用组的常规 sandwich SE 不假定有效，覆盖率只作为描述性诊断。

## 对上轮解释的修正

不能再仅用“Cross-fitting 每次少了一半训练数据”解释上一轮的不利结果。
在本轮同训练量、同模型银行下，不利偏差仍然存在。但本轮换用了训练半
样本 pilot，不能用两轮数值差直接量化训练样本量贡献。

保存的精确 IV score 分解也表明，排除样本复用不等于消除所有 nuisance
学习误差。设 a=ghat-g0、b=mhat-m0、V=D-m0、U=Y-theta0*D-g0，则真值处：

`mean(score) = mean(V*U) + mean(-V*a-b*U) + mean(b*a)`。

RF Cross-fitting 四个背景的一阶项 MC95 区间均包含零，而二阶项均显著
为正。例如 Figure 2 linear 的一阶项均值为 0.002481，区间
[-0.000190,0.005151]；二阶项均值为 0.088404，区间
[0.086649,0.090160]。这些是 score moment，不是直接等于 theta 偏差；
参数估计还涉及每次重复的 Jacobian。该分解提供残余学习误差的描述性
证据，不宣称已独立识别某一个机制导致所有性能差异。

因此本轮不能恢复“RF/XGB Cross-fitting 无偏”的旧强结论，也没有通过
继续调参把结果改成预期。上一轮正确正交 score 相对非正交 ML 的去偏
结果仍是另一项比较，不被本轮结果否定。

## 完整输出

- [全部单元和配对结果](../output/equal_training_iv_v1_server_20261002/output/equal_training_iv_v1_server_formal/analysis/RESULTS.md)
- [偏差与区间图](../output/equal_training_iv_v1_server_20261002/output/equal_training_iv_v1_server_formal/analysis/bias_comparison.png)
- 原版 3×2 直方图：
  [Figure 1 linear](../output/equal_training_iv_v1_server_20261002/output/equal_training_iv_v1_server_formal/analysis/figure1_linear.png)、
  [Figure 1 nonlinear](../output/equal_training_iv_v1_server_20261002/output/equal_training_iv_v1_server_formal/analysis/figure1_nonlinear.png)、
  [Figure 2 linear](../output/equal_training_iv_v1_server_20261002/output/equal_training_iv_v1_server_formal/analysis/figure2_linear.png)、
  [Figure 2 nonlinear](../output/equal_training_iv_v1_server_20261002/output/equal_training_iv_v1_server_formal/analysis/figure2_nonlinear.png)。
- [独立核验收据](../output/equal_training_iv_v1_server_20261002/SERVER_INDEPENDENT_VERIFICATION.json)

服务器独立目录：`/data2/lzx/dhy/dml_equal_training_v1_20261002`。
源码包 SHA-256：`87dee88642ca73926f6378ffbf0e59f288c66e03992e9e313bd36e28c4c3e71c`。
冻结身份：`7fed12c33a3df94bab223b1cba86e64988005ad81f9cc2b96f75ad6fb52b4701`。
已下载证据包 `output/equal_training_iv_v1_server_20261002/evidence.tgz`，SHA-256：
`801c7454c2cf59d8aed29403230b2201f91370ed354e3e047adb9be0e422fb10`。
旧结果均保留，不重启或混入被排除的本地模拟。
