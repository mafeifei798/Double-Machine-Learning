> Repository presentation copy: workspace links are normalized; offline raw-evidence links point to RAW_EVIDENCE.md. Original reports remain preserved separately. Scientific results and original execution identifiers are not edited.

# IV-type DML：补完敏感性分析 v6

完成日期：2026-10-03（America/Los_Angeles）。仅 simulation；全部拟合、测试、统计分析和绘图均在已授权 rna-server 完成。本地只编辑代码、读取证据、核对文件和目视检查图片。没有重新运行或覆盖任何旧实验，没有改简历或生成 PDF。

## 结论

在预先冻结的受控设定下，IV-type DML / Cross-fitting 的去偏机制在样本量、折数及参数变化后仍然存在；但结果不支持“所有 learner、所有有限样本设定都严格无偏”或“所有情形的 MAE/MSE 均显著改善”。RF 在较难的线性 DGP 下仍有可检测的残余偏差及部分欠覆盖；Lasso 的偏差收益不总能抵消方差成本。

[全部 30 张图片](SENSITIVITY_V6_FIGURES.md) · [全部 600 个统计单元](SENSITIVITY_V6_ALL_CELLS.md) · [全部统计与配对检验 JSON](../output/sensitivity_v6_server_20261002/evidence/output/formal/analysis/SUMMARY.json) · [完整证据包](RAW_EVIDENCE.md)

## 补完的范围与公平性

- 三类实际 learner：Lasso、Random Forest、XGBoost，同时学习两个 nuisance；最终只使用结构 IV-type score，没有把 Series OLS、Decision Tree 或 oracle 偷换进这轮三模型比较。
- n=300/1000/3000 × K=2/5 的完整 6 组合，采用 baseline 模型参数。
- n=1000、K=2 下增加 outcome_weak/outcome_strong/treatment_weak/treatment_strong，逐次只改变一个模型角色。不是所有因素与所有超参数的全因子交互实验。
- 原设定 baseline 与 v5 的 moderately harder DGP 同时运行。较难设定增加有效变量和有界交互，噪声与真实效应不变；信号方差会变化，不是固定 SNR 的难度单因素实验。
- 新独立基础种子 2026100861，R=200/单元；同次重复内共享 X/U/V、划分及模型随机状态，n 使用共同 n=3000 抽样的前缀。没有按最终偏差选参数、选种子、挑结果或延长重复次数。
- Figure 1 的 naive raw-D ratio 与 IV DML 使用同一个 held-out 结构 g 预测。Figure 2 同时保留 full No-split 与 equal-training-size same-bank reuse 对照。
- 模型库 j 在第 j 折之外训练：Cross-fit 预测第 j 折使用模型 j；reuse 使用模型 (j+1) mod K。因此二者训练规模、模型库、pilot 和目标相同，每个观测各使用一个模型预测。没有只对 No-split 注入额外响应压力。
- K=5 的训练比例为 0.8n，K=2 为 0.5n；K 对比不能解读为训练比例固定的纯折数效应。

本轮仍明确使用处理变量候选输入 [x0,x2,x4,x6] 和已知正向/单调结构先验，baseline 也保留两项无效候选变量；pilot 为训练数据内的 joint Lasso，alpha=.001。Lasso 的固定 raw/sigmoid 字典不包含新增交互。这是受控机制研究，不是 unrestricted variable discovery 或任意模型错配下的普遍保证。真正的 nuisance 只用于事后误差诊断，不进入拟合。

完整设定：[冻结协议](SENSITIVITY_V6_PROTOCOL.md)、[参数配置](../configs/sensitivity_v6.json)、[基础 DGP/模型配置](../configs/harder_v5.json)。

### 参数范围（weak / base / strong 指约束强度）

| 角色 | RF | XGBoost | Lasso |
| --- | --- | --- | --- |
| Figure 1 g | leaf=1 / 2 / 5 | depth=4 / 3 / 2 | 结果惩罚乘数=.5 / 1 / 2 |
| Figure 2 ell | leaf=1 / 1 / 5 | depth=7 / 6 / 4 | 结果惩罚乘数=.5 / 1 / 2 |
| m | leaf=5 / 10 / 20 | L2 lambda=2.5 / 5 / 10 | alpha=.0005 / .001 / .002 |

其他设置与 pilot 固定。Figure 2 RF 的 outcome_weak 因 base 已是 leaf=1，属于精确重复。Full No-split 在相同 n/preset 下的 K2/K5 标签也是精确重复。都保留并核对，不能当作独立支持证据。

## 样本量结果

下表仅 baseline 参数，合并展示两个难度、两个设计、两个 DGP、三个模型、两个 K：每个 n 共 48 个 Cross-fit 标签。它们通过共同随机数配对，不是 48 个独立实验。

| n | Cross-fit 平均偏差范围 | Coverage 范围 | 整段 MC95 落入 ±0.01 的标签数 |
| ---: | --- | --- | ---: |
| 300 | −0.01685 ～ +0.03050 | 88.0% ～ 98.0% | 1/48 |
| 1000 | −0.01209 ～ +0.01275 | 89.0% ～ 97.0% | 33/48 |
| 3000 | −0.00453 ～ +0.00845 | 90.5% ～ 96.0% | 46/48 |

Practical centering 要求平均偏差的整段单元级 Student-t MC95 都落入 ±0.01；未通过不等于证明偏差很大。尤其 n=300 的模拟均值区间较宽。不能据此宣称其余 47 个组合都“有严重偏差”。这是个别单元区间，不是同时置信区间。

n300→1000、n1000→3000 的全部 96 个 Cross-fit 配对 MAE/MSE 差值都显著下降（按预定 factor family 的 Holm 校正）。这支持本范围内样本增大改善总体精度，不代表每个绝对均值偏差严格单调。

n=3000 仍未通过 practical-zero 的两个标签均为 moderate / Figure1 / linear / RF：K2 bias=+0.00821，MC95=[+0.00550,+0.01093]；K5 bias=+0.00845，MC95=[+0.00579,+0.01111]。两者点偏差已小于 .01，但区间略越界且偏差仍可检测；覆盖率分别 91.0% 和 91.5%。

## 2-fold 与 5-fold

没有证据支持“5-fold 普遍优于 2-fold”。在 72 个 baseline-parameter Cross-fit K 对比中，只有两个配对 MAE/MSE 改善经全 factor family 的 Holm 校正成立：baseline 与 moderate 的 Figure2 / linear / Lasso / n300，MAE 差分别 −0.00988 和 −0.01127。其余多数差异未达到该校正阈值；这不是证明两种折数等价，也不保证偏差或覆盖率不变。

## 参数敏感性

去偏优势没有因测试范围内的参数变化整体消失，但不能称为“对所有参数不敏感”。例如 moderate / Figure1 / linear / RF / n1000 / K2：

| 处理模型设定 | DML mean bias | Coverage |
| --- | ---: | ---: |
| m weak（RF leaf=5） | +0.01042 | 95.0% |
| base（RF leaf=10） | +0.01275 | 93.5% |
| m strong（RF leaf=20） | +0.01715 | 91.0% |

这里“更强”表示更强约束、较大的叶节点，不是模型更好。较强平滑可降低过拟合却增加 nuisance 残差相关性，因此不必改善因果偏差。此解释与保存的二阶误差分解相容，但不构成对任意 RF 的定理，也不是重新调参后的新最优结果。

最差 Cross-fit 覆盖率为 87.5%，出现在 moderate / Figure2 / linear / RF / n1000 / K2 / outcome_strong：bias=+0.01514，MC95=[+0.00979,+0.02048]，coverage Wilson95=[82.2%,91.4%]。均值标准误约 .03084，重复间 SD 约 .03834。保留这一欠覆盖结果，不写成 uniformly valid inference。

## 两个去偏对比的完整检验

预先定义 360 个方法标签对比，MAE 和 MSE 分别进行配对单侧检验，并各自对完整 360 项做 Holm 校正。下列计数是固定网格清单，不是独立实验成功率；包含上述精确重复，不增加 R。

| 方法对比 | 对比标签数 | MAE 显著下降 | MSE 显著下降 |
| --- | ---: | ---: | ---: |
| Figure1 naive → DML | 120 | 91 | 94 |
| Figure2 full No-split → Cross-fit | 120 | 82 | 83 |
| Figure2 same-bank reuse → Cross-fit | 120 | 104 | 109 |

各对比的绝对 Monte Carlo 平均偏差点估计均下降；这不是经过校正的绝对偏差改善检验，也不意味着 MAE/MSE 每项都改善。same-bank reuse 对照的每项 MAE/MSE 点估计都下降，但部分 Lasso 优势尚不显著。full No-split 对比有 13 项 MAE、11 项 MSE 点估计反而更大，均集中在 Lasso；减小偏差与减小总误差不能混为一谈。

RF 全部方法对比的 MAE/MSE 改善通过预定校正；XGB 大部分通过；Lasso 的小偏差场景更容易受方差成本和多重校正影响。没有对负结果追加重复次数来获取显著性。

所有未显著、方向不利及良好结果都在 JSON 和 600 单元表中。960 个因素对比另按 fold / sample_size / preset / difficulty 及各指标分别 Holm 校正。R=200 的本轮不与 R=1000 的 v4 合并计算。

## 较难设定的独立 n1000、K2、base 直方图锚点

这不是重新选择最好的组合，而是协议预定的锚点。

### Figure1：Naive ML → IV DML

| DGP | Learner | Naive mean bias | DML mean bias | DML MC95 | DML coverage |
| --- | --- | ---: | ---: | --- | ---: |
| linear | LASSO | 0.01250 | 0.00082 | [-0.00362, 0.00525] | 95.0% |
| linear | RF | 0.05396 | 0.01275 | [0.00836, 0.01714] | 93.5% |
| linear | XGB | 0.04638 | 0.00496 | [0.00042, 0.00951] | 94.5% |
| nonlinear | LASSO | 0.03141 | -0.00116 | [-0.00584, 0.00352] | 93.5% |
| nonlinear | RF | 0.04050 | -0.00136 | [-0.00598, 0.00326] | 94.5% |
| nonlinear | XGB | 0.03458 | -0.00220 | [-0.00700, 0.00261] | 94.5% |

### Figure2：No-split / Same-bank reuse → Cross-fit

| DGP | Learner | Full No-split bias | Same-bank reuse bias | Cross-fit bias | Cross-fit MC95 | Coverage |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| linear | LASSO | -0.01301 | -0.03119 | 0.00501 | [-0.00021, 0.01023] | 91.5% |
| linear | RF | -0.27035 | -0.25758 | 0.01151 | [0.00614, 0.01687] | 89.0% |
| linear | XGB | -0.46249 | -0.46722 | 0.00193 | [-0.00380, 0.00766] | 90.5% |
| nonlinear | LASSO | -0.01797 | -0.03535 | -0.00103 | [-0.00565, 0.00360] | 95.0% |
| nonlinear | RF | -0.29405 | -0.28551 | -0.00732 | [-0.01184, -0.00281] | 96.0% |
| nonlinear | XGB | -0.47678 | -0.47937 | -0.01209 | [-0.01691, -0.00726] | 94.5% |

例如较难线性 Lasso 的 full No-split bias 从 −.01301 降到 Cross-fit +.00501，但 MAE 从 .02800 增到 .03051，RMSE 从 .03588 增到 .03766。same-bank reuse MAE=.03702、RMSE=.04557，Cross-fit 则均下降。明确区分这两个对照，不能为了讲去偏而删除 full No-split。

较难非线性 XGB Cross-fit 仍有 bias=−.01209，MC95 不含零且超出 practical-zero 范围；其 .945 覆盖率不错，但不是精确无偏。直方图看起来接近零不能代替数值判断。

## 与已完成工作的关系

v3 Series OLS + Decision Tree 的受控机制、v4 原设定的三模型正式主实验、v5 难度扩展和本轮 v6 敏感性分析都已完成，分开保留。v6 补上了当前 IV-only 三模型主线之前缺少的样本量 × 折数及角色专属参数分析；旧 partialling-out 敏感性不算作此次 IV-type 证据。

v6 baseline 保留四个处理候选输入，v4 是两候选主实验；同时随机样本和 R 不同，所以不能把 v4 与 v6 的数值差直接解释成性能退化或单纯难度效应。moderate 与 baseline 的难度差必须用本轮共享背景的配对比较。

可以收尾的结论是：在披露的结构先验和测试范围内，正交化与样本外预测提供明显去偏，并对若干设定变化保持优势；小样本、nuisance 二阶误差和推断覆盖率仍有限制。不是“保证四个口号在任意设定下都成立”。

## 完成与可复核证据

- 单次服务器任务 PID 1126430；完整退出 returncode=0，日志以 COMPLETE_VERIFIED_SENSITIVITY 结束。8000/8000 原始任务、120000 估计、600 单元、360 方法对比、960 因素对比。
- 正式软件预检：29 tests pass。实际 DoubleML external-prediction coefficient/SE parity：48 checks pass，容差 1e−12；DoubleML 有 192 条包弃用 FutureWarnings，单独保存，未混入科学拟合警告。
- [独立核验](../output/sensitivity_v6_server_20261002/evidence/INDEPENDENT_VERIFICATION.json)：PASS；覆盖 19 个冻结来源、任务集合/哈希、共享数据/模型库/成员关系、9600 个 alias 行、全部汇总/检验/图。正式拟合警告=0、数值失败=0。
- [服务器精确复算](../output/sensitivity_v6_server_20261002/evidence/SOFTWARE_REPLAY_CHECK.json)：4 个跨 n/K/preset/design/DGP 样本、共 60 个估计逐项 exact match，复算重复不加入 MC。
- 本地再次核对 19 个来源、8000 记录、30 原图及 24 排版修正版 SHA256；所有 30 原图和 24 修正版都已目视检查。原图中的标题排版问题通过独立 renderer 修复，没有覆盖主分析或再拟合。
- [排版数据核对](../output/sensitivity_v6_server_20261002/evidence/presentation/PRESENTATION_LEDGER.json)：540 条系列、1980 个点与原 SUMMARY 逐项相同；标题、留白与导出边界是唯一变化。

科学身份 SHA256：
```
8386d43620f5ad6b9df45cc88429245d708057e1a7d29ca90cce4d0bba8a8a5a
```

SUMMARY SHA256：
```
6b5e66c4ec9c8c53c0f5a0c2542c361deb8aad6a24afd17714bedbfd36c05117
```

完整下载证据包 SHA256：
```
3bdde95dfc1d016e0d8b98c151db432475c2dc6c1d7c0582c44ac01b0bee774a
```

服务器路径：/data2/lzx/dhy/dml_sensitivity_v6_20261002。启动目录的 20261002 为 PDT 启动日期，20261003 为完成日期；此前报告保留，不回填到毕设的历史日期。

