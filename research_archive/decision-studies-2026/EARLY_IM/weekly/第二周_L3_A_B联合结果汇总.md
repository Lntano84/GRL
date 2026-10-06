# 第二周 L3 A+B 联合结果汇总

运行日期：2026-08-31  
仓库：`C:\Users\windows\Desktop\GRL-main`

## 配置与统计口径

- 数据集：NetHEPT；IC 扩散，传播概率 `0.01`。
- Oracle：**restricted oracle（top-200-by-degree）**。每一步只在按度数选出的前 200 个节点中寻找真实边际增益最大节点，不是全图 oracle。
- 参数：`steps=10`，诊断 MC=`10`，端到端 marginal-gain MC=`10`，最终 spread MC=`100`，`M={10,20,50,100}`，Random 每个种子 `10` 次 repeat。
- 随机种子：`20260821`、`20260822`、`20260823`、`20260824`、`20260825`。
- 表中数值为跨 5 个种子的 `mean +/- sample std`。Random 先在同一随机种子的 10 个 repeat 内求均值，再跨 5 个种子统计。
- `Top-1`：所选节点是否为该候选池内真实 marginal gain 最大的节点的比例。
- 诊断表的总运行时间：该方法的检索和重排时间，加上每步一次共享真实增益评估时间。端到端表的总运行时间：selection、真实增益评估和最终 spread 评估的合计。单位均为秒。

## 原始输出路径

每个目录均含 `oracle_trajectory_diagnostic.json/.csv`、`end_to_end_sequential.json/.csv`、`end_to_end_summary.csv`、`validation.json`、`config.yaml` 和 `console.log`。

1. Seed `20260821`：`C:\Users\windows\Desktop\GRL-main\outputs\retrieval_reranking\nethept\20260831_152922\retrieval_reranking`
2. Seed `20260822`：`C:\Users\windows\Desktop\GRL-main\outputs\retrieval_reranking\nethept\20260831_170241\retrieval_reranking`
3. Seed `20260823`：`C:\Users\windows\Desktop\GRL-main\outputs\retrieval_reranking\nethept\20260831_170653\retrieval_reranking`
4. Seed `20260824`：`C:\Users\windows\Desktop\GRL-main\outputs\retrieval_reranking\nethept\20260831_171937\retrieval_reranking`
5. Seed `20260825`：`C:\Users\windows\Desktop\GRL-main\outputs\retrieval_reranking\nethept\20260831_172420\retrieval_reranking`

## Oracle-Trajectory Diagnostic

所有方法共享 restricted-oracle 的 seed 轨迹，因此此表用于拆分 retrieval 的 CandidateLoss 与池内 reranking 的 RankingLoss；不代表各方法最终端到端 spread。

| Retriever | Ranker | M | CandidateLoss | RankingLoss | TotalRegret | Top-1 | Total runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Degree | OriginalOrder | 10 | 4.894 +/- 0.743 | 25.970 +/- 1.171 | 30.864 +/- 0.955 | 0.000 +/- 0.000 | 4.684 +/- 0.459 |
| Degree | OriginalOrder | 20 | 3.228 +/- 0.435 | 27.636 +/- 0.980 | 30.864 +/- 0.955 | 0.000 +/- 0.000 | 4.684 +/- 0.459 |
| Degree | OriginalOrder | 50 | 1.244 +/- 0.394 | 29.620 +/- 0.987 | 30.864 +/- 0.955 | 0.000 +/- 0.000 | 4.684 +/- 0.459 |
| Degree | OriginalOrder | 100 | 0.362 +/- 0.333 | 30.502 +/- 0.854 | 30.864 +/- 0.955 | 0.000 +/- 0.000 | 4.684 +/- 0.459 |
| Degree | MarginalGainPredictor | 10 | 4.894 +/- 0.743 | 0.394 +/- 0.650 | 5.288 +/- 0.677 | 0.860 +/- 0.207 | 4.706 +/- 0.478 |
| Degree | MarginalGainPredictor | 20 | 3.228 +/- 0.435 | 1.202 +/- 0.604 | 4.430 +/- 0.375 | 0.680 +/- 0.110 | 4.690 +/- 0.460 |
| Degree | MarginalGainPredictor | 50 | 1.244 +/- 0.394 | 3.186 +/- 0.210 | 4.430 +/- 0.375 | 0.380 +/- 0.045 | 4.690 +/- 0.459 |
| Degree | MarginalGainPredictor | 100 | 0.362 +/- 0.333 | 4.068 +/- 0.604 | 4.430 +/- 0.375 | 0.360 +/- 0.055 | 4.690 +/- 0.459 |
| DegreeDiscount | OriginalOrder | 10 | 9.372 +/- 1.941 | 18.122 +/- 2.854 | 27.494 +/- 1.127 | 0.200 +/- 0.100 | 4.712 +/- 0.460 |
| DegreeDiscount | OriginalOrder | 20 | 5.692 +/- 0.729 | 21.802 +/- 1.481 | 27.494 +/- 1.127 | 0.000 +/- 0.000 | 4.712 +/- 0.460 |
| DegreeDiscount | OriginalOrder | 50 | 2.002 +/- 0.276 | 25.492 +/- 0.968 | 27.494 +/- 1.127 | 0.000 +/- 0.000 | 4.712 +/- 0.460 |
| DegreeDiscount | OriginalOrder | 100 | 0.000 +/- 0.000 | 27.494 +/- 1.127 | 27.494 +/- 1.127 | 0.000 +/- 0.000 | 4.712 +/- 0.460 |
| DegreeDiscount | MarginalGainPredictor | 10 | 9.372 +/- 1.941 | 5.592 +/- 1.268 | 14.964 +/- 3.138 | 0.600 +/- 0.200 | 4.718 +/- 0.461 |
| DegreeDiscount | MarginalGainPredictor | 20 | 5.692 +/- 0.729 | 7.420 +/- 3.129 | 13.112 +/- 3.480 | 0.600 +/- 0.071 | 4.717 +/- 0.461 |
| DegreeDiscount | MarginalGainPredictor | 50 | 2.002 +/- 0.276 | 2.428 +/- 0.436 | 4.430 +/- 0.375 | 0.580 +/- 0.110 | 4.717 +/- 0.460 |
| DegreeDiscount | MarginalGainPredictor | 100 | 0.000 +/- 0.000 | 4.430 +/- 0.375 | 4.430 +/- 0.375 | 0.360 +/- 0.055 | 4.718 +/- 0.461 |
| FeatureDQN | OriginalOrder | 10 | 32.930 +/- 1.151 | 6.814 +/- 0.624 | 39.744 +/- 1.079 | 0.020 +/- 0.045 | 4.697 +/- 0.463 |
| FeatureDQN | OriginalOrder | 20 | 9.098 +/- 2.236 | 30.646 +/- 2.458 | 39.744 +/- 1.079 | 0.000 +/- 0.000 | 4.697 +/- 0.463 |
| FeatureDQN | OriginalOrder | 50 | 8.640 +/- 1.726 | 31.104 +/- 1.889 | 39.744 +/- 1.079 | 0.000 +/- 0.000 | 4.697 +/- 0.463 |
| FeatureDQN | OriginalOrder | 100 | 8.164 +/- 1.852 | 31.580 +/- 2.315 | 39.744 +/- 1.079 | 0.000 +/- 0.000 | 4.697 +/- 0.463 |
| FeatureDQN | MarginalGainPredictor | 10 | 32.930 +/- 1.151 | 2.314 +/- 0.906 | 35.244 +/- 1.818 | 0.440 +/- 0.114 | 4.702 +/- 0.463 |
| FeatureDQN | MarginalGainPredictor | 20 | 9.098 +/- 2.236 | 10.752 +/- 2.353 | 19.850 +/- 0.680 | 0.200 +/- 0.200 | 4.702 +/- 0.463 |
| FeatureDQN | MarginalGainPredictor | 50 | 8.640 +/- 1.726 | 22.224 +/- 0.991 | 30.864 +/- 0.955 | 0.000 +/- 0.000 | 4.702 +/- 0.463 |
| FeatureDQN | MarginalGainPredictor | 100 | 8.164 +/- 1.852 | 22.700 +/- 1.267 | 30.864 +/- 0.955 | 0.000 +/- 0.000 | 4.703 +/- 0.463 |
| Random | OriginalOrder | 10 | 20.682 +/- 0.991 | 16.776 +/- 0.784 | 37.458 +/- 0.935 | 0.110 +/- 0.012 | 4.683 +/- 0.459 |
| Random | OriginalOrder | 20 | 16.142 +/- 0.682 | 21.317 +/- 0.615 | 37.458 +/- 0.935 | 0.064 +/- 0.015 | 4.683 +/- 0.459 |
| Random | OriginalOrder | 50 | 8.877 +/- 0.852 | 28.581 +/- 0.917 | 37.458 +/- 0.935 | 0.022 +/- 0.013 | 4.683 +/- 0.459 |
| Random | OriginalOrder | 100 | 3.987 +/- 0.433 | 33.471 +/- 1.038 | 37.458 +/- 0.935 | 0.010 +/- 0.012 | 4.683 +/- 0.459 |
| Random | MarginalGainPredictor | 10 | 20.682 +/- 0.991 | 7.778 +/- 0.370 | 28.460 +/- 0.791 | 0.386 +/- 0.015 | 4.688 +/- 0.459 |
| Random | MarginalGainPredictor | 20 | 16.142 +/- 0.682 | 9.503 +/- 0.661 | 25.644 +/- 0.969 | 0.320 +/- 0.025 | 4.688 +/- 0.459 |
| Random | MarginalGainPredictor | 50 | 8.877 +/- 0.852 | 8.794 +/- 0.644 | 17.671 +/- 0.973 | 0.332 +/- 0.052 | 4.688 +/- 0.459 |
| Random | MarginalGainPredictor | 100 | 3.987 +/- 0.433 | 6.835 +/- 0.719 | 10.822 +/- 0.512 | 0.326 +/- 0.027 | 4.689 +/- 0.459 |

## End-to-End Sequential Evaluation

每种方法按自己的选择轨迹更新 seed set；此表的 final spread 才是端到端结果，不能和上面的 oracle-trajectory diagnostic 混用。

| Retriever | Ranker | M | CandidateLoss | RankingLoss | TotalRegret | Top-1 | Final spread | Total runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Degree | OriginalOrder | 10 | 10.780 +/- 0.961 | 23.068 +/- 0.476 | 33.848 +/- 1.304 | 0.200 +/- 0.000 | 299.310 +/- 0.445 | 4.843 +/- 0.356 |
| Degree | OriginalOrder | 20 | 10.564 +/- 0.924 | 23.284 +/- 0.476 | 33.848 +/- 1.304 | 0.200 +/- 0.000 | 299.310 +/- 0.445 | 4.828 +/- 0.361 |
| Degree | OriginalOrder | 50 | 1.550 +/- 0.542 | 32.298 +/- 1.257 | 33.848 +/- 1.304 | 0.100 +/- 0.000 | 299.310 +/- 0.445 | 4.811 +/- 0.357 |
| Degree | OriginalOrder | 100 | 0.000 +/- 0.000 | 33.848 +/- 1.304 | 33.848 +/- 1.304 | 0.100 +/- 0.000 | 299.310 +/- 0.445 | 4.811 +/- 0.356 |
| Degree | MarginalGainPredictor | 10 | 18.184 +/- 1.011 | 8.390 +/- 0.440 | 26.574 +/- 1.104 | 0.440 +/- 0.055 | 316.506 +/- 0.459 | 4.339 +/- 0.299 |
| Degree | MarginalGainPredictor | 20 | 15.748 +/- 0.929 | 12.564 +/- 0.453 | 28.312 +/- 0.947 | 0.260 +/- 0.089 | 321.258 +/- 0.310 | 4.777 +/- 0.512 |
| Degree | MarginalGainPredictor | 50 | 1.682 +/- 1.197 | 25.800 +/- 1.041 | 27.482 +/- 1.140 | 0.100 +/- 0.000 | 311.202 +/- 0.241 | 4.725 +/- 0.420 |
| Degree | MarginalGainPredictor | 100 | 0.000 +/- 0.000 | 27.482 +/- 1.140 | 27.482 +/- 1.140 | 0.100 +/- 0.000 | 311.202 +/- 0.241 | 4.724 +/- 0.424 |
| DegreeDiscount | OriginalOrder | 10 | 3.804 +/- 0.499 | 53.334 +/- 1.141 | 57.138 +/- 0.878 | 0.100 +/- 0.000 | 230.004 +/- 0.294 | 4.755 +/- 0.477 |
| DegreeDiscount | OriginalOrder | 20 | 3.804 +/- 0.499 | 53.334 +/- 1.141 | 57.138 +/- 0.878 | 0.000 +/- 0.000 | 230.004 +/- 0.294 | 4.754 +/- 0.477 |
| DegreeDiscount | OriginalOrder | 50 | 1.320 +/- 0.551 | 55.818 +/- 1.173 | 57.138 +/- 0.878 | 0.000 +/- 0.000 | 230.004 +/- 0.294 | 4.754 +/- 0.478 |
| DegreeDiscount | OriginalOrder | 100 | 0.000 +/- 0.000 | 57.138 +/- 0.878 | 57.138 +/- 0.878 | 0.000 +/- 0.000 | 230.004 +/- 0.294 | 4.754 +/- 0.477 |
| DegreeDiscount | MarginalGainPredictor | 10 | 23.128 +/- 0.412 | 10.340 +/- 1.139 | 33.468 +/- 1.076 | 0.240 +/- 0.089 | 266.890 +/- 0.350 | 4.293 +/- 0.365 |
| DegreeDiscount | MarginalGainPredictor | 20 | 17.796 +/- 0.538 | 18.256 +/- 0.923 | 36.052 +/- 1.347 | 0.200 +/- 0.071 | 277.000 +/- 0.416 | 4.837 +/- 0.386 |
| DegreeDiscount | MarginalGainPredictor | 50 | 6.688 +/- 0.432 | 23.136 +/- 0.917 | 29.824 +/- 1.150 | 0.180 +/- 0.045 | 295.918 +/- 0.295 | 4.772 +/- 0.474 |
| DegreeDiscount | MarginalGainPredictor | 100 | 0.000 +/- 0.000 | 26.708 +/- 1.265 | 26.708 +/- 1.265 | 0.100 +/- 0.000 | 319.444 +/- 0.413 | 4.758 +/- 0.424 |
| FeatureDQN | OriginalOrder | 10 | 52.592 +/- 0.362 | 23.848 +/- 0.501 | 76.440 +/- 0.479 | 0.080 +/- 0.045 | 92.174 +/- 0.696 | 4.685 +/- 0.297 |
| FeatureDQN | OriginalOrder | 20 | 31.596 +/- 1.287 | 44.844 +/- 1.177 | 76.440 +/- 0.479 | 0.000 +/- 0.000 | 92.174 +/- 0.696 | 4.683 +/- 0.294 |
| FeatureDQN | OriginalOrder | 50 | 31.596 +/- 1.287 | 44.844 +/- 1.177 | 76.440 +/- 0.479 | 0.000 +/- 0.000 | 92.174 +/- 0.696 | 4.683 +/- 0.294 |
| FeatureDQN | OriginalOrder | 100 | 31.596 +/- 1.287 | 44.844 +/- 1.177 | 76.440 +/- 0.479 | 0.000 +/- 0.000 | 92.174 +/- 0.696 | 4.683 +/- 0.294 |
| FeatureDQN | MarginalGainPredictor | 10 | 62.966 +/- 0.480 | 3.176 +/- 0.082 | 66.142 +/- 0.470 | 0.440 +/- 0.089 | 181.446 +/- 0.500 | 4.503 +/- 0.605 |
| FeatureDQN | MarginalGainPredictor | 20 | 31.942 +/- 1.436 | 31.020 +/- 1.377 | 62.962 +/- 0.329 | 0.100 +/- 0.000 | 217.518 +/- 0.560 | 4.566 +/- 0.247 |
| FeatureDQN | MarginalGainPredictor | 50 | 26.622 +/- 1.314 | 30.488 +/- 0.963 | 57.110 +/- 0.520 | 0.000 +/- 0.000 | 193.358 +/- 0.176 | 4.931 +/- 0.418 |
| FeatureDQN | MarginalGainPredictor | 100 | 23.412 +/- 1.290 | 32.580 +/- 1.159 | 55.992 +/- 0.658 | 0.000 +/- 0.000 | 200.858 +/- 0.268 | 4.706 +/- 0.388 |
| Random | OriginalOrder | 10 | 43.667 +/- 1.042 | 26.451 +/- 1.038 | 70.118 +/- 0.410 | 0.092 +/- 0.022 | 147.165 +/- 1.859 | 4.738 +/- 0.407 |
| Random | OriginalOrder | 20 | 35.187 +/- 0.614 | 34.930 +/- 0.255 | 70.118 +/- 0.410 | 0.040 +/- 0.020 | 147.165 +/- 1.859 | 4.738 +/- 0.407 |
| Random | OriginalOrder | 50 | 23.510 +/- 0.844 | 46.607 +/- 0.658 | 70.118 +/- 0.410 | 0.012 +/- 0.004 | 147.165 +/- 1.859 | 4.738 +/- 0.407 |
| Random | OriginalOrder | 100 | 8.733 +/- 0.347 | 61.385 +/- 0.564 | 70.118 +/- 0.410 | 0.010 +/- 0.000 | 147.165 +/- 1.859 | 4.738 +/- 0.407 |
| Random | MarginalGainPredictor | 10 | 41.312 +/- 0.502 | 7.531 +/- 1.108 | 48.843 +/- 0.700 | 0.436 +/- 0.040 | 258.475 +/- 3.867 | 4.295 +/- 0.355 |
| Random | MarginalGainPredictor | 20 | 30.122 +/- 2.396 | 11.231 +/- 0.740 | 41.353 +/- 1.765 | 0.318 +/- 0.028 | 291.695 +/- 3.917 | 4.693 +/- 0.397 |
| Random | MarginalGainPredictor | 50 | 14.405 +/- 1.325 | 19.189 +/- 0.702 | 33.594 +/- 1.211 | 0.208 +/- 0.046 | 304.704 +/- 2.525 | 4.755 +/- 0.293 |
| Random | MarginalGainPredictor | 100 | 5.089 +/- 0.546 | 23.178 +/- 0.682 | 28.267 +/- 0.857 | 0.164 +/- 0.023 | 312.915 +/- 1.160 | 4.674 +/- 0.300 |

## 分解核对

逐条扫描诊断与端到端记录共 `10,400` 条。以 `1e-8` 为浮点容差，发现 `0` 条不满足：

```text
TotalRegret = CandidateLoss + RankingLoss
```

因此没有需要单独列出的异常种子或记录；没有修改任何原始数值。

## 可汇报的事实

- 五个种子均完成；每份输出均标注 `oracle_scope=top-200-by-degree`，FeatureDQN 与 MarginalGainPredictor 均为 `loaded`。
- 在 oracle-trajectory diagnostic 中，使用 MarginalGainPredictor 后，同一 retriever、M 下的 CandidateLoss 不变，而 RankingLoss 由池内排序决定。例如 Degree、M=20 的 RankingLoss 从 `27.636 +/- 0.980` 降至 `1.202 +/- 0.604`；该观察只说明共享 oracle 轨迹上的池内重排误差变化。
- 在本次端到端设置中，final spread 应只依据第二张表陈述。例如 Degree + MarginalGainPredictor、M=20 为 `321.258 +/- 0.310`；Degree 原顺序为 `299.310 +/- 0.445`。这与诊断表属于不同选择轨迹，不能互相替代。
- 本次运行未重训 FeatureDQN，只加载仓库原有的 `param/dqn_model.pth` 和已有 embedding。运行日志能证明 checkpoint 被成功读取，但不能证明该 checkpoint 按本次协议重新训练，也不应据此推断其训练质量或泛化性。

## 限制

- Oracle 是 restricted oracle（top-200-by-degree），不是全图 oracle。
- 结果只适用于当前 NetHEPT、IC 概率 `0.01`、本次 MC 设置、五个随机种子、候选范围和已有 checkpoint；不能直接外推到其他数据集、全图搜索或其他训练设置。
- Random 的数值包含每个种子内的 10 次随机 repeat；报告中的标准差仍以 5 个独立随机种子为单位。
- 本次任务未改动源码或正式配置；主仓库启动时已有未提交的联合实验实现文件，本次仅写入被忽略的 `outputs/` 与本报告。
