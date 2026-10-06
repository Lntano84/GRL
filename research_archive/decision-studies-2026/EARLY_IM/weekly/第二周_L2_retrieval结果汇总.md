# 第二周 L2 Candidate Retrieval Benchmark 结果汇总

运行日期：2026-08-31  
仓库：`C:\Users\windows\Desktop\GRL-main`  
基线提交：`bcc6686 feat: integrate candidate retrieval benchmark`

## 实验协议

- Oracle：**restricted oracle（top-200-by-degree）**。每一步只在度数最高的 200 个节点构成的候选范围内比较，不代表全图范围。
- 数据与扩散：NetHEPT、有向图、IC、边传播概率 `0.01`。
- 参数：`steps=10`，`mc_runs=10`，`random_repeats=10`，`pool_sizes=[10, 20, 50, 100]`。
- 种子：`20260715`、`20260716`、`20260717`、`20260718`、`20260719`。
- 统计：下表为 5 个种子的 mean +/- sample std；runtime 单位为秒。

## 运行命令

```powershell
$env:PYTHONPATH='src'
python scripts/run_candidate_benchmark.py --config outputs/candidate_benchmark/run_configs/nethept_seed_20260715.yaml
python scripts/run_candidate_benchmark.py --config outputs/candidate_benchmark/run_configs/nethept_seed_20260716.yaml
python scripts/run_candidate_benchmark.py --config outputs/candidate_benchmark/run_configs/nethept_seed_20260717.yaml
python scripts/run_candidate_benchmark.py --config outputs/candidate_benchmark/run_configs/nethept_seed_20260718.yaml
python scripts/run_candidate_benchmark.py --config outputs/candidate_benchmark/run_configs/nethept_seed_20260719.yaml
```

临时配置仅位于 `outputs/candidate_benchmark/run_configs/`；正式配置 `configs/candidate_benchmark_nethept.yaml` 未修改。

## 原始输出

每个目录均包含 `candidate_benchmark.json`、`candidate_benchmark.csv`、`config.yaml` 和 `console.log`。

1. `C:\Users\windows\Desktop\GRL-main\outputs\candidate_benchmark\20260831_143439\candidate_benchmark`
2. `C:\Users\windows\Desktop\GRL-main\outputs\candidate_benchmark\20260831_143506\candidate_benchmark`
3. `C:\Users\windows\Desktop\GRL-main\outputs\candidate_benchmark\20260831_143529\candidate_benchmark`
4. `C:\Users\windows\Desktop\GRL-main\outputs\candidate_benchmark\20260831_143553\candidate_benchmark`
5. `C:\Users\windows\Desktop\GRL-main\outputs\candidate_benchmark\20260831_143617\candidate_benchmark`

## 汇总结果

| Retriever | M | Recall@M (mean +/- std) | CandidateLoss (mean +/- std) | Retrieval runtime / s (mean +/- std) |
|---|---:|---:|---:|---:|
| Degree | 10 | 0.440 +/- 0.055 | 4.958 +/- 1.735 | 0.000108 +/- 0.000008 |
| Degree | 20 | 0.500 +/- 0.000 | 3.688 +/- 1.022 | 0.000108 +/- 0.000008 |
| Degree | 50 | 0.720 +/- 0.045 | 1.698 +/- 0.951 | 0.000108 +/- 0.000008 |
| Degree | 100 | 0.960 +/- 0.055 | 0.078 +/- 0.130 | 0.000108 +/- 0.000008 |
| DegreeDiscount | 10 | 0.100 +/- 0.000 | 12.042 +/- 1.013 | 0.001911 +/- 0.000136 |
| DegreeDiscount | 20 | 0.240 +/- 0.055 | 5.206 +/- 1.154 | 0.001911 +/- 0.000136 |
| DegreeDiscount | 50 | 0.680 +/- 0.045 | 2.362 +/- 0.276 | 0.001911 +/- 0.000136 |
| DegreeDiscount | 100 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.001911 +/- 0.000136 |
| FeatureDQN | 10 | 0.000 +/- 0.000 | 29.396 +/- 1.073 | 0.001308 +/- 0.000020 |
| FeatureDQN | 20 | 0.100 +/- 0.000 | 10.258 +/- 0.761 | 0.001308 +/- 0.000020 |
| FeatureDQN | 50 | 0.100 +/- 0.000 | 9.712 +/- 0.649 | 0.001308 +/- 0.000020 |
| FeatureDQN | 100 | 0.200 +/- 0.071 | 8.250 +/- 1.085 | 0.001308 +/- 0.000020 |
| Random | 10 | 0.078 +/- 0.013 | 19.806 +/- 0.809 | 0.000033 +/- 0.000002 |
| Random | 20 | 0.120 +/- 0.031 | 14.900 +/- 0.973 | 0.000033 +/- 0.000002 |
| Random | 50 | 0.290 +/- 0.042 | 7.753 +/- 0.567 | 0.000033 +/- 0.000002 |
| Random | 100 | 0.518 +/- 0.031 | 3.344 +/- 0.363 | 0.000033 +/- 0.000002 |

## FeatureDQN 状态

五个种子均成功加载：`status=loaded`。使用的 checkpoint 为 `param/dqn_model.pth`，embedding 为 `param/node2vec_NetHEPT.txt.pth`。本次没有失败状态，因此没有需要保留的失败原因。

## 核对与限制

- 五次运行的全部逐步 records 中，CandidateLoss 均不为负；`recalled=1` 的 records 均满足 CandidateLoss 为 0。
- 运行结束后 Git 工作区仍为 clean；未修改源码、正式配置、提交或 Git 历史。
- 这些数值只描述候选检索在当前 NetHEPT、IC 概率、预算与 **restricted oracle（top-200-by-degree）** 协议下的表现；不能外推为全图 Oracle 结果，也不直接评价完整影响力最大化流程或其他数据集上的表现。

