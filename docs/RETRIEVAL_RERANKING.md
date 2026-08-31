# Retrieval + Seed-conditioned Marginal Reranking

该实验把候选检索与池内重排组合起来，并严格分开两类结果：

1. **Oracle-trajectory diagnostic**：所有方法共享 restricted oracle 的 seed 轨迹，在同一个 `S` 下比较检索误差和重排误差。
2. **End-to-end sequential evaluation**：每种 retriever、`M` 和 ranker 按自己的选择结果更新 `S`，报告最终 spread 和运行时间。

## 运行

小规模 smoke：

```bash
PYTHONPATH=src python scripts/run_retrieval_reranking.py \
  --config configs/smoke/retrieval_reranking_nethept.yaml
```

单随机种子 NetHEPT：

```bash
PYTHONPATH=src python scripts/run_retrieval_reranking.py \
  --config configs/retrieval_reranking_nethept.yaml
```

每次运行在配置的 `outputs/` 目录下生成独立时间戳目录，包含：

- `oracle_trajectory_diagnostic.json` 与 `.csv`：诊断汇总和完整逐步记录；
- `end_to_end_sequential.json` 与 `.csv`：各方法自己的轨迹、最终 spread 和时间；
- `validation.json`：loss 分解检查；
- `config.yaml`：实际运行配置快照。

## 方法与指标

Retriever 包含 Random、Degree、动态 Degree Discount 和 FeatureDQN。每个候选池分别使用 `OriginalOrder` 与已训练 `MarginalGainPredictor` 排序。模型 checkpoint 或 embedding 缺失时，JSON 会记录 `status=skipped` 和原始原因，并跳过该方法。

对当前 seed set `S`，同一步所有候选的真实 marginal gain 使用同一批 live-edge realization：

- `CandidateLoss = restricted-oracle gain - pool 内真实最大 gain`
- `RankingLoss = pool 内真实最大 gain - 所选节点真实 gain`
- `TotalRegret = restricted-oracle gain - 所选节点真实 gain`

实现检查 `CandidateLoss + RankingLoss = TotalRegret`，允许 `1e-8` 以内浮点误差。

## 限制

Oracle 是配置指定的 **restricted oracle（默认 top-200-by-degree）**，仅在该节点范围内寻找真实边际增益最大的节点，不是全图 oracle。结果只能解释当前数据集、扩散参数、候选范围、MC 次数和随机种子下的 retrieval/reranking 表现。
