# 交接说明

仓库地址：<https://github.com/Lntano84/GRL>

本说明面向接手后续实验的同学。先确认本地分支处于最新 `main`：

```powershell
git clone https://github.com/Lntano84/GRL.git
cd GRL
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m pytest -q
```

已有结果和阶段报告保存在 `C:\Users\windows\Desktop\codex`，不在 Git 仓库内；`outputs/` 也被 Git 忽略。不要用新的运行结果覆盖已有报告。

## 已有实验入口

| 阶段 | 目的 | Smoke 命令 | 正式单种子命令 |
| --- | --- | --- | --- |
| S1 / L1-L2 | Candidate Retrieval Benchmark | `python scripts/run_candidate_benchmark.py --config configs/smoke/candidate_benchmark_nethept.yaml` | `python scripts/run_candidate_benchmark.py --config configs/candidate_benchmark_nethept.yaml` |
| L3 | retrieval + seed-conditioned reranking 联合评估 | `python scripts/run_retrieval_reranking.py --config configs/smoke/retrieval_reranking_nethept.yaml` | `python scripts/run_retrieval_reranking.py --config configs/retrieval_reranking_nethept.yaml` |
| S4 | Direct-MLP 与 Direct-MLP+Overlap 对照 | `python scripts/run_overlap_experiment.py --config configs/smoke/overlap_nethept.yaml` | `python scripts/run_overlap_experiment.py --config configs/overlap_nethept.yaml` |

所有命令均要求先设置 `$env:PYTHONPATH = "src"`。先跑 smoke，再决定是否运行正式配置；正式结果必须使用新的输出目录或新的时间戳，保留原始 JSON、CSV 和日志。

## L3 联合实验

实现文件：

- `scripts/run_retrieval_reranking.py`
- `src/grl/diagnostics/retrieval_reranking.py`
- `configs/retrieval_reranking_nethept.yaml`
- `configs/smoke/retrieval_reranking_nethept.yaml`
- `docs/RETRIEVAL_RERANKING.md`

检索器为 Random、Degree、动态 Degree Discount、FeatureDQN。池内排序器为原检索顺序和已训练的 `MarginalGainPredictor`。诊断实验和端到端实验必须分开解释：前者在同一 oracle seed 轨迹下分解检索和排序误差，后者每种方法沿自己的 seed 轨迹评价最终 spread。

指标定义：

```text
CandidateLoss = restricted-oracle gain - pool best gain
RankingLoss   = pool best gain - selected gain
TotalRegret   = restricted-oracle gain - selected gain
```

每条记录应满足 `CandidateLoss + RankingLoss = TotalRegret`（允许微小浮点误差）。同一步候选真实 marginal gain 必须使用同一批 live-edge realization。

Oracle 只能写成 `restricted oracle (top-200-by-degree)`，不可写成全图 oracle 或 global oracle。FeatureDQN 当前仅确认能够加载既有 checkpoint，尚未按这次协议重新训练，因此不能据此宣称 RL 有效或无效。

## 外部模型依赖

联合实验的 MarginalGainPredictor 不随仓库提交，因为模型文件在仓库外：

```text
C:\Users\windows\Desktop\codex\GRL_最终实验_v2\models\direct_marginal_nethept.pth
C:\Users\windows\Desktop\codex\GRL_最终实验_v2\models\comparison_node2vec_nethept.pth
```

正式和 smoke 配置通过以下相对路径引用它们：

```text
../codex/GRL_最终实验_v2/models/direct_marginal_nethept.pth
../codex/GRL_最终实验_v2/models/comparison_node2vec_nethept.pth
```

如果接手环境没有这些文件，程序会在 JSON 中记录 `status=skipped` 和原因，并跳过 MarginalGainPredictor；不得用随机初始化模型替代。FeatureDQN 还依赖仓库内的 `param/dqn_model.pth` 与 `param/node2vec_NetHEPT.txt.pth`。

## 后续建议

1. 先复现 smoke，再用 5 个新随机种子重复 L3 或 S4，保留每个种子的独立输出。
2. S4 优先做 overlap 特征消融：逐项移除最短路径、k-hop overlap 与覆盖代理，检查增益来自哪一类特征。
3. 跨图实验开始前，先明确训练图和测试图，并按图划分数据。测试图的标签、候选特征统计、归一化信息均不可参与训练或调参。
4. 若要重新评估 FeatureDQN，先固定训练协议和 5 个测试种子；无法稳定优于启发式时，保留它作为失败基线，转向 overlap-aware marginal ranking。

## 提交规范

每一类独立实验使用单独提交，提交前执行：

```powershell
python -m pytest -q
git diff --check
git status
```

不要提交 `outputs/`、checkpoint、embedding 或个人报告，除非明确确认了文件体积、许可证和公开范围。
