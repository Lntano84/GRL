# 第二周 S4：显式 overlap 特征实验汇总

日期：2026-08-31  
仓库：`C:\Users\windows\Desktop\GRL-main`

## 实验目的与公平设置

- 对照模型：原 `Direct-MLP`，结构未改。
- 新模型：`Direct-MLP+Overlap`；它仍是直接预测边际增益的 MLP，不是 GNN。
- 两个模型复用完全相同的 marginal dataset、train/validation/test context 划分和 live-edge 标签。
- 数据文件 SHA-256：`70ff9f53f98f425da22a425235367f390fbbc0dfc67792bb781f096f55f3f026`。
- split：训练 3200 条/160 contexts，验证 400 条/20 contexts，测试 400 条/20 contexts。
- 五个训练随机种子：`20260821` 至 `20260825`；每个 seed 都重新训练两个模型，训练轮数、batch 和学习率相同。
- Sequential 评价：每一步在 restricted top-200-by-degree 范围内重新构造 Degree Top-20 候选池；不是全图 oracle。

运行命令：

```powershell
$env:PYTHONPATH='src'
python scripts/run_overlap_experiment.py --config configs/smoke/overlap_nethept.yaml
python scripts/run_overlap_experiment.py --config configs/overlap_nethept.yaml
```

最终 smoke：`C:\Users\windows\Desktop\GRL-main\outputs\overlap_experiment\smoke_nethept\20260831_190510`  
最终正式结果：`C:\Users\windows\Desktop\GRL-main\outputs\overlap_experiment\nethept\20260831_190534`

## 9 个 overlap 特征

1. 到最近 seed 的无权最短路，截断后归一化；表示候选与已有 seed 的结构距离。
2. 不可达标记；区分“距离较远”和“根本不在同一弱连通区域”。
3. 空 seed set 标记；使第一步的特殊状态显式可见。
4. candidate 1-hop 闭邻域被 seed 1-hop 邻域并集覆盖的比例。
5. candidate 2-hop 闭邻域被 seed 2-hop 邻域并集覆盖的比例。
6. candidate 与 seed 的 2-hop 邻域 Jaccard。
7. candidate 2-hop 范围内包含的 seed 比例。
8. seed 2-hop 邻域并集占全图节点的比例，作为 seed set 局部覆盖代理。
9. candidate 1-hop 闭邻域中 seed 节点的比例。

这些特征近似已有 seed 和候选可能覆盖到相同局部区域的程度。第 4、5、8 项只是局部覆盖/社区覆盖代理，不是社区检测结果；本实验没有引入未经验证的 Louvain。

## Held-out test 结果

数值是五个训练随机种子的 `mean +/- sample std`。RankingLoss 在同一 test context 的候选组内计算。

| 模型 | MAE | Spearman | Top-1 | RankingLoss |
|---|---:|---:|---:|---:|
| Direct-MLP | 0.8724 +/- 0.0383 | 0.8573 +/- 0.0246 | 0.740 +/- 0.022 | 1.5270 +/- 0.0324 |
| Direct-MLP+Overlap | 0.8849 +/- 0.0283 | 0.8405 +/- 0.0105 | 0.750 +/- 0.035 | 1.4905 +/- 0.1171 |

按 `|S|` 分组：

| `|S|` | 模型 | MAE | Spearman | Top-1 | RankingLoss |
|---:|---|---:|---:|---:|---:|
| 0 | Direct-MLP | 1.341 | 0.853 | 0.500 | 6.675 |
| 0 | Direct-MLP+Overlap | 1.534 | 0.799 | 0.500 | 6.675 |
| 1 | Direct-MLP | 1.005 | 0.834 | 0.500 | 2.150 |
| 1 | Direct-MLP+Overlap | 0.955 | 0.826 | 0.600 | 1.720 |
| 2 | Direct-MLP | 0.485 | 0.922 | 0.400 | 1.470 |
| 2 | Direct-MLP+Overlap | 0.485 | 0.896 | 0.500 | 1.325 |
| 3 | Direct-MLP | 0.702 | 0.828 | 0.500 | 3.275 |
| 3 | Direct-MLP+Overlap | 0.710 | 0.793 | 0.400 | 3.485 |
| 4 | Direct-MLP | 0.949 | 0.834 | 1.000 | 0.000 |
| 4 | Direct-MLP+Overlap | 1.009 | 0.794 | 1.000 | 0.000 |
| 5 | Direct-MLP | 0.947 | 0.826 | 1.000 | 0.000 |
| 5 | Direct-MLP+Overlap | 0.866 | 0.833 | 1.000 | 0.000 |
| 6 | Direct-MLP | 0.957 | 0.885 | 1.000 | 0.000 |
| 6 | Direct-MLP+Overlap | 0.934 | 0.896 | 1.000 | 0.000 |
| 7 | Direct-MLP | 1.018 | 0.826 | 0.500 | 1.700 |
| 7 | Direct-MLP+Overlap | 0.993 | 0.832 | 0.500 | 1.700 |
| 8 | Direct-MLP | 0.716 | 0.829 | 1.000 | 0.000 |
| 8 | Direct-MLP+Overlap | 0.732 | 0.810 | 1.000 | 0.000 |
| 9 | Direct-MLP | 0.605 | 0.936 | 1.000 | 0.000 |
| 9 | Direct-MLP+Overlap | 0.630 | 0.926 | 1.000 | 0.000 |

## Sequential 结果

每个模型按自己的轨迹更新 `S`。RankingLoss 和 Top-1 为截至该预算的逐步均值，final spread 使用相同 MC 设置评价。

| Budget | 模型 | RankingLoss | Top-1 | Final spread |
|---:|---|---:|---:|---:|
| 1 | Direct-MLP | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 89.382 +/- 0.100 |
| 1 | Direct-MLP+Overlap | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 89.382 +/- 0.100 |
| 3 | Direct-MLP | 0.393 +/- 0.512 | 0.800 +/- 0.183 | 170.040 +/- 0.138 |
| 3 | Direct-MLP+Overlap | 2.913 +/- 0.520 | 0.667 +/- 0.000 | 160.804 +/- 0.341 |
| 5 | Direct-MLP | 2.492 +/- 0.931 | 0.560 +/- 0.219 | 242.836 +/- 0.428 |
| 5 | Direct-MLP+Overlap | 2.216 +/- 0.380 | 0.600 +/- 0.000 | 241.560 +/- 0.436 |
| 10 | Direct-MLP | 12.302 +/- 0.822 | 0.280 +/- 0.110 | 320.808 +/- 4.837 |
| 10 | Direct-MLP+Overlap | 10.478 +/- 0.336 | 0.300 +/- 0.000 | 329.528 +/- 1.002 |

## 结论

第一版 overlap 特征没有带来一致增益，不能表述为“overlap 已经有效”。总体 test MAE 增加约 1.4%，Spearman 下降约 0.017；Top-1 增加 0.01，RankingLoss 降低约 2.4%。按 `|S|` 看，有些组改善、有些组变差。

Sequential 在 budget=10 时，overlap 的 RankingLoss 更低，final spread 高 `8.720`（约 `2.7%`）；但 budget=3 和 5 的 spread 更低。因此当前只能认为显式 overlap 中可能存在局部有用信号，还没有稳定证据支持全面替换 Direct-MLP。下一步若继续，应先增加每个 `|S|` 的 test contexts，并做特征消融，而不是直接加入更复杂模型。

## 最小跨图 protocol

- 训练图固定为 NetHEPT；测试图从仓库已有 `epinions-d-5.txt`、`twitter-d.txt` 或老师指定图中确认一张。
- 测试图需要独立生成图内 embedding 和 marginal dataset，保持 IC 参数、9 维特征定义、归一化和 seed-size 分层规则一致。
- NetHEPT 训练结束后冻结模型；测试图 marginal labels 只用于最终评价，不能训练或调参。
- 禁止使用测试图 test split 选 epoch/超参数，禁止混用跨图 node ID，禁止复用携带标签信息的 cache。
- 当前未运行跨图结果。还缺：老师确认的测试图、该图方向/边权/IC 参数、期望节点边数、独立 embedding、按相同协议生成的 marginal labels，以及跨图维度对齐方案。

## 输出文件

- 配置：`configs/overlap_nethept.yaml`、`configs/smoke/overlap_nethept.yaml`
- 总结果：`results.json`
- held-out 汇总：`aggregate_metrics.csv`
- sequential 汇总：`sequential_aggregate.csv`
- 每个 seed 子目录：模型 checkpoint、按 `|S|` CSV、两个模型的 sequential JSON
- 测试：`34 passed`
