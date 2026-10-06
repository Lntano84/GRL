# 断点续跑：已完成并通过验证

**状态**：`ccim/xplore/progress.py` 实现，`scripts/xplore_official.py` 已接入（`--resume`，每 25 episode 原子落盘）。
**验证**：`scripts/resume_check.py` 两阶段对照**全部通过**，`results/resume_check.json`。
**未启动五种子长跑**（按你的指示，等批量化结论）。

---

## 1. 续跑点包含什么

仅存权重**不是**续跑点，已存下来的 `official_seed*_ep500.pt` 这类只含权重的文件**不得**当续跑点——`progress.load()` 会检查 `schema`，缺字段直接拒绝。

实际保存的内容（`progress.capture`）：

| 类别 | 字段 |
|---|---|
| 网络 | `actor_critic`、`target_actor_critic` 全部张量 |
| 优化器 | `critic_opt` 状态（Adam 一阶/二阶矩） |
| replay | `buffer` 内容、**`probs` 优先级**、`max_size`、`beta`、`eps` |
| 计数 | `episode`、`epsilon`、`selected_episode`、`val_curve` |
| **随机状态** | `rng`（numpy Generator：选图/选初始种子/ε/探索/嵌入种子）、`seed_rng`（每 episode 的 5 个免费调查节点）、`replay.rg`（batch 取样）、`random`（作者 `ge/walker.py` 游走与 `icm.sample_live_icm` 抽边）、`torch`、`np.random` 全局 |

**原子性**：每次先写 `*.tmp`，再 `os.replace` 覆盖目标。`os.replace` 在 Windows 与 POSIX 上都是原子的，因此写到一半崩溃**不可能**破坏已有的有效检查点——这正是「经常关机」场景下必须的。

## 2. 验证结果

### 阶段一：往返一致性（`replay=105 > 100`，更新已生效）

```
warmed 21 episodes, replay.size=105 (updates live)
[PASS] weights restored bit-exactly (checksum 15.903581)
[PASS] replay contents AND priorities restored (105 entries, prob sum 35.853409)
[PASS] all random streams resume identically (next draws [0.2763400329976994, 880700787, 0.34832991985182593])
episode counter resumes at 21, epsilon 0.1
```

### 阶段二：连续跑 vs 存档后重启跑

从同一初始状态出发，跑 2 个 episode → 存档 → **在全新对象里重建** → 再跑 3 个 episode：

```
continuous run  actions [[82,155,85,679,136],[329,243,224,17,241],[4,66,23,17,9]]
restarted run   actions [[82,155,85,679,136],[329,243,224,17,241],[4,66,23,17,9]]
[PASS] identical survey actions after restarting
[PASS] identical parameters (269.710479 vs 269.710479)
[PASS] identical replay size (130 vs 130)
[PASS] identical epsilon (0.09950100 vs 0.09950100)

OVERALL: resume is exact -- safe for the long run
```

**调查动作序列逐位相同**，这是最有说服力的一条：它同时覆盖网络权重、优化器、replay 优先级与全部随机流——任何一项没恢复对，动作序列都会分叉。

## 3. 顺带修掉的两个 bug

1. `resume_check.py` 里 `list(replay.buffer) == list(replay2.buffer)` 会抛
   `ValueError: truth value of an array is ambiguous`——replay 里每一条都是 numpy 数组。已改为 `deep_equal()`（逐元素比 numpy/tensor/嵌套结构）。
2. 用 PowerShell `Replace` 注入多行代码时，单引号里的 `` `r`n `` **不会**转义，被当字面量写进了源码，导致语法错误。已修复并改用文件方式打补丁。

## 4. 现状与下一步

| 项 | 状态 |
|---|---|
| 断点续跑 | ✅ **完成并验证**（每 25 episode 原子落盘，跨进程重启逐位一致） |
| `compute_loss=False` | 逐位等价但 **0.97×，不采纳**（你说得对：不为零提速付重跑成本） |
| 共享嵌入（`fast`） | 备选：更新 2.94×、episode 21 s → 6.19 s、整轮 58 h → 17.2 h；**float32 内同解但非逐位**，属新实现版本 |
| 16 线程 → 1 线程 | 免费，约 −20% |
| **padding＋mask 批量化** | **下一步**，按你的边界做：真实 16–63 节点混合 batch，对照 Q／损失／梯度／连续更新，再测完整 episode |

**批量化必须特别注意你指出的两点**（都已记下，不会预设结论）：
- `num_batches_tracked=0` 只说明**没有累积运行统计**，**不保证**批量计算的 BatchNorm 与逐图计算相同——同一批里的图会进入彼此的**即时均值/方差**；
- mask 必须覆盖**池化与填充节点**，不能只遮输入。

按你定的判据：对照通过且接近约 3 s/episode → 冻结优化版从头跑五种子；数值或决策明显改变 → 明确列为**新的批量化实现**，先做短程质量检查，**不接旧检查点、不称等价加速**。GPU 在批量化之后再定。
