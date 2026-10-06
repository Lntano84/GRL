# 交接记录（关机前）— 2026-09-22

**仓库根目录**：`C:\Users\windows\Desktop\im算法与基准阅读包\06_CCIM`
**ASCII 入口（推荐用这个，中文路径在某些控制台会被 GBK 错显成 `im绠楁硶涓庡熀鍑嗛槄璇诲寘`）**：
`C:\Users\windows\Desktop\ccim_work`（junction，同一份文件）
**不是 git 仓库，没有任何东西被推送。**

---

## 1. 正在跑的东西（关机会丢失）

`python -u scripts/shorthorizon.py --threads 1 --episodes 200`
- 后台 job id：`pwsh-2`
- 已经开始训练，预计总时长 **约 90 分钟**
- **关机即丢失。** 它只在**每个 (variant, seed) 跑完 200 episode 后**才落盘，中途死掉不保留该格。
- 恢复办法：重新执行上面这条命令即可（种子与设置都已写死在脚本里）。

**已经因为同类原因丢过三次进度**，恢复时注意：
1. 早期探针只热身 1 个 episode → `replay < 100` → **一次梯度更新都没触发**（测的是「无更新 episode」）；
2. `Tee-Object` 缓冲 → 中途死掉时输出全丢（现已改 `python -u`）；
3. `torch.save` 写在决策探针之后 → 训练跑完才崩，进度全丢（已把落盘移到探针之前）。

---

## 2. 已完成并落盘的结论

### 2.1 时代/成本链条（都在 CPU 1 线程、`replay>100`、真实更新计时下测得）

| 版本 | 每次更新 | 每 episode | 冻结预算 10,000 episode |
|---|---:|---:|---:|
| published（作者原样）@16 线程（生产实测） | — | 21.0 s | ≈ 58 h |
| published @1 线程 | 1170.0 ms | 13.04 s | 36.2 h |
| `fast`（共享嵌入）@1 线程 | 421.7 ms | 5.82 s | 16.2 h |
| **`batched`（padding＋mask）@1 线程** | **144.8 ms** | **3.07 s** | **8.5 h** |

- **线程数越多越慢**：1 → 2 → 4 → 16 线程分别为 1140 / 1390 / 1401 / 1667 ms 每次更新。**生产运行用的是默认 16 线程**，改 1 线程是免费的约 −20%。
- 更新耗时构成：**前向 75%、反向 26%**；replay 取样＋张量组装 0.4%、优化器 0.25%。
- profiler：**每次更新 554,927 次算子调用**，平均每次约 10 µs → **启动/Python 开销受限，不是算力受限**。
- 根因：作者 `dqn.py:647-653` 用列表推导**逐样本**调用，batch=100 被浪费；每样本 **6 次**图表示前向（`GraphTD3.forward` 每次调用跑两遍），共 **600 次/更新**。

### 2.2 等价性判定（分条，不可合并）

| 改动 | 数值地位 |
|---|---|
| `compute_loss=False`（关掉未使用的 `link_loss`/`entropy_loss`） | **逐位相同**，但 **0.97×，零提速 → 不采纳** |
| 共享嵌入（`dqn_opt.gradient_update_sarsa_fast`） | 与 published 差 **参数相对 4.0e-07**（float32 量级），**非逐位** |
| padding＋mask 批量编码器 | 表示层**通过**：`state_embed` 相对 1.78e-07、`node_embeddings` 5.32e-08 |
| padding＋mask 完整更新 | 梯度相对 6.65e-07，但**参数从第一步起漂移 2.9e-04**，每步约 +2.8e-04 近似线性 |

**第一步漂移的机制已查明**：Adam 首步是**符号步**（状态为空时 `Δθ ≈ −lr·sign(g)`），所以**每个参数张量的 max\|Δθ\| 恰好 = 学习率 1e-4**；任何梯度处于 float32 噪声水平的坐标可能**符号翻转**，该坐标各走一个 ±lr。**不是**「ε 主导分母导致连续放大」。10 步线性只是观察，**不据此推断长期轨迹**。

**因此 `batched` 是「经过对照的新批量化实现」，不是作者实现或 `fast` 的等价加速。**

### 2.3 断点续跑：**已完成并验证**

- `ccim/xplore/progress.py`：保存 online/target 网络、`critic_opt`（Adam 两矩）、replay **内容与优先级**、episode、epsilon、`val_curve`、以及**全部随机流**（`rng`、`seed_rng`、`replay.rg`、stdlib `random`、`torch`、`np.random` 全局）。
- `scripts/xplore_official.py` 已接入 `--resume`，**每 25 episode 原子落盘**（`*.tmp` + `os.replace`）。
- `scripts/resume_check.py` 两阶段**通过**：往返一致性（权重逐位、replay 内容与优先级、随机流下一抽样全同）；连续跑 vs 存档重启跑（**调查动作序列逐位相同**、参数 269.710479 相同、replay 130、epsilon 相同）。
- ⚠ 措辞已收紧：**动作相同本身不充分**（部分状态有差异也可能碰巧同动作）；是它与 replay、优先级、随机流的逐位一致**合起来**才支撑「可续跑」。
- ⚠ 仅含权重的 `official_seed*_ep*.pt` **不得**当续跑点，`progress.load()` 会检查 schema 拒绝。
- ⚠ **`shorthorizon.py` 还没接这套 `progress`**，这是它一崩就丢的根因。

### 2.4 官方架构低预算改编版（训练**未完成**）

- 作者五个组件全部组装可运行（14 处机械兼容补丁，见 `results/official_assembly.json`），**唯一缺 gensim**（pip 的镜像地址坏了；用 `--index-url https://pypi.org/simple` 可下载）。
- **偏差清单（7 条）**：负采样替换 `hs=1`（算法差异）、episode 2000 vs 10000、`emb_iters` 10 vs 50、**奖励重复实际 5 次 vs 作者代码实际 4 次**（行为修正）、串行替代 `Manager`、奖励用同目标替代实现、绘图/tensorboard 移除。
- **统一名称：官方架构低预算改编版**；主比较 = **改编版 − random** 与 **改编版 − degree_min**。
- 训练**跑了两次都被杀**（一次我主动停掉做无争用测速，一次 job runner `STATUS_CONTROL_C_EXIT`），**检查点为空**。
- 代码与论文不一致处（**按代码**）：`train.py:442` 把终止奖励 **除以 OPT**，所以训练信号**是被归一化的**（基准是 OPT 不是 CHANGE）；`--const_features 1` 时喂给编码器的是**全 1 节点特征**，DeepWalk 只从**动作表示**进入。

### 2.5 更早的探索式 IM 结果（已完成）

- **基线配对曲线**（3 张开发图 × 20 组 × 9 预算）：调查显著改善最终传播（1.29–2.02×）；**同权限下 T=5 未检出策略差异**；T=20 两项负向是**探索性、Holm 校正后不显著**（0.0169→0.2026、0.0489→0.5375）。
- **简化 Geometric-DQN 风格实现**：开发评估未建立超越简单启发式的收益（与 `degree_min` 打平 +0.17，p=0.92）。
- **训练贡献诊断**：相对未训练网络训练图 **+0.82**（p=0.0013）、验证图 **+1.60**（p<0.00001）均显著；`ep2000 − ep500` **未检出明确变化**。
- 与作者 `icm.py` 的**交叉核验最大相对差 1.85%**（**兼容性证据，不是实现相同的证明**）。

---

## 3. 关机后的恢复顺序

1. **重跑短程表**（唯一在飞的实验）：
   ```
   cd C:\Users\windows\Desktop\ccim_work
   $env:PYTHONUNBUFFERED='1'; python -u scripts/shorthorizon.py --threads 1 --episodes 200
   ```
   它产出：逐种子验证收益曲线、固定可见状态上的首选动作一致率与 Q 排序、两版 episode 实测秒数、批量版峰值内存。
2. **判定**：若验证收益明显恶化 → 停在批量版诊断；若走势相近 → 才决定是否用 `batched` 跑五种子。**200 episode 只能看有没有明显破坏，不能证明长期质量相同。**
3. **GPU** 等短程表出来后再决定。目前证据**不支持**为它降级 torch（本机 2.13.0+cpu，公开 cu128 索引最高 2.11.0），且该负载是**启动开销受限**，不是算力受限。
4. 若要走五种子：先给 `batched` 接上 `progress` 续跑（现在只有 `xplore_official.py` 有）。

---

## 4. 关键产物索引

| 文件 | 内容 |
|---|---|
| `CCIM_XPLORE_协议侦察.md` | 官方协议钉死（论文＋代码逐条） |
| `CCIM_XPLORE_基线曲线.md` | 3 图配对基线 + Holm 完整 12 行表 |
| `CCIM_XPLORE_训练贡献诊断.md` | epoch0/早期/选中/末期对照 |
| `CCIM_XPLORE_断点续跑.md` | 续跑实现与两阶段验证 |
| `CCIM_XPLORE_梯度更新耗时定位.md` | 线程数 + profiler |
| `CCIM_XPLORE_消除重复计算.md` | 两项改动的等价性与提速 |
| `CCIM_XPLORE_批量化前提.md` / `CCIM_XPLORE_批量化结果.md` | 批量化规格与结果 |
| `CCIM_XPLORE_第一步定位与短程.md` | 首步符号步定位（含本次失败原因） |
| `results/*.json` | 全部实测数据 |
| `tests/` | 104 条测试 |
