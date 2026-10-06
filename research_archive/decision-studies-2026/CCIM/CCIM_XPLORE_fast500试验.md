# fast @1线程 · 500 episode × 5 种子 开发试验（已启动）

## 0. 先回答「能否接着累计」：**不能，已从头跑**

你的条件是「已有 200-episode 完整续跑状态若能通过 schema 与随机状态检查，可接着累计」。实测结果：**没有可用的续跑状态**。

仓库里唯一的候选是 `results/official_checkpoints/official_seed0_ep0.pt`：

```
keys = ['actor_critic', 'target_actor_critic', 'critic_opt']
has schema: False | has replay: False | has rng: False
```

它只是**权重快照**（而且是未训练的 ep0）。短程检查留下的 `results/shorthorizon_ckpt/*.pt` 同样只有权重，**没有 replay、优化器状态或随机流**。

`progress.load()` 会检查 `schema`，这类文件会被直接拒绝——这正是当初设计闸门的目的。
**结论：按你的规则从头跑。**

## 1. 预先写死的计划（`results/xplore_official_plan.json`）

| 项 | 值 |
|---|---|
| 更新实现 | **`fast`**（共享嵌入；与作者版参数相对差 4.0e-07，非逐位，属**经过对照的优化实现**） |
| 预算 | **500 episode / 种子**，**5 个种子 (0,1,2,3,4)** |
| 检查点 | **ep0 / ep100 / ep250 / ep500**；**ep500 是本试验的固定终点** |
| 训练图 | `interact__ia-crime-moreno` / `ia-infect-dublin` / `ia-infect-hyper` |
| 验证图 | `interact__ia-enron-only` / `interact__rt-twitter-copen` |
| 线程 | **1**（实测 1 线程比默认 16 线程快约 20%） |
| 续跑 | **每 25 episode 原子落盘**，`--resume` 已启用 |
| 同权限协议 | 不变：前沿调查、T=5、IC p=0.1、100 样本贪心选 k=10、全局播种、完整图独立评估（1000 样本） |

**预计训练约 7.1 小时**（500 × 5 × 10.24 s = 25,600 s），4 个检查点 × 5 种子的验证另计。

## 2. 驱动改动（9 处，全部断言过锚点）

`xplore_official.py` 新增 `--variant {published,fast,batched}` 与 `--episodes`，检查点集改为 `(0,100,250,500)`，并在每个种子的模型构建后调用 `install_variant()`。已通过语法检查与 `--help` 验证。

`batched` 分支保留但**不用于本试验**——短程检查已判定它轨迹不同（固定状态首选动作一致率 5.6%），且真实混合下只快 1.19×。

## 3. 跑完要交什么

1. **五个种子的验证收益**（ep0/100/250/500 曲线）；
2. 用**冻结的同权限协议**与 **random**、**degree_min** 比较（degree_min 是按前轮结果选定的强启发式参照，如实注明）；
3. 据此**只决定一件事**：是否值得再投约 21 小时补足原预算。

**它不能当作完整的 2000-episode 结果，也不能当作独立投稿证据**；这是固定终点的开发试验。

## 4. 本轮明确记录为「尚未实测」

**GPU。** 未测。理由是批量版已停在诊断，而瓶颈是 episode 里两版**共有**的环境/奖励/嵌入开销，换 GPU 不改变这部分；且现有证据不支持为它把 torch 从 2.13.0 降到 ≤2.11.0。**记录为未实测，不是「已验证无收益」。**

## 5. 已复核并更正的数字

- `fast` 在**真实三图混合**下为 **10.24 s/episode**（→ 28.4 h / 10,000 episode），不是单图基准的 5.82 s（→ 16.2 h）。本轮 500 episode 的 7.1 h 估计用的是前者。
- **8 小时跑完原预算在这一族实现上不可达。**

## 6. 状态

进程存活，已过 OPT 计算、进入训练。日志：`results/official_fast500_log.txt`。
关机后直接重跑同一命令即可从最近一次 25-episode 检查点继续：

```
cd C:\Users\windows\Desktop\ccim_work
$env:PYTHONUNBUFFERED='1'; python -u scripts/xplore_official.py --resume
```
