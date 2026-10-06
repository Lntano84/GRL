# P0 模型合同修复报告

> 日期：2026-09-17
> 触发：外部审查提出两个 P0 反例
> 脚本：`scripts/audit/verify_p0_counterexamples.py`、`scripts/audit/measure_state_fix_impact.py`
> 结论：**两个反例都成立,已修复并固化回归测试。修复后此前所有 overexposure 数值结论作废,必须重做。**

---

## 1. P0-1：正激活节点永不转负 —— **确认成立,已修复**

### 反例（独立复现，无 MC）

图 `s→a=0.4`、`s→b=0.5`、`b→a=0.4`，窗口 `a=[0.2,0.6]`、`b=[0.2,0.9]`，种子 `{s}`。

```
修复前：ever_positive = {s,a,b}   negative = {}      spread = 3     ❌
修复后：ever_positive = {s,a,b}   negative = {a}     spread = 2     ✅
        final positive = {s,b}
```

### 根因

旧代码在每个节点**首次转移**时就把它加入 `settled`，并在评估时 `if state[node] != INACTIVE or node in settled: continue`。
**正激活节点因此再也不会被重新评估**，`δ` 继续累积但状态被冻结，`positive` 集合只增不减。

**这系统性地低估了过度暴露**：每个种子自身贡献的 +1 永远计入，而它造成的伤害永远不扣。

### 修复

- 区分三个集合：`positive`（**期末**正激活，即 `spread` 的口径）、`negative`、`ever_positive`（**曾经**正激活，含后来转负者）；
- 非种子节点每轮**重新评估**，若 `δ > θ^τ` 则从 `positive` 移出并进入 `negative`；
- 种子的影响只在其**首次转正**那一轮传播（`ever_positive` 驱动 `delta`，与论文 §3.1 一致）。

---

## 2. P0-2：`tau = 1` 边界 —— **确认成立,已修复**

### 反例（独立复现）

图 `s→v=1`，窗口 `v=[0.2, 1.0]`，种子 `{s}`。

```
修复前：spread = 1，v 属于 negative     ❌（δ=1 落在 [0.2,1] 内，按规则应激活）
修复后：spread = 2，v 属于 positive     ✅
```

### 根因（审查判断正确）

旧代码有 `current >= kappa and current < 1.0` 的额外守卫，理由是"`2δ(1−δ)` 在 δ=1 处为 0"。

**这个理由混淆了两件事**：`2δ(1−δ)` 是**采样窗口分布下**的窗口事件概率；状态规则是在**窗口已经抽定**之后的条件判断。不能用前者的边缘概率去决定后者的状态边界。窗口抽定为 `[0,1]` 时，`δ=1` 在窗口内。

### 我此前的辩护是错的

我曾在 `overexposure.py` 的注释里写过：

> *"Without the guard the literal rule would activate a node whenever tau == 1, contradicting the very probability the model was derived from."*

**这句话是错的**，已随修复删除。审查指出的正是这一点。

---

## 3. ⚠️ 修复对既有结论的影响：**全部作废**

`measure_state_fix_impact.py` 用**同一图、同一种子集、同一窗口、同一 trial 序号**，
对比"旧规则（在本脚本内局部重实现）"与"新规则"：

| k | 旧 spread | 新 spread | 变化 | 旧负边际占比 | 新负边际占比 |
| --- | --- | --- | --- | --- | --- |
| 1 | 299.35 | **111.44** | **−62.8%** | 34.8% | 27.1% |
| 2 | 325.71 | **123.34** | **−62.1%** | 36.1% | 42.3% |
| 5 | 358.52 | **137.28** | **−61.7%** | 40.2% | 35.2% |
| 10 | 365.95 | **143.43** | **−60.8%** | 39.8% | 36.4% |

### 这意味着什么

**Congress-Twitter 根本不在 `k ≥ 5` 饱和。** 旧数字（366 → 366 → 366）之所以"爬升后持平"，
正是因为正激活节点永不转负，`spread` 被单调累加到接近某条错误的上限。
修正后真相是：**过度暴露实际摧毁了一个种子的大部分可达性（366 → 143）**。

### 因此以下结论**全部作废，必须重做**

| 作废的结论 | 出处 |
| --- | --- |
| "Congress 在 `k≥5` 饱和,方法间无差别" | 阶段 4 §3 |
| "可用区间只有 3/18 格子(congress k=1,2 等)" | 阶段 4b §5 |
| "degree gap 在饱和区间为负是噪声" | 阶段 4 §3 |
| 阶段 5 的所有 spread / gap / 成本数字 | 阶段 5 §2 |
| 阶段 5 的"learning 输给 degree"（结论方向可能仍成立,但**数值与排序证据需重取**） | 阶段 5 §3 |
| 所有早期 Gate 1c / Gate 2 / Gate 3 的 spread 与边际增益数值 | `docs/GATE*_REPORT.md` |

**仍然成立**（不依赖状态机）：
- P0-3 已撤回的 RR 论证（本来就作废）；
- 数据集"负边际增益存在"这一**定性**事实（新旧都为正，占比 27~42%）；
- `ever_positive` 驱动 `delta` 这一机制本身。

---

## 4. 已固化的回归测试

新建 `tests/test_overexposure_state_machine.py`（9 个测试）：

- `test_p0_1_positive_node_can_later_be_overexposed` —— 反例本身；
- `test_p0_2_delta_one_with_tau_one_activates` —— 反例本身；
- `test_p0_2_boundary_is_literal_on_both_sides` —— `δ=κ` 激活 / `δ>τ` 转负；
- `test_seeds_are_never_overexposed`；
- `test_ever_positive_is_the_set_that_drives_exposure` —— 契约直述；
- `test_two_seeds_acting_in_the_same_round_overexpose_immediately` —— 记录我在写测试时**三次**犯的同一个错误：把贡献者都放在种子集里（或同一距离），导致它们同一轮到达，`m` 从未正激活；
- `test_positive_then_negative_keeps_promoting_its_neighbours` —— 加一跳延迟后的正确构造；
- `test_spread_counts_final_state_not_history`。

**改写** `tests/test_overexposure.py::test_delta_one_makes_positive_activation_impossible`
→ `test_delta_one_inside_window_activates`。
该测试原本在**固化错误规则**（"maximal exposure is maximal rejection"）。
**修正测试而非迁就代码**——这正是审查意见要求的。

**测试总数 139 → 149 passed。**

---

## 5. 我在本轮犯的错误（记录在案）

写这批回归测试时，我**连续三次**把"先正后负"的场景构造错：

1. 第一次：用一个种子加两条 `add_edge(s,m)` —— DiGraph 不允许平行边，第二次调用覆盖第一条；
2. 第二次：用两个种子 `s1,s2` —— 都在第 1 轮，`m` 一次收到 0.6 > τ，**从未正激活**；
3. 第三次：`s→p`、`s→q` —— `p`、`q` 同轮点亮，同样问题。

**每次都误以为是代码 bug，实际是我的测试模型错了。** 最终靠"给其中一条路径加一跳延迟"才构造正确。
这说明 `ever_positive` 的语义（"曾经正激活"而非"曾被评估"）很容易被误用，值得保留这些测试。

---

## 6. 尚未处理的事项（P0-3 / P0-4 / P1）

审查意见中其余部分我**尚未处理**，列出以免遗漏：

| 编号 | 内容 | 状态 |
| --- | --- | --- |
| **P0-3** | "RR 集恒为空"不是合法的 RR 检验；论文题目 *When Reverse Reachability Has No Domain* 必须撤回；替换为更窄的论证（非负覆盖表示 ⇒ 单调次模；而目标存在非单调实例） | ⏳ 未处理，**论文题目需改** |
| **P0-4** | surrogate 界隙实验未对齐目标集 D；`0/180` 不能作为反驳；我"逐样本不推出期望"的说明本身是错的；`Aτ ⊆ Aκ` 应可证 | ⏳ 未处理，需先统一目标集定义 |
| **P1-1** | `κ~2x−x²`、`τ~x²` 的分布不是两份均匀阈值 LT，`0.16+0.16<0.64` 已违反子模性；"可分别跑 RR"不成立 | ⏳ 未处理 |
| **P1-2** | 经验残差阈值不构成可靠性保证；反例 预测 `[10,9,0]` / 真实 `[10,9,100]` 会错误认证 | ⏳ 未处理 |
| **P1-3** | 预算/成本/划分/状态指标共 9 项混淆 | ⏳ 未处理 |

---

## 7. 下一步（按审查建议的顺序）

1. **先冻结模型合同**（本轮已做 P0-1/P0-2，还需统一目标集 D、种子资格、`≤k` 语义）；
2. **重建参照**：在修正模型下重算 spread / 边际 / 目标集口径；
3. **无学习自适应 MC 基线先行**：同批世界内比较候选增益，统计区间逐步排除，明确错误预算；
4. **解析分只决定评估顺序**（degree / delta2 / random 接同一验证框架）；
5. **学习只在公平消融中证明增量**；
6. **摊销必须计入离线成本** `C_train + Q × C_query`。

**预注册门槛（审查建议，需 PI 确认后冻结）**：
在 ≥3 张预先选定图的留出查询上，质量损失 ≤1%（相对高精度 MC 参考；目标值接近零时改用绝对容差），
且相对**最强无学习比较方法**节约 ≥30% 在线级联成本；报告配对置信区间、失败查询比例与运行时间。
