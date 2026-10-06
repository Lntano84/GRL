# Inf. Sci. 2026 (CCIM) 实验设置核对与对标计划

> 日期：2026-09-16
> 来源：`_extracted/1-s2.0-S0020025526003063-main.txt`（Information Sciences 744 (2026) 123375）
> 用途：DASFAA 2027 投稿必须能在**同一批数据集**上给出可比结果

---

## 1. 论文实际用的数据集（与我们现在用的完全不同）

| 网络 | `|V|` | `|E|` | `⟨k⟩` | 目标集 `|T|` | 性质 |
| --- | --- | --- | --- | --- | --- |
| **Congress-Twitter** | 333 | 13,289 | **27.98** | 80 | 稠密 |
| **Wiki-Vote** | 889 | 2,914 | 3.28 | 889（全部节点） | 即退化为传统 IM |
| **Occupywallstnyc** | 3,609 | 3,936 | 1.09 | 2,000 | 稀疏 |
| **Trust Bitcoin-Alpha** | 3,783 | 24,186 | 6.39 | 1,000 | 较稠密 |

⚠️ **我们现在跑的是 NetHEPT（15,233 节点 / 32,235 边），论文没有用。**
**八个 Gate 实验全部基于 NetHEPT，因此目前的结论无法直接对标论文。**

**好消息**：四个数据集都很小（最大 3,783 节点），
**完全可以跑精确 MC-greedy 与自适应基线** —— 这正是论文没做的（见下）。

---

## 2. 论文的基线里没有精确求解器

论文 §7.3 的对比算法（原文）：

> *"we compare proposed upper bound (UB) method with several representative algorithms
> for influence maximization problem, including **Random selected, Max degree, IMRank [36],
> PageRank [37] and CELF [38]** algorithms."*

- `Random` / `Max degree` / `IMRank` / `PageRank` 都是**启发式**；
- `CELF` 是**对单调子模函数**做 lazy greedy 的加速技巧 —— 而论文自己的 Lemma 2 只保证
  `σ^κ(·)`、`σ^τ(·)` 单调次模，`σ(·)` 本身不是；
- **没有任何一个是精确（或 `(1−1/e)`）求解器**；
- **没有报告 spread 数值**，只有 Fig. 9 的曲线。

⇒ **"最优解离基线有多远"在论文里是未知的。** 这是一个可以直接填的空白。

---

## 3. 论文自己承认的非单调性（与我们 Gate 1c 的差异）

论文 §7.3 (i) 原文：

> *"As evidenced by the non-monotonic property of the objective function, the overall
> influence does not invariably increase with the number of seed nodes. Notably,
> experiments conducted on the **Occupywallstnyc** dataset (Fig. 9(c)) demonstrate that the
> objective value exhibits **fluctuations across all algorithms**... once the seed set
> reaches a certain scale, the marginal contribution of new nodes diminishes, and
> **in some cases, may even lead to a decline in overall influence**."*

| | 论文 | 我们（NetHEPT） |
| --- | --- | --- |
| `σ(S)` 随 `|S|` | **会下降**（Occupywallstnyc） | **单调递增**，边际 −96.5% 但不转负 |
| 逐候选负边际 | 未量化 | **13/1440**，仅在权重放大 3~6 倍时 |

**推断**：非单调性的显现**依赖图的稠密度**。
Occupywallstnyc `⟨k⟩=1.09`（极稀疏，近似森林）却出现波动；
NetHEPT `⟨k⟩≈4.2` 且入权归一化，反而不波动。
**这个"哪种图会非单调"的问题本身就是论文没回答的问题，可以作为 §3 的分析素材。**

---

## 4. 论文自己承认的算法弱点

论文 §7.3 (ii) 原文：

> *"its performance is less favorable on the **Trust-Bitcoin-Alpha** dataset.
> The primary reason lies in the relatively **high edge density**...
> Such densely connected network structures significantly increase the complexity of the
> proposed algorithm, as the candidate solution space expands sharply with each iteration,
> leading to **exponential growth in computation time**.
> Moreover, the abundance of local optimal solutions in high-density networks makes the
> algorithm more prone to being **trapped in suboptimal solutions**...
> In addition, the excessive number of edge connections can **amplify cumulative errors**
> in the evaluation of the objective function."*

⇒ 论文亲口承认 **UB 在稠密图上退化**（计算爆炸 + 局部最优 + 评估误差累积）。
**Congress-Twitter `⟨k⟩=27.98` 比 Trust-Bitcoin-Alpha 更稠密**，但论文说它表现"更好" ——
这两句话之间存在张力，值得核实。

---

## 5. 理论保证（弱）

| 定理 | 内容 |
| --- | --- |
| Theorem 5 | IGA 的近似比只有 **`γ/k`** |
| Theorem 1 | NP-hard（归约到 set cover） |
| Theorem 2 | `σ(S)` 非单调 |
| Theorem 3 / 4 | 固定窗口 / 任意窗口下均**非次模也非超模** |
| Lemma 2 | `σ^κ(·)`、`σ^τ(·)` 单调且次模 |
| **Theorem 6** | **`λ(S) ≥ σ(S)`** |
| Theorem 12 | 仅在**内向树**上给出 `P_t(u,S)` 的近似闭式 |

摘要（第 128~129 行）称 UB "achieves an expected approximation ratio of
`σ^κ(S_k) − (e−1 − c_τ...)`" —— **这个表达式本身就是一个单调代理 `σ^κ(S_k)` 上的比值**，
进一步印证：**论文的求解路径绕开了非单调目标**（与 `docs/GATE1_REPORT.md` Gate 1d 的结论一致）。

---

## 6. 对标实验清单（按优先级）

| 优先级 | 动作 | 说明 |
| --- | --- | --- |
| **P0** | 取得四个数据集（Congress-Twitter / Wiki-Vote / Occupywallstnyc / Trust Bitcoin-Alpha） | 否则无法对标；均为公开数据集 |
| **P0** | 在四图上复现 **Gate 1a''（`degree_pool` 条件排序）** | 验证 +29% 的结论是否跨图成立 |
| **P0** | 在 **Occupywallstnyc** 上复现 **Gate 1c（饱和曲线）** | 论文说它会波动，我们必须能复现或反驳 |
| **P1** | 实现 **精确 MC-greedy + 自适应贪心**，作为论文缺失的 oracle | 量出"基线离最优有多远" |
| **P1** | 对 `σ^κ` / `σ^τ` 真跑 RR，量化 **`λ(S) − σ(S)` 的界隙** | 把 Gate 1d 的"RR 只能解上界"做成定量实验 |
| **P2** | 复现 UB / IGA / Max_Degree / IMRank / PageRank / CELF | 论文 baseline |
| **P2** | 在 **Trust Bitcoin-Alpha** 与 **Congress-Twitter** 上测稠密图退化 | 验证论文自承的弱点 |

---

## 7. 对 GRL 提案的影响

1. **不能再用 NetHEPT 作为唯一实验图**。DASFAA 审稿人会直接问"为什么不用原论文的数据集"。
2. **四图都很小（≤3,783 节点）**，这是**优势**：可以做精确 oracle，
   而 GRL 的卖点可以定位成 **"用学习替代不可负担的精确搜索"** ——
   在论文的图上精确搜索本来就该可行，但论文没做，说明他们的 UB 连小图都没跑出精确对比。
3. **稠密图退化是论文自承的弱点**，且 `⟨k⟩=27.98` 的 Congress-Twitter 属于稠密区间。
   如果 GRL 在稠密图上比 UB 更稳，这是一个论文自己递过来的卖点。
