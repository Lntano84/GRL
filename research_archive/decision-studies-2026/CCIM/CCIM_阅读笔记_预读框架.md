# Complex Contagion IM / 避免影响力过度暴露 —— 预读框架

> 状态：**预读框架（尚未读到原文）**。
> 本文中凡标 ✅ 的为可确证事实；标 ⚠️ 的为基于题目、摘要级信息与邻近文献的**推测**，必须在读到 PDF 后逐条核对或删除。
> 生成时间：2026-09-13，基于公开检索到的元数据。

---

## 0. 元数据与目录信息错配（需先处理）

### 0.1 目标论文（用户指定阅读）✅

| 项 | 值 |
| --- | --- |
| 标题 | Complex Contagion Influence Maximization: Leveraging threshold-dependent model for avoiding influence overexposure in online social networks |
| 期刊 | *Information Sciences*（Elsevier） |
| DOI | [10.1016/j.ins.2026.123375](https://doi.org/10.1016/j.ins.2026.123375) |
| 官方入口 | [ScienceDirect S0020025526003063](https://www.sciencedirect.com/science/article/abs/pii/S0020025526003063) |

**同源会议版（很可能是本文前身，值得对照阅读）** ✅

- 标题：*Target Influence Maximization Against Overexposure Under Threshold-Dependent Model in Online Social Networks*
- 出处：COCOON 2024（30th International Conference on Computing and Combinatorics，上海，2024-08-23~25），Proceedings Part II
- DOI：[10.1007/978-981-96-1093-8_10](https://dl.acm.org/doi/10.1007/978-981-96-1093-8_10)

> 注意标题差异：会议版强调 **Target**（面向目标集合的 IM），期刊版标题去掉了 Target 但保留 threshold-dependent + overexposure。
> 核对点：期刊版是"会议版扩充"还是"问题被重新定义"？这直接决定实验对比对象。

### 0.2 目录内已有文件是**另一篇论文** ✅

`CCIM_论文信息.md` 写的是 **Capacity Constrained Influence Maximization in Social Networks**（ACM 2023，DOI 10.1145/3580305.3599267）。
二者不是同一篇：

| | Capacity-Constrained IM (ACM 2023) | 本篇 (Inf. Sci. 2026) |
| --- | --- | --- |
| 约束来源 | 节点/传播路径的**容量**上限 | **过度暴露**（曝光过量导致效果衰减） |
| 触发模型 | 标准 IC 家族 | **阈值依赖模型**（threshold-dependent / 复杂传染） |
| 目标函数性质 | 容量约束下可能破坏单调性 | 过度暴露使收益**非单调** |

**结论**：`06_CCIM` 这个目录名（CCIM）对应的是"容量约束"那篇，而用户要读的是"复杂传染 + 过度暴露"这篇。两篇都属于"标准 IM 的单调性/次模性前提被打破"这一大类，但机制不同，**不能互相引用为同一工作**。建议把本篇另存目录或改名，避免后续引用串号。

### 0.3 当前环境的取证限制 ✅

以下站点在本环境中被网络策略阻断（DNS 解析到非公网地址 / TLS 连接被关闭），实测均失败：
`sciencedirect.com`、`dl.acm.org`、`link.springer.com`、`doi.org`、`api.crossref.org`、`api.openalex.org`、`semanticscholar.org`；shell 直连（`Invoke-WebRequest`、`curl.exe`）同样失败。
→ 因此本笔记**没有**原文摘要，**不包含**该论文的公式、算法名与实验数字。任何声称"该论文提出算法 X、近似比为 Y"的表述在读到原文前都不成立。

---

## 1. 领域坐标系：这篇论文站在哪里 ✅/⚠️

### 1.1 标准 IM 的两个默认前提

经典 IM（Kempe–Kleinberg–Tardos）的整套理论建立在两条性质上：

1. **单调性**：多加种子不会让影响力变小（S ⊆ T ⇒ σ(S) ≤ σ(T)）。
2. **次模性**：边际收益递减。

有了这两条，贪心算法才有 (1 − 1/e) 近似保证；IMM/SUBSIM 这类 RR 集（reverse-reachable set）方法才能用无偏估计器把复杂度压到近线性。

### 1.2 本论文的攻击点：**过度暴露（overexposure）**⚠️

信息在社交网络上重复触达同一个人，并非"越多越好"。过度曝光会引发反感、屏蔽、免疫，使**边际收益转负**。
一旦边际收益可为负：
- 单调性失效 → 贪心的 (1 − 1/e) 保证**不再成立**；
- 次模性失效 → 贪心的近似比论证**整体崩塌**；
- "种子越多越好"的直觉失效 → 存在一个**最优种子规模/强度**，而非越大越好。

这正是标题里 "avoiding influence overexposure" 的动机。⚠️（过度暴露的具体函数形式必须核对原文。）

### 1.3 与"目标 IM"（Target IM）的关系 ⚠️

会议版标题含 **Target**：目标不是全网激活数最大，而是**指定目标集合**被激活/被说服的效果最好。
与 overexposure 结合后，问题的直觉是：
- 对目标集合，曝光存在**最优强度区间**——不足则说服不了，过量则引起反感；
- 非目标节点被"误伤"式曝光同样消耗预算且可能产生负外部性。
⚠️ 期刊版是否保留 target 设定需核对。

### 1.4 邻近文献（可作背景与 baseline 线索）✅

| 工作 | 关系 |
| --- | --- |
| *Mitigating Overexposure in Viral Marketing* (AAAI/arXiv:1709.04123) | overexposure 作为**收益递减/负值**的早期形式化，重要前置 |
| *Cascades and Overexposure in Social Networks* (AAMAS 2022) | 级联与过度暴露的建模 |
| *Relieving Overexposure in Information Diffusion Through a Budget Multi-stage Allocation* (ACM TOIT 2025, [10.1145/3708537](https://dl.acm.org/doi/10.1145/3708537)) | 多阶段预算分配视角缓解过度暴露，**同期竞品，很可能在本篇实验对比里** |
| 经典 LT / 复杂传染（Centola & Macy） | threshold-dependent 模型的理论源头 |

---

## 2. 待核对问题清单（拿到 PDF 后按序回答）

### A. 模型与问题定义（对应目标 1）

- [ ] A1. **阈值模型的确切形式**：线性阈值（LT）？一般阈值？阈值是常数、按度数比例，还是从分布采样？是否有"阈值随时间/曝光次数变化"的机制？
- [ ] A2. **"暴露"如何计数**：统计邻居激活次数？统计收到的消息条数？是否有**跨轮次累积**的曝光计数？
- [ ] A3. **overexposure 如何形式化**（本篇最核心的一条）：
      - 是激活概率的**饱和函数**（先增后减）？
      - 还是"曝光次数超过上限则节点**免疫/永久拒绝**"的硬约束？
      - 还是目标函数里显式的**惩罚项**（如 −λ·overexposed 节点数）？
- [ ] A4. **目标函数**：单调吗？次模吗？**证明在哪一节**？（若声称次模，必须找到证明；若已非次模，看作者用什么替代性质，如 weak submodularity / 有界边际增益比 γ。）
- [ ] A5. **问题是否 NP-hard**，难度来自哪里（max coverage 归约？）。
- [ ] A6. 是否有 **Target 集合**？还是全网激活？
- [ ] A7. 预算是"种子个数 k"还是"总曝光量/多阶段预算"？

### B. 算法与理论（对应目标 2）

- [ ] B1. 主算法名称与骨架（贪心？RR 集？采样？启发式？RL/学习式？）。
- [ ] B2. **近似比**：有还是没有？如果是 (1−1/e)，**凭什么**——单调次模被破坏后这个界从哪来（是否加了约束使问题重回次模，或只对受限子类成立）？
- [ ] B3. **估计器**：若用 RR 集，在overexposure 语义下，标准 RR 覆盖等价关系是否仍成立？（详见 §3）
- [ ] B4. 时间复杂度与可扩展性声明，实验图规模上限。
- [ ] B5. 是否有下界/不可近似性结果。

### C. 实验与 baseline（对应目标 3）

- [ ] C1. 数据集（几个、规模、来源）。
- [ ] C2. baseline 列表——**必须确认是否包含**：Degree Discount、IMM/SUBSIM、经典贪心、以及 TOIT 2025 那篇 overexposure 竞品。
- [ ] C3. 评价指标：激活数？目标集合命中率？**overexposure 指标本身**如何度量？
- [ ] C4. 参数敏感性（阈值分布、曝光上限、k）。
- [ ] C5. 是否报告**方差/置信区间**与运行时间。

### D. 作为 GRL 认证 oracle 的可行性（对应目标 4）

- [ ] D1. 该问题是单调次模吗？（→ 决定能否用 IMM 式采样做 certified oracle）
- [ ] D2. 若否，是否存在**可证明的**近似算法可作为 oracle；还是只能退回 Monte Carlo？
- [ ] D3. 该算法在你们 GRL 框架里是**被认证的对象**还是**认证工具**？
- [ ] D4. 目标函数不同（overexposure-aware vs 你们的定义）时，能否直接复用其 oracle？

---

## 3. 承接原笔记的关键疑问：RR 估计器还有效吗？⚠️

原 `CCIM_论文信息.md` 提的疑问是对的，且在本篇上**更尖锐**：

> 普通 IC 模型中"一个节点被多条路径触达后只算一次"的 RR 覆盖等价关系可能需要重新证明。

推理链（待原文验证）：

1. 标准 IMM 的无偏性依赖 **RR 集覆盖 ⇔ 影响力** 的等价：随机取一个目标节点 r，反向 BFS 得到的集合 R，满足 σ(S) = n · Pr[S ∩ R ≠ ∅]。这一步**只用到了"激活与否"的二值性**。
2. 一旦引入 overexposure，节点收益**依赖被触达的次数/强度**，不再是布尔量。那么：
   - "覆盖即收益 1"的等价**失效**；
   - RR 集需要携带**多次触达的重数信息**（多重集？加权计数？），或其定义要改成能记录曝光量的反向结构；
   - 标准 IMM 的采样与阈值估计**不能默认无偏**。
3. **因此**：若本篇声称用了 RR 类方法并保持 (1−1/e)，必须找到它对等价关系的**重新证明**；找不到就说明它要么换了估计器，要么把 overexposure 建模成了不破坏覆盖语义的形式（例如只惩罚**种子侧**的重复，而非接收侧）。
   → **这是读这篇论文时最该盯住的技术点。**

---

## 4. 读完 PDF 后的产出计划

1. 用真实内容替换本笔记中所有 ⚠️ 段落，补齐公式与符号表。
2. 写出该论文的 **模型定义 → 目标函数 → 性质证明 → 算法 → 实验** 单页精读卡。
3. 与会议版（COCOON 2024）做差异对照表。
4. 给出 baseline 选用建议与复现风险点。
5. 回答 §3 的 oracle 可行性问题，并回填 `CCIM_论文信息.md` 的相应结论。
