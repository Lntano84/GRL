# FRAME相关论文阅读包：六篇优先读物

2026-10-17。目的：学习相近工作的贡献定位、机制与实验设计，明确FRAME与已有研究的联系。新增训练0，新增回放0，未改变任何冻结结果或论文原稿。

阅读范围：五篇取得PDF并重点核对方法/实验，实验页有可视预览；ESWA2024仅取得出版方摘要、亮点与书目信息。正式发表与预印本状态分别标注。检索覆盖规划/搜索元控制与电网拓扑控制两个方向。

## 1. 管计算的会议工作：Stop! Planner Time（AAAI2024）

Budd, Lacerda, Hawes. Stop! Planner Time: Metareasoning for Probabilistic Planning Using Learned Performance Profiles. AAAI38(18),20053–20060. https://doi.org/10.1609/aaai.v38i18.29983

**相近处：** 付出额外规划计算能否改善最终执行，以及何时停止。用算法特征和任务上下文学习性能信息，避免假设随时免费知道当前方案的真实质量。

**方法/实验：** WBRTDP上学习停止和调权；深海寻宝、赛车场两个随机MDP任务，8个训练种子、1000个留出问题。比较便宜MidBound、昂贵PolicyEval及NoFeatures/NoContext/NoTuning；以状态访问量定义计算预算。精确评估带来的额外计算计入目标。

**学习点：** 让每个消融回答一个贡献问题；展示计算代价变化时的停止行为。FRAME可学习它的证据组织，但现有固定核查没有学得性能模型，亦未复现其WBRTDP任务。

阅读位置：正文Experiments、Results，PDF5–7页；本地sources/AAAI2024_StopPlannerTime.pdf。

## 2. 轻量在线反馈控制：Online Performance Prediction（IJCAI2018）

Svegliato, Wray, Zilberstein. Meta-Level Control of Anytime Algorithms with Online Performance Prediction. IJCAI2018,1499–1505. https://www.ijcai.org/proceedings/2018/208

**相近处：** 直接利用当前运行反馈决定继续计算是否值得，减少离线建模负担。

**方法/实验：** 在线非线性回归预测性能，构建myopic/nonmyopic停止控制。与离线性能轮廓方法对比，覆盖TSP、JSP、QAP和机器人；TSP报告50–90城市，JSP、QAP也有不同规模，通常每设置50实例；另有三地图仿真及真实机器人案例。

**学习点：** 简单模块可通过清楚的问题假设、运行成本和应用效果形成贡献。适合学习任务规模表和真实执行案例；不能把“用在线反馈停止”本身当FRAME首创。

阅读位置：正文§5及表1–4；本地sources/IJCAI2018_OnlinePerformance.pdf。

## 3. 同领域最接近的工程与效率论文：Managing Power Grids（Energy and AI2023）

Lehna et al. Managing power grids through topology actions: A comparative study between advanced rule-based and reinforcement learning agents. Energy and AI14,100276. https://arxiv.org/abs/2304.00765

**相近处：** 电网仿真评估、分组候选搜索、规则/学习混合控制，以及运行时间。

**方法/实验：** 增加N-1策略与拓扑恢复；24个场景、30个环境随机种子、六种控制器。逐场景存活、分布、逐动作时间和拓扑变化案例一并呈现。时间只在三方法共同存活的步骤上比较。

**直接重叠：** 原Tutor已先查62个候选，找不到合适动作才查其余146个；改进版再加入300个N-1候选。因此“候选分阶段/先查小集合”已有明确前例。学习版主要利用策略加速508候选的搜索。

**学习点：** 共同存活时域计时、逐决策延迟分布、控制质量相近时的计算对比。FRAME需突出已获得可用前缀动作后的进一步核查、停止依据及别名补查的具体增量。

阅读位置：正文§6–7，尤其本地预印本PDF第11页图5/6和段落；本地sources/EnergyAI2023_ManagingGrids.pdf。

## 4. 同领域结构分解：Progressive Decision-Making（ESWA2024）

Liu et al. Progressive decision-making framework for power system topology control. Expert Systems with Applications235,121070. https://doi.org/10.1016/j.eswa.2023.121070

**目前核对范围：出版方摘要/亮点，无全文。** GCN提取局部与全局特征，按“是否干预→定位关键节点→母线配置动作”逐级控制；摘要报告两个电力系统及多种RL对照。

**学习点：** 如何将电网动作层次映射成有名称的模块。“局部/全局结构”“多阶段决策”已有工作，FRAME应解释结构信息如何用于分配已购买与待购买的预测查询。当前无法据摘要补出具体电网规模、种子数或消融细节。

## 5. 最新且高度相关：Gibbs Priors（2026预印本）

Dogoulis, Cordy. Physics Informed Reinforcement Learning with Gibbs Priors for Topology Control in Power Grids. arXiv:2604.01830v1,2026-04-02. https://arxiv.org/abs/2604.01830

**身份：** 本次只核实arXiv预印本，未确认正式会议/期刊，不能标作已发表会议SOTA。

**方法/实验：** 危险状态才干预；先训练GNN预测动作后过载，再以Gibbs先验选Top-K并重加权PPO。case14/36/118，报告奖励、存活、决策毫秒。候选库209/66978/73357，K为64/512/512；它是真实多电网覆盖，但电网、动作量和运行约束同时变化。

**关键读数：** case118方法/LJN奖励84643.5/137420.6、存活435.4/717.6、耗时32.405/81.495ms。更快同时质量更低，不能按同质量提速理解。正文case36声称864步评价，表II平均存活却超过864，复现前需核对；不据此否定全文。

**学习点：** 候选规模、实际决策时间、质量分开呈现；FRAME与其区别是基于实际在线仿真反馈控制查询，当前不依赖新训练的风险代理/PPO。没有共同协议复现，不能比较双方速度百分比或宣称胜过该法。

阅读位置：§III–IV、表I–III，本地PDF5页；sources/arXiv2026_GibbsPrior.pdf。

## 6. 计算分配的经典来源：Rational Lazy A*（IJCAI2013）

Tolpin et al. Towards Rational Deployment of Multiple Heuristics in A*. IJCAI2013,674–680. https://arxiv.org/abs/1305.5030

**相近处：** 决定什么时候值得付费计算昂贵启发函数，采用简单的价值/后悔估计；属于FRAME“计算选择”思想来源，不作为近期SOTA标签。

**方法/实验：** Lazy A*延迟计算昂贵启发式，Rational Lazy A*允许跳过计算。对比单启发式、直接取max和Sel-MAX；报告规划问题覆盖、共同求解集合几何平均时间、跳过昂贵计算的比例、展开/生成节点。15-puzzle部分明确展示节省计算却几乎没有实际提速的情形。

**学习点：** 同时呈现“省了多少次昂贵计算”和“总时间实际省多少”，并用计算组成解释两者差别。其最优搜索性质依赖原问题假设，不能移植成FRAME的闭环质量保证。

阅读位置：实验§5、表4–6；本地sources/IJCAI2013_RationalLazy.pdf。2018年Artificial Intelligence256,181–210有相关扩展版，本轮未通读该扩展版。

## 阅读优先级与我方待回答的问题

| 优先级 | 读物 | 我们应从中回答什么 |
|---|---|---|
| 1 | Lehna2023 | 与已有分组候选搜索相比，FRAME具体多了什么、为何保持质量？ |
| 2 | Budd2024 | 如何把计算控制的贡献拆成可检验的问题？ |
| 3 | Gibbs2026 | 与学习代理筛选相比，我们的质量、训练需求、评估成本边界是什么？ |
| 4 | Svegliato2018 | 如何把轻量在线反馈模块写成清楚的算法贡献？ |
| 5 | Liu2024 | 结构分解前例覆盖到哪里，哪些属于通用已有思想？ |
| 6 | Tolpin2013 | 为什么少算不一定等比例加速，怎样报告真实收益？ |

优先学习不等于全部移植成新对照。当前没有授权或启动新模型/训练/预算扫描。近期基线与经典思想来源应分别说明；质量、奖励、时域、动作库、硬件未统一时，论文原表的加速倍数只作其内部结果。

## 对当前FRAME定位的影响

我的判断：已有研究已覆盖阶段化搜索、动作层次分解、在线停止和学习筛选，因此这些词本身不足以概括新意。FRAME的论证应精确到实际机制：在可用前缀后继续核查的规则、同站别名如何补足已压缩候选、以及这套流程怎样在共同底层控制器上取得可测的查询/实际计算收益并满足既定质量条件。

这些工作证明这类问题已有成熟研究语境；我们的论文仍需凭具体方法差异和已完成的公平证据建立贡献。下一步先据这些读物整理相关工作对照表和运行过程图，再决定是否确需新增可扩展性实验。

## 材料与边界

source_receipt.json保存原文链接、身份、下载结果及SHA256。sources/内五份PDF与文字提取；previews/内五个实验页。对ESWA不声称阅读全文。所有点评是本次源文阅读后的判断，未独立复现文中成绩，也未改变FRAME历史裁决。
