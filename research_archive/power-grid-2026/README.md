# 最新：GRID29 共有模型复用试点（2026-10-11）

[报告](outputs/grid29/GRID29_report.md) · [匹配排序诊断](outputs/grid28_review/GRID28_ranker_review.md) · [归档清单](grid29_archive_manifest.json)

GRID29工程数值资格130状态流量差0；4条三月闭环逐步动作/观测/成本相同，查询不变。两组控制时间减少10.93%/0.39%，汇总点估计6.19%，幅度不稳定，不能认定跨日期稳定整体提速或GNN贡献。V0非DPP/键缺陷已保留，V1修订与协议不覆盖历史。

后验匹配GRID28的26次同状态/同候选池/同选择，4周GNN/PPO交付路径也相同；GNN比PPO少465次查询（5.98%），逐调用7少/4多/15平。只有已暴露开发信息，不是图架构或泛化证明。GRID29B冻结一条实际控制状态/记忆下配对DC诊断，未启动新训练或封存日期评价。

# 最新：GRID28 分批搜索闭环开发结果（2026-10-11）

[完整核对](outputs/grid28/GRID28_accounting_report.md) · [逐日期](outputs/grid28/all_comparison.csv) · [审计](outputs/grid28/audit.json) · [归档清单](grid28_archive_manifest.json)

GRID28完成12/12条开发回放，保存数据审计通过。GNN0查询节省50.90%、搜索时间减少43.18%、总控制减少8.83%；共同完成周成本平均增幅0.0454%、最大0.1361%。GNN0/PPO均完成3/4周，没有新增/提前失败；PPO同规则查询节省47.78%、搜索减少37.74%、控制减少3.45%。单次计时、开发数据及不同轨迹，不证明稳定端到端速度或GNN独有贡献；0.5%为投资筛查，不是等价。

GRID27简单停止遗漏关键动作，保留否定筛查；GRID28只加入当前最佳方案所在变电站补全。旧全池守卫仍保留，新规则不具有全池最优选择保证。当前拟合/RL/封存评价0。整体控制器线另进入GRID29原DC模型复用资格，不计为GNN贡献。

# 最新诊断：GRID26 补查瓶颈（2026-10-11）

[离线补查报告](outputs/grid26_review/FALLBACK_REVIEW.md) · [逐调用表](outputs/grid26_review/fallback_per_call.csv) · [独立复核](outputs/grid26_review/review_audit.json)

26个已暴露开发调用中16次全池补查消耗91.41%查询；这16个状态全池也没有rho<0.9合法候选。保持K128与当前守卫时，单靠更准排序不能进一步省查询。前128在14次有合法FULL最优动作的补查中，GNN0/GNN1/PPO覆盖8/7/5；属于事后开发诊断，不是图模块查询增量确认。没有新预测、训练、守卫修改或封存日期评价。下一步优先研究合法分批补查/停止，而非加层。

# 最新：GRID26 实际提速核验完成

[贡献核对](outputs/grid26/GRID26_contribution_report.md) · [48条串行结果](outputs/grid26/comparison.csv) · [保存数据审计](outputs/grid26/audit.json) · [增量归档清单](grid26_archive_manifest.json)

48/48条开发回放，95,424物理步与379,368公开预测；所有动作、观测和成本路径与FULL相同，均完成3/4日期，六月同一步失败。两GNN搜索时间减少19.55%/17.14%，总控制仅减少1.40%/0.88%，查询少34.09%。重复0总控制反向；不宣称稳定显著端到端提速。FULL搜索仅占总控制9.80%。原PPO同样省搜索18.31%；156/156 GNN–PPO逐调用选择与查询数量相同，图模块独有增量未建立，排序/查询集合并不都相同。

GRID25四条试点作为测量修订历史保留，不能拿其较有利时间替代正式矩阵。新拟合/RL/预留家族评价均为0。旧模型、旧失败和回放前协议不变。

## 既有记录

[GRID26 回放前封存协议](protocols/GRID26_before_runs/README.md)

# 最新：GRID24 质量补强（2026-10-10）

[完整报告](outputs/grid24/GRID24_report.md) · [逐运行结果](outputs/grid24/per_run_comparison.csv) · [审计](outputs/grid24/audit.json) · [更改记录](outputs/grid24/POWER_MODEL_CHANGELOG_after_GRID24.md) · [增量归档清单](grid24_archive_manifest.json)

沿用原 rho_safe=0.9，在高风险时缓存补查剩余候选。12/12 开发轨迹逐步重现 FULL，成本差为0且未丢失完整周；搜索查询汇总减少 34.0930%。此通用补救对两 GNN 和原 PPO 都有效，尚未证明图模型增量或正式端到端提速。旧权重、失败结果、GRID00–23 原归档文件与清单保持不变。未新训练或打开预留评价。

[会议与前作核查](outputs/grid_venue_review/CCF_B_GRID_VENUE_REVIEW_2026-10-10.md) · [更早投稿时间](outputs/grid24/METHOD_AND_VENUE_NOTE.md)

## GRID00–23 历史记录

# 电网研究归档（2026-10-10）

[最新完整裁决](outputs/grid_research_audit/GRID22_23_closeout.md) · [模型保留与改动](outputs/grid_research_audit/POWER_MODEL_CHANGELOG.md) · [机器归档清单](archive_manifest.json)

GRID22/23共52条预定闭环轨迹完成并通过保存数据审计。K128共同破并列修订缓解了提前失败，但三月成本仍高8.073%，GNN对同预算PPO/MLP/固定偏置的成本增量只有0.015%–0.031%，未过固定门槛。没有新增RL或最终封存评价。原模型和失败结果保留。

本目录是实验记录与源码快照，不是原IM模型在电网上的直接迁移成绩。当前电网模型是监督候选排序器，加在作者PPO先验及完整工程控制器上；两层图原型的候选解码器、先验残差及同信息MLP均单独记录。GNN＋RL并非首次出现。

## 包含范围

- GRID00–23的根目录报告、配置、汇总、审计、更正与原始清单，保留旧版本。部分早期阶段没有Markdown报告，以JSON和后继审计为准。
- 自有Python源码的历史目录布局快照；外部依赖与作者代码不捆绑。
- V0/V1共10个拟合版本的初始权重、最终权重、曲线和摘要（20份权重文件），逐文件SHA-256核对。
- 日期家族切分更正、预留8家族的元数据、当前未启动RL的说明。

原始大轨迹、逐调用向量、其他历史RL权重和中间epoch、RTE数据集、虚拟环境及作者PPO权重保留在本地，未上传本目录。原阶段delivery_manifest列出的完整文件并非全部包含；这份archive_manifest准确列出本目录实际包含项。不能直接在此运行依赖完整本地轨迹的封存审计。

## 关键边界

成本只比较共同完成的周，早失败的低累计成本不计优势。四个周是已暴露开发数据，两种拟合种子不是独立日期。原GRID03测试分区已部分用于后续开发；8个新日期家族的最终评价未开始。单次含记录开销的墙钟不是无日志生产速度基准；模型排名改善不等同于闭环质量改善。

GRID22曾中断，305个已保存步及至少4,253次预测另记；未落盘在途费用未知，不能当零成本。详情见该阶段报告。

## 一手背景与外部依赖

- [Grid2Op](https://github.com/Grid2op/grid2op) 与 [LightSim2Grid](https://github.com/Grid2op/lightsim2grid)。
- [L2RPN 2023 LJN 作者控制器](https://github.com/lajavaness/l2rpn-2023-ljn-agent)；原连续优化及恢复规则保留，外部权重原训练身份未知。
- [软标签 GNN](https://github.com/AI4REALNET/soft_label_gnn) 与[论文](https://arxiv.org/abs/2503.15190)。
- [GNN潜在连接表示](https://github.com/MatthijsdeJ/GNN_PN_Imitation_Learning) 与[论文](https://arxiv.org/abs/2501.07186)。
- [GNN风险代理与Gibbs先验RL](https://arxiv.org/abs/2604.01830)。
只记录已用的代码和已核查的研究背景；没有宣称完整复现上述每种学习方法。源码包含历史路径、数据提取和依赖设定，使用前需按原协议重建目录及外部依赖。

## 各阶段文件索引

| 阶段 | 已包含根结果文件 |
|---|---|
| GRID00 | [GRID00_report.md](outputs/grid00/GRID00_report.md) |
| GRID01 | [GRID01_report.md](outputs/grid01/GRID01_report.md) |
| GRID02 | [GRID02_report.md](outputs/grid02/GRID02_report.md) |
| GRID03 | [GRID03_report.md](outputs/grid03/GRID03_report.md) |
| GRID04 | [GRID04_report.md](outputs/grid04/GRID04_report.md) |
| GRID05 | [GRID05_report.md](outputs/grid05/GRID05_report.md) |
| GRID06 | [GRID06_report.md](outputs/grid06/GRID06_report.md) |
| GRID07 | [GRID07_report.md](outputs/grid07/GRID07_report.md) |
| GRID08 | [GRID08_report.md](outputs/grid08/GRID08_report.md) · [GRID08_review_zh.md](outputs/grid08/GRID08_review_zh.md) |
| GRID09 | [GRID09_report.md](outputs/grid09/GRID09_report.md) · [GRID09_review_zh.md](outputs/grid09/GRID09_review_zh.md) |
| GRID10 | [GRID10_report.md](outputs/grid10/GRID10_report.md) · [GRID10_review_zh.md](outputs/grid10/GRID10_review_zh.md) |
| GRID11 | 参见归档清单 |
| GRID12 | [GRID12_review_zh.md](outputs/grid12/GRID12_review_zh.md) |
| GRID13 | [GRID13_review_zh.md](outputs/grid13/GRID13_review_zh.md) |
| GRID14 | [GRID14_review_zh.md](outputs/grid14/GRID14_review_zh.md) |
| GRID15 | [GRID15_report.md](outputs/grid15/GRID15_report.md) |
| GRID16 | [GRID16_report.md](outputs/grid16/GRID16_report.md) |
| GRID17 | [GRID17_report.md](outputs/grid17/GRID17_report.md) |
| GRID18 | [GRID18_report.md](outputs/grid18/GRID18_report.md) |
| GRID19 | [GRID19_report.md](outputs/grid19/GRID19_report.md) |
| GRID20 | [GRID20_report.md](outputs/grid20/GRID20_report.md) |
| GRID21 | [GRID21_report.md](outputs/grid21/GRID21_report.md) |
| GRID22 | [GRID22_report.md](outputs/grid22/GRID22_report.md) |
| GRID23 | [GRID23_report.md](outputs/grid23/GRID23_report.md) |
