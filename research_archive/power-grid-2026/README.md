# 最新：GRID149输入与目标对齐

[报告](protocols/GRID149_input_inventory/complete/GRID149_report.md)；不同学习目标分开，下一步补同任务低容量基线。

# 最新：GRID148电网状态与当前标签核查

[报告](protocols/GRID148_state_context/complete/GRID148_report.md)；空间可恢复，12项汇总没有稳定几何改善，不启动新训练。

# 最新：GRID147跨日期原因核查

[报告](protocols/GRID147_generalization_diagnosis/complete/GRID147_report.md)；固定树袋外、相似状态与已保存电网观测核查，无新训练。

# 最新：GRID146单入口学习完成

[报告](protocols/GRID146_single_entry/complete/GRID146_report.md)；仅128处一次学习停止，原FRAME接管后续。全部结果和旧分数结构控制保留。

# 最新：GRID145只读风险控制可行性

[报告](protocols/GRID145_risk_feasibility/complete/GRID145_report.md)；三篇原文适用性、阈值非单调反例、日期聚类支持。无训练或策略更换。

# 最新：GRID39 complete

[方案/报告](protocols/GRID39_completed/outputs/grid39/GRID39_report.md)

GRID39十二条一版目标修订闭环及保存数据核验完成；REVISED_GNN0质量通过4/4、最大可比成本差0.136%、控制时间减少11.03%；REVISED_GNN1质量通过4/4、最大可比成本差0.136%、控制时间减少11.29%。原版查询7312/7484，新版7380/7381；种子0成本不变、种子1四周平均成本增幅0.03403%（以FULL归一化）。没有稳定改进，暂不替换原配置；全模型保留。SoftGNN必要官方源码/论文配置已保存，尚未安装/拟合/回放。仅单次计时开发测量，不把跨批旧/新时间当配对模型提升。

# 最新：GRID39 before

[方案/报告](protocols/GRID39_before_results/outputs/grid39/design.json)

GRID39已冻结12条有限闭环评价：一版目标修改的两个GNN种子与新运行FULL，4个已暴露开发周、单次计时、轮换顺序。控制/停止/守卫/缓存/物理求解不变；原模型和GRID37结果保留，新旧时间不作配对比较。不扩大规则或消融清单，最终8日期仍封存。

# 最新：GRID38完成

[完整报告](protocols/GRID38_completed/outputs/grid38/GRID38_report.md)

GRID38一版目标修订的四次拟合与104项已有反馈排序核验完成；GNN0查询7358→7490、同FULL选择24→24/26、关键动作True；GNN1查询7530→7426、同FULL选择25→24/26、关键动作True；MLP0查询7549→7663、同FULL选择25→24/26、关键动作True；MLP1查询7826→7651、同FULL选择26→24/26、关键动作True。原/新模型、全部种子和训练记录保留，未新增教师仿真。离线查询与排序变化尚不是闭环质量或完整提速。

# 最新：GRID37完成

[完整报告](protocols/GRID37_completed/outputs/grid37/GRID37_report.md)

GRID37全部32条公平静态/GNN/完整搜索比较及保存数据核验完成；LOCAL_GNN0质量通过8/8、最大可比成本差0.136%；LOCAL_GNN1质量通过8/8、最大可比成本差0.001%；STATIC_ID质量通过8/8、最大可比成本差0.372%。两模型种子和两计时重复均保留，详见质量—时间主表。静态顺序属于必要消融，不扩展为方法清单。

# 最新：主对照收敛，单一模型改进已冻结

[训练目标诊断](protocols/GRID_target_alignment/outputs/grid_target_audit/TARGET_ALIGNMENT.md) · [GRID38训练方案](protocols/GRID38_before_training/outputs/grid38/design.json)

按用户投稿目标收敛：主实验只保留2–3个最近且可用的成熟方法，MLP/静态顺序仅少量归因消融，不再扩规则组合。GRID37当前已启动批次跑完保留。训练集167状态/148有效目标峰值均与实际奖励最优对齐，没有代码错误；最优集合原目标均值权重24.44%。GRID38冻结仅改目标为一半原soft标签、一半奖励最优集合均匀标签，原初始化/数据顺序/网络/先验/训练轮数全部相同，两GNN两MLP公平处理；不覆盖原模型、不扫参数。

# 最新：静态规则资格通过，公平比较与训练目标核对继续

[GRID36完整结果](protocols/GRID36_completed/outputs/grid36/GRID36_report.md) · [10%幅度分解](protocols/GRID_search_speed_explanation/outputs/grid_speed_review/GRID34_INCREMENT.md) · [GRID37冻结](protocols/GRID37_before_results/outputs/grid37/design.json)

GRID36三月廉价规则闭环核验完成：STATIC_ID成本+0.37200%通过0.5%筛查；PHYSICS_TAIL/LOCAL_STRESS成本+3.61164%/+4.01870%未通过，所有结果保留。静态顺序仅单次资格，不能据跨批时间宣告赢过GNN。GRID37冻结同一共有优化下FULL、两GNN、STATIC_ID四日期两平衡重复共32条，公平区分便宜停止与图模型增量。

# 最新：共有速度与GNN增量已分开，廉价顺序闭环资格待查

[GRID35结果](protocols/GRID35_completed/outputs/grid35/GRID35_report.md) · [原非学习部分](protocols/GRID35B_completed/outputs/grid35b/GRID35B_report.md) · [实际资格试点](protocols/GRID36_before_results/outputs/grid36/design.json)

GRID35/35B廉价顺序独立诊断通过：STATIC_ID/LOCAL_STRESS/原既有PHYSICS_TAIL查询7879/6965/6748，同FULL选择21/17/17（共26状态）；均找回1733已知关键动作，但最大rho差更大。便宜规则省查询不等于长期质量，GRID36三条三月闭环质量试点已冻结。

# 最新：GRID34公平闭环完成，共有提速与模型增量分开

[完整结果](protocols/GRID34_completed/outputs/grid34/GRID34_report.md) · [独立保存数据审计](protocols/GRID34_completed/outputs/grid34/review.json)

GRID34全部48条公平闭环和保存数据审计完成；LOCAL_GNN0控制时间相对共有优化FULL/PPO/MLP0/MLP1分别减少11.34%/0.12%/1.36%/0.32%；LOCAL_GNN1控制时间相对共有优化FULL/PPO/MLP0/MLP1分别减少11.79%/0.62%/1.86%/0.82%。全部模型种子保留，六月共同1901步失败保留。两GNN均满足质量投资门槛，MLP0九月成本高0.60247%超过门槛但结果保留；共有编译复用不归于GNN，开发日期不作封存泛化。

# 最新：GRID34公平试点通过，其他开发周继续

[三月试点审计](protocols/GRID34_pilot_completed/outputs/grid34/pilot_audit.json)

GRID34三月12条公平试点质量门槛通过：GNN1/MLP0/MLP1两次均逐步等同FULL；GNN0/PPO共同成本增加0.1361%，全部2017步完成。优化后FULL141–142秒，各排序器125–128秒；架构间仅小差距，MLP强对照保留，不能把总体节省全部归于GNN。

# 控制提速差异的可核查解释

[为什么10%与0.39%差很多](protocols/GRID_common_speed_explanation/outputs/grid_speed_review/WHY_10_PERCENT.md)

已完成用户要求的幅度诊断：三月可复用DC+安全QP占原控制39.7%、安全QP缓存命中77.8%、未复用unsafe占38.6%；其他三周可复用占73.4%–88.1%、命中90.0%–93.8%、unsafe占3.3%–8.2%，支持22%vs62%–74%的组件解释。旧10.93%/0.39%缺逐solver计时，精确波动原因仍不可恢复。GRID32继承的repeats=2/旧四条文案为元数据残留，实际冻结6条execution_order及审计一贯是3单配对；原JSON不改。

# 最新：GRID33现成排序对照通过，GRID34公平闭环已冻结

[同状态结果](protocols/GRID33_completed/outputs/grid33/GRID33_report.md) · [公平闭环协议](protocols/GRID34_before_results/outputs/grid34/design.json)

GRID33五个现成排序器130项同状态诊断及独立停止复算通过；查询7358/7530(GNN0/1)、7549/7826(MLP0/1)、7823(PPO)。全部找回1733关键动作；差距仅开发线索，不证明长期质量或图结构必要性。

# 最新：GRID32其他开发周通过，GRID33现成排序对照已冻结

[结果](protocols/GRID32_completed/outputs/grid32/GRID32_report.md) · [下一步](protocols/GRID33_before_results/outputs/grid33/design.json)

GRID32三组其他已暴露周配对通过：2035-06-25_8减少63.26%；2035-09-03_4减少61.89%；2035-10-01_5减少74.46%。11870步动作/观测/成本逐步相同，六月共同1901步失败保留，未新增/提前失败；每周仅一次配对，不作为独立封存泛化或GNN贡献。

# 最新：GRID31完整控制器共享提速通过，GRID32已冻结

[完整报告](protocols/GRID31_completed/outputs/grid31/GRID31_report.md) · [跨开发周协议](protocols/GRID32_before_results/outputs/grid32/design.json)

GRID31六条完整三月回放及审计通过：DC+安全QP共享复用相对原完整控制，两组减少23.49%/21.33%，汇总点估计22.42%；12102步动作/观测/成本/67950次预测相同，所有6项配对solver状态/迭代比较无差别。局部数值资格与完整时间分开，原10.93%/0.39%及6.45%/6.51%全部保留。仅三月2重复，不是GNN独有或封存泛化证据。

# 最新：GRID30安全QP复用资格通过，GRID31完整比较已冻结

[资格报告](protocols/GRID30_qualified/outputs/grid30/GRID30_qualification_report.md) · [完整试点协议](protocols/GRID31_before_results/outputs/grid31/design.json)

GRID30安全QP共享编译复用资格通过：9项手工/复制检查、153次真实预求解参数/记忆三向量，最大差4.33e-10MW；306次动作重建相同，2017步实际路径/成本/查询仍原保护控制。隔离原QP42.45s、缓存15.24s，减少64.09%，不构成整套提速。V0原负值setter失败已保留，V1原位原样复制未裁剪。

# 最新：GRID29C组件解释通过，GRID30数值资格已冻结

[报告](protocols/GRID29C_completed/outputs/grid29c/GRID29C_report.md) · [下一资格协议](protocols/GRID30_before_results/outputs/grid30/design.json)

GRID29C完整组件配对4/4审计通过：两组完整控制提速6.45%/6.51%；DC省17.47/18.14s（9.61/9.94完整控制百分点），非DC部分分别多耗5.74/6.27s。680个配对solver调用的状态/迭代零改变，8068步动作/观测/成本/查询相同。重复编译节省机制成立，但旧10.93%/0.39%仅有逐步计时，具体历史波动成因无法完整追溯。

# 最新：GRID29B真实记忆诊断通过，GRID29C待运行

[诊断报告](protocols/GRID29B_completed/outputs/grid29b/GRID29B_report.md) · [下一组件协议](protocols/GRID29C_before_results/outputs/grid29c/design.json)

GRID29B真实控制记忆下153组DC配对：原34.611s、缓存13.268s，减少61.67%；154组保存三向量最大流量差0、2017步路径完全相同。实际原DC调用编译29.17s/后端0.070s，确认重复编译是这一DC组成的主要开销，不等于完整控制器根因。旧10.93%组恢复交付步骤省25.42s、其余多1.75s；旧0.39%组对应省10.32s、其余多9.62s。旧两组没有逐solver计时，不能事后识别各自DC收益。

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

## GRID44 before

GRID43并列补查试点关闭：6条保留、18条未运行；不改善原模型成本且增加查询，当前仍用原GRID21权重和GRID28规则。GRID44首次8预留日期家族确认已冻结：96串行回放、两计时重复、原GNN/SoftGNN各两种子、作者PPO及FULL；预处理/缓存/控制/停止一致，先冻结后解压。不重新训练、不按确认结果调参；全部家族及变体持续排除训练。 [协议与结果](protocols/GRID44_before/outputs/grid44/design.json)。

## GRID44 acquired

GRID44冻结8个首选序列已按官方索引安全解压并逐文件哈希，104文件、50541137字节；尚未运行控制器或读取评价成绩。96条冻结串行确认待运行，不重新训练。 [协议与结果](protocols/GRID44_acquired/outputs/grid44/design.json)。

## GRID44关机检查点

用户准备关机，GRID44在36/96条后保存停止，剩余60条。全部已完成轨迹、计量与哈希保留，调度器不再派发任务；未重训/调参、尚无整批确认裁决。下次按原 --phase all 跳过已完成身份续跑，不重新准备/解压。 [续跑说明](protocols/GRID44_shutdown_36/outputs/grid44/RESUME.md)。

## GRID44 complete

GRID44八预留日期家族96条串行确认及保存数据核验完成；FULL完成16/16、无提前失败16/16、质量筛查16/16；LOCAL_GNN0完成16/16、无提前失败16/16、质量筛查14/16；LOCAL_GNN1完成16/16、无提前失败16/16、质量筛查14/16；SOFTGNN0完成16/16、无提前失败16/16、质量筛查14/16；SOFTGNN1完成16/16、无提前失败16/16、质量筛查12/16；LOCAL_PPO完成16/16、无提前失败16/16、质量筛查14/16。两原模型平均控制提速14.624%（家族bootstrap95%[10.650%,18.543%]），各有一个日期成本+0.88509%超过原0.5%筛查，联合筛查未通过；不是研究绝对失败。原GNN对PPO宏平均提速3.211%/2.481%，不普遍支配SoftGNN。48/48计时重复物理路径一致；两次计时/两种子不是独立日期，SoftGNN为资源匹配适配。全部日期使用后不再叫封存，禁止据其择模型赢家。 [协议与结果](protocols/GRID44_complete/outputs/grid44/design.json)。

## GRID45–47 before

GRID45独立复算96条确认并定位4月23日540步：两GNN与PPO均在rho0.92207平台停查、漏掉FULL844（真实奖励差0.00686），不能单动作因果归整周成本。GRID46只恢复作者原0.9缓冲要求，旧26状态×5排序器130项序列/未观测反馈核验通过，原GNN/PPO同FULL选择26/26；查询14893。GRID47冻结24条旧开发回放，先6三月，原两GNN质量及正提速通过才18其他旧日期；所有基线共用保护，不重新训练/复用GRID44确认当未见评价。回补为已有保护的质量—时间端点，不包装新安全理论。

## GRID45–47 pilot

GRID47六条共同作者阈值回补试点完成；原两GNN门槛=True；FULL成本差0.000000、时间减少0.0000；LOCAL_GNN0成本差0.000000、时间减少0.0358；LOCAL_GNN1成本差0.000000、时间减少0.0375；SOFTGNN0成本差0.012900、时间减少0.0390；SOFTGNN1成本差0.061945、时间减少-0.0349；LOCAL_PPO成本差0.000000、时间减少0.0258。所有基线与种子保留，单次开发计时。

## GRID45–47 complete

GRID47共同作者阈值回补24条旧开发比较与保存数据核验完成；FULL质量4/4、查询22597、控制321.67秒；LOCAL_GNN0质量4/4、查询14893、控制310.55秒；LOCAL_GNN1质量4/4、查询14893、控制310.32秒；SOFTGNN0质量3/4、查询15251、控制301.07秒；SOFTGNN1质量3/4、查询16057、控制313.99秒；LOCAL_PPO质量4/4、查询14893、控制306.97秒。保护是已有作者缓冲的回补，不是新安全理论或图专属；没有训练或GRID44重评价。

## GRID48 before

按用户要求保留原快速结构。GRID48训练侧资格：±2残差导致72/148最佳目标无法排第一、70/148峰值概率质量超硬界；但没有任何目标数学上被排除128候选，不能把硬界当成闭环失败原因。仅移除末端tanh硬界，保留PPO先验KL0.1、原GNN结构、同初始权重/数据/目标/64轮/两种子；不采用严格GRID47保护，不训练或重评GRID44日期。序列化NumPy布尔缺陷原版保留，资格数值按相同规则修复。两次拟合尚未开始。

## GRID48 complete

GRID48移除残差硬界的两次固定拟合及旧26状态复核完成；GNN0查询7358→7385、同FULL选择24→25/26、1733匹配True；GNN1查询7530→7279、同FULL选择25→24/26、1733匹配True。原模型全部保留，未训练/评价GRID44日期；仅缓存开发比较，实际质量和提速尚未证明。

## GRID49 before

GRID49冻结原快速控制下的一次评分改动比较：原GNN两种子、新无硬界残差两种子、FULL；先5旧三月试点，两个新种子质量/存活合格才15其他旧开发回放。停止/保护/特征/缓存/控制均沿用原快版，不采用GRID47严格回补。没有GRID44训练/重评或新模型拟合；近期强基线原比较完整保留，不扩baseline菜单。

## GRID49 pilot

GRID49原快控制五条评分改动试点完成；两新种子质量=False；FULL成本差0.000000、查询9578、控制141.16s；LOCAL_GNN0成本差0.001361、查询3007、控制126.22s；LOCAL_GNN1成本差0.000000、查询3122、控制125.95s；FREE_GNN0成本差0.000000、查询2967、控制126.07s；FREE_GNN1成本差0.013741、查询2882、控制121.77s。全部原/新权重、结果保留，不选种子。

## GRID50 before

按用户保留原版优势：GRID49自由残差种子0成本零且约10.69%提速，种子1成本+1.374%故不替换原版、15条未跑。GRID50只做一版软压缩2*asinh(raw/2)，尺度仍原2、局部斜率1、先验KL0.1保留；同结构/初始化/数据/目标/64轮/两种子，原快停止不变。若仍混合，停止激活形式变更，转向训练覆盖；不在GRID44调参或择种子。

## GRID50 complete

GRID50一版软压缩两次拟合及旧26状态复核完成；GNN0查询7358→7728、同FULL选择24→25/26、1733True；GNN1查询7530→7728、同FULL选择25→25/26、1733True。不挑种子，不重训/评价GRID44，原快模型仍默认；缓存表现不等于闭环质量/提速。

## GRID51 before

GRID51冻结原快速控制下的一次评分改动比较：原GNN两种子、新软压缩残差两种子、FULL；先5旧三月试点，两个新种子质量/存活合格才15其他旧开发回放。停止/保护/特征/缓存/控制均沿用原快版，不采用GRID47严格回补。没有GRID44训练/重评或新模型拟合；近期强基线原比较完整保留，不扩baseline菜单。

## GRID51 pilot

GRID51原快控制五条评分改动试点完成；两新种子质量=True；FULL成本差0.000000、查询9578、控制153.76s；LOCAL_GNN0成本差0.001361、查询3007、控制130.89s；LOCAL_GNN1成本差0.000000、查询3122、控制127.29s；COMPRESSED_GNN0成本差0.000000、查询3309、控制128.01s；COMPRESSED_GNN1成本差0.000000、查询3213、控制126.15s。全部原/新权重、结果保留，不选种子。

## GRID51 complete

GRID51原快速控制20条评分改动比较与保存数据核验完成；FULL质量4/4、查询22597、控制328.62s；LOCAL_GNN0质量4/4、查询7312、控制295.48s；LOCAL_GNN1质量4/4、查询7484、控制279.75s；COMPRESSED_GNN0质量4/4、查询7682、控制278.91s；COMPRESSED_GNN1质量4/4、查询7682、控制281.65s。全部种子保留；旧开发非独立确认，实际改动不自动成为默认。

## GRID52 before

GRID51软压缩候选保留：旧开发完成周成本与FULL相同、控制减少14.3%–15.1%，不声称已胜PPO/SoftGNN。GRID52新版新八日期确认已在解压前冻结：两模型种子、三类主对照、两次平衡计时共96条；全部先前暴露/训练/开发/已提取日期排除，按季度固定哈希选日期和变体。Q1/Q3剩余各两日期，选择受限如实披露。无新训练/教师，原GRID44结果及质量损失完整保留。

## GRID52 acquired

GRID52八冻结首选序列解压核验完成：104文件、51506252字节。尚无控制器评价/新拟合，96条固定确认待运行。

## GRID52 complete

GRID52新版八新日期96条确认及保存数据核验完成；FULL完成16/16、质量16/16、控制1330.81s；SOFTGNN0完成16/16、质量16/16、控制1117.88s；SOFTGNN1完成16/16、质量16/16、控制1102.17s；LOCAL_PPO完成16/16、质量12/16、控制1095.09s；COMPRESSED_GNN0完成16/16、质量16/16、控制1164.43s；COMPRESSED_GNN1完成16/16、质量16/16、控制1152.88s。新版两种子均通过原质量—时间筛查，完整周最大成本增幅均0.08045%，对FULL宏平均控制时间减少14.465%/14.748%、查询少70.00%/70.62%；平均两种子时间减少14.606%（八家族bootstrap95%[7.240%,22.175%]）。但控制时间比PPO约长3.24%–3.57%、比SoftGNN1约长4.34%–4.60%，不能主张全面领先；PPO最大成本增幅12.434%，SoftGNN两种子最大增幅0.177%/0.169%且均通过质量筛查。48/48计时重复路径一致。全部种子/失败保留；两次重复不是独立家族；SoftGNN为资源匹配适配；GNN独有归因未自动建立，不把旧GRID44失误视为已经修复。

## GRID53 timing diagnosis

GRID53已有212份计时/1272项逐搜索账目核查完成：GRID44原版对PPO快2.48%–3.21%、对SoftGNN1快2.25%–2.93%，但慢于SoftGNN0；GRID52新版在不同日期上对这两项均略慢，不能据跨日期比较认定模型退步。GRID51同批新版seed0快5.606%、seed1慢0.678%，结果混合。新版前向仅约2.3ms/搜索、共同PPO先验约233ms；一份旧可见状态剖析确认909次as_dict重复静态拓扑展开占先验剖析约85%。相对SoftGNN额外先验及不同路径连续优化共同形成时间差；相对PPO主要差值在连续优化，不能只归于神经网络或用公共缓存冒充图贡献。未训练、未新候选simulate、未改模型或停止规则；原新全部结果保留。

## GRID54 before

GRID54先验静态元数据缓存资格已冻结：只替换909候选拓扑字典与动作映射的重复展开；每次仍读取新鲜动作字段验证快照，替换/原地变化即重建，不用可能陈旧的to_vect。130个旧开发状态精确比较先验分数/排序/最终模型排名与特征哈希，原新版与PPO公平共享；冷启动计入先验时间，无训练/候选simulate/保护改动。

## GRID54 complete

GRID54等价先验缓存资格通过：130状态分数/排序/最终排名/特征逐项一致，四项替换/原地修改/恢复重建通过；局部先验累计29.966→3.615s（减少87.94%），含首次冷构造与每次动作新鲜字段校验。无新训练/候选simulate；不是闭环速度或GNN独有贡献，公平给旧新PPO。

## GRID55 before

GRID55一次有限同日期旧新版及既有主对照比较已冻结：128条串行回放、8个既有暴露日期、两个平衡计时；原GNN/软压缩GNN/SoftGNN各两种子、作者PPO、FULL。共有GRID54先验缓存含首次构造和逐次内容核验，原GRID28快控制/缓存/合法信息不变。无新训练/参数/日期替换；此为配对选型与工程归因，不重新宣称独立确认。

## GRID55 complete

GRID55有限同日期旧新版及既有强对照比较与保存数据核验完成；FULL质量16/16、控制1313.24s；SOFTGNN0质量16/16、控制1101.60s；SOFTGNN1质量16/16、控制1103.91s；LOCAL_PPO质量12/16、控制1130.60s；COMPRESSED_GNN0质量16/16、控制1151.80s；COMPRESSED_GNN1质量16/16、控制2392.41s；LOCAL_GNN0质量14/16、控制1072.01s；LOCAL_GNN1质量14/16、控制1069.77s。各版本/全部种子/失败保留，实际质量—时间及配对差见报告；已暴露日期不当独立确认，共有缓存不作为图模型独有。 **计时资格未通过：** Windows现代待机事件与worker125（新版种子1）及127（PPO）重合；前者控制1354.35s/CPU96.94s，不能把巨幅负提速解释为模型退步。原128条、RAW统计与选型均未改；64/64重复路径一致，成本证据有效（原版最坏+1.740%/+1.704%，新版均+0.08045%）。最终速度与版本选型仍未解决。仅提出为该日期补一次完整八臂计时，不覆盖原记录、不变模型/规则；受本批不重复已完成项/不改预算约束，尚未执行，需用户决定。请先读GRID55_review_status.md，再读RAW报告。

## GRID55B before

用户2026-10-13继续后，按关机前具体方案冻结GRID55B：仅待机受影响日期2035-11-12_9全部八方法补新重复2，2400秒墙钟上限；原128条不覆盖，不改模型/控制/阈值/先验缓存，不新增训练。运行期临时防空闲待机，不修改永久电源计划。结束后核对电源日志、原两次物理路径/成本/合法反馈/查询，对全部八方法一致构建原重复0/新重复2计时视图，沿用原对称选型。不是新独立确认。

## GRID55B complete

GRID55B八条补测与保存数据核验通过：16/16与原两重复的路径、向量、成本、反馈和查询一致，运行期无电源待机事件。原RAW128保留，七日期原0/1加受影响日期原0/新2组成明确来源的128槽位计时视图。FULL质量16/16、控制1311.18s；SOFTGNN0质量16/16、控制1092.66s；SOFTGNN1质量16/16、控制1095.14s；LOCAL_PPO质量12/16、控制1064.52s；COMPRESSED_GNN0质量16/16、控制1133.24s；COMPRESSED_GNN1质量16/16、控制1124.13s；LOCAL_GNN0质量14/16、控制1072.05s；LOCAL_GNN1质量14/16、控制1071.05s。原对称选型=COMPRESSED_CANDIDATE_QUALITY_SCREEN；这是质量—时间投资筛查，不是等价或独立确认；公共缓存不归于GNN。

## GRID56 论文证据与审计收口

GRID56最终审计收口通过：84项冻结哈希、9项RAW锚点、174项新补测文件收据、3842项历史归档条目、128槽位来源与21项统计/bootstrap/选型公式。无新仿真或训练，不宣称独立AC重求解。完成质量—速度主表、科学图、方法结果工作稿、贡献边界与复现入口；训练教师费用按20训练周分离为50.68分钟/146521个N1查询，最终两拟合调用89.99秒，已有PPO成本仍未知。现成STATIC_ID开发消融接近GNN，下一项仅补最终配置的廉价顺序消融，不扩主对照。

[报告](protocols/GRID56_evidence_complete/outputs/grid56/GRID56_report.md) · [方法草稿](protocols/GRID56_evidence_complete/outputs/grid56/method_and_results_draft.md)

## GRID57 before

GRID57仅冻结一条必要廉价顺序消融：既有STATIC_ID在最终八个已暴露日期各一次回放，原控制/停止/合法信息/DC与安全QP缓存不变，排序是冻结的代表ID升序。无新训练/参数/日期/基线家族；主比较完整周质量、存活和查询，历史FULL/GNN只用于离线参照；新批墙钟不与旧批构成公平提速主张。

## GRID57 complete

GRID57八条既有STATIC_ID最终配置消融与保存数据审计完成：质量尺度通过8/8，GNN平均相对其查询减少1.897%，正向4/8；裁决MIXED。无新拟合；不作跨批速度或新独立确认。

## GRID57 频次先验来源

GRID57固定顺序来源已核实：909动作counts非增且全部八运行行hash等同作者资产顺序，等价历史教师频次先验。当前GNN查询宏平均仅少1.90%（7有调用日期），汇总反而多6.69%；成本1好/6差/1平、平均高0.509个FULL归一化百分点。现有图模块增量未成立，整体提速仍保留。只保留“让GNN校正较强频次先验”的单次改进假设，不继续基线扩散。

## GRID58 before

GRID58冻结一次先验替换：保留作者教师频次STATIC_ID为零残差基准，原软压缩GNN/尺度2/初始化/167状态148目标/64轮/两种子/优化器/停止/守卫均不变。仅两个拟合与26旧开发缓存状态资格；不训练最终八日期，不调参或新增教师。两种子须找回既知1733才另设有限旧开发闭环。

## GRID58 complete

GRID58两个固定频次先验GNN拟合及52项旧开发缓存核验完成；seed0查询7457、STATIC7879、同FULL22/26、1733=True；seed1查询7520、STATIC7879、同FULL22/26、1733=True。旧开发闭环资格=True，尚未运行闭环，无新教师；缓存结果不作质量/提速证明。

## GRID59 before

GRID59冻结16条四旧开发周有限配对试点：频次先验GNN两种子、强STATIC_ID、FULL；原快控制/停止/合法信息/DC安全QP缓存不变，单次平衡串行计时。无新拟合或最终八日期评价；先验证两种子质量，再按冻结查询条件看额外学习信号，不新增先验/常数变体。

## GRID59 complete

GRID59有限频次先验闭环及保存数据审计完成；STATIC_ID质量4/4、对STATIC查询少0.000%、对STATIC控制少0.000%；FREQUENCY_GNN0质量4/4、对STATIC查询少3.127%、对STATIC控制少-0.635%；FREQUENCY_GNN1质量4/4、对STATIC查询少2.675%、对STATIC控制少-0.506%；裁决WEAK_OR_MIXED_STOP_REVISION。单次旧开发配对，不是封存确认。原新先验/模型/失败全部保留。

## GRID60 before

用户已选择以框架作用为论文主线、GNN可替换辅助。GRID60仅做机制/近邻方法资格核对：共同频次排序下FULL、首个安全候选加增强回退、无变电站补全、现有完整框架；不新增训练/参数。

## GRID60 complete

用户已选择以框架作用为论文主线、GNN可替换辅助。GRID60仅做机制/近邻方法资格核对：共同频次排序下FULL、首个安全候选加增强回退、无变电站补全、现有完整框架；不新增训练/参数。104项合法反馈扰动与90项原FRAME复算通过；实际局部补查改变12次选择。缓存状态数量/选择尚不是完整周机制因果证据。

## GRID61 before

GRID61同频次顺序的框架机制矩阵：FULL/加强首个安全候选/去局部补全/现有FRAME，四旧开发日期、两个平衡重复32条；仅改变停止/补全策略。原GRID28基础控制、合法反馈和共有DC/safeQP缓存不变；无GNN拟合、阈值调节或外部基线扩散。

## GRID61 complete

GRID61同频次顺序的框架机制矩阵：FULL/加强首个安全候选/去局部补全/现有FRAME，四旧开发日期、两个平衡重复32条；仅改变停止/补全策略。原GRID28基础控制、合法反馈和共有DC/safeQP缓存不变；无GNN拟合、阈值调节或外部基线扩散。裁决LOCAL_COMPLETION_SUPPORTED_IN_MATRIX；FULL质量8/8、控制减少0.000%;FIRST_SAFE_BEST质量4/8、控制减少7.032%;NO_LOCAL质量6/8、控制减少14.251%;FRAME质量8/8、控制减少13.059%。保留全部共同失败、较慢和较差结果；同排序机制比较不是新独立确认。

## GRID62 before

GRID62冻结全部剩余三个未用日期、四既有同频次策略、两平衡计时重复，共24条。主方法FRAME、对照FULL/加强首个安全候选，NO_LOCAL只作已有内部消融。原控制/缓存/参数均不变，无训练；三个日期不能包装成八日期或跨电网确认。

## GRID62 complete

GRID62冻结全部剩余三个未用日期、四既有同频次策略、两平衡计时重复，共24条。主方法FRAME、对照FULL/加强首个安全候选，NO_LOCAL只作已有内部消融。原控制/缓存/参数均不变，无训练；三个日期不能包装成八日期或跨电网确认。保存数据审计通过，裁决FRAMEWORK_CONFIRMED_IN_LIMITED_FRESH_MATRIX。全部失败/较慢/较差结果保留；非独立AC重求解。

## 2026-10-13 — ICAPS 2027 guide-based first draft

用户明确确认目标ICAPS2027，按其提供的LAMDA-NeSy Research-Starter-Kit公开写作章节完成4340词英文初稿、中文导读、写作要求映射和两个月安排。正文以渐进搜索/变电站补全为主线，GNN辅助；保留FIRST的新日期竞争力、无搜索日期/共同失败和计时变异。八行统一主表复算通过，旧V1/V2与GRID61/62原材料不变；无新仿真或训练。官方AAAI2027样式原样保留，但应用内XeTeX被其PDFLaTeX要求拒绝，未宣称正式格式/8页认证；另交付9页可视核对的单栏PDF阅读版。当前是AI辅助作者重写工作稿，非直接提交或匿名认证。摘要/全文截止12/7与12/14 UTC-12，拟在最终政策和作者文本确认后精修。

[ICAPS2027初稿与导读](protocols/GRID_ICAPS_draft_v1/README.md)

## 2026-10-13 — complete Chinese author manuscript

用户要求先用中文通读重写，已制作完整中文正文、可编辑Word/五项原生OMML公式、中文框架图、10页同版PDF与Markdown。11个一级章节/全部二级章节、三表数值对应英文初稿；全部页面原尺寸视觉检查通过，旧英文和GRID61/62结果哈希不变。无新运行或拟合。Windows运行包缺转换器，使用官方校验后的任务目录行政解包作隔离排版依赖，未安装桌面软件或更改系统关联；大依赖不入Git。首轮日志GBK/UTF8解码错误和排版修订保留，最终渲染正常。下一步由作者中文重写，再同步英文；当前版仍为AI辅助工作稿，不是正式AAAI版或独立科学审计。

[完整中文作者稿](protocols/GRID_ICAPS_cn_v1/README.md)


## 2026-10-13 — Chinese manuscript V2 following author edits

按用户前四页删改重写完整中文论文，改用直接阐述问题、方法和收益的语气。按用户顺序设置摘要/引言/方法/实验/结论/相关工作/局限/参考文献；重读Research-Starter-Kit七项相关正文。删除重复防御性对比句和审计流水账，保留三表数据、失败与竞争结果及必要计量定义。用户编辑DOCX哈希不变，新V2独立保存；5项可编辑公式，8页阅读排版全部检查。无新实验或训练，科学裁决不变。


## 2026-10-14 — D drive storage migration

按用户要求将本课题157270个文件、35.430GB统一迁至D:/Codex/GRL研究。文件路径/数量/大小核对及1963项文档与抽样哈希检查通过。桌面GRL研究资料指向D盘；旧C路径仅保留outputs/tmp/work目录连接，原数据副本清理完成。当前应用占用目录导致直接改名失败，改用空目录原位连接保留旧路径；后续新材料默认D:/Codex，已更新用户与项目AGENTS。用户修改稿、历史结果均保留，无新实验或训练。

## GRID63 complete

GRID63同状态等额补查闭环比较完成：7个已有日期、FULL/FRAME/GLOBAL_MATCH、3次平衡计时，共63条；裁决GLOBAL_COMPETITIVE_NO_MATERIAL_STRUCTURE_GAIN。FULL质量7/7、完整周5/7、控制减少0.000%、查询减少0.000%；FRAME质量7/7、完整周5/7、控制减少14.016%、查询减少74.092%；GLOBAL_MATCH质量7/7、完整周5/7、控制减少13.721%、查询减少74.092%。FRAME相对GLOBAL仅三月+0.04677% FULL、四月+0.00782% FULL，其余完整日期成本相同；没有日期达到0.1%尺度。不能把旧NO_LOCAL消融单独解释为同站结构选择的实质优势。同状态配额在新增反馈前确定，轨迹分叉后全周查询总量不强行等量。三次重复用于计时，不增加独立日期。44条后外部会话中断，未完成片段封存；仅续跑未完成条目。保存数据审计，不是独立AC重求解，无新训练/RL。全区间原审计时间资格false保留；逐worker复核6条电源事件均位于中断空档，未重叠63条成功回放，不扣减任何原时间。

## GRID64 source qualification

用户要求补论文来源的自适应停止对照。GRID63等预算全局对照已完成，结构增量未达尺度。GRID64仅完成来源核对：ICAPS2022作者Metareasoning.jl锁定facb41ad；AAAI2024为上下文元MDP+DQN，不能用手写曲线阈值冒充。优先准备明确标注的META-DQN电网适配；正式元状态/动作/质量与时间效用映射尚未冻结、尚未拟合或回放。新基线训练须剔除旧训练池中与当前评价重合的2035-03-05及2035-06-25日期家族，18个非重合旧训练周可供后续资格核对。

## GRID64 complete

GRID64论文来源自适应停止基线六次固定拟合完成，CPU墙钟202.76s。排除全部七个评价日期家族，14训练/4校准家族、129/22状态；无新教师仿真，旧已付费反馈151状态/132570次源查询复用。ICAPS2022官方Double DQN核心Python移植，停止/继续限定，q为已观测一步N1奖励/2，不是全周价值。三个预定时间价格用校准集共享选择cost1，两个种子均满足预定一步校准条件；132项在线/缓存一致与132项隐藏后缀检查通过，全部六份初末权重与较差配置保留。尚无新闭环结果。

## GRID65 before

GRID65冻结70条串行闭环比较：七个已暴露日期、FULL/FRAME/GLOBAL_MATCH/两个META-DQN种子、两次反序计时重复。全部同原控制/排序/缓存；元策略首次加载和每次推理计入控制时间，无在线拟合。先检验成本/存活，再看查询与时间；不改变模型、时间价格、日期或择种子。

## GRID65 complete

GRID65论文来源自适应停止对照70条与保存数据审计完成，裁决FRAME_ADVANTAGE_OVER_ADAPTED_STOPPING_IN_MATRIX；FULL质量7/7、完整周5/7；FRAME质量7/7、完整周5/7；GLOBAL_MATCH质量7/7、完整周5/7；META_DQN0质量6/7、完整周5/7；META_DQN1质量5/7、完整周5/7。FRAME查询少74.092%、控制少13.583%、最坏成本+0.372%；元策略两种子查询少69.277%/63.045%、控制少13.542%/12.079%、最坏成本+2.973%/+3.983%。FRAME的优势主要是质量和查询；相对元策略宏平均控制仅少0.056%/1.743%，汇总秒数对种子0反而多0.634%、对种子1近乎相同，不能宣称稳定整套提速优势。GLOBAL_MATCH质量7/7、查询与FRAME相同、控制少13.787%，同站结构增量仍未建立。35/35物理重复一致、42项老控制质量/查询复现、无电源事件，276冻结哈希与独立汇总复算通过。逐日期、两个种子、两计时重复均保留。新方法是ICAPS2022停止限定适配，不冒充原A*/RRT*或AAAI2024完整方法，不据此宣称SOTA普遍支配；无独立AC重求解。

## GRID66 complete

GRID66离线整理GRID65的70条逐日期成本、计时离散度、全步骤与搜索步骤P95/P99、互斥计算分解。核查旧路径/训练及结果使用记录后确认6个日期家族仅数据检查或未入选候选池，未有本地控制/教师结果；GRID62曾保守地把所有已解压日期也排除。本次明确更正可用池，不改旧裁决；外部作者先验训练覆盖仍未知。

## GRID67 before

GRID67冻结24条四旧日期单参数敏感性：前缀64/128/256，停滞1/2/3，默认不变，不做交叉网格。原控制/排序/缓存/物理环境不变，无训练。

## GRID68 before

GRID68冻结36条六个本地新日期、FULL/FRAME/GLOBAL_MATCH、两次反序计时。日期不用于参数选择。原控制/排序/缓存/物理环境不变，无训练。

## GRID67 complete

GRID67完成24条，保存数据审计通过，裁决PARAMETER_SENSITIVITY_REPORTED_NO_SELECTION；FULL质量4/4、完成3/4、查询少0.000%、控制少0.000%；FRAME质量4/4、完成3/4、查询少69.367%、控制少15.280%；FRAME_P64质量3/4、完成3/4、查询少72.873%、控制少15.583%；FRAME_P256质量4/4、完成3/4、查询少57.936%、控制少12.997%；FRAME_S1质量4/4、完成3/4、查询少70.105%、控制少15.551%；FRAME_S3质量4/4、完成3/4、查询少65.870%、控制少14.790%。所有不利日期与配置保留；默认不择优替换，无独立AC重求解。

## GRID68 complete

GRID68完成36条，保存数据审计通过，裁决QUALITY_FAILURE_OR_LIMITED_COMPUTE_GAIN；FULL质量6/6、完成6/6、查询少0.000%、控制少0.000%；FRAME质量4/6、完成5/6、查询少66.640%、控制少16.375%；GLOBAL_MATCH质量4/6、完成5/6、查询少65.064%、控制少14.219%。所有不利日期与配置保留；默认不择优替换，无独立AC重求解。

## GRID69 before

GRID69单一修复冻结：取消rho安全时128查询直接返回；所有压缩返回先完成至少256查询与连续两批无改进，再同站补全。先七月、十二月及旧三月三日期FULL/修复版6条；仅质量与查询门槛通过后运行其余10已暴露日期。8项手算/权限检查、308条旧搜索复现通过。原版不改，0.5%旧筛查不改，0.64%仍可连续报告为小幅增加，不删不利日期；无训练。

## GRID69 pilot

GRID69 pilot完成保存数据核验，6条、质量2/3、完成3/3；裁决PILOT_NOT_PASSED_NO_AUTOMATIC_EXPANSION。所有不利结果保留，修复开发不作为独立确认；单次计时仅描述。

## GRID70 before

GRID70分支隔离修复v2冻结：只核查原128安全前缀提前返回；256最低核查和两批停滞后，仅当发现更好incumbent时同站补全。原不安全前缀回退/补全精确委托旧版，前缀缓存不重复计费。GRID69旧三月回退代价1.374%及其全部记录保留；诊断发现首次退步来自未改进安全incumbent也新增触发局部补全。先三问题/回归日期修复版3条，质量门槛通过才跑其余10已暴露日期。无训练，不改日期/阈值，不以历史计时宣称稳定加速。

## GRID70 pilot

GRID70 pilot完成保存数据核验，3条、质量3/3、完成3/3；裁决PILOT_PASSED_CONTINUE_FIXED_REGRESSION。所有不利结果保留，修复开发不作为独立确认；单次计时仅描述。

## GRID70 complete

GRID70 complete完成保存数据核验，13条、质量13/13、完成11/13；裁决REGRESSION_REPORTED_ALL_DATES。所有不利结果保留，修复开发不作为独立确认；单次计时仅描述。 全部13日期统一FRAME_GUARD：原版质量11/13→13/13，完整周10/13→11/13（FULL11/13）；最坏完整周成本+0.372%，12个有搜索等时域日期查询少61.999%。跨历史FULL同状态70个/24627反馈逐项相同。0.64%旧成本增幅修复至-0.031%，七月1901步恢复2017；没有据历史时间比更新整套加速结论。

## GRID76 before

GRID76廉价对照闭环冻结：仅普通电网上下文模型通过11.735%校准查询门槛，结构主候选未通过，不冒充结构创新。初始安全128可按固定普通模型提前返回，否则精确复用GRID70；特征构造、首个JSON加载和推理均计入控制。77合法轨迹重放、14/14校准教师动作复现及2实际观测图上下文重建通过。先七月/十二月/三月3条质量压力试验，质量通过再跑其余10日期；压力日期不要求省查询。未优于现有强对照且质量合格前不重训会议对照。

## GRID76 pilot

GRID76 pilot完成保存数据核验，3条、质量3/3、完成3/3；裁决PILOT_PASSED_CONTINUE_FIXED_REGRESSION。所有不利结果保留，修复开发不作为独立确认；单次计时仅描述。

## GRID76 complete

GRID76 complete完成保存数据核验，13条、质量13/13、完成11/13；裁决REGRESSION_REPORTED_ALL_DATES。所有不利结果保留，修复开发不作为独立确认；单次计时仅描述。

## GRID79 before

GRID79同批平衡计时：FULL、普通上下文门控、会议适配两种子；七个原日期×两次反向重复，共56条。模型/日期/质量阈值不改，完整控制计时含图上下文、首载模型、优化与预测；P95及不重叠计算分解同时报告。604个既有付费反馈原策略一致性检查通过。

## GRID79 complete

GRID79同批平衡计时：FULL、普通上下文门控、会议适配两种子；七个原日期×两次反向重复，共56条。模型/日期/质量阈值不改，完整控制计时含图上下文、首载模型、优化与预测；P95及不重叠计算分解同时报告。604个既有付费反馈原策略一致性检查通过。 全部56条与保存数据审计完成，门控质量7/7；相对FULL控制宏平均减少+12.056%，正日期7/7，5%稳定筛查True；相对META_CONTEXT0控制宏平均减少+7.241%，正日期6/7，5%稳定筛查True；相对META_CONTEXT1控制宏平均减少+2.976%，正日期3/7，5%稳定筛查False。查询和控制时间分开，所有负面结果保留。 前51完整条目保留，索引51未完成尝试归档、确认父子进程停止与冻结哈希后原方案续跑；正式56条另有1次不完整尝试，跨中断的计时边界披露。

## ICAPS计算贡献论证主线 — 2026-10-16

用户确认论文重点：条件核查减少昂贵候选预测，并带来实际完整控制提速。已形成独立中文论证方案与同七日期主表：查询总量减少66.89%，控制时间日期宏平均减少12.06%，质量条件7/7、完成5/7与FULL一致、最大完整周成本增幅0.372%。会议适配两种子时间7.24%/2.98%，后者波动保留。十三日期66.47%为另一口径的扩展开发结果，分开报告。当前方法对应CONTEXT_GATE，旧V5首批安全直接返回的分支必须在新稿重写；手写稿/V4/V5不覆盖。GNN辅助，方法与闭环计算证据为贡献主线。本轮无新训练、实验或配图。

## GRID80/81 before

按用户一二优先：现有2035官方52日期家族全部已暴露，改为预先锁定2036交替月份首个周一六场景，用官方Chronix2Grid、固定生成种子20261016和独立依赖目录。GRID81比较FULL、当前门控、始终核查和随机门控0/1，随机范围内跳过概率4/13只用旧校准集确定，范围外均委托核查；两次反向计时共60条。755项既有训练/校准付费反馈检查通过；不训练、不改模型/阈值、不加入SoftGNN新对照。 方案冻结，先通过生成数据资格再执行闭环；生成工程失败不解释为方法失败。

## GRID80/81 qualified

按用户一二优先：现有2035官方52日期家族全部已暴露，改为预先锁定2036交替月份首个周一六场景，用官方Chronix2Grid、固定生成种子20261016和独立依赖目录。GRID81比较FULL、当前门控、始终核查和随机门控0/1，随机范围内跳过概率4/13只用旧校准集确定，范围外均委托核查；两次反向计时共60条。755项既有训练/校准付费反馈检查通过；不训练、不改模型/阈值、不加入SoftGNN新对照。 六个生成场景及全部输入哈希已完成资格检查，即将启动串行闭环；没有按控制结局筛选场景。

## GRID80/81 complete

按用户一二优先：现有2035官方52日期家族全部已暴露，改为预先锁定2036交替月份首个周一六场景，用官方Chronix2Grid、固定生成种子20261016和独立依赖目录。GRID81比较FULL、当前门控、始终核查和随机门控0/1，随机范围内跳过概率4/13只用旧校准集确定，范围外均委托核查；两次反向计时共60条。755项既有训练/校准付费反馈检查通过；不训练、不改模型/阈值、不加入SoftGNN新对照。 已完成保存数据审计。新场景联合筛查=True，两随机种子识别增量筛查=False；全部较差结果和失败保留。

## GRID82 before

按用户先完成工作1和2：已形成固定核查版FRAME_GUARD的中文方法、完整伪代码和贡献证据对应表，旧稿不覆盖；学习门控保留扩展消融。同六个已暴露生成场景比较最终固定核查版与既有ICAPS2022内核电网适配两个冻结种子，两次反向计时共36条。全部沿用模型/参数/控制底座/排序/缓存与0.5%质量条件，原共同校准质量匹配False显式保留，质量不合格不称同质量加速。 453项既有训练/校准付费反馈策略一致性预检通过，禁止新增训练/调参/择种子。

## GRID82 complete

按用户先完成工作1和2：已形成固定核查版FRAME_GUARD的中文方法、完整伪代码和贡献证据对应表，旧稿不覆盖；学习门控保留扩展消融。同六个已暴露生成场景比较最终固定核查版与既有ICAPS2022内核电网适配两个冻结种子，两次反向计时共36条。全部沿用模型/参数/控制底座/排序/缓存与0.5%质量条件，原共同校准质量匹配False显式保留，质量不合格不称同质量加速。 全部回放及保存数据审计完成；META_CONTEXT0:SUPPORTED_EQUAL_QUALITY_SPEED,控制时间宏平均减少+6.435%；META_CONTEXT1:BOTH_PASS_TIMING_MIXED_OR_SMALL,控制时间宏平均减少+0.017%。

## GRID83 原文对标与贡献重写

GRID83原文/作者代码对标与历史行为归因完成，无新训练/回放。META_CONTEXT1在重复0的30次搜索调用中27次首决策委托FRAME_GUARD；META_CONTEXT0为4/32。适配含完整FRAME宏动作，原共同校准质量未匹配、所选费用0如实保留。贡献主线收敛为无需额外元策略训练的分阶段预测核查与实际完整控制提速，GNN/学习门控辅助；中文替换段落独立保存，不覆盖用户稿。新增种子尚未启动，种子研究只能检验该电网适配的训练波动。

[对标方案](protocols/GRID83_source_alignment/outputs/grid83_alignment/ICAPS2022对标与贡献重写方案.md) · [中文替换段落](protocols/GRID83_source_alignment/outputs/grid83_alignment/中文论证替换段落_供用户审改.md)

## GRID84 before

GRID84固定费用种子扩展：保留0/1，仅新增2/3/4各一次拟合，原129/22数据、网络、训练循环、costindex2=0均不变；不重选校准费用。FRAME与五种子同六个已暴露生成场景、两次反向计时，共72条串行回放。所有种子/失败保留，原校准质量匹配False保留；FULL只作质量/时域参照，不混旧计时。

[方案与结果](protocols/GRID84_before/outputs/grid84/)

## GRID84 models

GRID84固定费用种子扩展：保留0/1，仅新增2/3/4各一次拟合，原129/22数据、网络、训练循环、costindex2=0均不变；不重选校准费用。FRAME与五种子同六个已暴露生成场景、两次反向计时，共72条串行回放。所有种子/失败保留，原校准质量匹配False保留；FULL只作质量/时域参照，不混旧计时。 三次拟合与训练反馈预检已完成，尚无新闭环结果。

[方案与结果](protocols/GRID84_models/outputs/grid84/)

## GRID84 complete

GRID84固定费用种子扩展：保留0/1，仅新增2/3/4各一次拟合，原129/22数据、网络、训练循环、costindex2=0均不变；不重选校准费用。FRAME与五种子同六个已暴露生成场景、两次反向计时，共72条串行回放。所有种子/失败保留，原校准质量匹配False保留；FULL只作质量/时域参照，不混旧计时。 已完成保存数据审计；五种子平均控制变化+2.228%，逐种子提速筛查通过1/5，整体加强判据False。

[方案与结果](protocols/GRID84_complete/outputs/grid84/)

## GRID85 机制分析

GRID85已有日志机制分析完成，无新训练/物理回放。旧13+新6日期，131实际调用/47140查询，137完整反馈状态；1096状态方法组合与131调用经独立无策略导入逻辑重算。初始rho<0.9后仍改变动作12/39、2/12；7个旧日期首次分歧均源于该分支核查，完整周变化3低2高并有1恢复完成。同站等预算增量旧14胜0负、新3胜1负，后者存在更大反向奖励差，不宣称稳定结构优势。固定512在新状态一步一致率26/28高于FRAME23/28；下一步关键为固定预算与反馈核查的闭环比较，尚未启动。GRID81完整控制均值差约81.3%来自预测时间差。原裁决、稿件和模型保留。

[完整报告](protocols/GRID85_mechanism/outputs/grid85_mechanism/GRID85_report.md) · [可复算反馈](protocols/GRID85_mechanism/outputs/grid85_mechanism/mechanism_feedback.json.gz)

## GRID86 before

GRID86固定预算机制闭环：最终FRAME_GUARD对固定256/512查询，同六个已暴露2036生成日期、两次反向计时共36条串行回放。固定版无同站补查或安全反馈触发扩展，无可接纳候选时返回空给共同控制器；不新增训练/调参/日期。质量与成本门槛不变，完整同质量提速另要求六日期相同时域，避免把早失败当省时。FULL仅作质量/时域参照，不混旧计时。全部预算和不利结果保留。

[方案与结果](protocols/GRID86_before/outputs/grid86/)

## GRID86 complete

GRID86固定预算机制闭环：最终FRAME_GUARD对固定256/512查询，同六个已暴露2036生成日期、两次反向计时共36条串行回放。固定版无同站补查或安全反馈触发扩展，无可接纳候选时返回空给共同控制器；不新增训练/调参/日期。质量与成本门槛不变，完整同质量提速另要求六日期相同时域，避免把早失败当省时。FULL仅作质量/时域参照，不混旧计时。全部预算和不利结果保留。 保存数据审计完成，总体判读：FIXED_RULE_ALREADY_COMPETITIVE。

[方案与结果](protocols/GRID86_complete/outputs/grid86/)

## GRID87 before

GRID87固定256旧日期质量回归：全部13个原2035日期各新增1条FIXED_256，逐字复用GRID86规则与原物理环境，沿用模型/控制器/缓存/候选顺序；不训练、不调参或换预算。FULL65/68和GUARD70仅作质量/存活/完整周成本/查询参照，不混历史计时。逐日期保存所有结果，并定位相同前缀的首次动作分歧；已暴露开发回归，不称独立确认。

[方案与结果](protocols/GRID87_before/outputs/grid87/)

## GRID87 complete

GRID87固定256旧日期质量回归：全部13个原2035日期各新增1条FIXED_256，逐字复用GRID86规则与原物理环境，沿用模型/控制器/缓存/候选顺序；不训练、不调参或换预算。FULL65/68和GUARD70仅作质量/存活/完整周成本/查询参照，不混历史计时。逐日期保存所有结果，并定位相同前缀的首次动作分歧；已暴露开发回归，不称独立确认。 保存数据审计通过，冻结裁决：FIXED256_QUALITY_FAILURE_GUARD_PASSES。

[方案与结果](protocols/GRID87_complete/outputs/grid87/)

## GRID88 before

GRID88最后一组固定预算对照：原13个已暴露2035日期，FRAME_GUARD与固定512各两次反向计时，共52条串行回放。同原物理环境、控制器、候选顺序与公共缓存，无新增训练/调参/预算/日期。质量门槛沿用0.5%成本与不早于FULL终止；双方13/13质量且时域相同时，至少5%控制/查询减少、9/13日期同向和两重复同向才支持计算优势。固定512与FRAME的实际时间仅本批配对；FULL65/68只作质量/存活/查询参照。完整保留GRID86、GRID87裁决与所有不利结果。完成后转中文稿和机制证据整合，不自动追加模型或预算实验。

[方案与结果](protocols/GRID88_before/outputs/grid88/)

## GRID88 complete

GRID88最后一组固定预算对照：原13个已暴露2035日期，FRAME_GUARD与固定512各两次反向计时，共52条串行回放。同原物理环境、控制器、候选顺序与公共缓存，无新增训练/调参/预算/日期。质量门槛沿用0.5%成本与不早于FULL终止；双方13/13质量且时域相同时，至少5%控制/查询减少、9/13日期同向和两重复同向才支持计算优势。固定512与FRAME的实际时间仅本批配对；FULL65/68只作质量/存活/查询参照。完整保留GRID86、GRID87裁决与所有不利结果。完成后转中文稿和机制证据整合，不自动追加模型或预算实验。 保存数据审计通过，冻结裁决：QUALITY_DIFFERENCE_OR_INCOMPLETE_COMPUTE_EVIDENCE。

[方案与结果](protocols/GRID88_complete/outputs/grid88/)

## ICAPS2022实验源文核对 2026-10-17

复核对标ICAPS2022原文实验：AWA*四类问题与RRT*应用，每类1000随机测试实例；固定配置/DEC对照、停止与调权组件比较、训练/特征/运行案例图。原文无专门系统性规模增长曲线。可扩展性应按我方贡献决定，当前未新增实验/训练。正文与作者代码的滑块难度和minibatch差别已单列。

[核对材料](protocols/ICAPS2022_experiment_review_20261017/ICAPS2022_实验核对与FRAME启发.md)

## FRAME相关论文阅读 2026-10-17

完成FRAME六篇相关论文阅读筛查：AAAI2024 Stop Planner Time、IJCAI2018在线性能预测、EnergyAI2023电网规则/RL比较、ESWA2024渐进决策、2026 Gibbs先验预印本及IJCAI2013 Rational Lazy A*。取得五PDF并核对方法/实验页，ESWA仅出版方摘要。既有分组候选搜索与动作层次分解前例需明确；预印本速度和质量分开报告。新增训练/回放0；未决定新增基线或第二电网。

[六篇方法与实验对照](protocols/FRAME_related_reading_20261017/六篇相关论文_方法实验与FRAME对照.md)

## FRAME中文逐节重写 01摘要 2026-10-17

[新版摘要和指南核对](protocols/FRAME_CN_section01_abstract_20261017/)

五句中文版摘要，实际重读飞书指南三个案例，等待用户审改；本轮止于摘要，旧稿和结果保留。

## META_UNIFIED_20261019

[META配置资料](paper/META_unified_20261019/README.md)：统一已有学习方法的特征/动作、准确权重、费用和结果批次；修正新说明中的历史标签歧义，保留原冻结文件和全部较差结果。无新增训练、仿真或闭环。

## GRID90_EXPERT_frozen

ExpertOp4Grid外部结构启发式：同六个已暴露G2场景、同合法动作库/基础控制器/一步公共预测，FRAME与ExpertOp适配两次交错反向计时共24条。原作者影响图/专家评分保留，官方25候选/每站5/最多3线路默认固定，所有图构建查询计入。严格公共接口不改保护参数，原方法外围恢复重连不带入。三次真实校准状态及未来向量接口检查通过，不按评价结果调参、选日期或增预算。

[方案与结果](protocols/GRID90_expert/frozen/)

## GRID90_EXPERT_complete

ExpertOp4Grid外部结构启发式：同六个已暴露G2场景、同合法动作库/基础控制器/一步公共预测，FRAME与ExpertOp适配两次交错反向计时共24条。原作者影响图/专家评分保留，官方25候选/每站5/最多3线路默认固定，所有图构建查询计入。严格公共接口不改保护参数，原方法外围恢复重连不带入。三次真实校准状态及未来向量接口检查通过，不按评价结果调参、选日期或增预算。 保存数据/原生结构排名重执行审计通过，冻结判读：REPORT_QUALITY_COMPUTE_TRADEOFF_OR_COMPETITIVENESS。

[方案与结果](protocols/GRID90_expert/complete/)

## GRID91_MECHANISM_frozen

用户授权完成第1项机制去留。最终规则另存为模块化纯策略，默认删除冗余qmin条件；原13/G1/G2共25场景做NO_LOCAL删除测试，另3条FRAME分别资格验证三组环境。共28条串行、单次描述性计时；历史FRAME只复用质量和查询，不跨批声称提速。全部输入、日期、模型、阈值和运行上限冻结；不训练、不换网、不扩对照。

[方案与结果](protocols/GRID91_mechanism/frozen/)

## GRID91_MECHANISM_complete

第1项28条回放及保存向量/反馈/成本/独立策略审计完成，3个FRAME环境资格轨迹重现。最终补查删除判读：RETAIN_CONDITIONAL_COMPLETION_QUALITY_EVIDENCE；移除后失去预设质量条件的场景4个。前缀作为初始化，反馈核查为核心，冗余默认qmin作为派生量。所有不利结果保留，无新学习或第二网。

[方案与结果](protocols/GRID91_mechanism/complete/)

## GRID92_COMPLETION_ANALYSIS_complete

同站补查三问完成仅保存记录分析，无新仿真/训练。185个FULL状态、113个触发补查状态：等预算同站对全局后缀20好/1差/92相当；候选级改进密度三组更高，但G1奖励幅度均值反向，G2收益高度集中。预先列出的可见特征未建立稳定触发判据。G2七月一步rho仅降约0.000044而整周成本高78078.900，不补查低6.1231%，差额主要为损耗与再调度；后续258个不同动作，不作首动作全部归因。保留条件模块，不改阈值。

[报告与逐项证据](paper/GRID92_completion_analysis_20261020/GRID92_report.md)

## GRID93_TRIGGER_complete

单条站内支持否决规则完成：去掉最佳后以成对奖励胜率0.5作负证据门控，无拟合或扫描。185个FULL和179个实际FRAME调用筛查，19场景选择不变复用、6条新闭环。判读REJECT_STATION_SUPPORT_TRIGGER，原达标而新规则不达标的场景1个。一步搜索价值与长期动作采用分开，不加第二条规则救结果。所有混合结果保留，无新模型训练或同批提速主张。

[报告](protocols/GRID93_trigger/complete/GRID93_report.md)

## FRAME_TRIGGER_READING_20261020

按用户要求先回读ExpertOp2018、Evolving EnergyAI2026、CCMA2025及ICAPS2022/AAAI2024元推理原文。核对停止/层级排序/追加动作、物理影响图、区域提案与长期协调、计算加执行成本目标及关键表格。区别控制干预门控与搜索补查门控，DQN原文与本地DoubleDQN适配、静态拥塞成功与整周质量、CCMA最佳验证种子测试与全种子均值。不新增规则/训练/回放；未来方向为风险与额外计算价值分开定义，非照搬阈值。

[阅读笔记](paper/frame_trigger_reading_20261020/相近机制论文_方法结论与FRAME启示.md)

## FRAME_COMPLETION_ROLES_20261020

仅保存数据分析：179次实际FRAME调用中109次补查，4次仍过载、83次0.9到1、22次低于0.9；对应预测奖励改善1/4、21/83、10/22。删除补查后4个失去质量的场景，首次实际动作分歧3个已低于0.9，不能据不过载取消补查。低ρ小改进对应的长期成本有正反例；触发、查询位置与动作采用分开。没有新预测、规则试验或训练。

[探索报告](paper/frame_completion_roles_20261020/补查目的分解_探索报告.md)。

## GRID94_SKIP_PREDICTABILITY_complete

已完成固定两小模型的无效补查可预测性诊断。109样本/17日期，标签为返回同一动作77次，改变动作32次；15项补查前可见特征，按日期留组与old13到G1/G2。AUC约0.62；0.95阈值逻辑回归留组取消11误删3、迁移取消6误删1，浅层树均不取消。未建立可靠保守取消信号，原G2七月成本反例未解决。不追加模型/阈值，不部署，长期有害补查仍缺逐次分支价值标签；无新仿真/RL。

[报告](protocols/GRID94_predictability/complete/GRID94_report.md)。

## GRID95_LABELS_complete

学习暂停。固定4状态、2条原轨迹重现及4条只取消一次补查的完整前缀对照已核验；干预前动作/观测/反馈与审阅数值记忆一致，同动作取消29查询后整周轨迹一致。主标签为剩余整周存活/成本/查询向量，短期窗口均保留，权衡不强制二分；不把整策略结果贴给单次调用。具体正负与存活结论见GRID95_report.md，无新拟合。

[报告](protocols/GRID95_labels/complete/GRID95_report.md)。

- GRID96：单次补查价值标签与逻辑回归/64浅树固定评估，见`protocols/GRID96_value_learning/complete/GRID96_report.md`。

GRID96主要判读见 protocols/GRID96_value_learning/complete/学习结果解读_中文.md：增大到56状态和64浅树后仍无可靠门控证据；唯一KEEP为旧试点，未部署。

- [补查标签清理与预训练候选](paper/FRAME_label_cleanup_20261020/README_中文.md)：56例全保留，30/9/5/1/11分组；无新训练、仿真或模型运行。

- GRID97补查轨迹影响：相同56例/15特征/日期隔离的LR、浅树、TabPFNv2离线比较，53非试点主成绩，全部误判保留；无新物理回放或部署。见power-grid-2026/protocols/GRID97_effect_learning/complete/GRID97_report.md。

- GRID98：56例补查前特征由15维保留扩为35维，候选结构/同站反馈/当前状态；隐藏反馈隔离与1120独立复算通过，无新拟合或回放。见GRID98_feature_enrichment/complete/GRID98_report.md。

- GRID99：原15维/补充35维、固定LR/浅树/TabPFNv2日期隔离比较，108上下文、原15维重现及误判完整保留，无新物理回放或部署。见GRID99_feature_learning/complete/GRID99_report.md。

- [学习探索路线与训练/留出诊断](paper/FRAME_learning_route_20261021/学习探索路线_中文.md)：面向用户1月目标继续学习研究，本次无新拟合或回放。

- [GRID103学习补查机制](protocols/GRID103_learning_mechanism/complete/学习机制结论_中文.md)：收益预测与匹配gamma0对照；未建立后续规划优势，固定一批仍具竞争力，所有种子与反例保留。

- [GRID104学习补查闭环](protocols/GRID104_learning_closedloop/complete/GRID104_report.md)：四个已暴露验证日期，六臂两次计时，原模型固定，无扩大或最终测试。

- [GRID110结构核查](protocols/GRID110_structure_audit/complete/GRID110_report.md)：全部4438状态结构可还原，11项摘要区分变化混合，本地未过载不是取消补查条件；无新训练/回放。

- [GRID111动作—反馈对应关系](protocols/GRID111_action_feedback/complete/GRID111_report.md)：局部奖励关联有信号，但插值有上界、单关系对比不能直接推出组合增益；无新训练或物理回放。

- [GRID112配对学习方案](protocols/GRID112_relational_plan/complete/学习模块方案_中文.md)：完整分区—反馈条件化、同参数JOINT/FACTOR比较；输入合同已核验，神经模型与训练尚未执行。

- [GRID113配对学习结果](protocols/GRID113_relational_learning/complete/GRID113_report.md)：固定六次JOINT/FACTOR监督回归、全部种子和日期、离线完整补查评价与审计。

- [GRID114学习失败核对](protocols/GRID114_fit_diagnosis/complete/GRID114_report.md)：零训练，核对目标与正收益决策、配对敏感性及训练/验证完整序列，修正笼统训练不足解释。

- [GRID115保留原主干的匹配方案](protocols/GRID115_matched_module_plan/complete/匹配方案_中文.md)：三组对照、初始化与预算已核对；未训练。

- [GRID116保留原主干的模块训练](protocols/GRID116_matched_module/complete/GRID116_report.md)：固定六次JOINT/FACTOR训练、全部种子与原BASE参照、完整序列评价及审计。

- [GRID117模块原因核查](protocols/GRID117_module_diagnosis/complete/GRID117_report.md)：零训练，成本位置、首次分歧、固定权重函数诊断与加法分工不可辨识核验。

- [GRID118固定基础模型校正方案](protocols/GRID118_fixed_base_plan/complete/固定主干校正方案_中文.md)：固定旧预测参照，补充相同额外训练对照；工程核验完成，未训练。

- [GRID119固定基础模型校正训练](protocols/GRID119_fixed_base_learning/complete/GRID119_report.md)：九次固定训练、原模型额外训练对照、完整搜索效用及独立保存结果核验。

- [GRID120学习目标对照](protocols/GRID120_decision_objective/complete/GRID120_report.md)：普通分类与收益代价加权分类，六个固定训练终点及完整保存结果核验。

- [GRID121按日期留出的学习迁移诊断](protocols/GRID121_date_transfer/complete/GRID121_report.md)：24个固定训练，日期分组、折内标准化和完整折外搜索核验；开发诊断，非独立最终测试。

- [GRID122高代价误停的可见证据核查](protocols/GRID122_false_stop_audit/complete/GRID122_report.md)：折内训练覆盖、首次实际分歧和公开反馈描述；无新增训练或触发规则。

- [GRID123净收益回归与改进事件辅助监督](protocols/GRID123_event_auxiliary/complete/GRID123_report.md)：同容量两种辅助标签、30次固定训练、15份原回归复用及独立保存结果核验。

- [GRID124辅助学习为何未改善补查](protocols/GRID124_shared_gradient_audit/complete/GRID124_report.md)：零训练，完整训练分布梯度、辅助预测误差及全部首次分歧差值守恒核查。

- [GRID125高收益补查为何被低估](protocols/GRID125_value_underestimate/complete/GRID125_report.md)：零训练，训练覆盖/曝光、原Huber目标的稀有大收益影响、已知查询费下界与全部投影动作不变核验。

- [GRID126已知费用与非负改进量学习](protocols/GRID126_known_cost_gain/complete/GRID126_report.md)：30次固定匹配训练、已知费用外置/非负参数化、完整局部搜索与全种子核验。

- [GRID127同一非负链接的均值收益损失](protocols/GRID127_mean_gain_objective/complete/GRID127_report.md)：30次固定MSE/KL训练、30份既有模型复用、全种子与条件折外局部补查核验。

- [GRID127与不学习补查对齐比较](protocols/GRID127_mean_gain_objective/nonlearning_comparison/与不学习方法比较_中文.md)：完整同站、固定一批、旧回归及新KL的相同保存前缀比较；零训练，局部收益与整周质量分开。

- [GRID128学习补查的文献反思](protocols/GRID128_learning_rethink/complete/文献反思与主要问题_中文.md)：五篇论文的任务/方法对照、全案例即时与后续计算价值核验；不训练，区分有利误停、跨批机会与质量目标。

- [GRID129基础策略风险控制准备](protocols/GRID129_baseline_risk/complete/基础规则上的风险控制_中文.md)：原FRAME回退、训练整段损失、单调日期风险与小样本校准条件；尚未训练或新验证评分。

- [GRID130一次LR与经验校准门控](protocols/GRID130_calibrated_gate/complete/GRID130_report.md)：同权重固定/校准门槛、完整同站/固定一批对照；局部收益保护增强但查询仅少4.81%，动作一致筛查未过，全部反例保留。

- [GRID131大补查识别核查](protocols/GRID131_expensive_coverage/complete/GRID131_report.md)：只读训练/校准，大补查可省查询、统一门槛约束、组内重叠与Q-only排序参照；零训练/新验证评分。

- [GRID132分组校准与大补查专门模型](protocols/GRID132_conditional_gate/complete/GRID132_report.md)：两份匹配LR、原权重统一/分组控制；分组校准小幅增加覆盖，专门学习无进一步优势，全部失败与数值警告核查保留。

- [GRID133同信息浅森林与输入范围核查](protocols/GRID133_nonlinear_gate/complete/GRID133_report.md)：一份固定非线性分类未改善大组跳过；完整尾部输入与旧下一批关系任务区分，不把范围缺口当预测价值证明。

- [GRID134完整尾部公开输入资格](protocols/GRID134_full_tail_inputs/complete/GRID134_report.md)：2378原训练/校准门控前缀全候选结构还原，零拟合/新物理，历史64接口错误与恢复收据保留；尚无预测改进结论。

- [GRID135完整尾部/配对匹配分类](protocols/GRID135_matched_tail/complete/GRID135_report.md)：四组各三种子同容量预算，范围与配对未形成稳定优势；[结果解释](protocols/GRID135_matched_tail/interpretation/结果解释与后续取舍.md)保留单调用新误停、种子波动及归档清单修复。

- [GRID136学习目标与标签核查](protocols/GRID136_objective_support/publication/阶段结论.md)：只读奖励精度、旧代理费下分批机会、同站支持及56真实物理标签；区分局部增益与整周业务目标，保存后续标签使用规范建议，无新拟合/仿真。

- [GRID137整周质量标签映射](protocols/GRID137_quality_mapping/publication/阶段结论与训练建议.md)：56既有单次取消45通过/2失败/9参照不足；失败日期覆盖不足，分开质量与计算收益，零训练/回放。

- [GRID138计算空间与查前信息](protocols/GRID138_opportunity/publication/GRID138_report.md)：完整旧25与四开发日期站内预测占比约0.87%/1.86%；建议先评估全局分批学习资格，保留17分支与历史计时差异，零训练/回放。

- [GRID139全局分批学习资格](protocols/GRID139_global_feasibility/publication/GRID139_report.md)：305既有FULL状态、3788前缀；延迟收益与同原动作压缩任务草案，强简单对照齐备，零训练/仿真。

- [GRID140全局压缩一次训练](protocols/GRID140_global_compression/complete/GRID140_report.md)：一次森林+逐调用校准，验证动作差异0/62、查询减少4.141%，离线增量筛查False；零新物理。

- [GRID141全局反馈与高分错误诊断](protocols/GRID141_feedback_diagnosis/complete/GRID141_report.md)：加权先验、455叶支持及726公开前缀核验；28候选统计未训练，保留全部负例和标签边界。

- [GRID142六项批次信息训练](protocols/GRID142_batch_representation/complete/GRID142_report.md)：一次表示干预，开发验证动作差异1/62、少查2.788%，原筛查False。

- [GRID143继续少数类与校准日期诊断](protocols/GRID143_continue_risk/complete/GRID143_report.md)：两固定模型排序、8个校准日敏感性重放、122后续路径；零拟合，原阈值不变。

- [GRID144类别损失一次训练](protocols/GRID144_class_mass/complete/GRID144_report.md)：原18特征、两类质量平衡，开发动作差异0/62、少查0.155%，原筛查False。
