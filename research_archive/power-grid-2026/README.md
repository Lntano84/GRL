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
