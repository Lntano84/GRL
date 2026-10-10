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
