# FRAME：电网拓扑控制中的仿真搜索

FRAME 研究在线电网拓扑控制中如何分配有限的候选仿真。方法以分批全局搜索寻找可行方案，根据搜索反馈决定继续扩展，并在适用时补查当前最佳动作所属变电站的候选。动作去重时保留变电站归属，帮助补查正确覆盖候选。

本仓库保存 Grid2Op / L2RPN 环境中的方法、实验报告、对照、分析数据和复现材料。研究仍在开发阶段，结果需结合所测电网、场景及运行条件理解。

## 论文与实验

- [最新论文修订稿（PDF）](research_archive/power-grid-2026/paper/FRAME_eEnergy_revised_20261118/FRAME_eEnergy_revised.pdf)，同目录包含 LaTeX 源稿和表格数据。
- [电网实验与报告索引](research_archive/power-grid-2026/README.md)，涵盖 FRAME、学习停止策略、排序和独立对照。
- [等预算搜索机制分析](research_archive/power-grid-2026/paper/GRID92_completion_analysis_20261020/GRID92_report.md)，比较同站补查与相同查询数量的全局继续搜索。
- [GRID330 闭环对照](research_archive/power-grid-2026/protocols/GRID330_remaining19/results/GRID330_closed_loop_report.md)和 [GRID334 LogEIPC 对照](research_archive/power-grid-2026/protocols/GRID334_logeipc_closed_loop/results/GRID334_report.md)。
- [当前研究状态](docs/RESEARCH_STATE.md)与[后续工作](docs/NEXT_STEPS.md)。

## 目前结果

在 185 个已保存的穷举搜索状态中，113 个触发同站补查。与等预算继续原全局顺序比较，同站补查在当步预测奖励上为 20 胜、1 负、92 平。该分析使用同状态保存反馈。

在四个已使用的 2039 开发时域，混排与保护后的 261 方案相对原 FRAME 合计减少 20.95% 候选查询和 9.54% 全部预测。三个共同完周中，两周运行费用下降，一周持平。两个有同批计时的完整周控制时间分别增加 3.06% 和 0.90%。同配置 208 和 261 的四个时域轨迹与查询数一致。

## 仓库与复现

论文目录中的表格可由随附 CSV 和脚本重建。历史运行的完整软件版本与机器指纹仍有缺口，表格复算和完整控制器复现属于不同范围。

实验报告、配置、源码及归档清单位于 research_archive/power-grid-2026/。部分大型原始轨迹、场景和模型保存在本地 D 盘，清单中记录了对应路径和哈希。

2026 年 11 月整理时，早期 IM、CCIM、TAIM 探索和历史论文稿件从当前仓库目录移出，本地备份位于 D:/Codex/GRL研究/archives/github_cleanup_20261122/。Git 历史仍可查询早期提交。其他决策研究保留在 research_archive/decision-studies-2026/。
