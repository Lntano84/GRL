# 当前电网论文证据

用户核对材料（2026-10-13）：[六页数据工作簿](../research_archive/power-grid-2026/protocols/GRID_data_review_20261013/outputs/grid_data_review_20261013/GRID_数据与对比.xlsx)、[详细解释](../research_archive/power-grid-2026/protocols/GRID_data_review_20261013/outputs/grid_data_review_20261013/当前情况详细说明.md)、[128条正式逐次成绩](../research_archive/power-grid-2026/protocols/GRID_data_review_20261013/outputs/grid_data_review_20261013/GRID55B_128_runs.csv)、[16条新先验逐次成绩](../research_archive/power-grid-2026/protocols/GRID_data_review_20261013/outputs/grid_data_review_20261013/GRID59_16_runs.csv)。仅导出已有冻结数据，未新增实验。宏平均与汇总比例、完整周与共同失败、总控制与单步p95均分开展示。

**当前结论（2026-10-13，GRID57—59之后）：整体有界搜索框架的工程提速保留，GNN相对强作者频次顺序的实用增量未建立。GRID56的方法稿与贡献表属于较早证据快照，必须与本页及后续收尾一起读取。**

- GRID55B：原软压缩候选相对同共有缓存FULL完整控制时间宏平均减少15.752%；两种子均16/16完成，最坏完整周成本变化+0.08045%，守住0.5%投资尺度。PPO/SoftGNN仍略快，不能写成全面支配。
- GRID57：STATIC是作者历史教师频次降序动作库，经当前合法掩码与规范化；不是随机或无经验顺序。STATIC质量8/8通过；GNN相对它成本1好/6差/1平，平均高0.509个FULL归一化百分点；查询宏平均少1.90%，汇总反而多6.69%。此批不作跨批公平速度主张。
- GRID59：两个频次先验GNN种子质量4/4通过，六月与FULL/STATIC共同在1901步失败，其余三周完成。相对同批STATIC查询平均少2.901%、各2/4日期正向，实际控制时间宏平均多0.570%；未达冻结5%和至少3/4正向的继续门槛，停止这次先验改动。
- 进一步账目核对：九月GNN仅少查2.75%—3.26%，但相对STATIC有6步动作不同、818个后状态hash不同，完整周成本高1.867个FULL归一化百分点；N1搜索少0.30—0.34秒，搜索外控制多1.57—1.74秒。图/先验/前向约0.039秒，不能单独解释整体变慢。此为描述性账目分解，未识别独立因果机制。
- 新RL训练、图消息传递必要性、普遍泛化以及录用概率均未建立；不将公共缓存归为GNN独有，也不把70%查询减少写成70%完整控制提速。

最新材料：[GRID59审计](../research_archive/power-grid-2026/protocols/GRID59_complete/outputs/grid59/GRID59_report.md)、[更新后的证据与定位](../research_archive/power-grid-2026/protocols/GRID59_closeout/outputs/grid59_closeout/GRID59_closeout.md)、[当前贡献表](../research_archive/power-grid-2026/protocols/GRID59_closeout/outputs/grid59_closeout/claim_evidence_current.json)、[同批差异分解](../research_archive/power-grid-2026/protocols/GRID59_closeout/outputs/grid59_closeout/paired_diagnostics.csv)。

接下来首先是论文定位：若以整体框架为主，需核对有界搜索与现有方法的具体区别，并保留频次强基线；GNN暂列可替换组件。若坚持GNN为核心贡献，则需另立学习假设、重新冻结评价，不能直接扩大当前配置或进入RL。尚未安排新实验。

## 历史证据包

GRID56最终审计收口通过：84项冻结哈希、9项RAW锚点、174项新补测文件收据、3842项历史归档条目、128槽位来源与21项统计/bootstrap/选型公式。无新仿真或训练，不宣称独立AC重求解。完成质量—速度主表、科学图、方法结果工作稿、贡献边界与复现入口；训练教师费用按20训练周分离为50.68分钟/146521个N1查询，最终两拟合调用89.99秒，已有PPO成本仍未知。现成STATIC_ID开发消融接近GNN，下一项仅补最终配置的廉价顺序消融，不扩主对照。

完整材料：[证据报告](../research_archive/power-grid-2026/protocols/GRID56_evidence_complete/outputs/grid56/GRID56_report.md)、[方法草稿](../research_archive/power-grid-2026/protocols/GRID56_evidence_complete/outputs/grid56/method_and_results_draft.md)、[贡献检查](../research_archive/power-grid-2026/protocols/GRID56_evidence_complete/outputs/grid56/claim_evidence.json)。

历史关机交接仅记录当时状态，不表示当前仍有未完成审计。
