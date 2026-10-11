# GRID28 GNN 与 PPO 实际同状态诊断

4个日期中，交付轨迹逐步完全相同的有4个。匹配26次辅助调用；相同候选池26次，相同选择26次。

在这些匹配调用中，GNN查询7312次、PPO查询7777次，少465次（5.98%）。每调用查询方向：{'gnn_fewer': 7, 'ppo_fewer': 4, 'equal': 15}。公共反馈6178/6178逐项相同。

Posthoc exposed-date diagnostic. Matched query counts can show this frozen ranker configurations differ under the same stopping rule; not GNN architecture identification, independent generalization or stable whole-control timing.

不能将非搜索控制时间差解释为图模型效果；没有比较同信息MLP等强对照，也没有打开封存日期。查询节省是当前配置在已暴露状态上的读数，不升级为正式架构/泛化贡献。
