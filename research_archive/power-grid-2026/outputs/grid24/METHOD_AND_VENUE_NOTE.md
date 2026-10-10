# 质量补强与投稿时间核对（2026-10-10）

本轮先试已发表的验证—回退思路，不改监督模型或 PPO 权重。Imitation Learning for Intra-Day Power Grid Operation through Topology Actions 的 Verify+Greedy / Verify+N-1 在预测动作经仿真验证后仍高风险时使用规则专家。这里借用这类通用补救，具体阈值沿用当前 LJN 已有 rho_safe=0.9。该阈值是工程安全余量，不是正式安全保证。

原文：https://arxiv.org/html/2407.19865v1 （§5.3、Table 2）。

相关正式前作还包括 Action Set Based Policy Optimization for Safe Power Grid Management（ECML-PKDD 2021 ADS）：https://arxiv.org/html/2106.15200v1 。因此学习候选加仿真验证本身不作为新意。

本次动作：先预测排序并仿真 128 个去重合法候选；没有可接受候选或所选候选的仿真最大 rho>=0.9 时，只查询此前未查询的剩余候选，保持原搜索目标与精确破并列。仅使用当时合法公共反馈。不按“后半周”切换，不读取未来负荷或预留评价结果。

先做三月最差开发周的两个 GNN 种子。通过质量门槛才展开另外十条预先冻结的开发轨迹（原 PPO 与两个 GNN，同规则）。旧版本、参数、教师成本与结果全部保留。

## 更早投稿的官方时间

- ICAPS 2027 正式长文：摘要 2026-12-07，全文 2026-12-14，AoE。会议 2027-06-27 至 07-02。主会接受学习支持的规划；2027 还有 Planning Under a Different Name 赛道。只有在论文有明确的搜索/决策贡献时才考虑，不能为了日期勉强改定位。https://icaps27.icaps-conference.org/calls/cfp/
- ECAI 2027 官网公布全文日期 2027-04-14，会议 2027-10-02 至 10-07；包括 PAIS 应用活动。具体赛道要求和截止应等详细 CFP 核对，不能把主会日期自动视为 PAIS 日期。https://ecai2027.org/
- ECML-PKDD 2027 举办日期 2027-08-30 至 09-03，详细 ADS 截稿尚未找到。作为参考，2026 ADS 全文截止是 2026-03-12；不能据会议八月举办推断还有十个月才交论文。https://ecmlpkdd.org/2027/ 、https://ecmlpkdd.org/2026/submissions-ads-track/

不提前保证这些会议会接收当前贡献；不同时提交实质相同论文。投稿以正式长文为目标，Workshop/Short/Findings 不代替 CCF B 正式长文。
