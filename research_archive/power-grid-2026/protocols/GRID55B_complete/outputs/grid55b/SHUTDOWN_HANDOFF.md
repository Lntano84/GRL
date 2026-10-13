# 2026-10-13 GRID55B 关机交接

八条有限计时补测已全部完成2017步，墙钟841.98秒，保存数据审计通过。与原两次的16项比较中，物理向量、动作/观测哈希、逐步成本、合法反馈和查询全部一致。运行期电源事件0，临时防待机请求已释放，永久电源计划未改。

原GRID55的128条、RAW报告、受待机影响计时和旧选型均保留不改。修复视图只将受影响日期2035-11-12_9的全部八方法同样采用原重复0/新重复2；其余七日期仍为原重复0/1。每个计时槽位的真实来源保存在qualified_view/provenance.json，不减待机秒数、不替换为CPU秒数、不选择最快模型种子。这不是新独立确认。

## 当前结果

- 新版两种子相对FULL宏平均控制时间减少15.438%/16.065%，平均15.752%，八日期家庭bootstrap95%区间8.256%–23.657%，7/8日期均值为正；查询减少70.00%/70.62%。两种子完整周最大成本增幅均0.08045%，均通过冻结投资筛查。
- 原版两种子控制时间减少18.336%/18.429%，但完整周最大成本增幅1.74036%/1.70438%，各14/16质量合格；没有早于FULL失败。
- 新版相对原版平均控制时间多3.50%。按原对称质量—时间规则，工程主候选为软压缩新版，是质量筛查选择，不是新版更快。没有修改默认源码，原版仍作为速度/质量消融保留。
- 新版仍比PPO慢2.75%–3.53%，比SoftGNN慢0.89%–1.89%；不能宣称全面领先。PPO最坏成本增幅12.434%，SoftGNN两种子0.177%/0.169%。公共先验/DC/安全QP缓存不是GNN独有贡献。

## 审计工程记录

首次离线审计的WindowsPowerShell5诊断模块加载失败；System日志可读性守卫正确中止，未把空事件列表当作证明。保留audit.py原冻结源码、audit_v0_power_host_failure.json；audit_power_v1.py仅将两个事件日志子进程改用已安装的PowerShell7，科学检查/统计/选型未改。没有重复求解或训练。另用独立直接配对公式核对21项比较（实际数量见arithmetic_crosscheck.json），没有独立AC重求解。

## 交付及恢复

- 主报告：outputs/grid55b/GRID55B_report.md
- 审计：supplement_audit.json、review.json、arithmetic_crosscheck.json
- 计时主表与来源：qualified_view/review.json、method_summary.csv、family_comparisons.csv、provenance.json
- 原证据：outputs/grid55/；新八条完整向量仍在outputs/grid55b/runs/本地保存。
- Git小型归档：research_archive/power-grid-2026/protocols/GRID55B_before与GRID55B_complete。

没有待续跑实验；不要再次执行已完成run.py或重复完整归档。LG01、GRID52、GRID55自动跟进均保持暂停。下一次继续时先读本交接和最新研究状态，进入论文材料/必要归因证据整理，不自动启动新训练、扩基线或计时重跑。
