# LG01 真实监督训练试验

已完成并复核：480 次正式标签、192 次正式测试。裁决为 ACCEPT_CHEAP_STOP_GNN：采用验证封存的固定 LOSS-EXPAND，停止当前 GNN 配置，不进入 RL。家庭宏平均 G=-0.09%，95% 家庭 bootstrap 区间 [-0.51%, 0.31%]，正向家庭 3/8。原结果、模型和封存代码哈希未变；最终报告见 `lg01/LG01_report.md`。

结果跟进自动化 `lg01` 已暂停。交付清单阶段读取自身 Windows 字节锁产生的错误已留档，通过单独收尾脚本补齐清单，没有重做任何求解或模型拟合。复核证据见 `lg01/LG01_postrun_review.json` 和 `lg01/LG01_delivery_verification.json`。

本次固定执行 32 个新种子家庭（14–45），按原规格生成 64 个 NOM-160 名义实例和 128 个故障状态。训练/验证收集 480 个真实动作结果；封存 GNN 与 B* 后执行最多 192 次测试。程序自动训练、评价、独立验解并输出裁决，不自动启动 RL 或追加扫参。

实时进度：`lg01/RUN_STATE.json`。逐次持久日志：`lg01/events.jsonl`。`LG01_process.json` 指向当前进程、最新控制台和错误日志。早期日志按版本保留。

此前设置每 15 分钟的结果跟进（automation id：`lg01`），现已完成收口并暂停。

两个首次动作标签产生前的 I/O 缺陷已修复并留档：Windows 文件名中状态 ID 的 `|` 改为可逆 `%7C`，以及 NumPy 标量转为原生 JSON 数值/布尔。首次有效 NOM-160 求解保留，没有重跑求解器。详见 `lg01/LG01_path_amendment.json` 与 `lg01/LG01_serialization_amendment.json`，原源码和冻结文件均按版本保留。真实修复方案落盘往返已通过。

最终产物（完成前不会冒充结果）：

- `lg01/LG01_report.md`、`lg01/LG01_results.json`：指标、家庭区间、分组、开销、资格与裁决。
- `lg01/LG01_test_pairs.csv`、`lg01/LG01_test_families.csv`：原始成本和家庭宏平均。
- `lg01/LG01_independent_audit.json`：独立验解、所有 Y/Z 固定条件和 κ，以及 NOM-160 的逐阶段输入锚点。
- `lg01/LG01_freeze.json`、`lg01/LG01_policy_seal.json`：家庭划分、源码哈希、配置和测试前封存。
- `lg01/nominal/`、`lg01/states/`、`lg01/labels/`、`lg01/models/`、`lg01/test/`：计划、动作结果、逐 epoch 检查点与正式测试。
- `lg01/code/`：冻结的源码快照。来源目录没有 Git 提交，明确记录为 null 并保存文件哈希。

固定动作标签不支付网络专属计算开销。训练图从 RINS 标签运行自身合法 LP 重建，交付后缓存并单独记账。正式 GNN 与池化 MLP 在各自完整在线流程内重建输入并评分；廉价固定 FULL 不执行不需要的松弛或图构造。

断点恢复执行同一个命令，已完成动作不会重跑。进程异常中止的求解动作按完整预留预算保守记账，保留预验合法参考方案，不重新获得 20 秒；其严格计时资格失败并公开。若图缓存未完成，训练会停止并说明缺失，不伪造标签或偷偷追加动作。恢复中的模型训练使用保存的优化器、随机状态和 batch 游标，并保留原 30 分钟期限。系统文件锁阻止两个运行器同时执行。

```powershell
& 'C:\Users\windows\Documents\Codex\2026-10-06\lg01-gnn-rins-rl-lp-20\work\lg01\.venv\Scripts\python.exe' 'C:\Users\windows\Documents\Codex\2026-10-06\lg01-gnn-rins-rl-lp-20\work\lg01\run.py' all
```

若运行停止，先读取 `lg01/RUN_STATE.json` 的错误原因。资格未通过不算研究阴性结果，也不改家庭、预算或参数来补救收益。

这是一项已完成的单次真实求解筛查；八个家庭尚不能支持广泛的图模型或 RL 优劣结论。
