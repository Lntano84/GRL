# GRID100 第一阶段：训练与验证数据采集

用户已授权继续完成学习数据第一部分。当前批量采集已启动，方案与实现另行封存于 `bulk_plan.json`、`bulk_code_freeze.json`，不覆盖原试点冻结文件。

- 40个预定场景：32训练、8验证，20个日期组。已完成的 `2037-01-05_100` 直接复用，不重生成、不重放。
- 最终测试8场景仍为种子计划，不生成时序、不采集输入或标签。所有日期、种子、原控制器、候选顺序和反馈规则沿用既定方案。
- 正确解释器：`D:/Codex/GRL研究/work/grid02/.venv/Scripts/python.exe`。
- 有限隐藏流水线入口：`work/grid100_data_foundation/bulk_pipeline.py`。先串行生成剩余39场景，再逐场景串行FULL教师回放和保存数据核验，最后归档推送。不训练模型。
- 查看 `bulk_state.json`、`generation_progress.json`、`bulk_manifest.json`、`dataset_receipts/`、`bulk_error.json`（若有）。生成、教师与提取日志在 `bulk_logs/`。
- 启动前通过601项补充冻结核对、96条试点标签重现与收益复算、96条前缀输入核对、日期隔离检查；准备检查显示已完成1/40。
- 结束标志是 `bulk_complete.json`，需要核对 `bulk_review.json`、`GRID100_report.md` 及本地/远端SHA。采集结束不代表训练或方法验证完成。
- 失败时保留错误和部分输出，不自动删除、重试或替换场景。已有完整收据验证后跳过；只有未落盘收据的完整教师结果需要人工检查后做纯数据提取，不能直接重复物理回放。
- 总采集窗口86400秒，每个生成场景1920秒外部看门狗、教师1320秒外部窗口。没有无限重启逻辑。
- 数据标签为同一状态下补查即时奖励增益，不能直接当整周成本收益。前缀来自FULL路径，后续还需FRAME分布/闭环验证。完整观测包含日历，模型输入投影须按 `training_input_contract.json` 排除日期等标识。
- 不恢复旧自动化，不追加模型、预算、日期或第二电网实验。

启动核验：2026-10-21T13:48:52Z，唯一流水线实际PID4160（虚拟环境启动器3512），生成子进程6944/37720；后续以实时进程与bulk_state.json为准。当前处于generating_trainval，已有试点1/40，未开始新模型拟合。

## 用户关机断点 2026-10-21T15:29:24.261656+00:00

后台进程已停止。生成收据16/40，教师数据1/40；中断场景2037-06-01_100，其现有文件保留。601项冻结与所有完整生成收据哈希已核对。先读shutdown_checkpoint.json，不直接重启原pipeline；保持最终测试封存、不重新生成完整项。

## 2026-10-22关机续跑资格

16个生成收据及全部数据哈希、601项冻结、试点数据均核对通过；第17项尚未创建输出目录，不存在需要覆盖的部分数据。旧生成日志原样移至interrupted_attempts/user_shutdown_01。恢复入口resume_bulk_01.py，仅在内存中扣除关机前已用时间，原pipeline/生成器/模型/日期/种子/规则均不变。恢复后日志bulk_pipeline_resume01_stdout.log/stderr.log，当前生成日志仍在bulk_logs/generation_trainval.log。不启动第二运行器，不训练、不解封最终测试。

恢复已启动：2026-10-22T00:39:46.737392+00:00，实际流水线PID32804、生成启动器PID35964。仅一个采集流水线，旧PIDs已失效。当前继续第17项；实时进度以bulk_state.json和generation_progress.json为准。

## 日志属性工程恢复02

40生成通过、16标签通过后，第17教师在1157步后因OptimModule缺少logger停止，错误保存在interrupted_attempts/logger_failure_01。恢复入口resume_bulk_02.py，教师诊断包装teacher_logger_recovery.py；只补原模块logger，无数值/决策变更，601冻结不变。只重跑不完整第17项，接纳前须通过recovery_02_prefix_check.json的1157步精确核对。保留全部失败，原16完整项不重跑。新日志bulk_pipeline_resume02_stdout.log/stderr.log；bulk_logs仍为当前逐项日志。不再启动旧resume01。最终测试封存，模型拟合0。
