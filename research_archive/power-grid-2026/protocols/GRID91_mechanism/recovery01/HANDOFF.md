# GRID91 第1项机制去留：运行交接

用户2026-10-20要求把第1项完成后汇报，重点解决昨天提出的机制问题。本轮只完成机制去留与方法精简，不启动第二电网、比例验证、CCMA或任何新训练。

固定方案design.json。三条FRAME_GUARD先分别验证old13/G1/G2环境能逐步重现历史，然后25条NO_LOCAL_FINAL删除回放；原13+G1六+G2六全部保留，单次串行，共28条。可复用FRAME/FULL质量和查询记录在references中逐个指定；禁止混历史计时声称新提速。初始种子、控制器、模型/资产、候选顺序和所有阈值不变。仅删去默认冗余qmin条件，前缀128、批次64、停滞2、分支阈值0.9保持。

预检316次FRAME默认等价、316次独立NO_LOCAL核查、194次隐藏后缀扰动、114个合成边界（两策略228次）通过；code_freeze576项、data_freeze1817项。所有源码/输入哈希要保持，历史代码和结果未改。项目Python：D:/Codex/GRL研究/work/grid00/.venv/Scripts/python.exe。

已有隐藏pipeline.py串行负责worker.py、audit.py、finalize.py、archive.py、git_push.cmd和complete.json。正常运行不启动第二运行器，不重复审计；无计时重复，不为结果追加预算、训练、日期或阈值。流水线日志pipeline_stdout.log/stderr.log，单worker日志worker_NN.log，状态pipeline_state.json、manifest.json、active_worker.json、finished.json、pipeline_error.json、review.json、decision.json、complete.json。

错误时保留部分目录与日志。不能安全确定已完成条目、冻结哈希和进程时不恢复。资格回放必须先通过轨迹签名，工程故障不当方法失败。不得删除完整条目或按中间结局更改方案。总窗口3小时、每条1200秒、最多56476物理步、250000公共预测，无新训练。

审计在benchmark全部结束后检查保存动作/观测向量、逐步成本、合法反馈、候选与别名、最优已观测选择、独立策略重执行、公共上下文、第一实际动作分歧及互斥计时。无独立AC重求解。25场景条目有重复日历日期且都已暴露，不能称25独立场景确认或跨网泛化。

冻结去留判据：取消补查后，若有场景失去原FULL质量达标而FRAME保持，则保留补查为条件质量改进；否则不称它是该质量门槛所必需。质量沿用不早于FULL终止、共同完整周成本最多+0.5%；完整报告质量、存活、全部成本与查询正反例。不是普适质量保证，不根据结果另找新触发器。

完成应检查outputs/grid91_mechanism/GRID91_report.md、精简方法说明_中文.md、review.json、decision.json、first_divergences.json、D盘协作副本和Git本地/远端SHA。报告覆盖：前缀作为初始化、反馈核心、128/64经验来源及排序依赖、默认256冗余、最终补查去留和边界。正文旧稿不覆盖。通知用户第1项完成及主要数值；第2项不自动启动。


## 工程入口纠正01

首次启动误用grid00解释器，在import torch时退出，尚无回放目录、观测或控制步，完成0项。错误与冻结快照保存在interrupted_attempts/attempt01，2393项冻结哈希不变。正确既有控制运行时是D:/Codex/GRL研究/work/grid02/.venv/Scripts/python.exe（Torch2.8.0+cpu、Grid2Op1.12.5、LightSim1.1.0、CVXPY1.9.3），未安装任何依赖。原流水线以该解释器恢复，当前日志pipeline_resume01_stdout.log/stderr.log。前述grid00路径只适合不依赖Torch的预检，勿据旧日志判当前故障。
