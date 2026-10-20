# GRID96 当前执行与续接

用户重新授权用可靠单次取消标签试简单学习，规模稍大。方案和所有核心源码已在新回放前封存推送5c96a5fc。56个状态/21条已有场景记录/17个日历日期；21条事实资格+53条单次取消共74条新回放，3条已完成GRID95试点分支复用且不计主要预测成绩。

当前隐藏run.py按冻结顺序串行运行；不要启动第二运行器，不重跑完成项，不改design/code/data冻结或模型/阈值。先检查pipeline_state.json、manifest.json、pipeline_error.json与run_stdout.log/run_stderr.log及worker进程。正确运行时work/grid02/.venv/Scripts/python.exe。Python venv启动器与原生运行时构成多级父子进程是正常现象。

74条全部完成、finished.json通过且原worker/run进程退出后，依次执行work/grid96_value_learning/audit.py、learn.py、audit_learning.py、finalize.py。必须先保存数据核验，再拟合模型；不得在回放期间学习或重型审计。最终用work/GRL-results-integration/scripts/git_push.cmd发布归档，并确认远端SHA=本地SHA。

标签主时域计划整周末2017步。原事实提前失败也保留，不能截到其失败点冒充整周标签。完全同物理路径且取消不增查询记SKIP；不同路径双方失败记UNRESOLVED；存活优先，两条都完成时按成本/查询/过载向量Pareto分KEEP/SKIP/MIXED。无成本转换系数。15个补查前特征；逻辑回归C1和64棵深度4浅树森林；日期留一与old13→G1/G2，不调参，0.95固定取消阈值。已知试点可以在训练日期内出现，但不计主要评估。不得部署门控，不得相加重叠的单次反事实收益作整套策略收益。

完成应简短中文报告真实指标和局限，给GRID96_report.md链接。不要恢复旧自动化或旧任务，不改论文。

执行补充：80ac1fed已加入执行专用postprocess.py（独立收据orchestration_receipt.json，不改变冻结控制/标签/模型）。隐藏postprocess已启动，负责等待74条结束且所有run/worker进程退出，再顺序调用冻结的audit/learn/audit_learning/finalize并推送。应检查postprocess_state.json和postprocess_error.json；它正常运行时不要手动重复审计/拟合/归档。publication.json通过并远端SHA=本地SHA才算发布完成。

## COMPLETE
74条新回放及3个已核验分支审计完成；56标签和固定简单模型均完成，原始全部轨迹已归档推送08680831，解读/版本/描述性分层推送7becce6a。没有学习门控部署，不再运行audit/learn/finalize或新实验。主解读：学习结果解读_中文.md。唯一KEEP是旧试点，53新标签无KEEP；日期留出LR AUC0.383/RF0.587，不支持替换原补查规则。最终实际Git本地/远端核对见final_verification.json。

