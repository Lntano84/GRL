# GRID184待用户决定新物理范围

182六次完整训练小模型拟合及三旧F权重复用、183输入/补查核心、184独立模型与Search.choose保存状态资格均已通过。最新方案design.json经protocol_order_receipt.json将第二重复改为第一32条严格逆序，全部64身份与参数不变，旧案已保存。

当前没有physical_authorization.json、benchmark_started.json或runs目录，不能启动pipeline/worker。用户确认具体64条范围后，才可记录真实用户授权及当前design SHA，使用work/grid02/.venv/Scripts/python.exe启动唯一pipeline.py。不得重训/生成/恢复旧运行器。

四个已暴露开发日期2037-03-02/05-04/08-03/10-19，各scenario100。FRAME/BATCH1/F三种子/共识三种子，64串行，总窗口4h、每条1200s，保留部分输出与全部不利结果。首条计划内FRAME承担实际接口检查；尚无真实控制器/求解器资格运行，发生工程故障先保留再核对，不能当方法失败。最终8不生成，无论文改动。

authorization_guard_stderr.log是有意验证缺少授权时阻止worker的异常，只是资格测试，不是实验故障。offline_review不是整周质量结果。训练集工程前向不当新泛化成绩。

## 2026-10-25 用户已确认启动

用户直接回复“启动”，对应本聊天上一条64条范围。physical_authorization.json绑定最新design SHA，launch_preflight.json冻结核对通过。只启动唯一pipeline.py；当前原始pipeline日志pipeline_stdout.log/stderr.log。旧authorization_guard_stderr.log仍仅为有意阻止未授权worker的测试。首条FRAME实际运行须核对。

实际启动已核实：唯一pipeline实际PID 32568、worker launcher 39816；首条FRAME_GUARD已前进到2017步，当前无pipeline_error。启动收据launch_verification.json；grid184-consensus跟进已启用。正常不启动其他运行器、不安装警告建议包。record_launch_attempt01/02是收据小脚本自身在写入前的编码/游标问题备份，未影响benchmark，不当方法失败。

## 2026-10-25 用户即将关机

64条完整回放与320个必需文件已核对落盘，finished/manifest一致。当前原audit仍运行，无分段检查点，未主动停止进程。若用户关机中断，只续审计/归档/推送，先检查review/decision等是否已落盘；不得重跑任何物理条目或直接重启全pipeline。保留原审计日志，审计恢复不扩大已完成的benchmark窗口。见shutdown_readiness.json。

## 审计I/O低效已定位，修正版仅准备

原audit.py在逐步循环中访问z[action][j]/z[observation][j]，每次会重复解压完整NPZ成员，造成不必要的长时间。work/grid184_audit_io_fix/audit_cached.py只将两数组每run各读取一次，保留全部assert和原源哈希，未修改冻结源、未停止原审计、未启动第二审计。关机后先核对原审计及review/decision状态；需要重新审计时使用该修正版，不重复物理回放；若原审计已通过则不重复。见preparation.json，尚不代表通过。
