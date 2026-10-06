# LG01 训练与封存测试结果

裁决：**ACCEPT_CHEAP_STOP_GNN**。
主指标家庭宏平均 G：**-0.09%**；家庭级配对 bootstrap 95% 区间：[-0.51%, 0.31%]。
八个测试家庭中 3/8 为正；相对 RINS：0.63%。
最强廉价策略 B*：fixed；验证最佳固定动作：LOSS-EXPAND。

G 的分母是修复成本 max(1, |J_r|)，不是 B* 成本；这是八家庭的一次投资筛查。

| 家庭 | G | 相对 RINS |
|---|---:|---:|
| 16 | -0.10% | 0.57% |
| 20 | -0.32% | 0.62% |
| 23 | 0.84% | 1.25% |
| 30 | 0.50% | 0.19% |
| 31 | 0.04% | 0.16% |
| 35 | -0.07% | 1.57% |
| 42 | -0.31% | 0.86% |
| 45 | -1.32% | -0.22% |

ρ 分组：{'0.75': -0.005955288778825606, '1.1': 0.004133895351368568}；求解器种子分组：{'0': 0.0007394144494693861, '1': -0.0025608078769264234}；故障分组：{'D1_m0_2p': -0.0031628146098324826, 'D2_all_1p': 0.0013414211823754457}。
推理 p95：13.45 ms；图构造 p95：122.69 ms，均计入在线预算。
完整资格检查：{'independent_check': True, 'complete_fixing_sets': True, 'kappa': True, 'online_timing': True, 'inference': True, 'training': True, 'total_compute': True, 'uninterrupted_generation': True}。测试超时协议违例 0 次。
验证逐状态候选池 oracle 相对验证最佳固定动作：0.63%。
训练成本：GNN 37.4 s，MLP 5.8 s，树 0.2 s。
累计记账计算耗时 6.58 小时；正式标签 480 次，正式测试 192 次。
LP 资格兜底 0 次；交付来自自身 LP/修复方案 3 次；中断 0 次。
动作频率：{'GNN': {'RC-EXPAND': 48, 'RINS': 9, 'LOSS-EXPAND': 6, 'RANDOM-EXPAND': 1}, 'Bstar': {'LOSS-EXPAND': 64}, 'RINS': {'RINS': 64}}。

所有有效候选和交付方案均以原始参数独立验解，再验证完整 Y/Z 固定集与短期 Y 的 κ。没有测试全候选池，没有历史 60 秒见证标签。
继续门槛联合满足且验证候选池有足够选择空间才进入后续状态条件选择研究；本轮不训练 RL。
通过仍需成熟 LNS/ALNS 对照、新分布确认和具体新意。

来源目录没有 Git 提交；LG01_freeze.json 保存逐脚本 SHA-256 和代码快照。依赖版本与接口修正均公开。
中断时完整预算保守记账，保留预验合法参考方案，不重新给同一求解动作 20 秒。中断项不能获得严格计时资格。
标签图缓存在固定动作交付后构造、单列成本；正式 GNN/MLP 自行重建所需输入并在预算内推理。

工程与版本记录：
- Complete far Y fixes checked independently. State build timed online.
- FULL fixed policy skips unneeded relaxed LP/graph. Others pay only their required inputs.
- Interrupted solver reserves are charged once in full; delivered prevalidated reference is retained, timing unqualified.
- Tree early_stopping=False: fixed 100 iterations, no implicit validation split.
- Filesystem-only amendment v2: canonical state IDs contain |, encoded as %7C in Windows filenames. First completed NOM-160 retained; no action labels/test outcomes existed before this fix. No seed, action, model, budget or decision threshold changed.
- Filesystem-only amendment v3: convert NumPy scalar booleans/numbers to native JSON values. First repaired plan now passes real serialization roundtrip. Both pre-label I/O failures retained; NOM-160 solver work retained, no repeated solver actions, no model or threshold changes. Report prints all version notes.

交付收尾记录：192 次正式测试及独立审计完成后，原运行器在交付清单中读取自身 Windows 字节锁 RUN.lock 时触发 PermissionError。进程退出后所有文件可读，复现确认仅运行时锁文件受影响。保留 LG01_manifest_error.v3.json；独立收尾脚本排除运行时锁、状态及事件文件生成清单，未修改封存代码、模型、标签或测试结果，未重复求解或拟合。LG01_postrun_review.json 已复算主指标、家庭 bootstrap、分组、封存时序和资源资格。
