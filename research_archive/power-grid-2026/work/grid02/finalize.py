import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid02'
audit=json.loads((OUT/'GRID02_audit.json').read_text(encoding='utf-8'))
assert audit['status']=='PASS_SAVED_ARTIFACT_CONSISTENCY'
assert all(x['all_observations_actions_rewards_native_flags_identical'] for x in audit['NN20_NN352_comparison'])
resources=json.loads((OUT/'resource_manifest.json').read_text(encoding='utf-8'))
# Layered environments expose shadowed distribution metadata; report the first
# resolved distribution, consistent with Python's actual module search order.
resources['installed_versions']={name:importlib.metadata.version(name) for name in resources['installed_versions']}
resources['version_resolution']='importlib.metadata.version(name), first resolved distribution in the documented layer order; shadowed metadata is not a second active runtime.'
(OUT/'resource_manifest.json').write_text(json.dumps(resources,indent=2),encoding='utf-8')
differences=json.loads((OUT/'FULL_NN_offline_differences.json').read_text(encoding='utf-8'))
main=[x for x in audit['runs'] if x['mode']=='full']
policies={name:[x for x in main if x['policy']==name] for name in ['FULL','NN20','NN352']}
aggregate={}
for name,runs in policies.items():
    times=[]
    for run in runs:
        rows=[json.loads(x) for x in (OUT/f"{name}_scenario{run['scenario']}_full/steps.jsonl").read_text(encoding='utf-8').splitlines()]
        times.extend(x['act_wall_s'] for x in rows)
    aggregate[name]={'steps':sum(x['steps'] for x in runs), 'simulate_calls':sum(x['simulate_calls'] for x in runs),
                     'act_total_s':sum(x['act_wall_s']['sum'] for x in runs),
                     'act_p95_s':float(np.percentile(times,95)), 'act_p99_s':float(np.percentile(times,99)),
                     'act_max_s':max(times), 'act_over1s_steps':sum(t>1 for t in times),
                     'all_reached_native_horizon':all(x['reached_recorded_native_horizon'] and x['chronics_done_at_end'] for x in runs),
                     'all_native_errors_zero':all(x['illegal']==x['ambiguous']==x['exception_steps']==0 for x in runs),
                     'topo12_simulations':sum(x['module_summary']['topo_12_unsafe']['simulate_calls'] for x in runs),
                     'topo12_calls':sum(x['module_summary']['topo_12_unsafe']['calls'] for x in runs),
                     'optim_calls':sum(x['module_summary']['optim']['calls'] for x in runs),
                     'optim_wall_s':sum(x['module_summary']['optim']['wall_s'] for x in runs),
                     'N1_wall_s':sum(x['module_summary']['topo_n1_unsafe']['wall_s'] for x in runs)}
nn=aggregate['NN20']
all_nn=aggregate['NN352']
saved=all_nn['simulate_calls']-nn['simulate_calls']
assert saved==4980
time_reduction=(all_nn['act_total_s']-nn['act_total_s'])/all_nn['act_total_s']
summary={'status':'BASELINE_QUALIFIED_NO_TOPK_EXPANSION_BENEFIT_IN_BUNDLED_SCENARIOS',
         'aggregate':aggregate,'NN20_NN352_simulations_saved':saved,
         'NN20_NN352_act_total_reduction':time_reduction,
         'total_physical_steps':audit['actual_steps'],
         'all_process_wall_s':sum(x['wall_s'] for x in audit['runs']),
         'new_training':False,'formal_research_dataset_downloaded':False,
         'cumulative_dependency_payload_cap_met':resources['cumulative_dependency_payload_cap_met']}
(OUT/'GRID02_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

lines=['# GRID02：作者 NN Top-k 接入及同映射全扫描比较', '',
       '2026-10-08。**裁决：NN 对照资格通过；在两个随包场景中，Top-20 已复现同映射全扫描的全部最终动作，没有观察到扩大该候选池的控制收益。不启动 GRL。**', '',
       '这不是新的算法贡献，也不是正式封存确认。我们补齐了可以运行、有可核对动作语义的学习辅助对照；不能据此证明神经排序优于固定/规则短名单，后者本轮未比较。', '',
       '## 1. 三臂是什么', '',
       '| 配置 | 全连通拓扑分支 | 其余控制 |',
       '|---|---|---|',
       '| FULL | 作者非NN方案，421项资产，冷却过滤后搜索 | 作者重连、恢复、N−1的909项搜索、连续优化 |',
       '| NN20 | 作者 `make_agent_topoNN`，原模型从352项映射中排Top-20后模拟 | 同作者NN混合方案，其N−1仍完整搜索 |',
       '| NN352 | 同NN20模型、映射和类，仅 `top_k=352` 扫描全部映射项 | 与NN20相同 |', '',
       '来源固定为 `lajavaness/l2rpn-2023-ljn-agent` commit `ca0637eab9f098be7f206ed0e46a3900cd4deec0`。NN352是本轮明确的控制参数变体，不冒充作者推荐配置，也不是全局最优上界。[固定来源](https://github.com/lajavaness/l2rpn-2023-ljn-agent/tree/ca0637eab9f098be7f206ed0e46a3900cd4deec0)', '',
       '模型原始archive声明SB3 2.3.0、Gymnasium 0.29.1、PyTorch2.1.2+cu121；本轮SB3/Gym版本与其一致，借用已有PyTorch2.8.0+cpu。原权重不改，55,201参数、仅186维当前rho输入、352维离散输出。不是GNN，没有训练或调用learn/fit，也不把作者离线训练成本算成本轮零成本。发布模型的完整训练划分未在本轮核实。', '',
       'NN映射与FULL421项仅重合328项（NN另有24项，FULL另有93项），所以主控制比较是NN20对NN352；FULL为另一作者配置参照，不能把其差异全部归为神经排序。该控制臂在任何GRID02电网控制结果产生前加入并记录修订。', '',
       '## 2. 执行与结果', '',
       '两个公开随包场景分别为2035-01-15_0、2035-08-20_0，各自原生最大575步（约两天），不是完整周研究场景。seed=0、CPU LightSim、MaxRhoReward、原生参数/区域/故障/攻击/动作权限不改；每臂新建环境与agent。', '',
       '| 配置 | 主运行步数 | 模拟调用 | act累计秒 | act p95毫秒 | act最大秒 |',
       '|---|---:|---:|---:|---:|---:|']
for name,a in aggregate.items():
    lines.append(f"| {name} | {a['steps']} | {a['simulate_calls']:,} | {a['act_total_s']:.3f} | {1000*a['act_p95_s']:.2f} | {a['act_max_s']:.3f} |")
lines+=['', '六条均走到数据末端；`native_done=true`伴随`chronics_done=true`及达到原生最大步数，**不是六次提前失稳**。实际非法、歧义和异常步骤均为0。主运行3450步，两个NN冒烟各24步，合计3498步；没有失败后的控制重跑。', '',
        f"**NN20与NN352的1,150步最终动作、观察哈希、奖励及原生标志全部逐步相同。** 其15次全连通搜索分别模拟300和5,280个候选，少{saved:,}次；act累计减少{100*time_reduction:.2f}%。这是同输入映射下的实测计算差，包含被动记录器及不同扫描量带来的日志开销，不是纯生产推理性能或稳健置信界。只有一次运行种子，不作显著性声明。", '',
        f"NN20仍有{nn['act_over1s_steps']}步act超过1秒，最大{nn['act_max_s']:.3f}秒；1秒只是描述阈值，不是本轮部署deadline。N−1累计{nn['N1_wall_s']:.3f}秒、连续优化累计{nn['optim_wall_s']:.3f}秒，二者没有被Top-20方案删除。不能把低p95当作所有危险决策都很快，也没有证明已有策略超出了实际业务计算期限。", '',
        '连续求解状态也需保留：FULL共18次（optimal 14、optimal_inaccurate 1、user_limit 3）；NN20与NN352各12次（optimal 5、user_limit 7）。原策略按有限目标值接受部分限迭代结果，本轮没有把这些结果认证为QP最优或独立核对全部QP约束残差。原生执行错误为0不等于连续求解全部已证最优。', '',
        '## 3. 控制差异与可说的机制', '',
        '| 场景 | FULL最大rho / 过载后步数 | NN20最大rho / 过载后步数 | NN352 |',
        '|---|---|---|---|']
for scenario in [0,1]:
    f=next(x for x in policies['FULL'] if x['scenario']==scenario)
    n=next(x for x in policies['NN20'] if x['scenario']==scenario)
    lines.append(f"| {scenario} | {f['max_after_rho']:.4f} / {f['overloaded_after_steps']} | {n['max_after_rho']:.4f} / {n['overloaded_after_steps']} | 与NN20相同 |")
lines+=['', '原生规则允许一定持续期的过载。这些数值不是停电率、安全认证或完整运营得分；MaxRhoReward是作者搜索奖励，不冒充挑战最终运营评分。', '',
        'FULL与NN分支的首次动作分歧在场景0的第61步、场景1的第42步，两处决策前观察完全相同。已有模块日志显示，FULL返回的拓扑动作都不在352项NN映射中。这支持一个具体限定：**在保持该NN映射不变时，仅把20扩成352不能恢复那两个FULL拓扑选择**。它不证明这些动作是全局必要的、不证明加入93项会改善整条轨迹，也不能把后续1.7367峰值全部归因于资产覆盖。后续两策略状态已不同。详见`FULL_NN_offline_differences.json`。', '',
        '因此当前不应开发“NN漏掉Top-20之外的好动作”的学习器：对这批状态，直接完整扫描352项已经没有改变控制行为。也不把剩下的N−1/连续开销当作学习缺口，那里仍需要强便宜方法与真实期限依据。', '',
        '## 4. 兼容、错误与资源披露', '',
        '1. 原模型与源码/动作资产逐文件保留哈希与许可证。沿用GRID01的NumPy/LightSim/CVXPY兼容；本轮追加工厂导入文件大小写修正。LJN两类和NN工厂主体AST不变，NN模块/动作空间代码及模型ZIP与原源字节一致。',
        '2. 旧pickle对象直接装入Grid2Op1.12.5时，旧拓扑字段被新lazy-private描述符忽略，且as_dict会缺少新私有字段。最初资格断言因此失败，0实际步。352项旧__dict__的全部向量字段、缓存与独立vect_actions逐项一致；从同一行向量重建新动作对象，并再次逐行重建核对。没有改动作内容/顺序、模型输入或候选数量。原资产不改，适配资产及352条迁移证据另存。',
        '3. 首次pip安装遇临时目录权限失败；补齐SB3必需绘图库前有两次导入失败，均发生在控制前。路径清单先后混用了Windows反斜线/正斜线，留下一个迁移资产重复键；离线规范化路径并保留旧清单，所有实际资产/轨迹不改。不能写本轮工程零错误。',
        '4. 动作记录复用GRID01无缓存副作用的字段向量化。NN两个24步冒烟与主运行前24步逐项一致；FULL两个主运行的前258步与GRID01逐项复现。离线核对覆盖文件/向量哈希、观察链、完整动作连续字段、模拟次数和模型输入哈希。它是保存数据一致性及原生接口核对，不是第二套AC潮流验解。', '',
        f"本轮源码/模型/动作payload {resources['source_and_asset_bytes']:,}字节；新增独立依赖wheel {resources['unique_dependency_payload_bytes']:,}字节。**pip dry-run和安装均下载了绘图库，累计wheel载荷估计{resources['cumulative_wheel_payload_including_dry_run_bytes']:,}字节（约{resources['cumulative_wheel_payload_including_dry_run_bytes']/1024**2:.2f}MiB），超过原30MiB依赖包络。** 原协议保留，不用独立wheel大小掩盖重复下载。该量按Content-Length与日志保守记账，不是逐包网络计量。没有下载Torch/GPU包或完整研究数据；GRID01选定依赖版本不变。", '',
        f"八条实际进程墙钟合计{summary['all_process_wall_s']:.3f}秒（约{summary['all_process_wall_s']/60:.2f}分钟），包含导入/建环境/模型加载/记录；不含安装、资格与离线审计。实际步数3498达到修订上限3498，未超过原8112步包络，运行合计低于3600秒。安装资源偏差与仿真预算分开报告。", '',
        '## 5. 研究判断与下一步', '',
        '**保留Grid2Op应用，结束本轮接入；不启动GRL，不在这两个公开短场景上继续调k、换排序器或扩大策略矩阵。** 目前获得的是成熟学习辅助基线的资格和一个清晰反证：扩大当前NN候选池未产生控制收益。不是对整个领域或GNN/RL的否定。', '',
        '下一步真正有信息量的是正式数据资格与评价协议：固定新的场景分组、运营指标和训练/验证/测试权限，完整保留发布模型训练来源边界，先检查强基线的可避免运行损失、候选覆盖及预测误差，不预先把它们统一归为“缺少学习”。如比较计算期限，必须有业务/基准依据并公平计费，不能人为给FULL过小预算。', '',
        '正式数据入口已做只读资格探针：urllib HEAD遇证书链核验错误，requests在开启TLS验证下HEAD重定向返回403；不能由HEAD失败推断GET失败。随后同样开启验证的`GET Range: bytes=0-0`返回206，只读取1字节，`Content-Range`给出归档总长5,358,993,257字节（约5.36GB、4.99GiB）。HEAD与GET差异与重定向到预签GET链接的机制相符，但本轮不进一步证明服务器端原因。**入口可读不等于完整数据已下载/哈希或场景语义已验收**。完整数据获取须单独限定下载/磁盘成本，不能继续把两个随包场景称为封存数据；没有关闭TLS验证。索引、响应与探针另存`formal_dataset_metadata.json`。', '',
        '主要交付：`GRID02_audit.json`、`GRID02_per_run.csv`、`GRID02_summary.json`、352项迁移证据、原源/适配/依赖清单、逐步NPZ和日志、NN Top-k索引与合法rho输入哈希、首次分歧离线分析。全部作业完成，无自动后续训练或跟进任务。']
(OUT/'GRID02_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('# GRID02 run state\n\nCompleted 2026-10-08. No outstanding simulation, training or automation.\n\n- Qualification passed after exact legacy action-object storage migration.\n- 6 main trajectories: 3 policies × 2 bundled 575-step scenarios.\n- 2 NN smoke trajectories × 24 steps; total actual steps 3498.\n- All reach native data horizon, 0 native illegal/ambiguous/exception steps.\n- NN20 and NN352 all 1150 actions/observations/rewards/flags identical; 4980 simulations saved.\n- Frozen running code and source/assets checked. Offline manifest alias correction retained.\n- Dependency cumulative payload exceeds original 30MiB ceiling; fully disclosed.\n- No training/full research archive download. GRID03 not started.\n',encoding='utf-8')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for base in [OUT,ROOT/'work/grid02'] for p in sorted(base.rglob('*'))
          if p.is_file() and '.venv' not in p.parts and '__pycache__' not in p.parts and 'tmp' not in p.parts and 'mplconfig' not in p.parts
          and p.name!='delivery_manifest.json'}
(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(summary['status'],'NN20 act reduction',time_reduction,'report saved',flush=True)
