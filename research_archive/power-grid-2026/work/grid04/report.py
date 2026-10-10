"""Aggregate frozen development comparisons; never run an agent or choose new jobs."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid04'
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
audit=load('GRID04_audit.json');assert audit['passed']
design=load('design_freeze.json');runs=load('per_run.json');paired=load('paired_trajectories.json')
divergences=load('first_divergences.json')
scene=design['selected_development_weeks'];policies=design['policies']
lookup={(r['scenario'],r['policy']):r for r in runs}
all_identical=all(r['identical_complete_actions'] and r['identical_complete_observations'] for r in paired if r['other']=='NN352')
comparison=[]
for policy in ['NN352','FULL','LOCAL20','RANDOM20']:
    n=[lookup[(s,'NN20')] for s in scene];r=[lookup[(s,policy)] for s in scene]
    complete=all(x['completed_native_week'] and y['completed_native_week'] for x,y in zip(n,r))
    gain=[(y['native_raw_cost_sum']-x['native_raw_cost_sum'])/abs(y['native_raw_cost_sum']) for x,y in zip(n,r)] if complete else None
    no_extra=all(not y['completed_native_week'] or x['completed_native_week'] for x,y in zip(n,r))
    row={'NN20_vs':policy,'all_four_pairs_complete':complete,'NN20_has_no_extra_failures':no_extra,
        'NN20_completed_weeks':sum(x['completed_native_week'] for x in n),'other_completed_weeks':sum(x['completed_native_week'] for x in r),
        'equal_week_mean_relative_cost_improvement':float(np.mean(gain)) if gain is not None else None,
        'positive_cost_weeks':sum(g>1e-9 for g in gain) if gain is not None else None,
        'per_week_relative_cost_improvement':gain,
        'other_cost_mean_over_NN20':float(np.mean([y['native_raw_cost_sum'] for y in r])/np.mean([x['native_raw_cost_sum'] for x in n])) if complete else None}
    comparison.append(row)
cheap=[c for c in comparison if c['NN20_vs'] in ['LOCAL20','RANDOM20']]
existing_nn_signal=all(c['all_four_pairs_complete'] and c['NN20_has_no_extra_failures'] and c['equal_week_mean_relative_cost_improvement']>=.02 and c['positive_cost_weeks']>=3 for c in cheap)
accepted_cheap=[c['NN20_vs'] for c in cheap if c['all_four_pairs_complete'] and c['other_cost_mean_over_NN20']<=1.02]
omissions=[r for r in divergences if r.get('same_state_pool_omission_witness')]
complete_cost_examples=[]
for s in scene:
    nn=lookup[(s,'NN20')]
    for p in ['NN352','FULL','LOCAL20','RANDOM20']:
        other=lookup[(s,p)]
        if nn['completed_native_week'] and other['completed_native_week']:
            complete_cost_examples.append({'scenario':s,'other':p,
                'other_cost_reduction_relative_to_NN20':(nn['native_raw_cost_sum']-other['native_raw_cost_sum'])/abs(nn['native_raw_cost_sum']),
                'scope':'Single complete paired week, descriptive only; no aggregate screen or generalization.'})
summary={'same_pool_NN20_NN352_actions_and_states_identical_all_weeks':all_identical,
    'same_prior_observation_first_divergence_omission_witnesses':len(omissions),
    'cost_screens_applicable':all(c['all_four_pairs_complete'] for c in cheap),
    'cost_screen_status':'APPLICABLE' if all(c['all_four_pairs_complete'] for c in cheap) else 'NOT_APPLICABLE_INCOMPLETE_EPISODES',
    'existing_NN_shortlist_value_screen_passed':existing_nn_signal,'accepted_cheap_comparators_by_102pct_screen':accepted_cheap,
    'comparisons':comparison,'training_or_RL_justified':False,
    'descriptive_jointly_complete_week_cost_comparisons':complete_cost_examples,
    'reason':'This only assesses existing baselines on four development weeks; no new algorithm or learnable improvement demonstrated.'}
(OUT/'GRID04_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
table=[]
for p in policies:
    a=[lookup[(s,p)] for s in scene]
    mean_cost=f"{sum(x['native_raw_cost_sum'] for x in a)/4:.2f}" if all(x['completed_native_week'] for x in a) else '不聚合'
    table.append(f"| {p} | {sum(x['completed_native_week'] for x in a)}/4 | {sum(x['steps'] for x in a)} | {sum(x['simulate_calls'] for x in a):,} | {sum(x['act_time']['sum'] for x in a):.2f} | {mean_cost} |")
detail=[]
for s in scene:
    for p in policies:
        r=lookup[(s,p)]
        detail.append(f"| {s} | {p} | {r['steps']}/2017 | {'完整' if r['completed_native_week'] else '提前终止'} | {r['native_raw_cost_sum']:.3f} | {r['act_time']['sum']:.2f} | {r['max_after_rho']:.4f} |")
comparisons=[]
for c in comparison:
    gain=f"{100*c['equal_week_mean_relative_cost_improvement']:+.3f}%" if c['all_four_pairs_complete'] else '不比较'
    count=str(c['positive_cost_weeks'])+'/4' if c['all_four_pairs_complete'] else '—'
    comparisons.append(f"| {c['NN20_vs']} | {c['NN20_completed_weeks']}/4 vs {c['other_completed_weeks']}/4 | {gain} | {count} |")
solver={p:{} for p in policies}
for r in runs:
    for status,count in r['solver_statuses'].items():solver[r['policy']][status]=solver[r['policy']].get(status,0)+count
same_text='四周全部动作和观测逐步相同；当前数据不支持把同库 Top-20 遗漏当作新训练目标。' if all_identical else '并非四周全部相同；先检查逐周分歧与完整存活结果，不能仅凭动作不同就宣称排序损失。'
value_text='通过的是作者现成 NN 相对本轮两个便宜对照的开发筛查，不是我们的学习贡献。' if existing_nn_signal else ('存在不完整周，联合成本门槛不适用；这不是成本优势被否定。' if not summary['cost_screens_applicable'] else '作者 NN 未通过相对两个便宜对照的联合成本门槛。')
cheap_text='、'.join(accepted_cheap) if accepted_cheap else '无（或存活条件不足，不能应用成本门槛）'
april_full=next(r for r in complete_cost_examples if r['scenario']==scene[1] and r['other']=='NN352')['other_cost_reduction_relative_to_NN20']
october_random=next(r for r in complete_cost_examples if r['scenario']==scene[3] and r['other']=='RANDOM20')['other_cost_reduction_relative_to_NN20']
survival=['| 周 | FULL | NN20 | NN352 | LOCAL20 | RANDOM20 |','|---|---:|---:|---:|---:|---:|']
for s in scene:
    survival.append('| '+s+' | '+' | '.join(str(lookup[(s,p)]['steps'])+('（完整）' if lookup[(s,p)]['completed_native_week'] else '') for p in policies)+' |')
divergence_table=['| 周 | 首次分歧步 | 同一前置观测 | NN352候选排名 | NN20 / NN352候选最佳预测rho | NN20 / NN352最终实际rho |','|---|---:|---|---:|---:|---:|']
for r in divergences:
    if (r['left'],r['right'])!=('NN20','NN352'):continue
    left=r.get('left_topo12_best_eligible_forecast_rho');right=r.get('right_topo12_best_eligible_forecast_rho')
    forecast=f'{left:.6f} / {right:.6f}' if left is not None and right is not None else '未同时取得合格候选'
    actual=f"{r['left_after_rho']:.6f} / {r['right_after_rho']:.6f}" if 'left_after_rho' in r else '—'
    divergence_table.append(f"| {r['scenario']} | {r['first_different_action']} | {r.get('same_prior_observation')} | {r.get('right_topo12_rank_in_right_shortlist')} | {forecast} | {actual} |")
timing_table=['| 配置 | Topology-12累计秒 | N-1累计秒 | 连续优化累计秒 | act p95最坏周 / 单次最大秒 |','|---|---:|---:|---:|---:|']
for p in policies:
    current=[lookup[(s,p)] for s in scene]
    modules=[sum(r['modules'].get(m,{}).get('wall_s',0) for r in current) for m in ['topo_12_unsafe','topo_n1_unsafe','optim']]
    timing_table.append(f"| {p} | {modules[0]:.3f} | {modules[1]:.3f} | {modules[2]:.3f} | {max(r['act_time']['p95'] for r in current):.4f} / {max(r['act_time']['max'] for r in current):.4f} |")
prefix_table=['| 周 | 相同前置观测前缀步数 | NN20 / NN352模拟次数 | NN20 / NN352 act秒 |','|---|---:|---:|---:|']
for r in paired:
    if r['other']!='NN352':continue
    prefix_table.append(f"| {r['scenario']} | {r['identical_prior_observation_prefix_steps']} | {r['same_prior_prefix_NN20_simulates']} / {r['same_prior_prefix_other_simulates']} | {r['same_prior_prefix_NN20_act_s']:.3f} / {r['same_prior_prefix_other_act_s']:.3f} |")
report=f'''# GRID04 正式开发周的强基线比较

## 结论

**20 条正式轨迹及保存数据审计完成，保留电网控制方向继续诊断，仍不启动 GRL。** 本轮是四个开发周、一个环境种子的现成方法比较，不是新算法或封存确认。

- 同库全扫描：{same_text}
- 神经短名单的额外价值：{value_text}
- 进入102%筛查范围的便宜参照：{cheap_text}。筛查不等于等价检验，单个随机种子也不代表随机策略总体。
- 完整配对周的描述：4月NN352相对NN20的原始运行成本低{100*april_full:.2f}%；10月RANDOM20则比NN20低{100*october_random:.2f}%。两例均完整，但不是跨周平均结论。它们提示改进空间，也反对“全扫描或NN普遍最好”的解释。

## 五臂与公平范围

FULL 是作者421项资产的非NN方案；NN20、NN352、LOCAL20、RANDOM20 使用完全相同的352项映射、同一个 LJNAgentTopoNN、重连/恢复/N−1搜索/连续优化。NN352 仅扩大扫描数量。LOCAL20 按动作涉及变电站的当前邻接线路最大rho、过载平方和排序；RANDOM20 用固定种子和当前步生成均匀哈希排列。均无额外模拟或未来信息。原生动作权限和参数不变，没有人为给 FULL 设置单步截止。

来源：[作者固定版本](https://github.com/lajavaness/l2rpn-2023-ljn-agent/tree/ca0637eab9f098be7f206ed0e46a3900cd4deec0)。后两臂是明确标注的本轮非学习对照，不是作者发布配置，也不是最强物理近似筛选器的完整集合。便宜臂仍在共同工厂初始化时加载原模型，短名单阶段不调用模型，初始化时间单列。运行时间包括记录器开销，不是生产推理性能。

四周：{', '.join(scene)}，按1/4/7/10月及固定名字哈希从已冻结开发组选择。4月周的前12步曾用于GRID03接口检查，选择没有使用该结果。后缀分组不证明独立，发布模型训练场景身份未知；不能称四周对该模型未见。验证、测试场景未运行代理。

数据为公开l2rpn_idf_2023基准的合成2035场景，电网基于改造的IEEE118系统；有真实输电运行背景，但不是RTE生产运行日志。本轮不能作为实际电网收益或安全性的验证。

## 汇总与逐周结果

{chr(10).join(survival)}

单元格为执行到的结束步（包括发生终止的最后一次step），不是扣除失败步后的成功服务时长；2017为原生完整周。

| 配置 | 完整周 | 实际步数 | 模拟调用 | act累计秒 | 四周原始成本均值 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

如出现提前终止，不聚合四周成本作质量排名。逐轨迹原始成本仍保存作账目描述；成本门槛只在双方四周均完整时应用，生存优先。

| NN20 对照 | 完整周 NN20 vs 对照 | 等权逐周相对成本改善 | 正向周 |
|---|---:|---:|---:|
{chr(10).join(comparisons)}

相对成本为 `(J_对照−J_NN20)/|J_对照|`，四周等权。不是修复成本归一化，也不是比赛总分。102%便宜参照筛查使用两方法四周算术平均成本之比。聚合公式在主批次运行期间、查看成本/存活结果前写明，不伪装成独立预注册确认。

| 周 | 配置 | 步数 | 结束 | 原始成本 | act秒 | 最大rho |
|---|---|---:|---|---:|---:|---:|
{chr(10).join(detail)}

## 保存数据中的首次分歧

{chr(10).join(divergence_table)}

共 {len(omissions)} 个周的首次动作分歧满足“相同前置物理观测、NN352返回原池动作、该动作不在NN20短名单”这一见证。候选预测rho来自原策略已经执行的一步模拟，按原模块的0<rho<当前rho、未终止且无异常条件筛选；模拟中的非法动作次数另列，没有新增模拟或模型推理。完整记录含候选排名、合法性、优化器状态和最终动作预测，见first_divergences.json。

这只能定位短名单遗漏与即时预测差异，不能证明该动作全局最优、低rho总是长期更好，或将后来的停机因果归于这个动作。相同物理观测也不是对全部求解器内部状态一致的独立认证；后续分叉状态不作配对因果比较。

本批两处遗漏见证中，全扫描候选的最佳预测rho较低，但交付最终动作后的实际rho均较高。双方还调用了连续优化，4月全扫描侧包含user_limit状态；两侧最终交付动作都没有已记录的完整一步预测。因此不能直接把差异归为预报误差，也不能把拓扑候选的模拟rho与叠加连续控制后的实际rho当成同一动作的预测残差。现有记录不足以证明“更好地模仿全扫描候选”就能改善控制。

## 搜索开销的描述性分解

{chr(10).join(timing_table)}

总量包含各自实际运行长度，不能据此单独作速度排名。模块时间包含其内部模拟，不能再与模拟时间相加。下面仅保留从初始步起前置物理观测连续相同的NN20/NN352前缀，可能包括第一次动作分歧所在步；不延伸到后续不同状态。

{chr(10).join(prefix_table)}

这些是带记录器的单机计时，不是业务时限认证；本轮没有预设或人为强加每次决策3秒限制，也没有证明全扫描在实际电网部署中太贵。

## 计量、安全与计算边界

- 原生完整终点2017步；提前终止与数据终点分别判断，原生异常类型保留。最大rho/过载步数是描述量，不是独立安全认证。完整告警/可再生终端分项、物理弃能、再调度、储能另存于 per_run.json。
- 官方原始成本来自 Grid2Op1.12.5 L2RPNSandBoxScore，独立账本使用相邻已交付观测复算。当前费用中的弃能项为总弃能变化量，物理弃能能量单列。未认证的比赛归一化总分未使用，库未改。
- {audit['public_ledgers_recomputed']:,} 条合法物理账本独立复算，最大残差 {audit['max_cost_abs_residual']:.8g}，保持GRID03浮点容差。错误终止的回退费用与可复核物理费用分列，不能当作负荷服务收益。
- 全部 {audit['vectors_rehashed_and_rechecked']:,} 份逐步向量重算哈希、观察链、模拟次数；{audit['shortlists_independently_recomputed_or_membership_checked']} 次短名单重算规则/检查池内资格；19份复用源码/模型/动作及账本文件哈希未变。独立审计不调用代理、潮流或训练。
- 连续求解状态：`{json.dumps(solver,ensure_ascii=False)}`。原策略有限迭代解仍按原规则接受；未对每个QP做独立约束残差认证，原生执行可行与QP证最优不同。
- 主进程耗时合计 {audit['main_runner_process_wall_s']:.2f}秒，含编排主批次 {audit['main_runner_including_orchestration_wall_s']:.2f}秒；失败启动 {audit['startup_process_wall_s']:.2f}秒。物理调用 {audit['total_physical_steps']:,}（其中失败记录1步），在40,400步和3,600秒主运行包络内。审计另耗 {audit['audit_wall_s']:.2f}秒。

## 实现披露

首次启动在元数据序列化阶段失败（NumPy float32），0物理步；仅修一个字段后，逐步容差字段再触发同类错误，已执行1步且NPZ保留。随后统一采用严格NumPy标量JSON编码，验证数值/布尔转换和非有限拒绝，再运行冻结矩阵。两次失败日志/代码快照保留于 runs/ 与 runs_main/；有效数据为 runs_v3/。额外步骤和启动耗时全部计入原预算。没有按结果换周、种子、候选数、策略、计量容差或模型。

## 研究含义与下一步

这批数据用于识别值得进一步诊断的差距，不能将FULL与NN差异全归于排序：资产池和类路径不同。NN20与NN352的同库比较更直接，但动作不同也可能来自并列或后续连续优化，仍需同状态核对；轨迹分叉后的状态不能当作配对因果证据。低成本短名单接近NN也不否定全部学习方法。

下一步应根据保存的首次分歧状态，区分资产覆盖、候选排名、预测误差与连续优化造成的差异；先复核合法可辨识的具体损失，再考虑新方法。已有模拟记录可作为后续诊断材料，但尚未建立可学收益或相对成熟工作的算法新意，因此不启动训练，不自动增加场景/种子或扫描Top-k。

文献边界也保持不变：原作者已有模仿学习后接PPO以及神经Top-k搜索；[2025年GNN拓扑控制工作](https://arxiv.org/html/2501.07186v3)已比较全连接模型、同构/异构GNN并研究网络配置变化下的泛化。因此，“把MLP换为GNN”或“接RL”本身不能作为贡献。本轮仅识别基线缺口，还未证明合法观测能预测何时需要扩大搜索，或这种预测优于简单回退。
'''
(OUT/'GRID04_report.md').write_text(report,encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('# GRID04\n\nCOMPLETED. 20 formal trajectories and saved-data audit complete. No pending training, simulation or automation.\n\n- Development-only comparison; validation/test sealed.\n- Official raw costs audited; competition-exact normalization remains uncertified.\n- No training or new algorithm claim. Next stage not started.\n',encoding='utf-8')
files=['GRID04_report.md','GRID04_summary.json','GRID04_audit.json','GRID04_per_run.csv','per_run.json','paired_trajectories.json','first_divergences.json','design_freeze.json','GRID04_protocol.md','RUN_STATE.md']
(OUT/'delivery_manifest.json').write_text(json.dumps({f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
