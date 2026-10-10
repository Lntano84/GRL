"""Apply the pre-outcome validation screen without fitting or new simulations."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid06'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
audit=load(OUT/'GRID06_audit.json');assert audit['passed']
design=load(OUT/'design_freeze.json');runs=load(OUT/'per_run.json')
lookup={(r['scenario'],r['policy']):r for r in runs}
contrasts=[]
for scene in design['selected_weeks']:
    f=lookup[(scene,'FALLBACK90')]
    for ref in ['NN20','NN352']:
        b=lookup[(scene,ref)]
        joint=f['completed_native_week'] and b['completed_native_week']
        reduction=(b['native_raw_cost_sum']-f['native_raw_cost_sum'])/abs(b['native_raw_cost_sum']) if joint else None
        benefit=(f['completed_native_week'] and not b['completed_native_week']) or (joint and reduction>=.05)
        contrasts.append(dict(scenario=scene,reference=ref,reference_steps=b['steps'],fallback_steps=f['steps'],
            reference_complete=b['completed_native_week'],fallback_complete=f['completed_native_week'],
            fallback_earlier_end=f['steps']<b['steps'],jointly_complete=joint,
            reference_cost=b['native_raw_cost_sum'] if joint else None,fallback_cost=f['native_raw_cost_sum'] if joint else None,
            cost_reduction_relative_to_reference=reduction,
            more_than_2pct_complete_week_cost_regression=joint and reduction<-.02,
            meaningful_benefit_week=bool(benefit)))
vs20=[c for c in contrasts if c['reference']=='NN20'];vs352=[c for c in contrasts if c['reference']=='NN352']
no_retreat20=not any(c['fallback_earlier_end'] or c['more_than_2pct_complete_week_cost_regression'] for c in vs20)
retain352=not any(c['fallback_earlier_end'] or c['more_than_2pct_complete_week_cost_regression'] for c in vs352)
benefits=sum(c['meaningful_benefit_week'] for c in vs20)
if not no_retreat20:verdict='DO_NOT_PROMOTE_RULE_AS_GENERAL_IMPROVEMENT'
elif benefits>=2 and retain352:verdict='VALIDATION_QUALITY_SCREEN_SUPPORTED'
elif benefits>=2:verdict='BENEFIT_VS_NN20_WITH_FULLSEARCH_QUALITY_TRADEOFF'
elif benefits==0:verdict='NO_PRACTICAL_INCREMENTAL_BENEFIT_CONFIRMED'
else:verdict='LIMITED_SIGNAL_NOT_CONFIRMED'
summary=dict(verdict=verdict,no_regression_vs_NN20=no_retreat20,meaningful_benefit_weeks_vs_NN20=benefits,
    full_search_quality_retained_screen=retain352,contrasts=contrasts,additional_training=False,
    predefined_blind_to_our_validation_outcomes=True,pretrained_model_unseen_status_certified=False,
    remaining_validation_weeks_sealed=8,all_test_weeks_sealed=12,
    validation_suffix_groups=sorted({int(s.rsplit('_',1)[1]) for s in design['selected_weeks']}))
save('GRID06_summary.json',summary)

# Descriptive decomposition only, not a replacement objective or a new screen.
components=[]
for run in runs:
    folder=OUT/'runs'/f"{run['scenario']}__{run['policy']}"
    rows=lines(folder/'steps.jsonl');valid=[r for r in rows if r['ledger_eligible']]
    entry=dict(scenario=run['scenario'],policy=run['policy'],completed_native_week=run['completed_native_week'])
    for component,field in [('loss_cost','losses_mwh'),('redispatch_cost','redispatch_mwh'),
        ('curtailment_change_cost','curtailment_delta_mwh'),('storage_cost','storage_throughput_mwh')]:
        entry[component]=sum(r['ledger']['marginal_cost']*r['ledger'][field] for r in valid)
    entry['recomputed_verified_total']=sum(entry[k] for k in ['loss_cost','redispatch_cost','curtailment_change_cost','storage_cost'])
    entry['native_verified_total']=sum(r['raw_operational_cost'] for r in valid)
    entry['native_error_terminal_total']=sum(r['raw_operational_cost'] for r in rows if not r['ledger_eligible'])
    assert abs(entry['recomputed_verified_total']-entry['native_verified_total'])<=sum(r['cost_rounding_tolerance'] for r in valid)+1e-6
    components.append(entry)
save('operational_cost_components.json',components)
with (OUT/'operational_cost_components.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(components[0]));w.writeheader();w.writerows(components)

divergences=[]
for pair in load(OUT/'paired_trajectories.json'):
    if pair['first_action_difference'] is None:continue
    s=pair['scenario'];step=pair['first_action_difference']
    detail=dict(scenario=s,step=step,reference=pair['reference'],other=pair['other'],policies={})
    for policy in [pair['reference'],pair['other']]:
        folder=OUT/'runs'/f'{s}__{policy}'
        row=lines(folder/'steps.jsonl')[step-1]
        events=[e for e in lines(folder/'events.jsonl') if e['step']==step]
        detail['policies'][policy]=dict(before_observation_hash=row['before_observation_hash'],action_hash=row['action_hash'],
            before_rho_max=row['before_rho_max'],after_rho_max=row['after_rho_max'],
            delivered_action_forecast_available=row['selected_action_forecast_available'],
            delivered_action_forecast_rho_max=row['selected_forecast_rho_max'],
            simulated_calls=row['simulate_calls'],act_wall_s=row['act_wall_s'],
            module_returns=[e for e in events if e['kind']=='module'],
            fallback=[e for e in events if e['kind']=='fallback'],
            solver_statuses=[e for e in events if e['kind']=='solve'])
    detail['same_prior_observation']=len({x['before_observation_hash'] for x in detail['policies'].values()})==1
    divergences.append(detail)
save('first_divergences.json',divergences)

texts={
 'DO_NOT_PROMOTE_RULE_AS_GENERAL_IMPROVEMENT':'至少一周出现相对NN20更早终止，或共同完整周成本退化超过2%；不推广这条规则为普遍改善。',
 'VALIDATION_QUALITY_SCREEN_SUPPORTED':'固定规则通过本轮事先封存的验证质量筛查；仍不是统计泛化保证、学习优势或算法新意。',
 'BENEFIT_VS_NN20_WITH_FULLSEARCH_QUALITY_TRADEOFF':'相对NN20存在重复实际增益，但未保留全扫描质量；应报告质量权衡，不能宣称替代全扫描。',
 'NO_PRACTICAL_INCREMENTAL_BENEFIT_CONFIRMED':'没有达到预设实际增益的验证周；未确认新增价值，不代表等价或整个领域无效。',
 'LIMITED_SIGNAL_NOT_CONFIRMED':'只有一个验证周达到预设实际增益；属于有限信号，尚未通过重复确认。'}
def pct(x):return '不比较（至少一臂未完成）' if x is None else f'{100*x:+.4f}%'
outcomes=['| 验证周 | NN20结束步 | 回退结束步 | NN352结束步 | 回退相对NN20成本降幅 | 回退相对NN352成本降幅 |',
          '|---|---:|---:|---:|---:|---:|']
costtable=['| 验证周 | NN20成本（仅完整） | 回退成本（仅完整） | NN352成本（仅完整） |','|---|---:|---:|---:|']
counttable=['| 验证周 | NN20模拟数 | 回退模拟数 | NN352模拟数 | 回退扩展/检查 |','|---|---:|---:|---:|---:|']
for s in design['selected_weeks']:
    n,f,h=[lookup[(s,p)] for p in ['NN20','FALLBACK90','NN352']]
    a=next(c for c in vs20 if c['scenario']==s);b=next(c for c in vs352 if c['scenario']==s)
    outcomes.append(f"| {s} | {n['steps']} | {f['steps']} | {h['steps']} | {pct(a['cost_reduction_relative_to_reference'])} | {pct(b['cost_reduction_relative_to_reference'])} |")
    values=[f"{r['native_raw_cost_sum']:.6f}" if r['completed_native_week'] else '未完成，不排名' for r in [n,f,h]]
    costtable.append(f"| {s} | {' | '.join(values)} |")
    counttable.append(f"| {s} | {n['simulate_calls']} | {f['simulate_calls']} | {h['simulate_calls']} | {f['expansions']}/{f['fallback_decisions']} |")
prefix=['| 验证周 | 参照→另一臂 | 同状态前缀步数 | 参照模拟数/动作秒 | 另一臂模拟数/动作秒 |','|---|---|---:|---:|---:|']
for p in load(OUT/'paired_trajectories.json'):
    prefix.append(f"| {p['scenario']} | {p['reference']}→{p['other']} | {p['same_prior_prefix_steps']} | {p['same_prior_reference_simulates']}/{p['same_prior_reference_act_s']:.4f} | {p['same_prior_other_simulates']}/{p['same_prior_other_act_s']:.4f} |")
illegal=sum(r['illegal_steps'] for r in runs);ambiguous=sum(r['ambiguous_steps'] for r in runs)
report=f'''# GRID06：固定回退规则的封存验证

**裁决：{verdict}。{texts[verdict]}**

## 事前固定的比较

从GRID03已经封存的月度验证清单中，按一、四、七、十月各取一个周；命名选择、不读取负载难度或策略结果。此前未由我们运行。NN20、FALLBACK90、NN352三臂，原环境种子0，12条交错轨迹。固定回退源码直接复用GRID05：前20无候选或其选中拓扑已模拟rho≥0.9时，扩展剩余332项，缓存原20项float32奖励并按完整神经排序合并。0.9只是作者控制器rho_safe，不是安全证书。

相对NN20若任一周更早结束，或共同完整周成本退化>2%，不推广。否则至少2/4周出现“新增完整周”或“共同完整周成本降低≥5%”，并对NN352无更早结束/共同完整周>2%成本退化，才通过本轮质量筛查。只对NN20有重复收益而NN352质量不保留，单列权衡。没有实际增益记未确认，只有一周记有限信号。没有在看到结果后修改门槛或替换周。

## 完成情况与原始成本

{chr(10).join(outcomes)}

相对NN20非退化检查：**{no_retreat20}**；达到预设实际增益的周数：**{benefits}/4**；保留同时运行的NN352质量检查：**{retain352}**。它们是投资筛查，不是显著性、等价检验或安全保证。

{chr(10).join(costtable)}

累计原始运营成本只在共同完整周作方法比较。提前结束的低累计成本不算优势。原始成本有已审计的物理分解；native终止错误成本另列，未冒充物理账目。当前归一化比赛总分未认证，未用于本轮排序。

## 计算开销与作用范围

{chr(10).join(counttable)}

总模拟数包括恢复、重连、N−1与其他原作者模块。不同策略可能改变轨迹状态与长度，整周墙钟/模拟数不是自动成立的同质量加速。下面只列共同决策前状态的前缀；它包含已有遥测与固定规则的真实开销，但不是重复延迟实验，也没有生产计算期限认证。

{chr(10).join(prefix)}

首个动作分歧的同状态、模块返回、实际rho与求解器状态另存first_divergences.json。拓扑候选的预测是在后续连续优化前取得；不得把该预测与最终交付动作的实际rho直接当作同动作预测残差，亦不能仅凭首个分歧断言后续停电的单一原因。operational_cost_components.json/CSV将已核验成本按损耗、再调度、弃能变化及储能拆开，仅为描述诊断，不改评分或预设裁决；弃能变化成本不等于物理弃能总量。

## 审计与资源

- 12/12正式运行退出0；实际推进与向量重读 **{audit['physical_steps']}** 步。
- **{audit['public_ledgers_recomputed']}** 个合格物理账目独立重算；最大成本绝对残差 **{audit['max_cost_abs_residual']:.9f}**，均满足沿用的float32计量容差。原生非法/歧义实际步为 **{illegal}/{ambiguous}**，原生终止异常逐运行另列。
- **{audit['topology_module_calls_reconstructed']}** 次拓扑模块调用和 **{audit['fallback_decisions_reconstructed']}** 次回退判断重建；资格、奖励、完整排序、返回动作与缓存不重复模拟均核对。生产强制扩展开关全部False。
- 规则/作者控制器/模型/动作库/计量器等 **{audit['reused_assets_unchanged']}** 项复用文件未变；资格检查冻结的70项数据文件与正式运行源码哈希未变。三臂原生参数、通用模块配置和同周初始观测一致。
- 正式矩阵及资格检查共 **{audit['total_runner_including_qualification_s']:.3f}秒**，低于2700秒；审计另计 **{audit['audit_wall_s']:.3f}秒**。正式结果 **{audit['main_persisted_bytes']:,}字节**，低于2GiB。
- 新资格检查0模拟、0实际推进；原样复用GRID05的11个语义用例与704次真实零推进模拟对照，不重跑它们。没有训练、重新推断教师标签、扫阈值或改变连续优化策略。

每步native高精度调用数与事件计数一致由冻结worker在线断言；离线重数事件与步日志，不声称有第二份独立native计数器。本审计基于留存向量/算术/原生反馈，不是独立AC潮流求解、QP残差认证、神经排序重计算或完整比赛复现。作者对finite user_limit等连续求解状态的处理保持原样。

## 研究判断的边界

本轮对我们此前未运行的四个周作事前冻结验证，但作者发布NN的训练身份未知，不能认证模型未见；同一合成环境、一个种子、四个周也不构成总体泛化保证。这四个周只有两个后缀组（5与12，三个周来自5）；后缀分组只是保守划分约定，不证明样本统计独立。不把逐步观测当作独立实验样本。数据是法国2035设定和修改IEEE118系统的公开合成场景，不是RTE生产日志。

GRID05开发结论作为历史保留，不被本轮重写。无论裁决如何，均不自动启动GRL：固定规则的成功仅建立强基线，失败只收口其普遍有效主张。神经推荐、物理搜索与连续优化的组合已经是作者结构，不能作为我们的新贡献。不得为了获得学习优势排除本轮强对照。

其他8个验证周及全部12个测试周本轮仍未运行。此次使用的4个验证周以后视为已暴露，不能重新包装成独立确认。没有未结作业，不自动追加场景、种子或求解预算。

主要文件：GRID06_summary.json、GRID06_per_run.csv、GRID06_audit.json、GRID06_postrun_review.json、paired_trajectories.json、first_divergences.json、expansion_audit.json；原始逐步数据在runs/。交付封存单独执行于所有生成器日志关闭后。
'''
(OUT/'GRID06_report.md').write_text(report,encoding='utf-8')
(OUT/'RUN_STATE.md').write_text(f'# GRID06\n\nCOMPLETE: 12/12 runs; saved-data audit passed. Verdict: {verdict}.\nFour validation weeks consumed; remaining8 validation/all12 test not evaluated. No pending jobs, training or automation.\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
