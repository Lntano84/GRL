"""Account for the stopped matrix; do not impute missing runs or change screens."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid06'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
audit=load(OUT/'GRID06_partial_audit.json')
assert audit['completed_data_audit_passed'] and not audit['matrix_complete']
design=load(OUT/'design_freeze.json');runs=load(OUT/'per_run_partial.json')
lookup={(r['scenario'],r['policy']):r for r in runs}
weeks=audit['completed_weeks'];contrasts=[]
for s in weeks:
    f=lookup[(s,'FALLBACK90')]
    for ref in ['NN20','NN352']:
        b=lookup[(s,ref)];joint=f['completed_native_week'] and b['completed_native_week']
        g=(b['native_raw_cost_sum']-f['native_raw_cost_sum'])/abs(b['native_raw_cost_sum']) if joint else None
        contrasts.append(dict(scenario=s,reference=ref,reference_steps=b['steps'],fallback_steps=f['steps'],
            jointly_complete=joint,cost_reduction_relative_to_reference=g,
            earlier_end=f['steps']<b['steps'],cost_regression_exceeds_frozen_2pct=joint and g<-.02,
            meaningful_benefit=(f['completed_native_week'] and not b['completed_native_week']) or (joint and g>=.05)))
violations=[c for c in contrasts if c['reference']=='NN20' and (c['earlier_end'] or c['cost_regression_exceeds_frozen_2pct'])]
verdict='FROZEN_REJECTION_ITEM_CONFIRMED_ON_COMPLETED_DATA' if violations else 'FULL_MATRIX_UNDETERMINED'
summary=dict(matrix_status='INCOMPLETE_IMPORT_WATCHDOG_STOP',completed_runs=9,failed_attempts=1,not_started_runs=2,
    completed_data_audit_passed=True,resource_acceptance_passed=False,verdict=verdict,
    original_full_matrix_not_accepted=True,contrasts=contrasts,confirmed_rejection_witnesses=violations,
    no_model_training_or_retry=True,validation_weeks_with_strategy_outcomes=weeks,
    failed_week_no_strategy_outcome=design['selected_weeks'][3],remaining_unrun_validation_weeks=9,unrun_test_weeks=12,
    warning='The missing October outcomes cannot be imputed. A verified rejection witness in completed data suffices for the frozen any-week rejection item; it does not complete the four-week matrix or establish a population failure rate.')
save('GRID06_partial_summary.json',summary)

components=[]
for r in runs:
    rows=lines(OUT/'runs'/f"{r['scenario']}__{r['policy']}"/'steps.jsonl')
    valid=[x for x in rows if x['ledger_eligible']]
    c=dict(scenario=r['scenario'],policy=r['policy'],completed_native_week=r['completed_native_week'])
    for name,field in [('loss_cost','losses_mwh'),('redispatch_cost','redispatch_mwh'),('curtailment_change_cost','curtailment_delta_mwh'),('storage_cost','storage_throughput_mwh')]:
        c[name]=sum(x['ledger']['marginal_cost']*x['ledger'][field] for x in valid)
    c['recomputed_verified_total']=sum(c[k] for k in ['loss_cost','redispatch_cost','curtailment_change_cost','storage_cost'])
    c['native_verified_total']=sum(x['raw_operational_cost'] for x in valid)
    c['native_error_terminal_total']=sum(x['raw_operational_cost'] for x in rows if not x['ledger_eligible'])
    assert abs(c['recomputed_verified_total']-c['native_verified_total'])<=sum(x['cost_rounding_tolerance'] for x in valid)+1e-6
    components.append(c)
save('operational_cost_components_partial.json',components)
with (OUT/'operational_cost_components_partial.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(components[0]));w.writeheader();w.writerows(components)

divergences=[]
for pair in load(OUT/'paired_trajectories_partial.json'):
    if pair['first_action_difference'] is None:continue
    step=pair['first_action_difference'];s=pair['scenario']
    d=dict(scenario=s,step=step,reference=pair['reference'],other=pair['other'],policies={})
    for p in [pair['reference'],pair['other']]:
        folder=OUT/'runs'/f'{s}__{p}';r=lines(folder/'steps.jsonl')[step-1]
        events=[e for e in lines(folder/'events.jsonl') if e['step']==step]
        d['policies'][p]=dict(before_observation_hash=r['before_observation_hash'],action_hash=r['action_hash'],
            before_rho_max=r['before_rho_max'],after_rho_max=r['after_rho_max'],
            delivered_action_forecast_available=r['selected_action_forecast_available'],
            delivered_action_forecast_rho_max=r['selected_forecast_rho_max'],
            module_returns=[e for e in events if e['kind']=='module'],
            fallback=[e for e in events if e['kind']=='fallback'],
            solver_statuses=[e for e in events if e['kind']=='solve'])
    d['same_prior_observation']=len({v['before_observation_hash'] for v in d['policies'].values()})==1
    divergences.append(d)
save('first_divergences_partial.json',divergences)

def pct(x):return '不比较：未共同完成' if x is None else f'{x*100:+.4f}%'
table=['| 验证周 | NN20结束步 | 回退结束步 | NN352结束步 | 回退相对NN20成本降幅 |','|---|---:|---:|---:|---:|']
costtable=['| 验证周 | NN20完整成本 | 回退完整成本 | NN352完整成本 |','|---|---:|---:|---:|']
simtable=['| 验证周 | NN20模拟数 | 回退模拟数 | NN352模拟数 | 回退扩展/检查 |','|---|---:|---:|---:|---:|']
for s in weeks:
    n,f,h=[lookup[(s,p)] for p in ['NN20','FALLBACK90','NN352']]
    g=next(c['cost_reduction_relative_to_reference'] for c in contrasts if c['scenario']==s and c['reference']=='NN20')
    table.append(f"| {s} | {n['steps']} | {f['steps']} | {h['steps']} | {pct(g)} |")
    costs=[f"{r['native_raw_cost_sum']:.6f}" if r['completed_native_week'] else '不排名' for r in [n,f,h]]
    costtable.append(f"| {s} | {' | '.join(costs)} |")
    simtable.append(f"| {s} | {n['simulate_calls']} | {f['simulate_calls']} | {h['simulate_calls']} | {f['expansions']}/{f['fallback_decisions']} |")
table.append(f"| {design['selected_weeks'][3]} | 未启动 | 未启动 | 导入标记前终止，无策略结果 | 缺失，不补值 |")
prefix=['| 验证周 | 参照→另一臂 | 同状态前缀步数 | 参照模拟数/动作秒 | 另一臂模拟数/动作秒 |','|---|---|---:|---:|---:|']
for p in load(OUT/'paired_trajectories_partial.json'):
    prefix.append(f"| {p['scenario']} | {p['reference']}→{p['other']} | {p['same_prior_prefix_steps']} | {p['same_prior_reference_simulates']}/{p['same_prior_reference_act_s']:.4f} | {p['same_prior_other_simulates']}/{p['same_prior_other_act_s']:.4f} |")
jan=weeks[0];jn,jf,jh=[lookup[(jan,p)] for p in ['NN20','FALLBACK90','NN352']]
janreg=(jf['native_raw_cost_sum']/jn['native_raw_cost_sum']-1)*100
fullreg=(jh['native_raw_cost_sum']/jn['native_raw_cost_sum']-1)*100
report=f'''# GRID06 部分完成收尾：固定回退验证

**完整确认矩阵未完成，资源验收未通过；九条已完成轨迹的数据审计通过。已有核验反例触发原冻结的拒绝项：{verdict}。**

本轮按GRID03事先封存的月度验证清单选择一、四、七、十月四周，NN20、原样复用FALLBACK90、同库NN352，共计划12条。规则、0.9阈值、原控制器、模型、连续优化和计量全部保持不变。只取得前三个周的9条完整运行；十月NN352在import_ready标记出现前被看门狗终止，后两臂按协议未启动。没有重跑、补选、追加种子或训练。

## 已核验结果与缺失

{chr(10).join(table)}

一月两臂都完整，固定回退成本**增加{janreg:.4f}%**，超过结果前固定的2%退化线。即使缺失十月也不能消除这个已核验拒绝见证，但这不意味着四周确认已经完成，亦不提供总体失败率。四月三臂都在176步终止，不用低累计成本计作节省。

{chr(10).join(costtable)}

一月NN352也比NN20贵 **{fullreg:.4f}%**。更大即时搜索并不自动转化为整周成本优势；不能仅凭此把差距归因于某个单一动作、模型预测错误或连续优化模块。本轮没有建立学习优势。

## 模拟与计量

{chr(10).join(simtable)}

整周模拟数包含其他控制模块，不同状态轨迹的总数不构成同质量加速结论。共同决策前状态的前缀比较如下，仅作一次运行的开销描述。

{chr(10).join(prefix)}

成本只比较共同完整周；原始native成本与物理账本核对，终止错误成本另列。operational_cost_components_partial.json/CSV按损耗、再调度、弃能变化与储能拆开，是描述诊断，不改变评分或判据。弃能变化费用不等于物理弃能总量。归一化比赛综合分数仍未认证，未用于排序。

首个动作分歧、同状态、模块选择和求解状态保存在first_divergences_partial.json。拓扑预测先于最终连续控制，不直接拿拓扑候选预测rho与最终动作实际rho作同动作残差解释；首个分歧也不单独证明后续费用差的因果机制。

## 超时与资源：不能写成全部通过

十月失败目录只有空console.log与watchdog_timeout.json，没有导入标记、初始状态、动作、反馈或实际推进日志。按冻结源码，环境构造与实际推进在写入导入标记之后。故它不构成十月策略成绩，不计作停电或算法失败。

设定导入上限180秒，实际检测记录 **{audit['import_watchdog']['wall_s']:.3f}秒**，子进程记录墙钟 **{audit['failed_process_wall_s']:.3f}秒**。看门狗没有有效满足导入上限，这是执行控制边界，原因未确定；不能仅凭空日志指认为PyTorch错误，也不确认主机休眠或调度暂停的成因。等待工具也曾长时间没有返回，这一时间现象不替代原因证据。

已记录子进程墙钟合计 **{audit['recorded_process_wall_including_failure_s']:.3f}秒**（含失败），完整九条合计 **{audit['completed_process_wall_s']:.3f}秒**。父进程没有正常MATRIX_COMPLETE时钟记录，因此不把子进程总和冒充完整调度墙钟，整体验收记未确认。

实际推进 **{audit['physical_steps']}步**、结果 **{audit['main_persisted_bytes']:,}字节**，这两项在原包络内。十月失败前未形成实际推进记录；不自动重试。审计另计 **{audit['audit_wall_s']:.3f}秒**。

## 数据审计范围

- 九条完成轨迹的全部向量重读、哈希和状态链核对。
- {audit['public_ledgers_recomputed']}个合格物理账目重算，最大原始成本残差 {audit['max_cost_abs_residual']:.9f}，满足沿用的float32容差。
- {audit['topology_module_calls_reconstructed']}次拓扑模块、{audit['fallback_decisions_reconstructed']}次回退判断重建；缓存前20项未重复模拟，原奖励/资格/完整排序/返回动作一致，生产forced_test全False。
- {audit['reused_assets_unchanged']}项复用文件、70项选定数据文件及4项启动时正式执行源码未变；三臂同周初始观测和通用参数一致。
- 每步native调用与事件计数由冻结worker在线断言，离线重数事件与步日志；不是独立的第二套native计数器。审计也不是独立AC求解、QP残差认证、神经模型重推断或训练。

完整矩阵用的audit_results.py/analyse.py等原脚本保留，未绕过其12条完整性条件来制造“全部通过”。本收尾使用明确命名的audit_partial.py/analyse_partial.py，局限随文件保存。

## 研究含义与后续边界

GRID05的开发正结果保留；本轮拒绝把固定回退推广为普遍改善。它不否定电网控制方向，也不支持“简单规则失败所以GRL会赢”。接下来的必要信息是这个已核验成本反例的可解释限制，而不是调整阈值找赢家。本轮已保存首分歧和费用分解，只作离线诊断，不启动新比较或模型。

前三周对我们此前未运行，但作者NN训练场景身份未知，不能认证模型未见。三个周来自后缀5和12；逐步观测不是独立样本。环境是公开合成法国2035/修改IEEE118场景，不是RTE生产日志。其余9个验证周没有取得策略成绩，全部12个测试周未运行；前三个周以后视为已暴露。十月存在启动失败记录，不称已经完成评估。

本轮到此按原冻结停止，无未结求解或训练、无自动化。主要交付：GRID06_partial_summary.json、GRID06_partial_audit.json、GRID06_partial_per_run.csv、GRID06_partial_postrun_review.json、first_divergences_partial.json、operational_cost_components_partial.json、原始runs/。
'''
(OUT/'GRID06_report.md').write_text(report,encoding='utf-8')
(OUT/'RUN_STATE.md').write_text(f'# GRID06\n\nSTOPPED_AND_CLOSED_PARTIAL: 9 complete runs audited; 1 import watchdog failure; 2 not started. Full quality/resource confirmation incomplete. Verdict on completed data: {verdict}.\nNo retries, pending jobs, training or automation. See GRID06_report.md.\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
