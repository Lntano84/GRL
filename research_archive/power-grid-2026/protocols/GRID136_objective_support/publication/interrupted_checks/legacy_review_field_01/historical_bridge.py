"""Compare immediate gains with existing56 single-omission physical labels."""
from run import *
def main():
    verify();assert read(O/'support_detail.json')['passed'];assert not (O/'historical_bridge_review.json').exists()
    old=R/'outputs/grid96_value_learning';design=read(old/'design.json');labels=read(old/'labels.json');assert len(labels)==56
    cases={m['id']:m for m in design['cases']};assert len(cases)==56;assert read(old/'review.json')['passed']
    paths=[old/n for n in ['labels.json','design.json','review.json','GRID96_report.md']]+[R/'outputs/grid95_labels/GRID95_report.md']
    paths += [R/m['source'] for m in cases.values()]
    frozen={p.relative_to(R).as_posix():sha(p) for p in paths};save('historical_bridge_sources.json',frozen)
    save('historical_bridge_plan.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True,
        purpose='All56 existing physical intervention labels versus their already-saved immediate full-local gain; separate old cohorts from2378teacher prefixes.',
        known='All historical records exposed;not a new test,trainingset or causal experiment. Do not drop pilots/unfavorable/MIXED/unfinished outcomes.',
        prohibitions=['fit','new label','price/risk scan','new physical replay','pool with2037data','claim positiveG implies economy or survival gain']))
    records=[];maxrhoerror=0.
    for l in labels:
        c=cases[l['case_id']];assert c['before_choice']==l['before_choice'] and c['after_choice']==l['after_choice'];call=read(R/c['source']);outcomes={r['pool_id']:r for r in call['outcomes']}
        before=outcomes[c['before_choice']];after=outcomes[c['after_choice']];rb=float(np.float32(before['reward']));ra=float(np.float32(after['reward']));G=ra-rb;assert G>=0
        rho_reduction=float(before['rho'])-float(after['rho']);maxrhoerror=max(maxrhoerror,abs(G-rho_reduction))
        primary=l['horizons']['primary'];comparable=bool(primary['completion']['reaches_horizon'] and primary['skip']['reaches_horizon'])
        delta=float(primary['skip']['cost']-primary['completion']['cost']) if comparable else None
        if comparable:assert abs(delta-l['cost_skip_minus_completion'])<1e-6
        records.append(dict(case_id=l['case_id'],calendar_date=l['calendar_date'],cohort=l['cohort'],pilot_known=l['pilot_known'],label=l['label'],
            immediate_full_gain=G,immediate_positive=G>0,initial_reward=rb,after_reward=ra,maxrho_reduction=rho_reduction,
            local_queries=l['local_queries'],before_choice=c['before_choice'],after_choice=c['after_choice'],physical_path_identical=l['physical_path_identical'],
            completion_complete=l['completion_complete'],skip_complete=l['skip_complete'],completion_steps=l['completion_steps'],skip_steps=l['skip_steps'],
            week_end_cost_comparable=comparable,cost_skip_minus_complete=delta,queries_saved=l['queries_saved']))
    tables={}
    for scope,rr in [('all56',records),('nonpilot53',[r for r in records if not r['pilot_known']])]:
        tables[scope]={}
        for label in ['SKIP','MIXED','KEEP']:
            aa=[r for r in rr if r['label']==label];tables[scope][label]=dict(cases=len(aa),positive_immediate_gain=sum(r['immediate_positive'] for r in aa),zero_immediate_gain=sum(not r['immediate_positive'] for r in aa),
                different_physical_paths=sum(not r['physical_path_identical'] for r in aa),complete_cost_pairs=sum(r['week_end_cost_comparable'] for r in aa))
        assert sum(v['cases'] for v in tables[scope].values())==len(rr)
    save('historical_bridge_records.json',records);save('historical_bridge_summary.json',tables)
    review=dict(passed=True,cases=56,nonpilot_cases=53,new_physical_replays=0,new_fits=0,new_labels=0,
        original_label_definitions_unchanged=True,original_action_reward_lookup_checked=True,
        matched_window_cost_arithmetic_checked=True,max_gain_minus_rho_reduction_error=maxrhoerror,
        cohorts_not_pooled_with2037=True)
    for p,h in frozen.items():assert sha(R/p)==h
    save('historical_bridge_review.json',review)
    lines=['# GRID136历史对照：即时正收益是否意味着补查对运行更好','',
        '真实运行标签并非从未做过。GRID95/96已经在56个既有状态上进行过单次取消补查的对照，并跟随原FRAME运行到固定终点；本页复用这些保存结果，不新增回放。','',
        '逐个从原候选反馈提取补查前后最佳奖励，计算G，再与原多维SKIP/MIXED/KEEP标签并列。不同2035/2036旧数据与当前2037教师前缀分开，不当新测试或混合训练集。','',
        '| 范围 | 原物理标签 | 例数 | G>0 | G=0 | 双方到达原整周终点可比成本 |','|---|---|---:|---:|---:|---:|']
    for scope,t in tables.items():
        for label,v in t.items():lines.append(f"| {scope} | {label} | {v['cases']} | {v['positive_immediate_gain']} | {v['zero_immediate_gain']} | {v['complete_cost_pairs']} |")
    lines+=['',
        '原标签SKIP包括取消后物理路径完全相同、或在规定存活/成本/计算/风险维度上占优；MIXED保留不同维度的权衡；KEEP保留补查带来的优势。路径相同的共同提前失败不等于完成整周。不能把SKIP全部当作“完整周安全取消”，也不能把MIXED全部算补查有益。','',
        '这组旧结果可以直接检验：G>0并不等价于整周应该保留补查；一些正G案例取消后仍属于SKIP，另一些是MIXED或KEEP。负载率下降是有意义的局部信号，但不能单凭其符号给运行经济收益贴标签。',
        '全部56条逐例G、原标签、存活、共同终点成本差及查询保存在historical_bridge_records.json。这里只作并列，不把多个单次干预的收益相加，也不训练新映射。','',
        '重要反例仍是GRID95的尚存过载案例：取消当步省成本，随后很快提前终止。因此即使未来补充候选的一步运行成本，也不能仅用一步成本差取代存活与后续时域。',
        '原GRID96/97在这些少量真实标签上学习识别并未成功；只有1个KEEP且主要非试点不含新KEEP，不能靠多数类准确率、换成TabPFN或加入当前大量相关前缀就宣称问题已解决。',
        '后续应复用已有标签流程与经验，明确需要什么新的独立运行证据；当前不自动扩大、重训或重放。']
    (O/'即时收益与历史物理标签对照.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    dest=A/'historical_bridge';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*[O/n for n in ['historical_bridge_sources.json','historical_bridge_plan.json','historical_bridge_records.json','historical_bridge_summary.json','historical_bridge_review.json','即时收益与历史物理标签对照.md']]]:shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8')
    verify();print(json.dumps(dict(summary=tables,review=review),ensure_ascii=False))
if __name__=='__main__':main()
