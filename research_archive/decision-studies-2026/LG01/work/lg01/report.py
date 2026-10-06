"""Family-level paired evaluation, qualification and independent plan rechecking."""
import csv, math, re, time
from collections import Counter
import numpy as np
from core import *

def audit_report(d,ledger):
    from run import state_recs,status
    seal=read(OUT/'LG01_policy_seal.json')
    assert all(sha(OUT/p)==h for p,h in seal['artifacts'].items())
    assert all(sha(ROOT/p)==h for p,h in d['code_hashes'].items())
    assert all(sha(OUT/'states'/p)==h for p,h in read(OUT/'LG01_states_seal.json')['states'].items())
    assert all(sha(OUT/'labels'/p)==h for p,h in read(OUT/'LG01_labels_seal.json')['files'].items())
    validation=read(OUT/'LG01_validation_selection.json')
    rows=[read(p) for p in sorted((OUT/'labels').glob('*.json'))]+[read(p) for p in sorted((OUT/'test').glob('*.json'))]
    independent=[];contexts={}
    nominal_audit=[]
    for p in sorted((OUT/'nominal').glob('*.final.json')):
        final=read(p);meta=final['meta'];inst,_=build_instance('large',meta['rho'],meta['seed'])
        name=inst.name;initial=solution(read(OUT/'nominal'/f'{name}.initial.json'))
        assert verify(inst,initial)['ok']
        current=initial;records=[]
        bm=build_model(inst,MODE_AUDITED);slots=col_map(bm)
        for phase in [1,2]:
            r=read(OUT/'nominal'/f'{name}.polish{phase}.json');s=solution(r)
            fixes={} if r.get('fixed_sets') is None else {int(c):v for c,v in r['fixed_sets'][0].items()}
            v=verify(inst,s,current,fixes,slots)
            if not v['ok']: raise RuntimeError(f'NOM-160 own-stage anchor recheck {name} phase {phase}: {v}')
            records.append(v);current=s
        assert all(np.array_equal(getattr(current,k),np.asarray(final['plan'][k])) for k in KEYS)
        nominal_audit.append(dict(instance=name,stage_checks=records,flips_vs_initial=final['flips_vs_initial']))
    for r in rows:
        state=r['state']
        if state not in contexts:
            rec=read(OUT/'states'/f'{state}.json');inst,_=build_instance('large',rec['rho'],rec['family'])
            dis=rec['disruption'];pert=apply_disruption(inst,Disruption(dis['name'],{int(j):ts for j,ts in dis['down'].items()}))
            bm=build_model(pert,MODE_AUDITED);contexts[state]=(pert,bm,solution(rec['repair']),col_map(bm))
        pert,bm,anchor,slots=contexts[state]
        sol=solution(r)
        if r.get('fixed_sets') is None: fixed={}
        else: fixed={int(c):v for c,v in r['fixed_sets'][ACTIONS.index(r['action'])].items()}
        v=verify(pert,sol,anchor,fixed,slots)
        if not v['ok']: raise RuntimeError(f'Independent delivered recheck failed: {state} {v}')
        assert abs(v['cost']-r['objective'])<=1e-6*(1+abs(v['cost']))
        for label,c in r.get('candidate_plans',{}).items():
            cv=verify(pert,solution(c),anchor,fixed,slots)
            if not cv['ok']:raise RuntimeError(f'Independent candidate recheck failed {state} {label} {cv}')
        # Every imposed fixed condition is at the repaired reference, with only removals.
        if r.get('fixed_sets') is not None:
            fr={int(c):v for c,v in r['FR'].items()}
            for f in r['fixed_sets']:
                assert set(map(int,f))<=set(fr)
                assert all(fr[int(c)]==x for c,x in f.items())
            assert not r['fixed_sets'][4]
            assert not fixes_ok(anchor,fr,slots)
            assert all(bm.integrality[c] and bm.var_lb[c]!=bm.var_ub[c] for c in fr)
        independent.append(dict(state=state,policy=r['policy'],action=r['action'],seed=r['seed'],**v))
    save(OUT/'LG01_independent_audit.json',dict(ok=True,runs=len(rows),checks=independent,
         nominal_checks=nominal_audit,
         mechanism='Raw-parameter independent lsp_checker plus full Y/Z fixings and kappa; no matrix feasibility reuse.',
         policy_hashes_verified=True,state_seal_verified=True,label_seal_verified=True))
    test=[r for r in rows if r['family'] in d['split']['test']]
    expected=128 if seal['Bstar']=='RINS' else 192;assert len(test)==expected
    # Filename provides policy identity even when B* uses the RINS action.
    lookup={}
    for p in sorted((OUT/'test').glob('*.json')):
        r=read(p);policy=p.stem.split('__')[-2];lookup[(r['state'],r['seed'],policy)]=r
    paired=[]
    for rec in state_recs(d,'test'):
        for seed in [0,1]:
            state=rec['state'];g=lookup[state,seed,'GNN'];b=lookup[state,seed,'RINS' if seal['Bstar']=='RINS' else 'Bstar'];r=lookup[state,seed,'RINS']
            paired.append(dict(family=rec['family'],rho=rec['rho'],fault=rec['disruption']['name'],seed=seed,state=state,
                               G=(b['objective']-g['objective'])/g['D'],vs_RINS=(r['objective']-g['objective'])/g['D'],
                               J_GNN=g['objective'],J_Bstar=b['objective'],J_RINS=r['objective'],J_repair=g['repair_cost'],
                               baseline_relative=(b['objective']-g['objective'])/max(1,abs(b['objective'])),
                               GNN_action=g['action'],Bstar_action=b['action']))
    families=[]
    for f in sorted(d['split']['test']):
        rr=[r for r in paired if r['family']==f];assert len(rr)==8
        families.append(dict(family=f,G=float(np.mean([r['G'] for r in rr])),vs_RINS=float(np.mean([r['vs_RINS'] for r in rr]))))
    values=np.array([r['G'] for r in families]);mean=float(values.mean())
    draws=np.random.default_rng(20261006).choice(values,size=(20000,8),replace=True).mean(1)
    interval=np.quantile(draws,[.025,.975]).tolist()
    def group(key,vals):
        # Equal repeats per family in every group; explicitly macro-average by family.
        return {str(x):float(np.mean([np.mean([r['G'] for r in paired if r[key]==x and r['family']==f]) for f in sorted(d['split']['test'])])) for x in vals}
    rho=group('rho',d['rhos']);seeds=group('seed',[0,1]);faults=group('fault',d['faults'])
    inference=[r['inference_s'] for r in test if r['policy']=='gnn'];p95=float(np.quantile(inference,.95))
    vsr=float(np.mean([r['vs_RINS'] for r in families]));timingbad=[r for r in test if not r['timing_ok']]
    interrupted=[r for r in rows if r.get('interrupted')]
    training=validation['training'];costtrain=sum(training[n]['wall_s'] for n in ['gnn','mlp','tree'])
    qualification=dict(independent_check=True,complete_fixing_sets=not interrupted,
           kappa=True,online_timing=not timingbad,inference=p95<=.1,
           training=costtrain<=3600 and all(training[n]['validation_training_cap_ok'] for n in ['gnn','mlp']),
           total_compute=ledger.charge()<=43200,uninterrupted_generation=not any(read(p).get('interrupted') for p in (OUT/'nominal').glob('*.final.json')))
    gates=dict(mean_G=mean>=.02,positive_families=int(np.sum(values>0))>=6,
               rho_all_positive=all(x>0 for x in rho.values()),solver_seed_all_positive=all(x>0 for x in seeds.values()),
               versus_RINS=vsr>0,**qualification)
    oracle=validation['validation_oracle_vs_fixed'];allpass=all(gates.values())
    if not all(qualification.values()): verdict='QUALIFICATION_FAILED'
    elif mean<=.005: verdict='ACCEPT_CHEAP_STOP_GNN'
    elif allpass and oracle>=.02: verdict='CONTINUE_STATE_SELECTION'
    elif oracle<.02: verdict='LOW_ACTION_SELECTION_SPACE'
    else: verdict='UNDETERMINED'
    frequencies={p:dict(Counter(r['action'] for r in rs)) for p,rs in {
           'GNN':[r for r in test if r['policy']=='gnn'],
           'Bstar':[v for (s,k,p),v in lookup.items() if p==('RINS' if seal['Bstar']=='RINS' else 'Bstar')],
           'RINS':[v for (s,k,p),v in lookup.items() if p=='RINS']}.items()}
    timing={}
    for phase in ['build_s','fixed_lp_s','relax_s','actions_s','graph_s','select_s','start_verify_s','mip_s','delivery_s']:
        vals=[r['timing'].get(phase,0.) for r in test if r['policy']=='gnn']
        timing[phase]=dict(mean=float(np.mean(vals)),p95=float(np.quantile(vals,.95)))
    result=dict(verdict=verdict,mean_G=mean,family_bootstrap_95=interval,positive_families=int(np.sum(values>0)),
         n_families=8,families=families,rho=rho,solver_seed=seeds,fault=faults,mean_vs_RINS=vsr,
         inference_p95_s=p95,Bstar=seal['Bstar'],best_fixed_action=seal['fixed_action'],
         validation_oracle_vs_fixed=oracle,gates=gates,qualification=qualification,frequencies=frequencies,
         stage_timing_GNN=timing,training=training,compute_charged_s=ledger.charge(),
         formal_test_runs=expected,formal_label_runs=480,interrupted_runs=len(interrupted),
         fallback_LP=sum(r['fallback'] for r in test),fallback_delivery=sum(r['fallback_delivery'] for r in test),
         timing_violations=[dict(state=r['state'],policy=r['policy'],seed=r['seed'],wall_s=r['wall_s']) for r in timingbad],
         baseline_relative_gain=float(np.mean([r['baseline_relative'] for r in paired])),
         initial_policy_diagnostics={n:read(OUT/'models'/f'{n}.initial.json') for n in ['gnn','mlp']},
         labels_degeneracy=dict(distinct_actions=dict(Counter(r['action_meta']['distinct_actions'] for r in rows if 'action_meta' in r and r['policy']=='RINS')),
             rc_zero_fraction_mean=float(np.mean([r['action_meta']['rc_zero_fraction'] for r in rows if r['policy']=='RINS' and r.get('action_meta',{}).get('rc_zero_fraction') is not None]))))
    for name,data in [('LG01_test_pairs.csv',paired),('LG01_test_families.csv',families)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save(OUT/'LG01_results.json',result)
    lines=['# LG01 训练与封存测试结果','',f'裁决：**{verdict}**。',
       f'主指标家庭宏平均 G：**{mean:.2%}**；家庭级配对 bootstrap 95% 区间：[{interval[0]:.2%}, {interval[1]:.2%}]。',
       f'八个测试家庭中 {int(np.sum(values>0))}/8 为正；相对 RINS：{vsr:.2%}。',
       f'最强廉价策略 B*：{seal["Bstar"]}；验证最佳固定动作：{seal["fixed_action"]}。',
       '', 'G 的分母是修复成本 max(1, |J_r|)，不是 B* 成本；这是八家庭的一次投资筛查。',
       '', '| 家庭 | G | 相对 RINS |','|---|---:|---:|']
    lines += [f'| {r["family"]} | {r["G"]:.2%} | {r["vs_RINS"]:.2%} |' for r in families]
    lines += ['',f'ρ 分组：{rho}；求解器种子分组：{seeds}；故障分组：{faults}。',
        f'推理 p95：{1000*p95:.2f} ms；图构造 p95：{1000*timing["graph_s"]["p95"]:.2f} ms，均计入在线预算。',
        f'完整资格检查：{qualification}。测试超时协议违例 {len(timingbad)} 次。',
        f'验证逐状态候选池 oracle 相对验证最佳固定动作：{oracle:.2%}。',
        f'训练成本：GNN {training["gnn"]["wall_s"]:.1f} s，MLP {training["mlp"]["wall_s"]:.1f} s，树 {training["tree"]["wall_s"]:.1f} s。',
        f'累计记账计算耗时 {ledger.charge()/3600:.2f} 小时；正式标签 480 次，正式测试 {expected} 次。',
        f'LP 资格兜底 {result["fallback_LP"]} 次；交付来自自身 LP/修复方案 {result["fallback_delivery"]} 次；中断 {len(interrupted)} 次。',
        f'动作频率：{frequencies}。',
        '', '所有有效候选和交付方案均以原始参数独立验解，再验证完整 Y/Z 固定集与短期 Y 的 κ。没有测试全候选池，没有历史 60 秒见证标签。',
        '继续门槛联合满足且验证候选池有足够选择空间才进入后续状态条件选择研究；本轮不训练 RL。',
        '通过仍需成熟 LNS/ALNS 对照、新分布确认和具体新意。',
        '', '来源目录没有 Git 提交；LG01_freeze.json 保存逐脚本 SHA-256 和代码快照。依赖版本与接口修正均公开。',
        '中断时完整预算保守记账，保留预验合法参考方案，不重新给同一求解动作 20 秒。中断项不能获得严格计时资格。',
        '标签图缓存在固定动作交付后构造、单列成本；正式 GNN/MLP 自行重建所需输入并在预算内推理。']
    lines += ['', '工程与版本记录：'] + ['- '+note for note in d['engineering_notes']]
    (OUT/'LG01_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'LG01_delivery_manifest.json',dict(files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['RUN_STATE.json','events.jsonl','LG01_delivery_manifest.json']}))
    status('complete',verdict=verdict,mean_G=mean,positive_families=int(np.sum(values>0)),charged_s=ledger.charge())
    return result
