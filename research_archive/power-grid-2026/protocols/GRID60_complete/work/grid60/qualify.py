"""Same-state public-feedback replay and existing-path mechanism accounting."""
import copy,csv,hashlib,json,math,time
from pathlib import Path
from policies import run,key,hand_checks
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid60'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mean=lambda v:math.fsum(v)/len(v) if v else None
def main():
    started=time.perf_counter();D=read(OUT/'design.json');assert not (OUT/'qualification.json').exists()
    for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
    subs=read(ROOT/D['substation_metadata'])['public_subids'];assert len(subs)==909
    rows=[];trace_rows=[];permission_checks=0
    for p in D['sources']:
        c=read(ROOT/p);truth={r['pool_id']:r for r in c['outcomes']};order=sorted(c['representatives'])
        assert set(truth)==set(order);aliases={int(k):int(v) for k,v in c['aliases'].items()}
        full_choice=c['chosen_pool_id'];full_selected=truth.get(full_choice)
        for policy in D['policies']:
            seen=[]
            def observe(i,phase):
                assert i not in seen;seen.append(i);return {**truth[i],'phase':phase}
            result=run(order,observe,policy,subs,aliases)
            assert seen==[r['pool_id'] for r in result['outcomes']]
            altered=copy.deepcopy(truth)
            for i in set(truth)-set(seen):altered[i].update(reward=99.,rho=.01,strict_admissible=True)
            again=run(order,lambda i,p:{**altered[i],'phase':p},policy,subs,aliases)
            assert result==again;permission_checks+=1
            own=truth.get(result['chosen_pool_id']);assert own is None or own['strict_admissible']
            if policy=='FULL':assert result['chosen_pool_id']==full_choice
            rows.append({'source':p,'week':p.split('__')[1],'step':c['step'],'policy':policy,
                'queries':len(seen),'full_queries':len(order),'local_queries':result['local_queries'],
                'choice':result['chosen_pool_id'],'full_choice':full_choice,'same_full_choice':result['chosen_pool_id']==full_choice,
                'reward_difference_full':None if own is None or full_selected is None else key(own)[0]-key(full_selected)[0],
                'rho_difference_full':None if own is None or full_selected is None else own['rho']-full_selected['rho'],
                'stop_reason':result['stop_reason']})
    for p in D['frame_traces']:
        c=read(ROOT/p);truth={r['pool_id']:r for r in c['outcomes']};order=sorted(c['representatives']);aliases={int(k):int(v) for k,v in c['aliases'].items()}
        seen=[]
        def observe(i,phase):
            assert i in truth and i not in seen;seen.append(i);assert truth[i]['phase']==phase;return {**truth[i],'phase':phase}
        result=run(order,observe,'FRAME',subs,aliases)
        assert result['chosen_pool_id']==c['chosen_pool_id'] and result['public_queries']==c['public_queries']
        assert result['local_queries']==c['local_queries'] and result['stop_reason']==c['stop_reason']
        assert seen==[r['pool_id'] for r in c['outcomes']]
        before=run(order,lambda i,p:{**truth[i],'phase':p},'NO_LOCAL',subs,aliases)
        a=truth.get(before['chosen_pool_id']);b=truth.get(result['chosen_pool_id'])
        assert a is None or b is not None and key(b)>=key(a)
        trace_rows.append({'source':p,'step':c['step'],'queries':c['public_queries'],'local_queries':c['local_queries'],
            'pre_local_choice':before['chosen_pool_id'],'final_choice':result['chosen_pool_id'],
            'local_changed_choice':before['chosen_pool_id']!=result['chosen_pool_id'],
            'local_reward_improvement':None if a is None or b is None else key(b)[0]-key(a)[0],
            'stop_reason':c['stop_reason']})
    summaries=[]
    for policy in D['policies']:
        a=[r for r in rows if r['policy']==policy]
        summaries.append({'policy':policy,'states':len(a),'queries':sum(r['queries'] for r in a),
            'query_reduction_full':1-sum(r['queries'] for r in a)/sum(r['full_queries'] for r in a),
            'same_full_choice':sum(r['same_full_choice'] for r in a),'local_queries':sum(r['local_queries'] for r in a),
            'max_rho_difference_full':max((r['rho_difference_full'] or 0.) for r in a)})
    assert len(rows)==104 and permission_checks==104
    R={'passed_engineering':True,'hand_checks':hand_checks(),'permission_checks':permission_checks,
        'existing_frame_replay_equivalent':len(trace_rows),'summaries':summaries,
        'existing_local_changed_calls':sum(r['local_changed_choice'] for r in trace_rows),
        'existing_local_query_count':sum(r['local_queries'] for r in trace_rows),
        'wall_s':time.perf_counter()-started,'new_fits':0,'new_forecasts':0,'new_physical_rollouts':0,
        'warning':'Olddevelopment FULL states and already-exposed FRAME traces. Same-state choice/query diagnostics are not causal whole-week cost or complete control speed. Published first-safe fallback strengthened to best admissible;not complete paper reproduction.'}
    assert R['wall_s']<D['caps']['wall_s']
    (OUT/'qualification.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
    for name,a in [('per_state.csv',rows),('existing_frame_trace.csv',trace_rows)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(a[0]));w.writeheader();w.writerows(a)
    lines=['# GRID60 框架机制资格检查','',R['warning'],'','|策略|固定状态|查询|对FULL查询减少|同FULL选择|局部补查数|最大rho差|','|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"|{s['policy']}|{s['states']}|{s['queries']}|{s['query_reduction_full']:.2%}|{s['same_full_choice']}|{s['local_queries']}|{s['max_rho_difference_full']:.6f}|")
    lines+=['',f"已有FRAME调用复算完全一致：{len(trace_rows)}；其中局部补查改变选择{R['existing_local_changed_calls']}次，补查总数{R['existing_local_query_count']}。这是实际路径内单步证据，不能自动转成全周因果收益。",'',
        '普通Top-k/物理验证/首个安全候选停止已有文献；当前贡献候选聚焦反馈驱动渐进补查与变电站语义补全的质量—计算权衡。共有编译复用已有同路径组件证据，但不作为新GNN或新求解算法。',
        '', '来源：[LJN官方](https://github.com/lajavaness/l2rpn-2023-ljn-agent)、[SoftGNN2025](https://arxiv.org/html/2503.15190v2#S4.SS4)、[Verify/Hybrid方法](https://arxiv.org/html/2501.07186v1#S4.SS9)。',
        '', '代码、手工边界、104项隐藏反馈扰动与现有FRAME复算通过后，下一步只冻结同排序的有限闭环机制比较。无GNN训练/调参。']
    (OUT/'GRID60_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('COMPLETE: saved-feedback mechanism qualification;no new rollout or fit.\n',encoding='utf-8')
    print(json.dumps(R))
if __name__=='__main__':main()
