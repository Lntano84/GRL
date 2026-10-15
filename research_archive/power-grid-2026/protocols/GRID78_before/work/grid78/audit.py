"""Saved vectors, legal causal feedback, public context and matched quality queries."""
from pathlib import Path
import csv,gzip,hashlib,importlib.util,json,math,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid78'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not (OUT/'RUN.lock').exists();D=read(OUT/'design.json');S=read(OUT/'finished.json')['summaries']
    assert len(S)==14 and {(s['week'],s['policy']) for s in S}=={(w,m['policy']) for w in D['weeks'] for m in D['methods']}
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    sys.path[:0]=[str(ROOT/'work/grid08'),str(ROOT/'work/grid78')]
    sp=importlib.util.spec_from_file_location('grid78_explicit_policy_audit',ROOT/'work/grid78/policies.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
    import torch;torch.set_num_threads(1)
    from common import make_env,graph_features
    env=make_env();obs0=env.reset();nb=int(env.nb_highres_called)
    full={s['week']:s for s in read(ROOT/'outputs/grid65/finished.json')['summaries'] if s['method']=='FULL' and s['repeat']==0}
    context={s['week']:s for s in read(ROOT/'outputs/grid76/finished.json')['summaries']};guard={s['week']:s for s in read(ROOT/'outputs/grid70/finished.json')['summaries']}
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];checks=dict(physical_rows=0,causal_calls=0,public_contexts=0,meta_decisions=0,terminal_exceptions=0);dates=[]
    try:
      for s in S:
        folder=OUT/'runs'/f"r0__{s['week']}__{s['method']}";assert s==read(folder/'summary.json') and s['illegal']==s['ambiguous']==0
        with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(l) for l in f]
        assert len(steps)==s['steps'] and abs(math.fsum(x['raw_cost'] for x in steps)-s['cost'])<1e-6
        assert abs(math.fsum(x['decision_s'] for x in steps)-s['controller_s'])<1e-6
        assert sum(x['simulations'] for x in steps)==s['public_forecasts'] and s['exception_steps']==sum(bool(x['exceptions']) for x in steps)
        assert all(x['done'] for x in steps if x['exceptions'])
        with np.load(folder/'vectors.npz') as z:actions=z['action'];observations=z['observation']
        assert len(actions)==len(observations)==len(steps)
        for j,x in enumerate(steps):
            assert digest(actions[j])==x['action_hash'] and digest(observations[j])==x['after_hash']
            if x['ledger'] is not None:assert abs(x['ledger']['recomputed_raw_cost']-x['raw_cost'])<=x['ledger_tolerance']
        calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
        assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
        for j,c in enumerate(calls):
            truth={r['pool_id']:r for r in c['outcomes']};seen=[];assert len(truth)==c['public_queries'] and c['order']==sorted(c['representatives'])
            for r in c['outcomes']:assert r['strict_admissible']==bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous'])
            with np.load(folder/f'call{j:03d}_inputs.npz') as z:cx={k:z[k] for k in ['x','edges','glob']}
            h=hashlib.sha256()
            for k in sorted(cx):v=np.ascontiguousarray(cx[k]);h.update(k.encode());h.update(str(v.shape).encode());h.update(v.tobytes())
            assert h.hexdigest()==c['input_hash']
            with np.load(folder/f'call{j:03d}_state.npz') as z:obs=obs0.copy();obs.from_vect(z['observation'])
            xx,ee,gg=graph_features(obs,env,dict(offered=False,predictions=[None,None]))
            assert np.array_equal(xx,cx['x']) and np.array_equal(ee,cx['edges']) and np.array_equal(gg,cx['glob'])
            def reveal(i,p):
                assert i in truth and i not in seen and truth[i]['phase']==p;seen.append(i);return truth[i]
            got=P.run(c['order'],reveal,s['policy'],subs,{int(k):int(v) for k,v in c['aliases'].items()},limit=c['rho_limit'],context=cx)
            for k in ['chosen_pool_id','public_queries','stop_reason','local_queries','completion_budget','pre_extra_choice','extra_selected_ids','meta_events']:assert got[k]==c[k],(s['week'],k)
            assert seen==[r['pool_id'] for r in c['outcomes']]
            checks['causal_calls']+=1;checks['public_contexts']+=1;checks['meta_decisions']+=len(c['meta_events'])
        checks['physical_rows']+=len(steps);checks['terminal_exceptions']+=s['exception_steps']
        cp=read(folder/'components.json');outer=math.fsum(x['wall_s'] for x in cp['calls'] if x['in_control']);assert s['controller_s']-outer-s['all_forecast_s']>=-1e-6
        f=full[s['week']];b=context[s['week']];g=guard[s['week']];complete=s['complete'] and f['complete'];delta=s['cost']/f['cost']-1 if complete else None
        survive=s['complete'] if f['complete'] else s['steps']>=f['steps'];equal=s['steps']==f['steps'] and s['complete']==f['complete']
        dates.append(dict(date=s['week'],method=s['method'],steps=s['steps'],complete=s['complete'],quality_pass=bool(survive and (delta is None or delta<=D['quality_threshold'])),cost=s['cost'],cost_delta_full=delta,queries=s['n1_forecasts'],full_queries=f['n1_forecasts'],context_queries=b['n1_forecasts'],guard_queries=g['n1_forecasts'],query_reduction_full=1-s['n1_forecasts']/f['n1_forecasts'] if equal and f['n1_forecasts'] else None,controller_s=s['controller_s'],latency_p95_s=float(np.percentile([x['decision_s'] for x in steps],95)),context_cost=b['cost'],context_steps=b['steps']))
      assert int(env.nb_highres_called)==nb
    finally:env.close()
    byseed=[]
    for method in ['META_CONTEXT0','META_CONTEXT1']:
        rows=[r for r in dates if r['method']==method];assert len(rows)==7
        eligible=[r['query_reduction_full'] for r in rows if r['query_reduction_full'] is not None]
        total=sum(r['queries'] for r in rows);ctotal=sum(r['context_queries'] for r in rows)
        byseed.append(dict(method=method,quality_dates=sum(r['quality_pass'] for r in rows),complete_dates=sum(r['complete'] for r in rows),queries=total,context_queries=ctotal,context_query_reduction_vs_meta=1-ctotal/total,macro_query_reduction_full=statistics.mean(eligible),max_complete_cost_delta=max(r['cost_delta_full'] for r in rows if r['cost_delta_full'] is not None),macro_controller_s=statistics.mean(r['controller_s'] for r in rows)))
    result=dict(passed_saved_data=True,runs=14,checks=checks,dates=dates,by_seed=byseed,calibration_quality_matched=D['calibration_quality_matched'],independent_ac_resolve=False,new_fits=0,timing='Single-run descriptive absolute times. Historical context gate/FULL are not contemporaneous matched pairs.',scope='Exposed-date development, same legal context and exact guarded mode. Source-style conference adaptation, not original task/SOTA reproduction.')
    save(OUT/'review.json',result)
    with (OUT/'per_date.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(dates[0]));w.writeheader();w.writerows(dates)
    lines=['# GRID78 相同上下文与核查权限：会议适配闭环比较','',result['scope'],'',
        '| 方法 | 质量日期 | 完成日期 | 查询总量 | 上下文门控相对该方法省查询 | 最大完整周成本增幅 |','|---|---:|---:|---:|---:|---:|']
    for s in byseed:lines.append(f"| {s['method']} | {s['quality_dates']}/7 | {s['complete_dates']}/7 | {s['queries']} | {100*s['context_query_reduction_vs_meta']:+.3f}% | {100*s['max_complete_cost_delta']:+.4f}% |")
    lines+=['','上下文门控旧七日期质量7/7、完成5/7、查询10171；FRAME_GUARD查询11618。两个模型种子共同校准、全部结果保留，不能按日期择优。','',
        '这项比较直接检查相同合法电网上下文与完整核查选择下的质量/查询关系。控制时间及P95为单次描述，未与旧记录组成新平衡计时，不据此声称实际控制提速。未独立重求解AC。','',
        '普通上下文门控使用成熟随机森林，结构特征增量未建立；胜过本适配版本也不能证明停止/参数控制本身新颖。贡献须继续围绕完整框架与预算内核查作用表述。','']
    (OUT/'GRID78_report.md').write_text('\n'.join(lines),encoding='utf-8');(OUT/'RUN_STATE.md').write_text('COMPLETE:14 runs, saved-feedback/vector/context audits passed. See all seed results.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='dates'}),flush=True)
if __name__=='__main__':main()
