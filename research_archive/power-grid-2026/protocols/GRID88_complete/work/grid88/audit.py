"""Balanced repeated timing: saved vectors, causal feedback, quality and accounting."""
from pathlib import Path
import csv,gzip,hashlib,importlib.util,json,math,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid88'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not (OUT/'RUN.lock').exists();D=read(OUT/'design.json');S=read(OUT/'finished.json')['summaries'];assert len(S)==52
    assert {(s['repeat'],s['week'],s['method']) for s in S}=={(x['repeat'],x['week'],x['method']) for x in D['execution_order']}
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    sys.path[:0]=[str(ROOT/'work/grid08'),str(ROOT/'work/grid88')]
    sp=importlib.util.spec_from_file_location('grid88_explicit_audit',ROOT/'work/grid88/policies.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
    import torch;torch.set_num_threads(1)
    from common import graph_features
    from environment import make_env
    env=make_env();obs0=env.reset();nb=int(env.nb_highres_called);subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    full_reference=[v for stage in [65,68] for v in read(ROOT/f'outputs/grid{stage}/finished.json')['summaries'] if v['method']=='FULL' and v['repeat']==0]
    for p,h in read(OUT/'data_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    rows=[];checks=dict(physical_rows=0,causal_calls=0,public_contexts=0,terminal_exceptions=0,unchanged_repeat_steps=0);repeat_reference={}
    try:
      for s in S:
        folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}";assert s==read(folder/'summary.json') and s['illegal']==s['ambiguous']==0
        with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(l) for l in f]
        assert len(steps)==s['steps'] and abs(math.fsum(x['raw_cost'] for x in steps)-s['cost'])<1e-6
        assert abs(math.fsum(x['decision_s'] for x in steps)-s['controller_s'])<1e-6 and sum(x['simulations'] for x in steps)==s['public_forecasts']
        assert s['exception_steps']==sum(bool(x['exceptions']) for x in steps) and all(x['done'] for x in steps if x['exceptions'])
        with np.load(folder/'vectors.npz') as z:actions=z['action'];observations=z['observation']
        assert len(actions)==len(observations)==len(steps)
        for j,x in enumerate(steps):
            assert digest(actions[j])==x['action_hash'] and digest(observations[j])==x['after_hash']
            if x['ledger'] is not None:assert abs(x['ledger']['recomputed_raw_cost']-x['raw_cost'])<=x['ledger_tolerance']
        rk=(s['week'],s['method']); signature=[(x['action_hash'],x['after_hash'],x['raw_cost'],x['done'],x['complete']) for x in steps]
        if rk in repeat_reference:
            assert signature==repeat_reference[rk];checks['unchanged_repeat_steps']+=len(steps)
        else:repeat_reference[rk]=signature
        if s['method']=='FRAME_GUARD':
            refpath=ROOT/'outputs/grid70/runs'/f"r0__{s['week']}__{s['method']}"/'steps.jsonl.gz'
            with gzip.open(refpath,'rt',encoding='utf-8') as ff: prior=[json.loads(l) for l in ff]
            assert signature==[(x['action_hash'],x['after_hash'],x['raw_cost'],x['done'],x['complete']) for x in prior]
        calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))];assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
        for j,c in enumerate(calls):
            truth={r['pool_id']:r for r in c['outcomes']};seen=[];assert len(truth)==c['public_queries'] and c['order']==sorted(c['representatives'])
            for v in c['outcomes']:assert v['strict_admissible']==bool(0<v['rho']<c['rho_limit'] and not v['done'] and not v['exceptions'] and not v['illegal'] and not v['ambiguous'])
            context=None
            if s['method']!='FULL':
                with np.load(folder/f'call{j:03d}_inputs.npz') as z:context={k:z[k] for k in ['x','edges','glob']}
                h=hashlib.sha256()
                for k in sorted(context):v=np.ascontiguousarray(context[k]);h.update(k.encode());h.update(str(v.shape).encode());h.update(v.tobytes())
                assert h.hexdigest()==c['input_hash']
                with np.load(folder/f'call{j:03d}_state.npz') as z:obs=obs0.copy();obs.from_vect(z['observation'])
                xx,ee,gg=graph_features(obs,env,dict(offered=False,predictions=[None,None]))
                assert np.array_equal(xx,context['x']) and np.array_equal(ee,context['edges']) and np.array_equal(gg,context['glob']);checks['public_contexts']+=1
            def reveal(i,p):
                assert i in truth and i not in seen and truth[i]['phase']==p;seen.append(i);return truth[i]
            got=P.run(c['order'],reveal,s['policy'],subs,{int(k):int(v) for k,v in c['aliases'].items()},limit=c['rho_limit'],context=context)
            for k in ['chosen_pool_id','public_queries','stop_reason','local_queries','completion_budget','pre_extra_choice','extra_selected_ids']:assert got[k]==c[k],(s['week'],s['method'],k)
            for k in ['meta_events','gate_input','gate_decision']:
                if k in got:assert got[k]==c[k]
            if s['method'].startswith('FIXED_'):
                cap=int(s['method'].split('_')[1]);assert seen==c['order'][:cap]
                legal=[v for v in c['outcomes'] if v['strict_admissible']]
                independent=max(legal,key=lambda v:(float(np.float32(v['reward'])),-v['pool_id']),default=None)
                assert c['chosen_pool_id']==(None if independent is None else independent['pool_id'])
            assert seen==[v['pool_id'] for v in c['outcomes']];checks['causal_calls']+=1
        cp=read(folder/'components.json');optim=math.fsum(x['wall_s'] for x in cp['calls'] if x['in_control']);fore=s['all_forecast_s'];search=s['n1_source_s']-s['n1_forecast_s'];rest=s['controller_s']-optim-fore-search
        assert min(optim,fore,search,rest)>=-1e-6,(s['method'],rest)
        assert abs(optim+fore+search+rest-s['controller_s'])<1e-6
        full=next(v for v in full_reference if v['repeat']==0 and v['week']==s['week'] and v['method']=='FULL')
        delta=s['cost']/full['cost']-1 if s['complete'] and full['complete'] else None;survive=s['complete'] if full['complete'] else s['steps']>=full['steps']
        same=s['steps']==full['steps'] and s['complete']==full['complete'];active={c['step'] for c in calls};n1lat=[x['decision_s'] for x in steps if x['step'] in active];lat=[x['decision_s'] for x in steps]
        rows.append(dict(date=s['week'],method=s['method'],repeat=s['repeat'],steps=s['steps'],complete=s['complete'],quality_pass=bool(survive and (delta is None or delta<=D['quality_threshold'])),cost=s['cost'],cost_delta_full=delta,equal_horizon_full=same,queries=s['n1_forecasts'],all_forecasts=s['public_forecasts'],controller_s=s['controller_s'],controller_cpu_s=s['controller_cpu_s'],loop_wall_s=s['loop_wall_s'],setup_s=s['setup_s'],bookkeeping_s=s['bookkeeping_s'],physical_s=s['physical_s'],decision_p50_s=float(np.percentile(lat,50)),decision_p95_s=float(np.percentile(lat,95)),decision_p99_s=float(np.percentile(lat,99)),decision_max_s=max(lat),n1_decision_p95_s=float(np.percentile(n1lat,95)) if n1lat else None,optimizer_s=optim,forecast_s=fore,n1_nonforecast_s=search,remainder_s=rest))
        checks['physical_rows']+=len(steps);checks['terminal_exceptions']+=s['exception_steps']
      assert int(env.nb_highres_called)==nb
    finally:env.close()
    groups=[]
    for method in [m['method'] for m in D['methods']]:
      for week in D['weeks']:
        own=[r for r in rows if r['method']==method and r['date']==week];assert len(own)==2
        fields=['controller_s','controller_cpu_s','loop_wall_s','decision_p95_s','decision_p99_s','optimizer_s','forecast_s','n1_nonforecast_s','remainder_s']
        v=dict(method=method,date=week,quality_pass=all(r['quality_pass'] for r in own),complete=own[0]['complete'],queries=own[0]['queries'],cost_delta_full=own[0]['cost_delta_full'])
        for f in fields:v[f]=statistics.mean(r[f] for r in own)
        v['controller_min_s']=min(r['controller_s'] for r in own);v['controller_max_s']=max(r['controller_s'] for r in own);groups.append(v)
    comparisons=[]
    for baseline in ['FIXED_512']:
        paired=[];repeat=[]
        for w in D['weeks']:
            aa=next(g for g in groups if g['method']=='FRAME_GUARD' and g['date']==w)
            bb=next(g for g in groups if g['method']==baseline and g['date']==w)
            ar=next(r for r in rows if r['method']=='FRAME_GUARD' and r['date']==w)
            br=next(r for r in rows if r['method']==baseline and r['date']==w)
            equal=ar['steps']==br['steps'] and ar['complete']==br['complete']
            paired.append(dict(date=w,controller_reduction=1-aa['controller_s']/bb['controller_s'],p95_reduction=1-aa['decision_p95_s']/bb['decision_p95_s'],query_reduction=1-aa['queries']/bb['queries'] if equal and bb['queries'] else None,equal_horizon=equal,context_s=aa['controller_s'],baseline_s=bb['controller_s']))
        for rep in [0,1]:repeat.append(statistics.mean(1-next(x for x in rows if x['method']=='FRAME_GUARD' and x['date']==w and x['repeat']==rep)['controller_s']/next(x for x in rows if x['method']==baseline and x['date']==w and x['repeat']==rep)['controller_s'] for w in D['weeks']))
        macro=statistics.mean(p['controller_reduction'] for p in paired);positive=sum(p['controller_reduction']>0 for p in paired)
        q=[p['query_reduction'] for p in paired if p['query_reduction'] is not None]
        comparisons.append(dict(baseline=baseline,macro_controller_reduction=macro,pooled_controller_reduction=1-sum(p['context_s'] for p in paired)/sum(p['baseline_s'] for p in paired),positive_dates=positive,repeat_macro_reductions=repeat,macro_p95_reduction=statistics.mean(p['p95_reduction'] for p in paired),macro_query_reduction_equal_horizons=statistics.mean(q) if q else None,quality_dates=sum(g['quality_pass'] for g in groups if g['method']==baseline),stable_5percent_screen=bool(macro>=.05 and positive>=9 and all(v>0 for v in repeat)),dates=paired))
    assert checks['unchanged_repeat_steps']==sum(v['steps'] for v in S if v['repeat']==0)
    result=dict(passed_saved_data=True,runs=52,checks=checks,comparisons=comparisons,guard_quality_dates=sum(g['quality_pass'] for g in groups if g['method']=='FRAME_GUARD'),guard_complete_dates=sum(g['complete'] for g in groups if g['method']=='FRAME_GUARD'),all_timing_screens_passed=all(x['stable_5percent_screen'] for x in comparisons),groups=groups,rows=rows,independent_ac_resolve=False,new_fits=0,scope=D['scope'])
    save(OUT/'review.json',result)
    for name,data in [('per_run.csv',rows),('per_date.csv',groups)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
    (OUT/'RUN_STATE.md').write_text('COMPLETE:52 balanced runs; saved vectors, causal decisions, quality and disjoint timing accounting checked.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','rows','comparisons']},ensure_ascii=False));print(json.dumps([{k:v for k,v in c.items() if k!='dates'} for c in comparisons]))
if __name__=='__main__':main()
