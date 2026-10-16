"""Balanced repeated timing: saved vectors, causal feedback, quality and accounting."""
from pathlib import Path
import csv,gzip,hashlib,importlib.util,json,math,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid87'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not (OUT/'RUN.lock').exists();D=read(OUT/'design.json');S=read(OUT/'finished.json')['summaries'];assert len(S)==13
    assert {(s['repeat'],s['week'],s['method']) for s in S}=={(x['repeat'],x['week'],x['method']) for x in D['execution_order']}
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    sys.path[:0]=[str(ROOT/'work/grid08'),str(ROOT/'work/grid87')]
    sp=importlib.util.spec_from_file_location('grid87_explicit_audit',ROOT/'work/grid87/policies.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
    import torch;torch.set_num_threads(1)
    from common import graph_features
    from environment import make_env
    env=make_env();obs0=env.reset();nb=int(env.nb_highres_called);subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    full_reference=[v for stage in [65,68] for v in read(ROOT/f'outputs/grid{stage}/finished.json')['summaries'] if v['method']=='FULL' and v['repeat']==0]
    guard_reference=read(ROOT/'outputs/grid70/finished.json')['summaries']
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
            refpath=ROOT/'outputs/grid82/runs'/f"r0__{s['week']}__{s['method']}"/'steps.jsonl.gz'
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
        full=next(v for v in full_reference if v['repeat']==s['repeat'] and v['week']==s['week'] and v['method']=='FULL')
        delta=s['cost']/full['cost']-1 if s['complete'] and full['complete'] else None;survive=s['complete'] if full['complete'] else s['steps']>=full['steps']
        same=s['steps']==full['steps'] and s['complete']==full['complete'];active={c['step'] for c in calls};n1lat=[x['decision_s'] for x in steps if x['step'] in active];lat=[x['decision_s'] for x in steps]
        rows.append(dict(date=s['week'],method=s['method'],repeat=s['repeat'],steps=s['steps'],complete=s['complete'],quality_pass=bool(survive and (delta is None or delta<=D['quality_threshold'])),cost=s['cost'],cost_delta_full=delta,equal_horizon_full=same,queries=s['n1_forecasts'],all_forecasts=s['public_forecasts'],controller_s=s['controller_s'],controller_cpu_s=s['controller_cpu_s'],loop_wall_s=s['loop_wall_s'],setup_s=s['setup_s'],bookkeeping_s=s['bookkeeping_s'],physical_s=s['physical_s'],decision_p50_s=float(np.percentile(lat,50)),decision_p95_s=float(np.percentile(lat,95)),decision_p99_s=float(np.percentile(lat,99)),decision_max_s=max(lat),n1_decision_p95_s=float(np.percentile(n1lat,95)) if n1lat else None,optimizer_s=optim,forecast_s=fore,n1_nonforecast_s=search,remainder_s=rest))
        checks['physical_rows']+=len(steps);checks['terminal_exceptions']+=s['exception_steps']
      assert int(env.nb_highres_called)==nb
    finally:env.close()
    # Historical references provide quality/query evidence only; never timing ratios.
    comparisons=[];divergences=[]
    def load_steps(folder):
        with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(l) for l in f]
    for s in S:
        fixed=next(r for r in rows if r['date']==s['week'])
        guard=next(v for v in guard_reference if v['week']==s['week'] and v['method']=='FRAME_GUARD' and v['repeat']==0)
        full=next(v for v in full_reference if v['week']==s['week'])
        delta_guard=guard['cost']/full['cost']-1 if guard['complete'] and full['complete'] else None
        guard_quality=(guard['complete'] if full['complete'] else guard['steps']>=full['steps']) and (delta_guard is None or delta_guard<=D['quality_threshold'])
        equal=guard['steps']==s['steps'] and guard['complete']==s['complete']
        comparisons.append(dict(date=s['week'],fixed_quality=fixed['quality_pass'],guard_quality=bool(guard_quality),fixed_complete=s['complete'],guard_complete=guard['complete'],full_complete=full['complete'],fixed_steps=s['steps'],guard_steps=guard['steps'],full_steps=full['steps'],fixed_cost_delta_full=fixed['cost_delta_full'],guard_cost_delta_full=delta_guard,fixed_cost_delta_guard=s['cost']/guard['cost']-1 if s['complete'] and guard['complete'] else None,fixed_queries=s['n1_forecasts'],guard_queries=guard['n1_forecasts'],full_queries=full['n1_forecasts'],equal_horizon_guard=equal,fixed_query_ratio_guard=s['n1_forecasts']/guard['n1_forecasts'] if equal and guard['n1_forecasts'] else None,query_reduction_full=1-s['n1_forecasts']/full['n1_forecasts'] if fixed['equal_horizon_full'] and full['n1_forecasts'] else None))
        gf=ROOT/'outputs/grid70/runs'/f"r0__{s['week']}__FRAME_GUARD"
        ff=OUT/'runs'/f"r0__{s['week']}__FIXED_256"
        gs=load_steps(gf);fs=load_steps(ff);first=None
        for j,(a,b) in enumerate(zip(gs,fs)):
            if a['action_hash']!=b['action_hash']:first=j;break
            # A common action/state prefix must reproduce exact physical outcomes.
            assert (a['after_hash'],a['raw_cost'],a['done'])==(b['after_hash'],b['raw_cost'],b['done']),(s['week'],j)
        v=dict(date=s['week'],first_action_divergence_step=None if first is None else fs[first]['step'],shared_action_prefix_steps=first if first is not None else min(len(gs),len(fs)),equal_horizon=equal)
        if first is not None:
            assert first==0 or gs[first-1]['after_hash']==fs[first-1]['after_hash']
            cg=[(p,read(p)) for p in gf.glob('call[0-9][0-9][0-9].json') if read(p)['step']==fs[first]['step']]
            cf=[(p,read(p)) for p in ff.glob('call[0-9][0-9][0-9].json') if read(p)['step']==fs[first]['step']]
            v['both_have_search_call']=bool(cg and cf)
            if cg and cf:
                pg,a=cg[0];pf,b=cf[0]
                with np.load(pg.with_name(pg.stem+'_state.npz')) as za,np.load(pf.with_name(pf.stem+'_state.npz')) as zb:
                    v['same_observation']=digest(za['observation'])==digest(zb['observation'])
                    v['same_base_action']=digest(za['base_action'])==digest(zb['base_action'])
                assert v['same_observation'] and v['same_base_action'] and a['order']==b['order']
                ta={x['pool_id']:x for x in a['outcomes']};tb={x['pool_id']:x for x in b['outcomes']};shared=set(ta)&set(tb)
                for i in shared:
                    assert all(ta[i][k]==tb[i][k] for k in ['rho','reward','done','exceptions','illegal','ambiguous','strict_admissible'])
                def chosen(c):
                    z=next((x for x in c['outcomes'] if x['pool_id']==c['chosen_pool_id']),None)
                    return None if z is None else {k:z[k] for k in ['pool_id','reward','rho','strict_admissible']}
                v.update(guard_queries=a['public_queries'],fixed_queries=b['public_queries'],guard_choice=chosen(a),fixed_choice=chosen(b),shared_feedback_checked=len(shared),guard_only_revealed_ids=sorted(set(ta)-set(tb)),fixed_only_revealed_ids=sorted(set(tb)-set(ta)),guard_stop=a['stop_reason'],fixed_stop=b['stop_reason'])
            v['interpretation']='First divergence at same physical prefix; subsequent week differences may compound and are not assigned solely to this call.'
        divergences.append(v)
    ratios=[c['fixed_query_ratio_guard'] for c in comparisons if c['fixed_query_ratio_guard'] is not None]
    qr=[c['query_reduction_full'] for c in comparisons if c['query_reduction_full'] is not None]
    result=dict(passed_saved_data=True,runs=13,checks=checks,rows=rows,comparisons=comparisons,first_divergences=divergences,fixed_quality_dates=sum(r['quality_pass'] for r in rows),guard_quality_dates=sum(c['guard_quality'] for c in comparisons),fixed_complete_dates=sum(r['complete'] for r in rows),full_complete_dates=sum(c['full_complete'] for c in comparisons),equal_horizon_guard_dates=sum(c['equal_horizon_guard'] for c in comparisons),macro_fixed_query_ratio_guard=statistics.mean(ratios) if ratios else None,macro_query_reduction_full=statistics.mean(qr) if qr else None,query_comparable_guard_dates=len(ratios),query_comparable_full_dates=len(qr),independent_ac_resolve=False,new_fits=0,timing_comparison=False,scope=D['scope'])
    save(OUT/'review.json',result);save(OUT/'first_divergences.json',divergences)
    for name,data in [('per_run.csv',rows),('per_date_comparison.csv',comparisons)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','comparisons','first_divergences']},ensure_ascii=False))
if __name__=='__main__':main()
