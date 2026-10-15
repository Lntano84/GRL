"""Replay only revealed feedback and verify saved vectors/costs. No AC re-solve."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid76';sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','all'],default='pilot');a=ap.parse_args()
    D=read(OUT/'design.json');S=read(OUT/'manifest.json')['summaries']
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    assert not (OUT/'RUN.lock').exists()
    if a.phase=='pilot':assert (OUT/'pilot_finished.json').exists() and len(S)==3
    else:assert len(S)==len(D['execution_order'])
    refs={s['week']:s for stage in [65,68] for s in read(ROOT/f'outputs/grid{stage}/finished.json')['summaries'] if s['method']=='FULL' and s['repeat']==0}
    current={s['week']:s for s in S if s['method']=='FULL'}
    public_subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    checks=dict(physical_rows=0,causal_calls=0,full_reference_feedback=0,physical_terminal_exceptions=0)
    families=[];cache={}
    for s in S:
        folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
        assert s==read(folder/'summary.json') and s['illegal']==s['ambiguous']==0
        with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(l) for l in f]
        assert len(steps)==s['steps'] and abs(math.fsum(x['raw_cost'] for x in steps)-s['cost'])<1e-6
        assert abs(math.fsum(x['decision_s'] for x in steps)-s['controller_s'])<1e-6
        assert sum(x['simulations'] for x in steps)==s['public_forecasts']
        assert s['exception_steps']==sum(bool(x['exceptions']) for x in steps)
        assert all(x['done'] for x in steps if x['exceptions'])
        with np.load(folder/'vectors.npz') as z:actions=z['action'];observations=z['observation']
        assert len(actions)==len(observations)==len(steps)
        for j,x in enumerate(steps):
            assert digest(actions[j])==x['action_hash'] and digest(observations[j])==x['after_hash']
            if x['ledger'] is not None:assert abs(x['ledger']['recomputed_raw_cost']-x['raw_cost'])<=x['ledger_tolerance']
        checks['physical_rows']+=len(steps);checks['physical_terminal_exceptions']+=s['exception_steps']
        calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))];cache[s['week'],s['method']]=calls
        assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
        for j,c in enumerate(calls):
            truth={r['pool_id']:r for r in c['outcomes']};seen=[]
            assert len(truth)==c['public_queries'] and c['order']==sorted(c['representatives'])
            for r in c['outcomes']:assert r['strict_admissible']==bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous'])
            def reveal(i,phase):
                assert i in truth and i not in seen and truth[i]['phase']==phase
                seen.append(i);return truth[i]
            with np.load(folder/f"call{j:03d}_inputs.npz") as z:context={k:z[k] for k in ['x','edges','glob']}
            result=P.run(c['order'],reveal,s['policy'],public_subs,{int(k):int(v) for k,v in c['aliases'].items()},limit=c['rho_limit'],context=context)
            for k in ['chosen_pool_id','public_queries','stop_reason','local_queries','completion_budget','pre_extra_choice','extra_selected_ids']:assert result[k]==c[k],(s['week'],k)
            assert seen==[r['pool_id'] for r in c['outcomes']]
            if s['policy']=='CONTEXT_GATE' and c['verification_guard']:
                assert c['public_queries']>=min(256,len(c['order']))
                assert not c['stop_reason'].startswith('initial_safe')
            checks['causal_calls']+=1
        cp=read(folder/'components.json');outer=math.fsum(x['wall_s'] for x in cp['calls'] if x['in_control'])
        assert s['controller_s']-outer-s['all_forecast_s']>=-1e-6
        if s['method']=='FULL':
            ref=refs[s['week']];assert all(s[k]==ref[k] for k in ['steps','complete','cost','n1_forecasts'])
        if s['method']!='CONTEXT_GATE':continue
        f=current.get(s['week'],refs[s['week']]);matched=s['week'] in current
        complete=bool(s['complete'] and f['complete']);equal=s['steps']==f['steps'] and s['complete']==f['complete']
        delta=s['cost']/f['cost']-1 if complete else None
        survive=s['complete'] if f['complete'] else s['steps']>=f['steps']
        old=next(x for stage in [65,68] for x in read(ROOT/f'outputs/grid{stage}/finished.json')['summaries'] if x['week']==s['week'] and x['method']=='FRAME' and x['repeat']==0)
        families.append(dict(date=s['week'],steps=s['steps'],full_steps=f['steps'],original_steps=old['steps'],complete=s['complete'],
            quality_pass=bool(survive and (delta is None or delta<=D['quality_threshold'])),cost_delta_full=delta,
            cost=s['cost'],queries=s['n1_forecasts'],full_queries=f['n1_forecasts'],original_queries=old['n1_forecasts'],
            query_reduction_full=1-s['n1_forecasts']/f['n1_forecasts'] if equal and f['n1_forecasts'] else None,
            controller_s=s['controller_s'],controller_reduction_full=1-s['controller_s']/f['controller_s'] if matched and equal else None,
            matched_timing_pilot=matched,original_cost_delta_full=old['cost']/f['cost']-1 if old['complete'] and f['complete'] else None))
    for s in S:
        if s['method']!='CONTEXT_GATE' or s['week'] not in current:continue
        fdir=OUT/'runs'/f"r0__{s['week']}__FULL";fmap={}
        for path in sorted(fdir.glob('call[0-9][0-9][0-9].json')):
            c=read(path)
            with np.load(path.with_name(path.stem+'_state.npz')) as z:key=(digest(z['observation']),digest(z['base_action']))
            fmap[key]={r['pool_id']:r for r in c['outcomes']}
        folder=OUT/'runs'/f"r0__{s['week']}__FRAME_VERIFY"
        for j,c in enumerate(cache[s['week'],s['method']]):
            with np.load(folder/f'call{j:03d}_state.npz') as z:key=(digest(z['observation']),digest(z['base_action']))
            if key not in fmap:continue
            for r in c['outcomes']:
                assert all(r[k]==fmap[key][r['pool_id']][k] for k in ['rho','reward','done','exceptions','illegal','ambiguous','strict_admissible']);checks['full_reference_feedback']+=1
    compare=[x['query_reduction_full'] for x in families if x['query_reduction_full'] is not None]
    proceed=len(families)==3 and all(x['quality_pass'] for x in families)
    result=dict(passed_saved_data=True,phase=a.phase,runs=len(S),checks=checks,dates=families,
        quality_dates=sum(x['quality_pass'] for x in families),complete_dates=sum(x['complete'] for x in families),
        macro_query_reduction=statistics.mean(compare) if compare else None,
        continue_regression=proceed if a.phase=='pilot' else None,
        verdict=('PILOT_PASSED_CONTINUE_FIXED_REGRESSION' if proceed else 'PILOT_NOT_PASSED_NO_AUTOMATIC_EXPANSION') if a.phase=='pilot' else 'REGRESSION_REPORTED_ALL_DATES',
        new_fits=0,independent_ac_resolve=False,
        timing_scope='Single pilot timing repetitions only; descriptive, not stable superiority. No historical timing ratios for remaining dates.')
    save(OUT/('pilot_review.json' if a.phase=='pilot' else 'review.json'),result)
    with (OUT/('pilot_dates.csv' if a.phase=='pilot' else 'per_date.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(families[0]));w.writeheader();w.writerows(families)
    (OUT/'RUN_STATE.md').write_text(result['verdict']+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
