"""Explain saved stopping decisions and actual trajectory differences; no fitting/physics."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,math,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid185_stopping_mechanism';S=R/'outputs/grid184_search_adapter';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID185_stopping_mechanism'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    p=A/phase;p.mkdir(parents=True,exist_ok=False)
    for src in [*W.glob('*.py'),*O.iterdir()]:
        if src.is_file():shutil.copy2(src,p/src.name)
    (p/'archive_manifest.json').write_text(json.dumps({q.name:sha(q) for q in p.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for name in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/name).items():assert sha(R/p)==h,p
    assert read(S/'complete.json')['passed'] and read(S/'review.json')['passed_saved_data']
def prepare():
    assert not O.exists();O.mkdir(parents=True);assert read(S/'complete.json')['passed'];assert read(S/'publication_verification.json')['passed']
    base=read(S/'baseline_aligned_call_tradeoffs.json')['families']['FACTOR']['pairs'];counts=collections.Counter(v['class_name'] for v in base);assert counts==dict(fewer_same_chosen=23,fewer_different_chosen=10,same_same_chosen=29)
    active=[v for v in read(S/'first_divergences.json')['entries'] if not v['physical_signature_equal']];assert len(active)==3
    save('design.json',dict(id='GRID185_SAVED_STOPPING_MECHANISM',authorization='User:好的继续吧; continue explanation of23savings and3physicaldivergences. No new experiment scope.',
        population='All62originalFACTOR/FRAMEcommon-state+base pairs from repeat0/all3seeds/all4exposeddevelopmentdates. Retain23same-actionquerysavings,10changed-actionfewerquery,29samequery/action. Separately all3C/FACTfirstphysical-actiondifferences,not pickedbyresult.',
        questions=['Whichbatcheswereskipped,anddidtheyreallyfailtoimprovebest-observedreward?','DoescheapconsensusmisestimateimmediateNET,orimprovecurrentrewardwhiletheactualclosedloopcostworsens?','Whichrecordedcostcomponentsaccountfortheobservedcostdifference?'],
        task_units='OriginalGRID102fee/gainscale only. Q1-Q0 isestimatednext-batchNET,not calibratedprobability,wholeweekvalue,secondsorcurrency.',
        information_boundary='Decisionfeatures/predictions are the alreadyloggedones; extraactuallypaidFRAME/Csuffixfeedback is postdecisiondiagnostic only. No recomputedforecast or hiddenunqueriedoutcomes. No newmodelforward.',
        method='Matchactualobservationhash+baseactionhash,requireidenticalpaidprefixandsharedfeedback,lockedcandidateorder. Float32reward/lowIDbest as deployed. Score next<=64purchasedbatch and allactualextras separately. Compare nextphysicalaction/observation but do not claim wholeweekcausality from one difference.',
        accounting='Use savedcostledger:price*(losses+redispatch+curtailmentchange+storage). Retainfloatroundingresidual and originaltolerances. Selectedexecutedbranchesonly,not costlabelsforunexecutedcandidates.',
        uniqueness='Report23seed-paired occurrences,uniquephysicalstate+base,uniqueSTOPprefix,anddate counts separately;no binomial/significance claim.',
        no_selection='No newthreshold,gate,feature/loss/seedselection or accept/reject benchmark gate. Allcounterexamples retained.',
        prohibited=['training/modelinference','environmentconstruction/reset/step/simulate','newdata/final8','newbranchrollouts','paper/figures','claimforecastorfullweekoptimality']))
    sources=[S/n for n in ['complete.json','publication_verification.json','review.json','decision.json','results_summary.json','baseline_aligned_call_tradeoffs.json','aligned_call_tradeoffs.json','first_divergences.json','design.json']]+[R/'outputs/grid102_search_rl/design.json',R/'work/grid03/score_adapter.py',R/'work/grid101_gain_learning/features.py',O/'design.json']
    archived=read(REPO/'research_archive/power-grid-2026/protocols/GRID184_search_adapter/raw_records/records_manifest.json')['files']
    for rel,record in archived.items():
        if not rel.startswith('runs/r0__'):continue
        name=Path(rel).name
        if name=='steps.jsonl.gz' or name=='summary.json' or name.startswith('call') and (name.endswith('.json') or name.endswith('_state.npz')):
            p=S/rel;assert sha(p)==record['sha256'];sources.append(p)
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sources});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();archive('frozen');print('GRID185_EXPLANATION_FROZEN',dict(counts))

FIELDS=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible']
def key(row):return float(np.float32(row['reward'])),-row['pool_id']
def best(rows):return max((v for v in rows if v['strict_admissible']),key=key,default=None)
def rew(row):return float(np.float32(row['reward']))
def inspect():
    verify();assert not (O/'review.json').exists();task=read(R/'outputs/grid102_search_rl/design.json');fee=task['cost_per_query'];scale=task['gain_scale'];checks=collections.Counter();callcache={};stepcache={}
    def calls(week,method):
        k=week,method
        if k not in callcache:
            out={};folder=S/'runs'/f'r0__{week}__{method}'
            for p in folder.glob('call[0-9][0-9][0-9].json'):
                c=read(p)
                with np.load(p.with_name(p.stem+'_state.npz')) as z:b=hashlib.sha256(z['base_action'].tobytes()).hexdigest()
                c['_base_hash']=b;c['_source']=p.relative_to(R).as_posix();out[c['step'],c['state_hash'],b]=c
            callcache[k]=out
        return callcache[k]
    def steps(week,method):
        k=week,method
        if k not in stepcache:
            with gzip.open(S/'runs'/f'r0__{week}__{method}'/'steps.jsonl.gz','rt',encoding='utf8') as f:stepcache[k]=[json.loads(x) for x in f]
        return stepcache[k]
    def shared(a,b,n):
        assert [v['pool_id'] for v in a['outcomes'][:n]]==[v['pool_id'] for v in b['outcomes'][:n]]
        assert [{k:r[k] for k in FIELDS} for r in a['outcomes'][:n]]==[{k:r[k] for k in FIELDS} for r in b['outcomes'][:n]]
        assert a['order']==b['order'] and a['aliases']==b['aliases'] and a['pre_extra_choice']==b['pre_extra_choice'];checks['shared_prefixes']+=1
    def batches(prefix,tail):
        incumbent=best(prefix);initial=rew(incumbent);rows=[]
        for start in range(0,len(tail),64):
            chunk=tail[start:start+64];before=rew(incumbent)
            for r in chunk:
                if r['strict_admissible'] and key(r)>key(incumbent):incumbent=r
            gain=rew(incumbent)-before;net=(gain-fee*len(chunk))/scale;rows.append(dict(ids=[v['pool_id'] for v in chunk],queries=len(chunk),best_reward_gain=gain,normalized_net=net,chosen=incumbent['pool_id']))
        return rows,dict(gain=rew(incumbent)-initial,queries=len(tail),normalized_net=(rew(incumbent)-initial-fee*len(tail))/scale,chosen=incumbent['pool_id'])
    def stop_detail(learner,other):
        n=len(learner['outcomes']);shared(learner,other,n);tail=other['outcomes'][n:];assert tail
        stop=learner['learned_decisions'][-1];assert stop['action']==0 and stop['paid_queries']==n and [v['pool_id'] for v in tail[:64]]==stop['next_ids']
        assert all(r['phase']=='local_completion' for r in tail)
        labels,total=batches(learner['outcomes'],tail);margins={name:q[1]-q[0] for name,q in stop['scores'].items()};assert all(v<=0 for v in margins.values()) if 'FACTOR' not in margins else margins['FACTOR']<=0
        margin=margins.get('FACTOR')
        raw=stop['raw_features'];assert int(round(math.expm1(raw[39])))==len(stop['next_ids'])
        return dict(stop_batch=stop['batch'],paid_queries=n,next_queries=len(stop['next_ids']),remaining_candidates=stop['remaining'],factor_margin=margin,stopping_model_margins=margins,raw_features={name:raw[j] for j,name in enumerate(task['input']['features'])},next_batch=labels[0],all_extra=total,batches=labels,extra_recorded_forecast_s=math.fsum(v['forecast_s'] for v in tail))
    rows=[]
    for pair in read(S/'baseline_aligned_call_tradeoffs.json')['families']['FACTOR']['pairs']:
        week=pair['date']+'_100';seed=int(pair['seed']);fa=calls(week,f'FACTOR_seed{seed}');fb=calls(week,'FRAME_GUARD');keys=[k for k in fa.keys()&fb.keys() if k[0]==pair['step']];assert len(keys)==1;k=keys[0];a,b=fa[k],fb[k]
        assert a['public_queries']-b['public_queries']==pair['query_delta'];same=a['chosen_pool_id']==b['chosen_pool_id'];assert same==pair['class_name'].endswith('same_chosen')
        item=dict(pair,physical_state_hash=k[1],base_action_hash=k[2],factor_source=a['_source'],frame_source=b['_source'])
        if pair['query_delta']<0:
            details=stop_detail(a,b);item.update(details)
            if same:assert details['all_extra']['gain']==0. and all(v['best_reward_gain']==0. and v['normalized_net']<0 for v in details['batches']);checks['unchanged_action_zero_gain_savings']+=1
            a_step=steps(week,f'FACTOR_seed{seed}')[pair['step']-1];b_step=steps(week,'FRAME_GUARD')[pair['step']-1]
            item.update(applied_physical_action_equal=a_step['action_hash']==b_step['action_hash'],next_observation_equal=a_step['after_hash']==b_step['after_hash'])
            item['stop_context_hash']=hashlib.sha256(json.dumps([k,[v['pool_id'] for v in a['outcomes']],a['chosen_pool_id']],sort_keys=True).encode()).hexdigest()
        else:shared(a,b,len(a['outcomes']));assert a['chosen_pool_id']==b['chosen_pool_id'];item['completion_budget']=a['completion_budget'];item['learned_batches_requested']=[v['action'] for v in a['learned_decisions']]
        rows.append(item)
    assert len(rows)==62;safe=[v for v in rows if v['class_name']=='fewer_same_chosen'];changed=[v for v in rows if v['class_name']=='fewer_different_chosen'];assert len(safe)==23 and len(changed)==10
    stats=dict(paired_calls=23,unique_physical_state_base=len({(v['physical_state_hash'],v['base_action_hash']) for v in safe}),unique_stop_prefixes=len({v['stop_context_hash'] for v in safe}),dates=sorted({v['date'] for v in safe}),skipped_queries_seed_paired=sum(v['all_extra']['queries'] for v in safe),skipped_batches_seed_paired=sum(len(v['batches']) for v in safe),zero_gain_all23=all(v['all_extra']['gain']==0 for v in safe),same_applied_action=sum(v['applied_physical_action_equal'] for v in safe),same_next_observation=sum(v['next_observation_equal'] for v in safe),stop_batch_counts=dict(collections.Counter(str(v['stop_batch']) for v in safe)),per_date={date:dict(paired_calls=sum(v['date']==date for v in safe),skipped_queries=sum(v['all_extra']['queries'] for v in safe if v['date']==date)) for date in sorted({v['date'] for v in safe})},other10_next_batch_positive_NET=sum(v['next_batch']['normalized_net']>0 for v in changed),other10_all_extra_positive_NET=sum(v['all_extra']['normalized_net']>0 for v in changed))
    save('factor_frame_cases.json',dict(cases=rows,stats=stats,labels_are_postdecision_diagnostics_only=True))
    comparisons=[];ledgerfields=['losses_mwh','redispatch_mwh','curtailment_delta_mwh','storage_throughput_mwh']
    for item in read(S/'first_divergences.json')['entries']:
        if item['physical_signature_equal']:continue
        week=item['week'];seed=item['seed'];step=item['first_physical_action_divergence_step'];a_calls=calls(week,f'CONSENSUS_seed{seed}');b_calls=calls(week,f'FACTOR_seed{seed}');kk=[k for k in a_calls.keys()&b_calls.keys() if k[0]==step];assert len(kk)==1;k=kk[0];a,b=a_calls[k],b_calls[k]
        da=next(x for x in a['learned_decisions'] if any(x['batch']==y['batch'] and x['action']!=y['action'] for y in b['learned_decisions']));db=next(x for x in b['learned_decisions'] if x['batch']==da['batch']);assert da['action']!=db['action'] and not da['delegated'];assert da['raw_features']==db['raw_features'] and da['next_ids']==db['next_ids']
        detail=stop_detail(b,a) if da['action']==1 else stop_detail(a,b);assert detail['next_batch']['ids']==da['next_ids'];cheap={name:q[1]-q[0] for name,q in da['scores'].items()};assert set(cheap)=={'STATE4','POWER8'} and all((v>0)==bool(da['action']) for v in cheap.values())
        factor_margin=db['scores']['FACTOR'][1]-db['scores']['FACTOR'][0];assert (factor_margin>0)==bool(db['action'])
        sa=steps(week,f'CONSENSUS_seed{seed}');sb=steps(week,f'FACTOR_seed{seed}');assert len(sa)==len(sb) and all(x['action_hash']==y['action_hash'] for x,y in zip(sa[:step-1],sb[:step-1]));assert sa[step-1]['before_hash']==sb[step-1]['before_hash']
        full_a=math.fsum(x['raw_cost'] for x in sa);full_b=math.fsum(x['raw_cost'] for x in sb);assert math.fsum(x['raw_cost'] for x in sa[:step-1])==math.fsum(x['raw_cost'] for x in sb[:step-1])
        def account(seq):
            assert all(x['ledger'] is not None for x in seq);values={field:math.fsum(x['ledger']['marginal_cost']*x['ledger'][field] for x in seq) for field in ledgerfields}
            reconciled=math.fsum(x['ledger']['recomputed_raw_cost'] for x in seq);assert abs(math.fsum(values.values())-reconciled)<1e-5
            for x in seq:assert abs(x['raw_cost']-x['ledger']['recomputed_raw_cost'])<=x['ledger_tolerance']
            return dict(components=values,recomputed=reconciled,raw=math.fsum(x['raw_cost'] for x in seq),rounding_residual=math.fsum(x['raw_cost'] for x in seq)-reconciled)
        aa=account(sa[step-1:]);bb=account(sb[step-1:]);deltas={f:aa['components'][f]-bb['components'][f] for f in ledgerfields};residual=aa['rounding_residual']-bb['rounding_residual'];assert abs(math.fsum(deltas.values())+residual-(full_a-full_b))<1e-5
        windows=[]
        for n in [1,6,12,48,len(sa)-step+1]:
            n=min(n,len(sa)-step+1);windows.append(dict(steps_from_divergence=n,cost_delta=math.fsum(x['raw_cost'] for x in sa[step-1:step-1+n])-math.fsum(x['raw_cost'] for x in sb[step-1:step-1+n])))
        na=best(a['outcomes']);nb=best(b['outcomes']);local_right=(detail['next_batch']['normalized_net']>0)==bool(da['action']);classification='consensus_agrees_with_realized_next_NET_but_higher_week_cost' if local_right else 'consensus_disagrees_with_realized_next_NET_and_higher_week_cost'
        out=dict(seed=seed,week=week,step=step,direction='F_STOP_to_C_CONTINUE' if da['action']==1 else 'F_CONTINUE_to_C_STOP',continuing_method='CONSENSUS' if da['action']==1 else 'FACTOR',shared_observation_hash=k[1],shared_base_action_hash=k[2],factor_margin=factor_margin,cheap_margins=cheap,next_batch=detail['next_batch'],all_actual_extra=detail['all_extra'],consensus_agrees_with_next_NET=local_right,factor_chosen=b['chosen_pool_id'],consensus_chosen=a['chosen_pool_id'],selected_reward_delta=rew(na)-rew(nb),selected_forecast_rho_delta=na['rho']-nb['rho'],common_current_raw_features=detail['raw_features'],classification=classification,whole_week_cost_delta=full_a-full_b,whole_week_cost_relative_delta=full_a/full_b-1,cost_component_deltas=deltas,cost_rounding_residual_delta=residual,cost_windows=windows)
        assert out['whole_week_cost_delta']>0;comparisons.append(out);checks['physical_divergence_common_prefix_and_ledger']+=1
    save('consensus_override_cases.json',dict(cases=comparisons,no_single_divergence_causality_claim=True,selected_realized_paths_only=True))
    summary=dict(savings=stats,direction_counts=dict(collections.Counter(v['direction'] for v in comparisons)),override_counts=dict(collections.Counter(v['classification'] for v in comparisons)),override_cases=[{k:v[k] for k in ['seed','week','step','direction','continuing_method','factor_margin','cheap_margins','next_batch','all_actual_extra','selected_reward_delta','whole_week_cost_relative_delta','cost_component_deltas']} for v in comparisons],checks=dict(checks),new_fits=0,new_model_forwards=0,new_physics=0)
    save('summary.json',summary);verify();save('review.json',dict(passed=True,all62pairs_retained=True,savings23=len(safe),different_action10=len(changed),same_query29=29,all3physicaldivergences_retained=True,checks=dict(checks),frozen_sources_unchanged=True,new_fits=0,new_model_forwards=0,new_physics=0))
    save('state.json',dict(phase='analysed'));print('GRID185_ANALYSED',json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':globals()[sys.argv[1]]()
