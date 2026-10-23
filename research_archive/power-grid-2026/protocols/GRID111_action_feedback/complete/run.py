"""All saved train/validation states; fixed arithmetic, no fitting or physical calls."""
from common111 import R,O,W,read,save,sha
import collections,copy,datetime,gzip,json,pickle,time
import numpy as np
from public import Decoder,best
from association import describe,evaluate

def main():
    assert not (O/'analysis_receipt.json').exists()
    inherited=read(R/'outputs/grid110_structure_audit/input_manifest.json')
    assert all(sha(R/p)==h for p,h in inherited.items())
    extra=[R/'outputs/grid108_loss_budget/probe_sources.json',R/'outputs/grid108_loss_budget/critical_probes.json']+list(W.glob('*.py'))
    probes=read(extra[0]);extra += [R/p for p in probes.values()]+[(R/p).with_name(__import__('pathlib').Path(p).stem+'_state.npz') for p in probes.values()]
    freeze={**inherited,**{p.relative_to(R).as_posix():sha(p) for p in extra}};save('input_manifest.json',freeze)
    save('plan.json',dict(id='GRID111_ACTION_FEEDBACK_ASSOCIATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Read-only saved-data diagnostics. No learned model fit/inference, control change, new simulation, test access or scaling.',
        geometry='Same bus-label-invariant device-pair connection signature as GRID110. Keep association of each paid same-station candidate with its reward; next<=64 fixed candidates only.',
        question1='Does nearest known partition have more similar next-candidate reward than all-known mean or exact expected random assignment of the same known rewards?',
        comparison='One fixed nearest structural distance, average all exact nearest ties. Exact null over all reward permutations; preserve histogram and tie size. No chosen k/bandwidth/seeds/sweep.',
        question2='Can a convex combination of already observed legal rewards predict improvement beyond the current admissible incumbent? Check boundedness independently.',
        question3='Are changed device-pair relations supported by both0/1 outcomes in already bought feedback? Record marginal reward contrasts and their agreement with hindsight best next candidate; descriptive, not causal or a deployed scoring policy.',
        samples='All3582 training/856 validation reachable states; ALL15 prior positive-flip states and3 live probes retained for detail. No filtering by outcome.',
        grouping='Primary error means: average candidates within state, prefixes within physical call, calls within date, then dates. Train/validation separate; raw pooled summaries and per-call/date reversals retained.',
        future='Future teacher outcomes used only by separate evaluator, never describe(). Probe next-outcome vectors not borrowed from another path.',
        limitation='Recorded forecast reward/old query fee, not physical week economics. Observational marginal contrasts are confounded by other connections, not identified causal effects.',
        outputs='Raw public contexts separate from future evaluation targets; IDs for association only, not numeric learned features. All adverse results preserved.'))
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cs={c['sample_id']:c for c in cases};D=read(R/'outputs/grid102_search_rl/design.json')
    meta=read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples'];bycall=collections.defaultdict(list)
    for m in meta:bycall[m['source_call']].append(m)
    decoder=Decoder();checks=collections.Counter();records=[];details=[]
    flips=read(R/'outputs/grid109_input_coverage/positive_flip_details.json');selected={(r['sample_id'],r['position']) for r in flips}
    start=time.perf_counter()
    with gzip.open(O/'public_inputs.jsonl.gz','wt',encoding='utf8') as inputs,gzip.open(O/'future_targets.jsonl.gz','wt',encoding='utf8') as targets:
        for no,(path,ms) in enumerate(bycall.items()):
            decoded=decoder.decode(path);call=decoded[0];truth={r['pool_id']:r for r in call['outcomes']};aliases=decoded[-1]
            for m in ms:
                case=cs[m['sample_id']];n=m['visible_count'];visible=[truth[i] for i in call['order'][:n]];station=decoder.subs[best(visible)['pool_id']]
                target={int(v) for k,v in aliases.items() if decoder.subs[k]==station};pending=[i for i in call['order'][n:] if i in target]
                for j in range(len(case['states'])):
                    nxt=pending[:64];assert len(nxt)==case['queries'][j] and best(visible)['pool_id']==case['chosen'][j]
                    context=decoder.context(decoded,visible,nxt,station);description=describe(context)
                    future=[{k:truth[i][k] for k in ['rho','reward','done','illegal','ambiguous','exceptions','strict_admissible']} for i in nxt]
                    evaluation=evaluate(context,description,future,D['cost_per_query'],D['gain_scale'])
                    assert evaluation['next_gain']==case['gains'][j]
                    assert not evaluation['positive_interpolation_gain_possible'];checks['convex_bound_states']+=1
                    # Redaction: the function has no future feedback/label fields and rejects their injection.
                    try:describe(dict(context,future_feedback=future))
                    except AssertionError:checks['future_field_rejections']+=1
                    else:raise AssertionError('Future feedback accepted')
                    reverse=copy.deepcopy(context)
                    reverse['seen_plans']=[[3-v if v>0 else v for v in p] for p in context['seen_plans']]
                    reverse['next_plans']=[[3-v if v>0 else v for v in p] for p in context['next_plans']]
                    reverse['incumbent_plan']=[3-v if v>0 else v for v in context['incumbent_plan']]
                    assert describe(reverse)==description;checks['bus_label_invariance']+=1
                    info=dict(sample_id=m['sample_id'],position=j,date=case['date'],partition=case['partition'],physical_state_group=case['physical_state_group'],source_call=path,step=m['step'])
                    inputs.write(json.dumps(dict(**info,public_input=context),ensure_ascii=False,allow_nan=False)+'\n')
                    targets.write(json.dumps(dict(sample_id=m['sample_id'],position=j,future_outcomes=future),ensure_ascii=False,allow_nan=False)+'\n')
                    records.append(dict(**info,**evaluation))
                    if (m['sample_id'],j) in selected:details.append(dict(**info,public_input=context,association=description,evaluation=evaluation))
                    visible.extend(truth[i] for i in nxt);pending=pending[len(nxt):]
                assert not pending
            if no%60==0:print('analysed',no+1,'/',len(bycall),'calls,',len(records),'states',flush=True)
    assert len(records)==4438 and len(details)==15
    # No future candidate outcome borrowing for exposed live probes, only the established batch-level target.
    critical=[]
    for probe in read(R/'outputs/grid108_loss_budget/critical_probes.json'):
        decoded=decoder.decode(probe['source']);call=decoded[0];dec=call['learned_decisions'][0]
        visible=call['outcomes'][:dec['paid_queries']];assert [v['pool_id'] for v in visible]==call['order'][:dec['paid_queries']]
        station=decoder.subs[best(visible)['pool_id']];context=decoder.context(decoded,visible,dec['next_ids'],station);d=describe(context)
        critical.append(dict(context=probe['context'],source=probe['source'],public_input=context,association=d,known_batch_normalized_net=probe['actual_next_normalized_net'],
            future_candidate_outcomes_borrowed=False))
    save('state_results.json',records);save('flip_details.json',details);save('critical_details.json',critical)
    def summarise(rs):
        metrics=['nearest_mse','mean_mse','exact_unpaired_mse','contrast_coverage','mean_unseen_relation_fraction']
        calls=collections.defaultdict(list)
        for r in rs:calls[r['physical_state_group']].append(r)
        callstats=[]
        for g,rr in sorted(calls.items()):
            v={k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None for k in metrics}
            callstats.append(dict(physical_state_group=g,date=rr[0]['date'],states=len(rr),**v))
        dates={}
        for date in sorted({r['date'] for r in callstats}):
            rr=[r for r in callstats if r['date']==date]
            dates[date]={k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None for k in metrics}
        macro={k:float(np.mean([r[k] for r in dates.values() if r[k] is not None])) for k in metrics}
        valid=[r for r in callstats if r['nearest_mse'] is not None];pos=[r for r in rs if r['actual_positive']]
        return dict(states=len(rs),physical_calls=len(calls),positive_states=len(pos),
            positive_calls=len({r['physical_state_group'] for r in pos}),date_call_balanced_macro=macro,per_date=dates,per_call=callstats,
            calls_nearest_better_than_mean=sum(r['nearest_mse']<r['mean_mse'] for r in valid),
            calls_nearest_better_than_exact_unpaired=sum(r['nearest_mse']<r['exact_unpaired_mse'] for r in valid),
            legal_error_calls=len(valid),reward_assignment_sensitive_states=sum(r['reversed_reward_assignment_changes_prediction'] for r in rs),
            positive_states_best_next_direction_positive=sum(r['best_next_contrast_direction'] is not None and r['best_next_contrast_direction']>0 for r in pos),
            positive_states_best_next_changes_fully_supported=sum(r['best_next_changed_pairs']==r['best_next_supported_changed_pairs'] for r in pos),
            positive_interpolation_gain_states=sum(r['positive_interpolation_gain_possible'] for r in rs))
    result={p:summarise([r for r in records if r['partition']==p]) for p in ['train','validation']};save('summary.json',result)
    assert all(sha(R/p)==h for p,h in freeze.items())
    save('analysis_receipt.json',dict(passed=True,states=len(records),detailed_flip_states=len(details),live_probes=len(critical),
        checks=dict(checks),public_command_checks=dict(decoder.checks),sources_unchanged=True,
        environment_constructions=0,new_forecasts=0,new_physical_steps=0,new_model_fits=0,learned_model_inferences=0,
        fixed_arithmetic_diagnostics=True,existing_learner_unchanged=True,elapsed_s=time.perf_counter()-start))
    print(json.dumps({p:{k:v for k,v in s.items() if k not in ['per_call','per_date']} for p,s in result.items()},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
