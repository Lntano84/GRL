"""Read-only global batch qualification on complete saved teacher feedback."""
from pathlib import Path
import collections,datetime,gzip,hashlib,importlib.util,json,math,shutil,statistics,struct,sys
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid139_global_feasibility';W=Path(__file__).parent
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID139_global_feasibility'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def key(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
def best(rows):return max((r for r in rows if r['strict_admissible']),key=key,default=None)
def bid(r):return None if r is None else r['pool_id']
def improves(a,b):return b is not None and (a is None or key(b)>key(a))
def gain(a,b):return None if a is None else key(b)[0]-key(a)[0]
def checkpoints(m):return list(range(min(128,m),m,64))+[m]
def sealed():
    d=read(R/'outputs/grid100_data_foundation/design.json')
    for c in d['scenario_plan']:
        if c['partition']=='test':
            for p in [R/d['environment']/'chronics'/c['scenario'],R/'outputs/grid100_data_foundation/datasets'/c['scenario'],R/'outputs/grid100_data_foundation/generation_receipts'/(c['scenario']+'.json'),R/'outputs/grid100_data_foundation/dataset_receipts'/(c['scenario']+'.json')]:assert not p.exists(),p
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    sealed()
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
def sources():
    d=read(R/'outputs/grid100_data_foundation/design.json');out=[]
    for c in d['scenario_plan']:
        if c['partition'] not in ['train','validation']:continue
        runtime='teacher_runtime' if c['scenario']==d['pilot']['scenario'] else 'bulk_runtime'
        p=R/'outputs/grid100_data_foundation'/runtime/'runs'/f"r0__{c['scenario']}__FULL_TEACHER";s=read(p/'summary.json');cc=sorted(p.glob('call[0-9][0-9][0-9].json'))
        assert s['week']==c['scenario'] and s['partition']==c['partition'] and len(cc)==s['n1_calls']
        out.append(dict(plan=c,summary=s,folder=p,calls=cc))
    assert len(out)==40 and sum(len(x['calls']) for x in out)==305;return out
def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed();assert read(R/'outputs/grid100_data_foundation/bulk_review.json')['passed']
    save('design.json',dict(id='GRID139_GLOBAL_SAVED_FEEDBACK_FEASIBILITY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User continues afterGRID138: inspect existing complete feedback before training or collecting.',
        population='All40 original train32/validation8 teacher scenarios,20 calendar dates,305 full calls. Include prefixes with no incumbent,not only previous local-eligible prefixes.',
        fixed='Original order,aliases,admissibility,float32/low-id selection;p128,batch64,patience2,safe0.9. Official GRID86/70 plus independent manual reconstruction.',
        baselines=['FULL','PREFIX128','PREFIX_PLUS_ONE192','FIXED_256','FIXED_512','FRAME_GLOBAL_ONLY','FRAME_GUARD'],
        baseline_scope='Query and reward references at saved states.192 is128+one original batch,not a new benchmarked method or chosen budget.',
        inputs='Only revealed prefix feedback and public M/rho_limit;metadata date/scenario/id for groups only. Hidden suffix poisoning checked.',
        targets='Separate next-batch and all-remaining gain;missing-incumbent recovery separate,gain null when incumbent absent. No wholeweek economic labels from search reward.',
        summaries='Train/validation separate;scenario ratios then same-date replicate mean then date macro. Report all comparisons,strata and stop failures;no new learned policy or cutoff.',
        hindsight='Earliest prefix matching original global incumbent is future-informed global-stage opportunity;extra local work may offset savings. Not deployable strategy or full-control saving.',
        public_strata='Original entry-safe flag,current incumbent availability,current safe0.9,and stale>=2. All strata retained,not feature/rule selection.',
        limits='FULL physical paths differ from FRAME;reward2-maxrho not weekly cost;original query fee not used as real time or money.',
        prohibited=['fit','model forward','new forecast/physical/scenario','final tests','threshold search','online rule change','automatic scale','paper edit']))
    paths=[R/p for p in ['outputs/grid100_data_foundation/design.json','outputs/grid100_data_foundation/bulk_review.json','outputs/grid100_data_foundation/bulk_complete.json','outputs/grid100_data_foundation/bulk_dataset_manifest.json','outputs/grid28/offline.json',
        'outputs/grid138_opportunity/final_verification.json','work/grid86/policies.py','work/grid70/policies.py','work/grid67/policies.py','work/grid63/policies.py','work/grid60/policies.py','work/grid27/policy.py']]
    for s in sources():paths += [s['folder']/'summary.json',*s['calls']]
    paths.append(O/'design.json');save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(set(paths))});save('code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))})
    save('state.json',dict(stage='frozen'));verify();archive('frozen');print('GRID139_FROZEN')
def input_at(c,n):
    visible=c['outcomes'][:n];assert [r['pool_id'] for r in visible]==c['order'][:n]
    cp=[min(128,n)]+list(range(192,n+1,64))
    if cp[-1]!=n:cp.append(n)
    b0=best(visible[:cp[0]]);b=best(visible);prev=b0;stale=0;count=0;lastgain=0.;repaired=False
    for end in cp[1:]:
        now=best(visible[:end]);ok=improves(prev,now);stale=0 if ok else stale+1;count+=ok;lastgain=gain(prev,now);repaired=prev is None and now is not None;prev=now
    valid=sorted((r for r in visible if r['strict_admissible']),key=key,reverse=True)
    f=dict(candidate_count=len(c['order']),visible_count=n,queried_fraction=n/len(c['order']),remaining_candidates=len(c['order'])-n,
        incumbent_available=b is not None,incumbent_reward=None if b is None else key(b)[0],incumbent_rho=None if b is None else b['rho'],rho_limit=c['rho_limit'],
        prefix_available=b0 is not None,prefix_safe=b0 is not None and b0['rho']<.9,incumbent_changed_since_prefix=bid(b0)!=bid(b),observed_admissible_fraction=len(valid)/n,
        best_two_reward_gap=key(valid[0])[0]-key(valid[1])[0] if len(valid)>1 else None,last_batch_reward_gain=lastgain,last_batch_repaired=repaired,
        improving_batch_fraction=count/(len(cp)-1) if len(cp)>1 else 0.,stale_batches=stale,reward_gain_since_prefix=gain(b0,b))
    payload=[{k:r[k] for k in ['pool_id','rho','reward','strict_admissible','done','illegal','ambiguous','exceptions']} for r in visible]
    return dict(features=f,visible_feedback_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest())
def manual_guard(c,subs):
    order=c['order'];m=len(order);n=min(128,m);b0=best(c['outcomes'][:n]);b=b0;safe=b is not None and b['rho']<.9;stale=0;visited=[n]
    for offset in range(n,m,64):
        before=b;n=min(offset+64,m);b=best(c['outcomes'][:n]);stale=0 if improves(before,b) else stale+1;visited.append(n)
        if not safe and b is not None and b['rho']<.9:break
        if b is not None and n>=256 and stale>=2 and n<m:break
    pre=b;local=[]
    if b is not None and n<m and (not safe or bid(b)!=bid(b0)):
        station=subs[b['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if subs[int(alias)]==station};local=[i for i in order[n:] if i in ids]
        truth={r['pool_id']:r for r in c['outcomes']};b=best([pre]+[truth[i] for i in local])
    return dict(global_n=n,pre=pre,after=b,local=local,visited=visited)
def metric(full,picked):
    return dict(exact_full_action=bid(full)==bid(picked),full_has_admissible=full is not None,lost_admissible=full is not None and picked is None,
        reward_regret=key(full)[0]-key(picked)[0] if full is not None and picked is not None else (0. if full is None else None))
def run():
    verify();assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(fits=0,physical=0))
    spec=importlib.util.spec_from_file_location('grid139_reference',R/'work/grid86/policies.py');G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
    subs=read(R/'outputs/grid28/offline.json')['public_subids'];calls=[];methods=[];metas=[];inputs=[];targets=[];checks=collections.Counter()
    oldhash={r['source_call']:r['source_sha256'] for r in read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']}
    for s in sources():
        plan=s['plan'];totalq=0
        for path in s['calls']:
            c=read(path);order=c['order'];truth={r['pool_id']:r for r in c['outcomes']};m=len(order);assert m>0 and len(truth)==m==c['public_queries'] and [r['pool_id'] for r in c['outcomes']]==order
            identity=plan['scenario']+'|'+path.stem;rel=path.relative_to(R).as_posix();totalq+=m
            if rel in oldhash:assert sha(path)==oldhash[rel]
            for r in c['outcomes']:
                assert r['strict_admissible']==bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['illegal'] and not r['ambiguous'] and not r['exceptions']);checks['admissibility']+=1
            full=best(c['outcomes']);assert bid(full)==c['chosen_pool_id'];seen=[]
            def reveal(i,phase):
                assert i not in seen;seen.append(i);return dict(truth[i],phase=phase)
            official=G.run(order,reveal,'FRAME_GUARD',subs,c['aliases'],limit=c['rho_limit']);manual=manual_guard(c,subs);ng=manual['global_n']
            assert official['pre_extra_choice']==bid(manual['pre']) and official['chosen_pool_id']==bid(manual['after'])
            assert seen==order[:ng]+manual['local'] and official['extra_selected_ids']==manual['local'];checks['independent_guard_calls']+=1
            cp=checkpoints(m);bprefix=best(c['outcomes'][:min(128,m)]);earliest=next(n for n in cp if bid(best(c['outcomes'][:n]))==bid(manual['pre']));assert earliest<=ng
            calls.append(dict(call_id=identity,source=rel,partition=plan['partition'],date=plan['date'],scenario=plan['scenario'],step=c['step'],teacher_complete=s['summary']['complete'],M=m,
                prefix_available=bprefix is not None,prefix_safe=bprefix is not None and bprefix['rho']<.9,full_available=full is not None,full_choice=bid(full),global_n=ng,global_choice=bid(manual['pre']),
                local_queries=len(manual['local']),frame_choice=bid(manual['after']),guard_stop_reason=official['stop_reason'],earliest_guard_prefix=earliest,global_queries_hindsight_redundant=ng-earliest,
                earliest_full_prefix=next(n for n in cp if bid(best(c['outcomes'][:n]))==bid(full)),guard_metric=metric(full,manual['pre']),frame_metric=metric(full,manual['after'])))
            baselines={'FULL':(m,full),'PREFIX128':(min(128,m),bprefix),'PREFIX_PLUS_ONE192':(min(192,m),best(c['outcomes'][:min(192,m)])),
                'FIXED_256':(min(256,m),best(c['outcomes'][:min(256,m)])),'FIXED_512':(min(512,m),best(c['outcomes'][:min(512,m)])),
                'FRAME_GLOBAL_ONLY':(ng,manual['pre']),'FRAME_GUARD':(ng+len(manual['local']),manual['after'])}
            for method,(q,b) in baselines.items():methods.append(dict(call_id=identity,partition=plan['partition'],date=plan['date'],scenario=plan['scenario'],method=method,queries=q,full_queries=m,**metric(full,b)))
            for n in cp[:-1]:
                before=best(c['outcomes'][:n]);nextbest=best(c['outcomes'][:min(n+64,m)]);x=input_at(c,n)
                poisoned=dict(c,outcomes=c['outcomes'][:n]+[dict(r,reward=999.,rho=-999.,strict_admissible=True) for r in c['outcomes'][n:]])
                assert input_at(poisoned,n)==x;checks['hidden_suffix_checks']+=1;sid=hashlib.sha256((identity+'|'+str(n)).encode()).hexdigest()[:24]
                metas.append(dict(sample_id=sid,call_id=identity,partition=plan['partition'],date=plan['date'],scenario=plan['scenario'],physical_state_group=identity,step=c['step'],visible_count=n,
                    teacher_complete=s['summary']['complete'],visited_by_replayed_guard_global=n in manual['visited'],original_guard_stop=n==ng,source=rel))
                y=dict(sample_id=sid,next_queries=min(64,m-n),remaining_queries=m-n,next_batch_improves=improves(before,nextbest),any_remaining_improves=improves(before,full),
                    next_batch_repairs_missing=before is None and nextbest is not None,remaining_repairs_missing=before is None and full is not None,no_current_incumbent=before is None,
                    next_batch_reward_gain=gain(before,nextbest),remaining_reward_gain=gain(before,full),full_choice=bid(full),current_choice=bid(before),next_choice=bid(nextbest))
                assert y['next_batch_reward_gain'] is None or y['next_batch_reward_gain']>=0
                assert y['remaining_reward_gain'] is None or y['remaining_reward_gain']>=0
                assert not set(x['features'])&set(y);inputs.append(dict(sample_id=sid,**x));targets.append(y)
        assert totalq==s['summary']['n1_forecasts'];checks['scenarios']+=1
    assert len(calls)==305 and checks['admissibility']==268356
    for name,rows in [('public_inputs.jsonl.gz',inputs),('targets.jsonl.gz',targets),('metadata.jsonl.gz',metas)]:
        with gzip.open(O/name,'wt',encoding='utf8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    save('call_ledger.json',calls);save('baseline_call_metrics.json',methods);summaries=[];scenario_rows=[]
    for partition in ['train','validation']:
        for method in baselines:
            rr=[r for r in methods if r['partition']==partition and r['method']==method];date_rows=[]
            for scenario in sorted({r['scenario'] for r in rr}):
                ss=[r for r in rr if r['scenario']==scenario];reg=[r['reward_regret'] for r in ss if r['reward_regret'] is not None]
                scenario_rows.append(dict(partition=partition,method=method,date=ss[0]['date'],scenario=scenario,calls=len(ss),queries=sum(r['queries'] for r in ss),full_queries=sum(r['full_queries'] for r in ss),
                    query_reduction=1-sum(r['queries'] for r in ss)/sum(r['full_queries'] for r in ss),full_action_match=sum(r['exact_full_action'] for r in ss)/len(ss),lost_admissible=sum(r['lost_admissible'] for r in ss),mean_comparable_regret=statistics.mean(reg) if reg else None))
            sr=[r for r in scenario_rows if r['partition']==partition and r['method']==method]
            for date in sorted({r['date'] for r in sr}):
                dd=[r for r in sr if r['date']==date];date_rows.append(dict(date=date,query_reduction=statistics.mean(r['query_reduction'] for r in dd),full_action_match=statistics.mean(r['full_action_match'] for r in dd)))
            reg=[r['reward_regret'] for r in rr if r['reward_regret'] is not None]
            summaries.append(dict(partition=partition,method=method,calls=len(rr),scenarios=len(sr),dates=len(date_rows),queries=sum(r['queries'] for r in rr),exact_full_actions=sum(r['exact_full_action'] for r in rr),
                lost_admissible=sum(r['lost_admissible'] for r in rr),regret_defined=len(reg),mean_comparable_regret=statistics.mean(reg) if reg else None,
                date_macro_query_reduction=statistics.mean(r['query_reduction'] for r in date_rows),date_macro_full_action_match=statistics.mean(r['full_action_match'] for r in date_rows),by_date=date_rows))
    save('baseline_scenario_metrics.json',scenario_rows);save('baseline_summary.json',summaries)
    combined=[dict(meta=m,x=x['features'],y=y) for m,x,y in zip(metas,inputs,targets)];situations=[];strata=[]
    for partition in ['train','validation']:
        rr=[r for r in combined if r['meta']['partition']==partition]
        scopes=[('all_prefixes',rr),('guard_visited',[r for r in rr if r['meta']['visited_by_replayed_guard_global']]),('guard_stops',[r for r in rr if r['meta']['original_guard_stop']]),('two_stale_after256',[r for r in rr if r['x']['stale_batches']>=2 and r['x']['visible_count']>=256])]
        for scope,ss in scopes:
            gg=[r['y']['remaining_reward_gain'] for r in ss if r['y']['remaining_reward_gain'] is not None]
            situations.append(dict(partition=partition,scope=scope,prefixes=len(ss),calls=len({r['meta']['call_id'] for r in ss}),dates=len({r['meta']['date'] for r in ss}),no_incumbent=sum(r['y']['no_current_incumbent'] for r in ss),
                remaining_repairs=sum(r['y']['remaining_repairs_missing'] for r in ss),next_repairs=sum(r['y']['next_batch_repairs_missing'] for r in ss),remaining_improvement=sum(r['y']['any_remaining_improves'] for r in ss),
                next_improvement=sum(r['y']['next_batch_improves'] for r in ss),delayed_improvement=sum(r['y']['any_remaining_improves'] and not r['y']['next_batch_improves'] for r in ss),mean_known_remaining_gain=statistics.mean(gg) if gg else None))
        buckets=collections.defaultdict(list)
        for r in rr:buckets[(r['x']['prefix_safe'],r['x']['incumbent_available'],False if r['x']['incumbent_rho'] is None else r['x']['incumbent_rho']<.9,r['x']['stale_batches']>=2)].append(r)
        for k,ss in sorted(buckets.items()):strata.append(dict(partition=partition,prefix_safe=k[0],incumbent_available=k[1],current_safe=k[2],two_stale=k[3],prefixes=len(ss),calls=len({r['meta']['call_id'] for r in ss}),
            remaining_improves=sum(r['y']['any_remaining_improves'] for r in ss),no_remaining_improvement=sum(not r['y']['any_remaining_improves'] for r in ss),next_improves=sum(r['y']['next_batch_improves'] for r in ss)))
    save('decision_situations.json',situations);save('all_public_feedback_strata.json',strata)
    save('all_guard_stop_remaining_improvements.json',[dict(call_id=r['meta']['call_id'],partition=r['meta']['partition'],date=r['meta']['date'],visible_count=r['meta']['visible_count'],public=r['x'],targets=r['y']) for r in combined if r['meta']['original_guard_stop'] and r['y']['any_remaining_improves']])
    save('review.json',dict(passed=True,checks=dict(checks),full_calls=305,train_calls=sum(c['partition']=='train' for c in calls),validation_calls=sum(c['partition']=='validation' for c in calls),global_prefixes=len(metas),
        data_partitions={p:dict(prefixes=sum(m['partition']==p for m in metas),dates=len({m['date'] for m in metas if m['partition']==p}),scenarios=len({m['scenario'] for m in metas if m['partition']==p})) for p in ['train','validation']},
        independent_official_policy_matches=True,no_incumbent_not_zero_gain=True,public_input_target_separation=True,source_hashes_unchanged=True,new_fits=0,new_model_forwards=0,new_physical=0,final_tests_absent=True))
    verify();save('complete.json',dict(passed=True,analysis_only=True,new_fits=0,new_physical=0,ready_to_scale=False,adopted=False));save('state.json',dict(stage='analysis_complete'));archive('complete')
    print(json.dumps(dict(summaries=[{k:v for k,v in r.items() if k!='by_date'} for r in summaries],situations=situations,review=read(O/'review.json')),ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
