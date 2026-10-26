"""Describe paired GRID187 successes/failures using existing records only."""
from pathlib import Path
import collections,gzip,hashlib,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid188_repeated_cases';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID188_repeated_cases';P=R/'outputs/grid187_action_invariance_target'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):
    (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else (_ for _ in ()).throw(TypeError(type(v).__name__)))+'\n',encoding='utf8',newline='\n')
def sealed():
    d=read(R/'outputs/grid100_data_foundation/design.json')
    for v in d['scenario_plan']:
        if v['partition']=='test':
            assert not (R/d['environment']/'chronics'/v['scenario']).exists()
            assert not (R/'outputs/grid100_data_foundation/datasets'/v['scenario']).exists()
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    sealed()
def archive(phase):
    dst=A/phase;dst.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dst/p.name)
    (dst/'archive_manifest.json').write_text(json.dumps({p.relative_to(dst).as_posix():sha(p) for p in dst.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed();assert read(P/'complete.json')['passed']
    save('design.json',dict(id='GRID188_REPEATED_CASES',authorization='User continues:继续吧. Saved-record diagnosis only.',posthoc=True,
        population='All16oldtrainingdates/3seeds retained. Focus on originalGRID187 all3seed restored24starts/8groups and lost28starts/17groups, with overlapping groups explicitly checked; not independent newvalidation.',
        questions=['Where is first actual policy divergence: local entry or after purchased batches?','Does final full-local action appear in next64 explicitly encoded plans or only later?','Do missed reward increments occur in next batch and what is oldNET sign?','How concentrated are date-balanced reward changes by physical group?','Do observed feedback/depth and fold-fit support differ, descriptively?'],
        analysis='Use original saved scores/trajectories and raw public rows. First divergence=min(old/new purchased batches); native decisions checked from OOF scores. Future gains/IDs strictly outcome-only. Report all41observed summaries, no predictor/threshold/search/featureselection. Collapse repeated seed states; balance summary by physicalgroup; report overlapping groups.',
        controls='No model forwards, optimizer, weight/fee/threshold changes, newphysics, final8, new ensemble or publication changes to old frozen sources.',
        limits='Outcome-selected developer subsets; first-divergence descriptors are not validated triggers; input-horizon limitation is only an explanation if candidate truly beyond next64. No full-week cost or time claim.'))
    files=[W/'run.py',O/'design.json',P/'interpretation.json',P/'oof_trajectories.json',P/'oof_scores.npz',P/'paired_summary.json',P/'family_summary.json',P/'complete.json',P/'delivery_receipt.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid113_relational_learning/train_public.pkl',R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz',R/'outputs/grid28/offline.json',R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json']
    for f in range(4):files.append(R/('outputs/grid168_one_sided_transfer/fold_arrays.npz' if f==0 else f'outputs/grid170_remaining_date_folds/fold{f}/fold_arrays.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('frozen');print('GRID188_FROZEN')

def describe(rows,names):
    out={}
    for k in names:
        groups=collections.defaultdict(list)
        for v in rows:groups[v['physical_state_group']].append(float(v['public'][k]))
        vals=np.array([np.mean(a) for a in groups.values()])
        out[k]=dict(physical_group_mean=float(vals.mean()),physical_group_median=float(np.median(vals)),min_group_mean=float(vals.min()),max_group_mean=float(vals.max()))
    return out

def analyze():
    verify();assert not (O/'analysis.json').exists()
    inter=read(P/'interpretation.json');tr=read(P/'oof_trajectories.json');by={}
    for v in tr:by.setdefault(v['method'],{})[v['sample_id']]=v
    cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'}
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};scores=np.load(P/'oof_scores.npz')
    names=read(R/'outputs/grid102_search_rl/design.json')['input']['features'];assert len(names)==41
    public={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            v=json.loads(line)
            if v['partition']=='train':public[v['sample_id'],v['position']]=v
    assert len(public)==3582
    tensor_rows=pickle.loads((R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes());subids=read(R/'outputs/grid28/offline.json')['public_subids']
    folds=[];heldfold={}
    for f in range(4):
        p=R/('outputs/grid168_one_sided_transfer/fold_arrays.npz' if f==0 else f'outputs/grid170_remaining_date_folds/fold{f}/fold_arrays.npz');z=np.load(p);fi=z['fit_ids'];hi=z['held_ids'];raw=np.stack([cases[meta[i]['sample_id']]['states'][meta[i]['position']] for i in fi]);ds={meta[i]['date'] for i in hi};assert not ds & {meta[i]['date'] for i in fi}
        for d in ds:assert d not in heldfold;heldfold[d]=f
        # Canonical plan content, excluding relative-to-incumbent slots; IDs are not model input.
        plan_set={row[np.r_[0:17,34:68]].tobytes() for i in fi for row in tensor_rows[i]['next_action']}
        folds.append(dict(lo=raw.min(0),hi=raw.max(0),plan_set=plan_set))
    focus={v['sample_id']:'RESTORED' for v in inter['all3seed_restored_starts']};focus.update({v['sample_id']:'LOST' for v in inter['all3seed_lost_starts']});assert len(focus)==52
    events=[];all_actions=[];full_groups=collections.defaultdict(lambda:dict(query=0.,gain=0.,restore_pairs=0,lost_pairs=0,date=None))
    dates=sorted({c['date'] for c in cases.values()});nc=collections.Counter(c['date'] for c in cases.values())
    for seed in range(3):
        for sid,c in cases.items():
            old=by[f'NET_seed{seed}'][sid];new=by[f'CHANGE_seed{seed}'][sid];weight=1/(3*16*nc[c['date']]);gg=full_groups[c['physical_state_group']];gg['date']=c['date'];gg['query']+=weight*(new['queries']-old['queries']);gg['gain']+=weight*(new['gain']-old['gain'])
            oc=bool(old['action_matches_full_local']);nn=bool(new['action_matches_full_local'])
            if nn and not oc:gg['restore_pairs']+=1
            if oc and not nn:gg['lost_pairs']+=1
            if sid not in focus:continue
            category=focus[sid];j=min(old['batches'],new['batches']);assert old['batches']!=new['batches'] and j<len(c['states'])
            idx=lookup[sid,j];pub=public[sid,j]['public_input'];raw=c['states'][j];q_old=scores[f'NET_seed{seed}'][idx];q_new=scores[f'CHANGE_seed{seed}'][idx]
            old_go=bool(q_old[1]>q_old[0]);new_go=bool(q_new[1]>q_new[0]);assert old_go!=new_go
            assert (category=='RESTORED')==(new_go and not old_go)
            assert c['chosen'][j]==pub['incumbent_id'];assert len(pub['next_ids'])==c['queries'][j]
            final=int(c['chosen'][-1]);k=next(t for t in range(j+1,len(c['chosen'])) if c['chosen'][t]==final)
            in_next=final in pub['next_ids'];assert in_next==(k==j+1)
            future_gain=float(c['gains'][j:].sum());next_gain=float(c['gains'][j]);first_improve=next((t-j+1 for t in range(j,len(c['gains'])) if c['gains'][t]>0),None)
            f=heldfold[c['date']];support=folds[f];tol=1e-6*(1+np.maximum(abs(support['lo']),abs(support['hi'])));outside=(raw<support['lo']-tol)|(raw>support['hi']+tol)
            row=tensor_rows[idx];seen_patterns=[a[np.r_[0:17,34:68]].tobytes() in support['plan_set'] for a in row['next_action']]
            v=dict(sample_id=sid,seed=seed,category=category,physical_state_group=c['physical_state_group'],date=c['date'],fold=f,position=j,global_visible_count=None,
                public={**{n:float(v) for n,v in zip(names,raw)},'station_id':int(subids[pub['incumbent_id']]),'seen_station_count':len(pub['seen_ids']),'next_queries':len(pub['next_ids']),'remaining_queries':int(c['queries'][j:].sum()),'next_plan_in_fit_fraction':float(np.mean(seen_patterns)),'outside_fit_feature_count':int(outside.sum())},
                model_decision=dict(old_continue=old_go,new_continue=new_go,old_margin=float(q_old[1]-q_old[0]),new_margin=float(q_new[1]-q_new[0])),
                outcome_only=dict(final_chosen=final,final_in_next64=in_next,batches_until_final=k-j,batches_until_first_gain=first_improve,next_gain=next_gain,suffix_gain=future_gain,next_NET=(next_gain-read(P/'design.json')['cost_per_query']*len(pub['next_ids']))/read(P/'design.json')['gain_scale'],old_final_gain=old['gain'],new_final_gain=new['gain']),
                outside_fit_features=[n for n,b in zip(names,outside) if b])
            events.append(v)
    assert len(events)==156
    # Collapse the same prefix reached by multiple seeds; retain seed decisions separately.
    unique={}
    for v in events:
        key=v['sample_id'],v['position']
        if key in unique:
            assert unique[key]['public']==v['public'] and unique[key]['outcome_only']==v['outcome_only'];unique[key]['seeds'].append(v['seed'])
        else:unique[key]={k:x for k,x in v.items() if k not in ['seed','model_decision']};unique[key]['seeds']=[v['seed']]
    rows=list(unique.values());summaries={}
    for category in ['RESTORED','LOST']:
        rr=[v for v in rows if v['category']==category];gs=sorted({v['physical_state_group'] for v in rr});sids=sorted({v['sample_id'] for v in rr})
        summaries[category]=dict(starts=len(sids),unique_divergence_states=len(rr),physical_groups=len(gs),dates=sorted({v['date'] for v in rr}),
            divergence_position=dict(collections.Counter(str(v['position']) for v in rr)),final_in_next64=sum(v['outcome_only']['final_in_next64'] for v in rr),
            first_gain_next64=sum(v['outcome_only']['batches_until_first_gain']==1 for v in rr),next_NET_positive=sum(v['outcome_only']['next_NET']>0 for v in rr),
            suffix_gain_group_mean=float(np.mean([np.mean([v['outcome_only']['suffix_gain'] for v in rr if v['physical_state_group']==g]) for g in gs])),
            station_counts=dict(collections.Counter(str(v['public']['station_id']) for v in rr)),all41_observed_group_balanced=describe(rr,names),
            supplemental_group_balanced=describe(rr,['remaining_queries','seen_station_count','next_plan_in_fit_fraction','outside_fit_feature_count']),
            outside_fit_feature_counts=dict(collections.Counter(n for v in rr for n in v['outside_fit_features'])),
            physical_group_details=[dict(group=g,starts=len({v['sample_id'] for v in rr if v['physical_state_group']==g}),unique_states=sum(v['physical_state_group']==g for v in rr),entry_states=sum(v['physical_state_group']==g and v['position']==0 for v in rr),next64_final_states=sum(v['physical_state_group']==g and v['outcome_only']['final_in_next64'] for v in rr),suffix_gain_mean=float(np.mean([v['outcome_only']['suffix_gain'] for v in rr if v['physical_state_group']==g]))) for g in gs])
    overlap=sorted(set(v['physical_state_group'] for v in rows if v['category']=='RESTORED')&set(v['physical_state_group'] for v in rows if v['category']=='LOST'))
    assert abs(sum(v['gain'] for v in full_groups.values())-(read(P/'family_summary.json')['CHANGE']['search_macro']['mean_gain']-read(P/'family_summary.json')['NET']['search_macro']['mean_gain']))<1e-12
    loss=sorted([dict(group=k,**v) for k,v in full_groups.items() if v['gain']<0],key=lambda x:x['gain']);total_negative=-sum(v['gain'] for v in loss)
    concentration=dict(total_negative_date_balanced_gain=total_negative,top1_fraction=-loss[0]['gain']/total_negative,top3_fraction=-sum(v['gain'] for v in loss[:3])/total_negative,all_groups=sorted([dict(group=k,**v) for k,v in full_groups.items()],key=lambda x:x['gain']))
    save('events_all_seeds.json',events);save('unique_divergence_states.json',rows);save('analysis.json',dict(groups=summaries,overlap_physical_groups=overlap,loss_concentration=concentration,scope='Descriptive outcome-selected subsets; no new predictions or trigger claimed'))
    verify();save('review.json',dict(passed=True,paired_decisions_checked=156,unique_states_checked=len(rows),final_in_next_batch_ID_vs_trajectory_checked=True,public_future_fields_separated=True,all41features_reported=True,all222groups_gain_decomposition=True,new_fits=0,new_forwards=0,new_physics=0,final8_absent=True));archive('analyzed');print('GRID188_ANALYZED',json.dumps(dict(groups={k:{x:y for x,y in v.items() if x not in ['all41_observed_group_balanced','supplemental_group_balanced','physical_group_details']} for k,v in summaries.items()},overlap=overlap,loss_top3=concentration['top3_fraction']),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
