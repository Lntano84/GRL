"""One-sided learned stop correction. Algebraic development check, zero fits."""
from pathlib import Path
import collections,hashlib,importlib.util,json,pickle,shutil,sys,math
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid167_one_sided_stop';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID167_one_sided_stop'
sp=importlib.util.spec_from_file_location('g167_majority',R/'work/grid166_fixed_vote/run.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P);G=P.G;H=P.H;read=P.read;sha=P.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
 P.verify()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
 assert not O.exists();O.mkdir(parents=True);P.verify();assert read(P.O/'complete.json')['passed_audit'];known=read(P.O/'direction_decomposition.json');base=read(P.O/'family_summary.json')['FACTOR']['mean_net_utility']
 save('design.json',dict(id='GRID167_ONE_SIDED_STOP',authorization='User continue amplifying local improvements;single direction-constrained development check,no fit.',
  rule='go_new=go_originalFACTOR AND(go_routedFACTOR OR go_smallCOST42). All old decisions strictq1>q0. Only allow earlier stop;never turn originalSTOP intoCONTINUE. No thresholds/weights/alternate gate.',
  proof='Same original candidate order and positive per-batch query counts:shared-prefix induction implies new stopping position<=originalFACTOR,so queries per identical saved-state search cannot increase. Not a physical wholeweek guarantee;reward/action can change.',
  provenance='Chosen fromGRID166 exposed direction decomposition. Saved-state mean effect is already inferable;must not present as blind prediction/independent confirmation.',
  known_expected_validation_net_delta=known['earlier_stop_contribution'],known_expected_validation_utility=base+known['earlier_stop_contribution'],
  evaluation='All3seeds/all2378train590validationstarts,original16/4localNETtask. Verify monotone query bound andactual equality to prior earlier-stop cases. Report alllostgain/newstopcases andphysical-groupbalancedreadout.',
  forbidden=['newfits/forwards','gateorweightthresholdscan','futurelabels in decisions','newphysics/data/test8','paper','automaticadoptionorscale']))
 paths=[P.O/n for n in ['fixed_vote_actions.npz','summary.json','family_summary.json','direction_decomposition.json','train_trajectories.json','validation_trajectories.json','complete.json']]+[R/'outputs/grid116_matched_module/state_predictions.npz',G.O/'state_scores.npz',H.O/'state_scores.npz',R/'outputs/grid102_search_rl/cases.pkl']+[R/'outputs/grid113_relational_learning'/n for n in ['train_metadata.json','validation_metadata.json']]+[O/'design.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();save('state.json',dict(stage='frozen_exploratory'));archive('frozen');print('GRID167_FROZEN_EXPECTED',known['earlier_stop_contribution'])
def analyse():
 verify();assert not (O/'review.json').exists();orig=np.load(R/'outputs/grid116_matched_module/state_predictions.npz');route=np.load(G.O/'state_scores.npz');small=np.load(H.O/'state_scores.npz');major=np.load(P.O/'fixed_vote_actions.npz');meta={p:read(R/f'outputs/grid113_relational_learning/{p}_metadata.json') for p in ['train','validation']};ix={p:{(r['sample_id'],r['position']):i for i,r in enumerate(rows)} for p,rows in meta.items()};cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());task=read(G.O/'design.json');fee=task['cost_per_query'];scale=task['gain_scale'];go={};checks=collections.Counter()
 for seed in [0,1,2]:
  for part in meta:
   a=orig[f'BASE_FACTOR_seed{seed}__{part}'];b=route[f'COST_MAIN_seed{seed}__{part}'];c=small[f'COST42_seed{seed}__{part}'];base=a[:,1]>a[:,0];propose=b[:,1]>b[:,0];helper=c[:,1]>c[:,0];decision=base&(propose|helper);assert np.all(decision<=base);assert np.array_equal(decision,base&major[f'MAJORITY_seed{seed}__{part}']);go[f'ONE_SIDED_seed{seed}__{part}']=decision;checks['state_no_reopening']+=len(decision)
 np.savez_compressed(O/'fixed_actions.npz',**go);records={p:[] for p in meta};summary={}
 for part in meta:
  oldrows=read(P.O/f'{part}_trajectories.json');oldmap={(r['method'],r['sample_id']):r for r in oldrows}
  for seed in [0,1,2]:
   name=f'ONE_SIDED_seed{seed}';rr=[]
   for case in cases:
    if case['partition']!=part:continue
    ids=[ix[part][case['sample_id'],j] for j in range(len(case['states']))];r=H.trajectory(case,go[name+'__'+part][ids],name,fee,scale);base=oldmap[f'ZERO_MAIN_seed{seed}',case['sample_id']];vote=oldmap[f'MAJORITY_seed{seed}',case['sample_id']];assert r['batches']<=base['batches'] and r['queries']<=base['queries'];assert r['gain']<=base['gain']+1e-12;checks['search_no_query_increase']+=1
    expected=vote if vote['batches']<base['batches'] else base
    for k in ['batches','queries','gain','chosen_pool_id','actions']:assert r[k]==expected[k]
    checks['matches_prior_direction_case']+=1;rr.append(r)
   records[part]+=rr;summary[name+'__'+part]=H.summarize(rr)
  for method in [*[f'ZERO_MAIN_seed{s}' for s in [0,1,2]],*[f'MAJORITY_seed{s}' for s in [0,1,2]],*[f'COST_MAIN_seed{s}' for s in [0,1,2]],*[f'COST42_seed{s}' for s in [0,1,2]],*[f'ZERO42_seed{s}' for s in [0,1,2]]]:
   rr=[r for (m,sid),r in oldmap.items() if m==method];assert len(rr)==(2378 if part=='train' else 590);summary[method+'__'+part]=H.summarize(rr);records[part]+=rr
  save(part+'_trajectories.json',records[part])
 family={}
 for key,fmt in [('ONE_SIDED','ONE_SIDED_seed{}'),('FACTOR','ZERO_MAIN_seed{}'),('MAJORITY','MAJORITY_seed{}'),('ROUTED','COST_MAIN_seed{}'),('COST42','COST42_seed{}'),('BASE','ZERO42_seed{}')]:family[key]={k:float(np.mean([summary[fmt.format(s)+'__validation']['macro'][k] for s in [0,1,2]])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']}
 v={(r['method'],r['sample_id']):r for r in records['validation']};pairs=[];events=[];counts=collections.Counter(c['date'] for c in cases if c['partition']=='validation')
 for seed in [0,1,2]:
  a=summary[f'ONE_SIDED_seed{seed}__validation'];b=summary[f'ZERO_MAIN_seed{seed}__validation'];pairs.append(dict(seed=seed,utility=a['macro']['mean_net_utility'],factor_utility=b['macro']['mean_net_utility'],net_delta=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],relative_change=(a['macro']['mean_net_utility']/b['macro']['mean_net_utility']-1),per_date=[dict(date=x['date'],net_delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
  for case in cases:
   if case['partition']!='validation':continue
   aa=v[f'ONE_SIDED_seed{seed}',case['sample_id']];bb=v[f'ZERO_MAIN_seed{seed}',case['sample_id']];dg=aa['gain']-bb['gain'];qs=bb['queries']-aa['queries'];nd=aa['net_utility']-bb['net_utility'];assert qs>=0 and dg<=1e-12;cat='UNCHANGED' if qs==0 else ('SAME_GAIN_LESS_QUERY' if dg==0 else ('SMALL_GAIN_LOSS_NET_POSITIVE' if nd>0 else 'LOST_GAIN_TOO_MUCH'));events.append(dict(seed=seed,sample_id=case['sample_id'],date=case['date'],physical_state_group=case['physical_state_group'],category=cat,saved_queries=qs,gain_delta=dg,net_delta=nd,same_choice=aa['chosen_pool_id']==bb['chosen_pool_id'],weight=1/(3*4*counts[case['date']])))
 eventsummary=[]
 for cat in sorted({r['category'] for r in events}):
  rr=[r for r in events if r['category']==cat];eventsummary.append(dict(category=cat,pairs=len(rr),starts=len({r['sample_id'] for r in rr}),physical_groups=len({r['physical_state_group'] for r in rr}),dates=sorted({r['date'] for r in rr}),same_choice_pairs=sum(r['same_choice'] for r in rr),net_contribution=math.fsum(r['weight']*r['net_delta'] for r in rr)))
 nd=family['ONE_SIDED']['mean_net_utility']-family['FACTOR']['mean_net_utility'];known=read(O/'design.json');assert abs(nd-known['known_expected_validation_net_delta'])<1e-12;assert abs(family['ONE_SIDED']['mean_net_utility']-known['known_expected_validation_utility'])<1e-12;assert abs(nd-sum(r['net_contribution'] for r in eventsummary))<1e-12;dates=[dict(date=pairs[0]['per_date'][i]['date'],mean_delta=float(np.mean([r['per_date'][i]['net_delta'] for r in pairs]))) for i in range(4)];save('summary.json',summary);save('family_summary.json',family);save('paired_summary.json',dict(pairs=pairs,net_delta=nd,relative_change_vs_factor=nd/family['FACTOR']['mean_net_utility'],relative_change_vs_base=family['ONE_SIDED']['mean_net_utility']/family['BASE']['mean_net_utility']-1,positive_seeds=sum(r['net_delta']>0 for r in pairs),per_date=dates,known_effect_not_new_confirmation=True));save('all1770_paired_events.json',events);save('paired_event_summary.json',eventsummary);verify();save('review.json',dict(passed=True,checks=dict(checks),known_arithmetic_effect_reproduced=True,new_fits=0,new_model_forwards=0,no_new_data=True,new_physics=0,final8_absent=True,quality_not_guaranteed=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(family=family,paired=read(O/'paired_summary.json'),events=eventsummary),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
