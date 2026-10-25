"""One exploratory fixed majority vote of already frozen local policies, zero fits."""
from pathlib import Path
import collections,hashlib,importlib.util,json,pickle,shutil,sys,math
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid166_fixed_vote';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID166_fixed_vote'
sp=importlib.util.spec_from_file_location('g166_models',R/'work/grid165_factor_cost_routing/run.py');G=importlib.util.module_from_spec(sp);sp.loader.exec_module(G);H=G.H
read=G.read;sha=G.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
 G.verify_trained()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
 assert not O.exists();O.mkdir(parents=True);G.verify_trained();assert read(G.O/'complete.json')['passed_audit']
 save('design.json',dict(id='GRID166_ONE_FIXED_MAJORITY',authorization='User continue pursuing local positive improvements;no new fitting.',
  hypothesis='One fixed vote may retain some routedFACTOR useful stops while preserving valuable continuation recognized by existing smallCOST42.',
  rule='At each actual local query state,strictq1>q0 yields each frozen decision. Take majority of originalFACTOR,routedCOST_MAIN,COST42 same seed. No averaging,weights,threshold/calibration,seedchoice or alternate rule.',
  equivalence='If routed policy differs from originalFACTOR,change original decision only ifCOST42 agrees with routed policy. Otherwise keep original. This is ordinary3waymajority,notnew architecture/safety guarantee.',
  provenance='Chosen after exposed validation contrast inGRID165. Explicitly posthoc development exploration,not independent confirmation. All original train/validation states andstarts retained.',
  population='3seeds,old16train4validationdates,2378/590starts,3582/856states,originalNET objective andfee.',
  outcome='Allseeds/dates utility,gain,queries plusall1770 paired changes against originalFACTOR. No all-seedwin gate for exploratory merit;no adoption.',
  prohibited=['newfits/forwards','ensemblegrid','futurelabels in votes','newdata/physics/test8','paper','automatic scale']))
 paths=[G.O/n for n in ['state_scores.npz','summary.json','validation_trajectories.json','train_trajectories.json','existing_expert_summary.json','complete.json']]+[H.O/n for n in ['state_scores.npz','summary.json','validation_trajectories.json','train_trajectories.json','complete.json']]+[R/'outputs/grid113_relational_learning'/n for n in ['train_metadata.json','validation_metadata.json']]+[R/'outputs/grid116_matched_module/state_predictions.npz',R/'outputs/grid102_search_rl/cases.pkl',O/'design.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();save('state.json',dict(stage='frozen_exploratory'));archive('frozen');print('GRID166_FROZEN')
def analyse():
 verify();assert not (O/'review.json').exists();old=np.load(R/'outputs/grid116_matched_module/state_predictions.npz');routed=np.load(G.O/'state_scores.npz');small=np.load(H.O/'state_scores.npz');plan=read(G.O/'design.json');fee,scale=plan['cost_per_query'],plan['gain_scale'];meta={p:read(R/f'outputs/grid113_relational_learning/{p}_metadata.json') for p in ['train','validation']};ix={p:{(r['sample_id'],r['position']):i for i,r in enumerate(rows)} for p,rows in meta.items()};cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());actions={};checks=collections.Counter()
 # Votes formed solely from saved predictions. Targets or future gains are not accessed here.
 for seed in [0,1,2]:
  for part in meta:
   base=old[f'BASE_FACTOR_seed{seed}__{part}'];candidate=routed[f'COST_MAIN_seed{seed}__{part}'];helper=small[f'COST42_seed{seed}__{part}'];a=base[:,1]>base[:,0];b=candidate[:,1]>candidate[:,0];c=helper[:,1]>helper[:,0];go=(a.astype(int)+b.astype(int)+c.astype(int))>=2;assert np.array_equal(go,np.where((a!=b)&(b==c),b,a));actions[f'MAJORITY_seed{seed}__{part}']=go;checks['vote_states']+=len(go)
 np.savez_compressed(O/'fixed_vote_actions.npz',**actions);summary={};tr={p:[] for p in meta}
 for seed in [0,1,2]:
  for part in meta:
   name=f'MAJORITY_seed{seed}';go=actions[name+'__'+part];rr=[]
   for case in cases:
    if case['partition']!=part:continue
    ii=[ix[part][case['sample_id'],j] for j in range(len(case['states']))];rr.append(H.trajectory(case,go[ii],name,fee,scale))
   summary[name+'__'+part]=H.summarize(rr);tr[part]+=rr
 # Prespecified saved references only;no alternate ensemble is evaluated.
 for part in meta:
  raw=read(G.O/f'{part}_trajectories.json')+read(H.O/f'{part}_trajectories.json');by={(r['method'],r['sample_id']):r for r in raw};names=[*[f'ZERO_MAIN_seed{s}' for s in range(3)],*[f'COST_MAIN_seed{s}' for s in range(3)],*[f'COST42_seed{s}' for s in range(3)],*[f'ZERO42_seed{s}' for s in range(3)]]
  for name in names:
   rr=[r for (method,sid),r in by.items() if method==name];assert len(rr)==(2378 if part=='train' else 590);summary[name+'__'+part]=H.summarize(rr);tr[part]+=rr
 for part in tr:save(part+'_trajectories.json',tr[part])
 save('summary.json',summary);family={}
 for mode,fmt in [('MAJORITY','MAJORITY_seed{}'),('FACTOR','ZERO_MAIN_seed{}'),('ROUTED','COST_MAIN_seed{}'),('COST42','COST42_seed{}'),('BASE','ZERO42_seed{}')]:family[mode]={k:float(np.mean([summary[fmt.format(s)+'__validation']['macro'][k] for s in [0,1,2]])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']}
 pairs=[];events=[];v={(r['method'],r['sample_id']):r for r in tr['validation']};datecounts=collections.Counter(c['date'] for c in cases if c['partition']=='validation')
 for seed in [0,1,2]:
  aa=summary[f'MAJORITY_seed{seed}__validation'];bb=summary[f'ZERO_MAIN_seed{seed}__validation'];pairs.append(dict(seed=seed,majority_utility=aa['macro']['mean_net_utility'],factor_utility=bb['macro']['mean_net_utility'],net_delta=aa['macro']['mean_net_utility']-bb['macro']['mean_net_utility'],per_date=[dict(date=x['date'],net_delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(aa['per_date'],bb['per_date'])]))
  for case in cases:
   if case['partition']!='validation':continue
   a=v[f'MAJORITY_seed{seed}',case['sample_id']];b=v[f'ZERO_MAIN_seed{seed}',case['sample_id']];gain=a['gain']-b['gain'];savedq=b['queries']-a['queries'];net=a['net_utility']-b['net_utility'];assert abs(net-(gain+fee*savedq)/scale)<1e-12
   if gain==0 and savedq==0:cat='UNCHANGED'
   elif savedq>0 and gain>=0:cat='LESS_QUERY_NO_GAIN_LOSS'
   elif savedq>0 and net>0:cat='LESS_QUERY_SMALL_GAIN_SACRIFICE_NET_POSITIVE'
   elif savedq<0 and net>0:cat='MORE_QUERY_GAIN_PAYS'
   elif savedq>0:cat='LESS_QUERY_GAIN_LOSS_TOO_LARGE'
   elif savedq<0:cat='MORE_QUERY_COST_NOT_REPAID'
   else:cat='SAME_QUERY_DIFFERENT_GAIN'
   events.append(dict(seed=seed,sample_id=case['sample_id'],date=case['date'],physical_state_group=case['physical_state_group'],category=cat,net_delta=net,gain_contribution=gain/scale,query_contribution=fee*savedq/scale,weight=1/(3*4*datecounts[case['date']]),same_action=a['chosen_pool_id']==b['chosen_pool_id']))
 eventsummary=[]
 for cat in sorted({r['category'] for r in events}):
  rr=[r for r in events if r['category']==cat];eventsummary.append(dict(category=cat,seed_start_pairs=len(rr),starts=len({r['sample_id'] for r in rr}),physical_groups=len({r['physical_state_group'] for r in rr}),dates=sorted({r['date'] for r in rr}),net_contribution=math.fsum(r['weight']*r['net_delta'] for r in rr),gain_contribution=math.fsum(r['weight']*r['gain_contribution'] for r in rr),query_contribution=math.fsum(r['weight']*r['query_contribution'] for r in rr),same_action_pairs=sum(r['same_action'] for r in rr)))
 delta=family['MAJORITY']['mean_net_utility']-family['FACTOR']['mean_net_utility'];assert abs(delta-sum(r['net_contribution'] for r in eventsummary))<1e-12;dates=[dict(date=pairs[0]['per_date'][i]['date'],mean_delta=float(np.mean([x['per_date'][i]['net_delta'] for x in pairs]))) for i in range(4)];save('family_summary.json',family);save('paired_summary.json',dict(pairs=pairs,net_delta=delta,relative_change_vs_factor=delta/family['FACTOR']['mean_net_utility'],positive_seeds=sum(x['net_delta']>0 for x in pairs),per_date=dates));save('all1770_paired_events.json',events);save('paired_event_summary.json',eventsummary);verify();save('review.json',dict(passed=True,checks=dict(checks),new_fits=0,new_model_forwards=0,one_rule_only=True,future_data_not_used_in_votes=True,all_seeds_dates=True,all1770_payoffs_reconciled=True,new_physics=0,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(family=family,paired=read(O/'paired_summary.json'),events=eventsummary),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
