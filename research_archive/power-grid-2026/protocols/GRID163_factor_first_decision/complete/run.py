"""Trace saved FACTOR improvements and decompose frozen main/correction outputs."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,gzip,hashlib,importlib.util,json,math,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid163_factor_first_decision';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID163_factor_first_decision'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def sealed():
 spec=importlib.util.spec_from_file_location('g163_sealed',R/'work/grid139_global_feasibility/run.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.sealed()
def verify():
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
 sealed()
def prepare():
 assert not O.exists();O.mkdir(parents=True);sealed();assert read(R/'outputs/grid162_positive_change_audit/complete.json')['passed_accounting'];assert read(R/'outputs/grid116_matched_module/review.json')['passed']
 save('design.json',dict(id='GRID163_FACTOR_FIRST_DECISION',authorization='User asks continue with modest repeatable improvements;trace FACTOR existing benefits before training.',
  population='All3 frozen FACTOR seeds,all590 old validation starts/856 states,all4 exposed old dates. Main-only diagnostic also all2378 train starts/3582 states. Both keep original16/4 local-task split.',
  trace='For all differing FACTOR/BASE saved validation trajectories,locate first common-prefix decision difference,read saved Q margins,public41 and known next-batch cost;future gain/query arithmetic separated as audit only. Equal-gain lower-query events and costly continuations both retained.',
  decomposition='Frozen FACTOR checkpoint output=co-trained main+correction. Recompute both,verify full output against saved scores and main against independentNumPy. Record whether correction flips first-difference decision. Main-only saved-sequence replay all starts as fixed functional ablation,no new fitting/threshold.',
  limits='Main was co-trained with correction,not separately trained BASE;removing correction cannot isolate training-time causal contribution. All dates exposed,not new confirmation. Old local net utility is not current global FRAME action-identity or weekly quality.',
  exploration='Report gains,repeatability and costs;no all-seed/all-date gate for worth studying. No model adoption from this audit.',
  prohibited=['new optimizer/fit','threshold search','new physics/scenario/data','final8','change fee/labels/order','choosebestseed','paper','automatic scale']))
 paths=[R/'outputs/grid102_search_rl'/n for n in ['cases.pkl','normalization.npz']]
 paths +=[R/'outputs/grid113_relational_learning'/n for n in ['train_public.pkl','validation_public.pkl','train_metadata.json','validation_metadata.json','design.json']]
 paths +=[R/'outputs/grid116_matched_module'/n for n in ['design.json','fits.json','state_predictions.npz','train_trajectories.json','validation_trajectories.json','train_results.json','validation_results.json','review.json']]
 paths +=[R/'outputs/grid116_matched_module/models'/f'BASE_FACTOR_seed{s}.pt' for s in range(3)]
 paths +=[R/'outputs/grid107_direct_value/state_predictions.jsonl.gz',R/'outputs/grid162_positive_change_audit/factor_all1770_paired_sequences.json',R/'outputs/grid162_positive_change_audit/factor_event_summary.json',R/'outputs/grid162_positive_change_audit/factor_equal_gain_action_check.json']
 paths +=[R/'work'/n for n in ['grid115_matched_module_plan/matched_model.py','grid113_relational_learning/model.py','grid113_relational_learning/common.py','grid116_matched_module/common116.py','grid101_gain_learning/features.py','grid102_search_rl/observations.py']];paths.append(O/'design.json');save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen'));verify();archive('frozen');print('GRID163_FROZEN')
def loaders():
 old=R/'outputs/grid113_relational_learning';meta={p:read(old/f'{p}_metadata.json') for p in ['train','validation']};inputs={p:pickle.loads((old/f'{p}_public.pkl').read_bytes()) for p in meta};lookup={p:{(r['sample_id'],r['position']):i for i,r in enumerate(mm)} for p,mm in meta.items()};cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());return meta,inputs,lookup,{c['sample_id']:c for c in cases}
def trace():
 verify();assert not (O/'trace_review.json').exists();meta,inputs,lookup,cases=loaders();saved=np.load(R/'outputs/grid116_matched_module/state_predictions.npz');oldq={}
 with gzip.open(R/'outputs/grid107_direct_value/state_predictions.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   x=json.loads(line)
   if x['partition']=='validation' and x['method'] in ['DIRECT0','DIRECT1','DIRECT2']:oldq[x['method'],x['sample_id'],x['position']]=[x['q0'],x['q1']]
 trajectories=read(R/'outputs/grid116_matched_module/validation_trajectories.json');by={(x['method'],x['sample_id']):x for x in trajectories};plan=read(R/'outputs/grid116_matched_module/design.json');fee=plan['unchanged']['cost_per_query'];scale=plan['unchanged']['gain_scale'];category={(r['seed'],r['sample_id']):r for r in read(R/'outputs/grid162_positive_change_audit/factor_all1770_paired_sequences.json')};names=read(O/'design.json');feature_names=[]
 fs=importlib.util.spec_from_file_location('g163_public_names',R/'work/grid101_gain_learning/features.py');ff=importlib.util.module_from_spec(fs);fs.loader.exec_module(ff);feature_names=ff.NAMES+['local_batches_done','local_budget_used_fraction','log_next_batch_queries','locked_remaining_fraction'];assert len(feature_names)==41
 out=[]
 for (seed,sid),event in category.items():
  a=by[f'BASE_FACTOR_seed{seed}',sid];b=by[f'DIRECT{seed}',sid];c=cases[sid]
  if a['batches']==b['batches']:assert a['actions']==b['actions'];continue
  p=min(a['batches'],b['batches']);assert p<len(c['states']);assert a['actions'][:p]==b['actions'][:p]==[1]*p;assert a['actions'][p]!=b['actions'][p]
  i=lookup['validation'][sid,p];fq=saved[f'BASE_FACTOR_seed{seed}__validation'][i];bq=np.asarray(oldq[f'DIRECT{seed}',sid,p]);assert int(fq[1]>fq[0])==a['actions'][p] and int(bq[1]>bq[0])==b['actions'][p];features={k:float(v) for k,v in zip(feature_names,c['states'][p])};planned=len(inputs['validation'][i]['next_action']);assert planned==int(c['queries'][p])==meta['validation'][i]['next_queries'];qcost=float(inputs['validation'][i]['known_query_cost'][0]);assert abs(qcost-fee*planned/scale)<1e-7
  last_index=a['batches'] if a['batches']>b['batches'] else b['batches'];future_gain=float(c['gains'][p:last_index].sum());future_queries=int(c['queries'][p:last_index].sum());net=(future_gain-fee*future_queries)/scale;assert abs(abs(event['net_delta'])-abs(net))<1e-10
  out.append(dict(seed=seed,sample_id=sid,date=c['date'],scenario=c['scenario'],physical_state_group=c['physical_state_group'],category=event['category'],first_difference_position=p,factor_action=int(a['actions'][p]),base_action=int(b['actions'][p]),factor_margin=float(fq[1]-fq[0]),base_margin=float(bq[1]-bq[0]),factor_q0=float(fq[0]),factor_q1=float(fq[1]),base_q0=float(bq[0]),base_q1=float(bq[1]),known_next_queries=planned,known_normalized_query_cost=qcost,public_features=features,audit_next_batch_gain=float(c['gains'][p]),audit_next_batch_net=float((c['gains'][p]-fee*planned)/scale),audit_extra_path_gain=future_gain,audit_extra_path_queries=future_queries,audit_extra_path_net=net,net_delta=event['net_delta'],macro_weight=event['macro_weight'],future_values_audit_only=True))
 summary=[]
 for cat in sorted({x['category'] for x in out}):
  rr=[x for x in out if x['category']==cat];summary.append(dict(category=cat,events=len(rr),start_prefixes=len({x['sample_id'] for x in rr}),physical_state_groups=len({x['physical_state_group'] for x in rr}),dates=sorted({x['date'] for x in rr}),at_start=sum(x['first_difference_position']==0 for x in rr),after_some_batches=sum(x['first_difference_position']>0 for x in rr),next_gain_zero=sum(x['audit_next_batch_gain']==0 for x in rr),next_net_positive=sum(x['audit_next_batch_net']>0 for x in rr),factor_margin_median=float(np.median([x['factor_margin'] for x in rr])),base_margin_median=float(np.median([x['base_margin'] for x in rr])),known_next_queries_median=float(np.median([x['known_next_queries'] for x in rr]))))
 assert len(out)==603;assert next(x for x in summary if x['category']=='LESS_QUERY_NO_GAIN_LOSS')['events']==260
 # Exact next-batch cost is already recoverable from the old public41 log count.
 norm=np.load(R/'outputs/grid102_search_rl/normalization.npz');costchecks=0
 for part,rows in inputs.items():
  for x in rows:
   raw=float(x['global41'][39])*float(norm['scale'][39])+float(norm['mean'][39]);n=int(round(np.expm1(raw)));assert n==len(x['next_action']);assert abs(float(x['known_query_cost'][0])-fee*n/scale)<1e-7;costchecks+=1
 save('first_difference_records.json',out);save('first_difference_summary.json',summary);save('cost_expression_audit.json',dict(passed=True,records=costchecks,old_global_feature='log_next_batch_queries',index=39,new_explicit_cost='fee*next_queries/gain_scale',query_count_exactly_recovered=True,new_information=False,representation_change_only=True,no_new_fit=True));verify();save('trace_review.json',dict(passed=True,paired_sequences=1770,changed_sequences=603,first_difference_margin_checks=603,query_cost_public_checks=costchecks,new_fits=0,new_physics=0));save('state.json',dict(stage='traced'));print(json.dumps(summary,ensure_ascii=False))
def parts():
 verify();assert read(O/'trace_review.json')['passed'];assert not (O/'parts_review.json').exists();import torch
 sys.path.insert(0,str(R/'work/grid116_matched_module'));import common116 as Old
 Old.configure();meta,inputs,lookup,cases=loaders();saved=np.load(R/'outputs/grid116_matched_module/state_predictions.npz');fits=read(R/'outputs/grid116_matched_module/fits.json');reexecuted={};main={};delta={};maxerr=0.;numpyerr=0.
 for seed in range(3):
  name=f'BASE_FACTOR_seed{seed}';path=R/'outputs/grid116_matched_module/models'/f'{name}.pt';receipt=next(x for x in fits['records'] if x['arm']=='BASE_FACTOR' and x['seed']==seed);assert sha(path)==receipt['checkpoint_sha256'];cp=torch.load(path,weights_only=False,map_location='cpu');assert cp['updates']==7250;net=Old.build(seed,'BASE_FACTOR');net.load_state_dict(cp['state_dict']);net.eval()
  for part,rows in inputs.items():
   total=[];base=[];correction=[]
   with torch.no_grad():
    for j in range(0,len(rows),32):
     pub=Old.collate(rows[j:j+32]);bb=net.base(pub['global41']);dd=net.delta(**pub);total.append((bb+dd).numpy());base.append(bb.numpy());correction.append(dd.numpy())
   qq=np.concatenate(total);bb=np.concatenate(base);dd=np.concatenate(correction);actual=saved[name+'__'+part];e=float(np.max(abs(qq-actual)));maxerr=max(maxerr,e);assert np.allclose(qq,actual,atol=3e-5,rtol=3e-5);assert np.array_equal(qq[:,1]>qq[:,0],actual[:,1]>actual[:,0]);assert np.array_equal(qq,bb+dd)
   xx=np.stack([x['global41'] for x in rows]).astype(float)
   for layer in [0,2,4]:
    ww=cp['state_dict'][f'base.q_net.{layer}.weight'].numpy().astype(float);bias=cp['state_dict'][f'base.q_net.{layer}.bias'].numpy().astype(float);xx=xx@ww.T+bias
    if layer!=4:xx=np.maximum(xx,0)
   ne=float(np.max(abs(xx-bb)));numpyerr=max(numpyerr,ne);assert np.allclose(xx,bb,atol=3e-5,rtol=3e-5);reexecuted[name+'__'+part]=qq;main[name+'__'+part]=bb;delta[name+'__'+part]=dd
 np.savez_compressed(O/'main_outputs.npz',**main);np.savez_compressed(O/'correction_outputs.npz',**delta);save('parts_source_freeze.json',{n:sha(O/n) for n in ['main_outputs.npz','correction_outputs.npz']});save('parts_review.json',dict(passed=True,seeds=3,states_per_seed=4438,total_scores_reexecuted=13314,independent_main_scores=13314,max_total_abs_error=maxerr,max_numpy_main_abs_error=numpyerr,saved_total_decisions_exact=True,new_fits=0,new_physics=0,warning='Main backbone was co-trained with correction,not independently trainedBASE.'));save('state.json',dict(stage='parts_reexecuted'));print('GRID163_PARTS',maxerr,numpyerr)
def replay(c,actions,method,fee,scale):
 p=0;seq=[]
 while p<len(c['states']):
  go=int(actions[p]);seq.append(go)
  if not go:break
  p+=1
 gain=float(c['gains'][:p].sum());queries=int(c['queries'][:p].sum());return dict(sample_id=c['sample_id'],date=c['date'],scenario=c['scenario'],physical_state_group=c['physical_state_group'],method=method,batches=p,actions=seq,gain=gain,queries=queries,net_utility=(gain-fee*queries)/scale,chosen_pool_id=c['chosen'][p])
def evaluate():
 verify();assert read(O/'parts_review.json')['passed'];assert not (O/'review.json').exists();meta,inputs,lookup,cases=loaders();main=np.load(O/'main_outputs.npz');delta=np.load(O/'correction_outputs.npz');saved=np.load(R/'outputs/grid116_matched_module/state_predictions.npz');plan=read(R/'outputs/grid116_matched_module/design.json');fee=plan['unchanged']['cost_per_query'];scale=plan['unchanged']['gain_scale'];effects=[]
 for r in read(O/'first_difference_records.json'):
  seed=r['seed'];name=f'BASE_FACTOR_seed{seed}__validation';i=lookup['validation'][r['sample_id'],r['first_difference_position']];b=main[name][i];d=delta[name][i];q=saved[name][i];bm=float(b[1]-b[0]);dm=float(d[1]-d[0]);tm=float(q[1]-q[0]);assert abs(bm+dm-tm)<1e-6;effects.append(dict(seed=seed,sample_id=r['sample_id'],date=r['date'],physical_state_group=r['physical_state_group'],category=r['category'],position=r['first_difference_position'],main_margin=bm,correction_margin=dm,total_margin=tm,correction_flips_main=bool(bm>0)!=bool(tm>0),main_action=int(bm>0),factor_action=int(tm>0),original_base_action=r['base_action']))
 groups=[]
 for cat in sorted({r['category'] for r in effects}):
  rr=[r for r in effects if r['category']==cat];groups.append(dict(category=cat,events=len(rr),correction_flips_main=sum(x['correction_flips_main'] for x in rr),main_already_factor_action=sum(x['main_action']==x['factor_action'] for x in rr),main_margin_median=float(np.median([x['main_margin'] for x in rr])),correction_margin_median=float(np.median([x['correction_margin'] for x in rr]))))
 records=[];summaries=[]
 for part in ['train','validation']:
  original=read(R/f'outputs/grid116_matched_module/{part}_trajectories.json');by={(r['method'],r['sample_id']):r for r in original}
  for seed in range(3):
   for mode in ['ORIGINAL_BASE','FACTOR_TOTAL','FACTOR_MAIN_ONLY']:
    rr=[]
    for c in cases.values():
     if c['partition']!=part:continue
     if mode=='FACTOR_MAIN_ONLY':
      qq=main[f'BASE_FACTOR_seed{seed}__{part}'];ii=[lookup[part][c['sample_id'],j] for j in range(len(c['states']))];row=replay(c,qq[ii,1]>qq[ii,0],f'{mode}_seed{seed}',fee,scale)
     else:
      n=f'DIRECT{seed}' if mode=='ORIGINAL_BASE' else f'BASE_FACTOR_seed{seed}';old=by[n,c['sample_id']];row=replay(c,old['actions'],f'{mode}_seed{seed}',fee,scale)
      for k in ['batches','gain','queries','chosen_pool_id']:assert row[k]==old[k]
     rr.append(row)
    dd=[]
    for date in sorted({r['date'] for r in rr}):
     xx=[r for r in rr if r['date']==date];dd.append(dict(date=date,cases=len(xx),mean_net_utility=float(np.mean([x['net_utility'] for x in xx])),mean_gain=float(np.mean([x['gain'] for x in xx])),mean_queries=float(np.mean([x['queries'] for x in xx]))))
    summaries.append(dict(partition=part,seed=seed,mode=mode,dates=len(dd),starts=len(rr),mean_net_utility=float(np.mean([x['mean_net_utility'] for x in dd])),mean_gain=float(np.mean([x['mean_gain'] for x in dd])),mean_queries=float(np.mean([x['mean_queries'] for x in dd])),per_date=dd));records+=rr
 save('first_difference_component_effects.json',effects);save('component_effect_summary.json',groups);save('fixed_main_only_trajectories.json',records);save('main_only_summary.json',summaries);verify();save('review.json',dict(passed=True,first_difference_cases=603,all_component_sums_checked=True,paired_original_trajectories_reproduced=True,all3seeds_all_dates_retained=True,new_fits=0,new_physics=0,no_threshold_search=True,final8_absent=True,not_new_training_causal_test=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(effects=groups,validation=[r for r in summaries if r['partition']=='validation']),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
