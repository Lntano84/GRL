"""Paired small extension: route known cost into FACTOR main, correction unchanged."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid165_factor_cost_routing';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID165_factor_cost_routing';ARMS=['ZERO_MAIN','COST_MAIN']
sp=importlib.util.spec_from_file_location('g165_cost_helpers',R/'work/grid164_explicit_cost/run.py');H=importlib.util.module_from_spec(sp);sp.loader.exec_module(H)
read=H.read;sha=H.sha;digest=H.digest
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 for f in sorted((O/'models').glob('*')) if (O/'models').exists() else []:
  if f.is_file():q=d/'models';q.mkdir(exist_ok=True);shutil.copy2(f,q/f.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
 H.verify_trained()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
 H.sealed()
def configure():
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def old_api():
 sys.path.insert(0,str(R/'work/grid116_matched_module'));import common116
 return common116
def build(seed,arm):
 import torch
 from torch import nn
 Old=old_api();old=Old.build(seed,'BASE_FACTOR');base42,_,_=H.build(seed)
 class Routed(nn.Module):
  def __init__(self):super().__init__();self.base=base42;self.delta=old.delta;self.use_cost=arm=='COST_MAIN'
  def forward(self,**public):
   cost=public['known_query_cost'] if self.use_cost else torch.zeros_like(public['known_query_cost']);return self.base(torch.cat([public['global41'],cost],-1))+self.delta(**public)
 return Routed(),old
def inputs(part):return pickle.loads((R/f'outputs/grid113_relational_learning/{part}_public.pkl').read_bytes())
def prepare():
 assert not O.exists();O.mkdir(parents=True);H.verify_trained();assert read(H.O/'complete.json')['passed_audit'];configure();Old=old_api();task=read(R/'outputs/grid116_matched_module/design.json');plan=read(H.O/'design.json');source=R/'outputs/grid113_relational_learning'
 d=dict(id='GRID165_FACTOR_COST_ROUTING',authorization='User continue;pretraining notice delivered for a paired main-cost routing trial.',
  rationale='COST42 has modest repeatable utility gains whileFACTOR has different positive cases. Test whether direct main access to known cost helps within FACTOR,not assume additive gains.',
  arms=ARMS,seeds=[0,1,2],fit_order=[dict(seed=s,arm=a) for s in [0,1,2] for a in ARMS],fits_permitted=6,
  model='Both12980parameters:42->64->64->2 main7042 plus original FACTOR correction5938. Correction sees same original41/actions/paidfeedback/knowncost in both. Main42extra0 versus knowncost.',
  initialization='Same original random41seed transplanted into main42with extra column0. Original random FACTOR correction/offset seed/headzero unchanged. Both arms tensor-identical initially,not fitted model reuse.',
  fixed='Old16train/4validation dates,3582/856states,2378/590starts,NET[0,net]Huber beta1,oldpricescale/sampler,7250updates,batch32,Adam. Only cost access to main differs.',
  cost_per_query=task['unchanged']['cost_per_query'],gain_scale=task['unchanged']['gain_scale'],training=dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,clip=10.,weight_decay=0.,sampler_seed_offset=20261022,wall_cap_s=1800,trace_every=500),
  evaluation='All6final weights published before validation;strictq1>q0,no thresholds. Allseed/date gain/query/utility and paired changes,oldFACTOR,COST42,BASE,BATCH1 reference. No all-seedwin gate for exploratory merit.',
  limitations='Exposed samegrid oldlocal task,not independent confirmation,weekly cost/quality or measuredcontroltime. Current global12/4/4 target not silently changed;this is separate local exploratory line.',
  prohibited=['additionalfits/seed/price/threshold/checkpoint search','interimvalidation','newdata/physics/test8','grossgainlabel orcostoutside change','paper','automatic scale'])
 save('design.json',d);rng=np.random.default_rng(165);synthetic=[]
 for n,m in [(7,5),(3,2)]:synthetic.append(dict(context_action=rng.normal(size=(n,68)).astype('f'),context_feedback=rng.normal(size=(n,8)).astype('f'),next_action=rng.normal(size=(m,68)).astype('f'),anchor_action=rng.normal(size=68).astype('f'),global41=rng.normal(size=41).astype('f'),known_query_cost=np.array([.02],'f')))
 import torch
 checks=[];pub=Old.collate(synthetic)
 for seed in [0,1,2]:
  a,old=build(seed,'ZERO_MAIN');b,_=build(seed,'COST_MAIN');assert digest(a)==digest(b);assert sum(p.numel() for p in a.parameters())==12980
  with torch.no_grad():qa=a(**pub);qb=b(**pub);qo=old(**pub);assert torch.equal(qa,qb);assert torch.allclose(qa,qo,atol=1e-6,rtol=1e-6)
  a(**pub).square().mean().backward();b(**pub).square().mean().backward();assert a.base.q_net[0].weight.grad[:,41].abs().sum()==0;assert b.base.q_net[0].weight.grad[:,41].abs().sum()>0
  expected=next(x for x in read(H.O/'preflight.json')['paired'] if x['seed']==seed);checks.append(dict(seed=seed,initial_hash=digest(a),sampling_hash=expected['sampling_hash']))
 save('preflight.json',dict(passed=True,paired=checks,parameters=12980,initial_outputs_match=True,no_optimizer_instances=True,new_fits=0))
 paths=[H.O/n for n in ['design.json','preflight.json','family_summary.json','complementarity_summary.json','complete.json']]
 paths +=[source/n for n in ['train_public.pkl','validation_public.pkl','train_metadata.json','validation_metadata.json','train_targets.npz','validation_targets.npz']]
 paths +=[R/'outputs/grid107_direct_value'/n for n in ['train.npz','validation_trajectories.json']]
 paths +=[R/'outputs/grid116_matched_module'/n for n in ['design.json','preflight.json','validation_results.json','validation_trajectories.json','review.json']]
 paths +=[R/'outputs/grid102_search_rl/cases.pkl',R/'work/grid113_relational_learning/model.py',R/'work/grid113_relational_learning/common.py',R/'work/grid115_matched_module_plan/matched_model.py',R/'work/grid164_explicit_cost/run.py',O/'design.json',O/'preflight.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();save('state.json',dict(stage='frozen'));archive('frozen');print('GRID165_FROZEN')
def train():
 verify();assert not (O/'training_started.json').exists();configure();import torch
 Old=old_api();d=read(O/'design.json');rows=inputs('train');meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');assert len(rows)==3582 and all(x['partition']=='train' for x in meta);z=np.load(R/'outputs/grid107_direct_value/train.npz');target=np.load(R/'outputs/grid113_relational_learning/train_targets.npz')['target'];assert np.array_equal(target,z['target']);yt=torch.tensor(np.column_stack([np.zeros_like(target),target]));take=Old.tensor_cache(rows);prob=z['sampling_probability'];start=time.perf_counter();save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fits_permitted=6,wall_cap_s=1800));(O/'models').mkdir();receipts=[]
 for item in d['fit_order']:
  if time.perf_counter()-start>1800:raise TimeoutError('Preserve partial task;no automatic extension')
  seed,arm=item['seed'],item['arm'];net,_=build(seed,arm);initial=digest(net);expected=next(x for x in read(O/'preflight.json')['paired'] if x['seed']==seed);assert initial==expected['initial_hash'];opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);count=np.zeros(3582,np.int64);losses=[];stream=hashlib.sha256();fs=time.perf_counter();net.train()
  for update in range(1,7251):
   if time.perf_counter()-start>1800:raise TimeoutError('Keep partial output,not a method failure')
   ix=rng.choice(3582,32,replace=True,p=prob);np.add.at(count,ix,1);stream.update(ix.astype('<i8').tobytes());q=net(**take(ix));loss=torch.nn.functional.smooth_l1_loss(q,yt[ix],beta=1.);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
   if update%500==0 or update==7250:
    p=dict(seed=seed,arm=arm,update=update,completed=len(receipts),fit_wall_s=time.perf_counter()-fs,total_wall_s=time.perf_counter()-start,mean_last_loss=float(np.mean(losses[-500:])));save('fit_progress.json',p);print(json.dumps(p),flush=True)
  assert stream.hexdigest()==expected['sampling_hash'];assert np.array_equal(count,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws']);name=f'{arm}_seed{seed}';path=O/'models'/f'{name}.pt';torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),arm=arm,seed=seed,updates=7250),path);np.savez_compressed(O/f'exposure_{name}.npz',state_draws=count,losses=np.asarray(losses));rec=dict(seed=seed,arm=arm,updates=7250,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(path),sampling_hash=stream.hexdigest(),parameters=12980,fit_wall_s=time.perf_counter()-fs);receipts.append(rec);save('fit_receipts.json',receipts)
 save('fits.json',dict(passed=True,new_fits=6,records=receipts,total_wall_s=time.perf_counter()-start,validation_used_for_fit=False));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*O.glob('exposure_*.npz'),O/'fits.json',O/'fit_receipts.json']});save('state.json',dict(stage='trained_before_validation'));archive('trained');print('GRID165_SIX_FITS_COMPLETE')
def verify_trained():
 verify()
 for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def publication():
 verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote;rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json');save('prevalidation_publication.json',dict(passed=True,local_sha=local,remote_sha=remote));print('GRID165_WEIGHTS_PUBLISHED')
def evaluate():
 verify_trained();assert read(O/'prevalidation_publication.json')['passed'];assert not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all6models_published=True));configure();import torch
 Old=old_api();sp=importlib.util.spec_from_file_location('g165_numpy',R/'work/grid113_relational_learning/audit.py');Aud=importlib.util.module_from_spec(sp);sp.loader.exec_module(Aud);d=read(O/'design.json');fee=d['cost_per_query'];scale=d['gain_scale'];cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());public={p:inputs(p) for p in ['train','validation']};meta={p:read(R/f'outputs/grid113_relational_learning/{p}_metadata.json') for p in public};lookup={p:{(r['sample_id'],r['position']):i for i,r in enumerate(meta[p])} for p in public};scores={};summary={};tr={p:[] for p in public};checks=collections.Counter();maxerr=0.
 for rec in read(O/'fits.json')['records']:
  seed,arm=rec['seed'],rec['arm'];name=f'{arm}_seed{seed}';net,_=build(seed,arm);cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu');net.load_state_dict(cp['state_dict']);assert digest(net)==rec['final_hash'];assert all(int(x['step'])==7250 for x in cp['optimizer']['state'].values());w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};dw={k.removeprefix('delta.'):v for k,v in w.items() if k.startswith('delta.')}
  for part,rows in public.items():
   q=Old.predict(net,rows,batch=32);scores[name+'__'+part]=q;y=np.load(R/f'outputs/grid113_relational_learning/{part}_targets.npz')['target'];ids=sorted(set([0,len(y)-1]+list(range(0,len(y),151))+list(np.flatnonzero(y>0))))
   for i in ids:
    row=rows[i];cost=row['known_query_cost'] if arm=='COST_MAIN' else np.zeros(1,'f');x=np.concatenate([row['global41'],cost]).astype(float)
    for j in [0,2,4]:
     x=x@w[f'base.q_net.{j}.weight'].T+w[f'base.q_net.{j}.bias']
     if j!=4:x=np.maximum(x,0)
    nq=x+Aud.numpy_forward(dw,row,'FACTOR');e=float(np.max(abs(nq-q[i])));maxerr=max(maxerr,e);assert np.allclose(nq,q[i],atol=3e-5,rtol=3e-5);assert bool(nq[1]>nq[0])==bool(q[i,1]>q[i,0]);checks['independent_numpy_full_forwards']+=1
   rr=[]
   for c in cases:
    if c['partition']!=part:continue
    ix=[lookup[part][c['sample_id'],j] for j in range(len(c['states']))];rr.append(H.trajectory(c,q[ix,1]>q[ix,0],name,fee,scale))
   tr[part]+=rr;summary[name+'__'+part]=H.summarize(rr)
 # Saved reference decisions,all six old seeds and BATCH1,with exact fee accounting.
 references=read(R/'outputs/grid116_matched_module/validation_trajectories.json')+read(H.O/'validation_trajectories.json');cm={c['sample_id']:c for c in cases};refnames=['DIRECT0','DIRECT1','DIRECT2','BASE_FACTOR_seed0','BASE_FACTOR_seed1','BASE_FACTOR_seed2','COST42_seed0','COST42_seed1','COST42_seed2','BATCH1'];oldmap={(r['method'],r['sample_id']):r for r in references}
 for name in refnames:
  rr=[]
  for (method,sid),r in oldmap.items():
   if method!=name:continue
   t=H.trajectory(cm[sid],r['actions'],name,fee,scale)
   for k in ['gain','queries','chosen_pool_id','batches']:assert t[k]==r[k]
   assert abs(t['net_utility']-r['net_utility'])<1e-12;rr.append(t)
  assert len(rr)==590;tr['validation']+=rr;summary[name+'__validation']=H.summarize(rr)
 np.savez_compressed(O/'state_scores.npz',**scores);save('summary.json',summary)
 for part,rr in tr.items():save(part+'_trajectories.json',rr)
 family={}
 for mode,fmt in [('ZERO_MAIN','ZERO_MAIN_seed{}'),('COST_MAIN','COST_MAIN_seed{}'),('OLD_FACTOR','BASE_FACTOR_seed{}'),('OLD_BASE','DIRECT{}'),('COST42','COST42_seed{}')]:family[mode]={k:float(np.mean([summary[fmt.format(s)+'__validation']['macro'][k] for s in [0,1,2]])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']}
 pairs=[]
 for seed in [0,1,2]:
  a=summary[f'COST_MAIN_seed{seed}__validation'];b=summary[f'ZERO_MAIN_seed{seed}__validation'];pairs.append(dict(seed=seed,treatment_utility=a['macro']['mean_net_utility'],control_utility=b['macro']['mean_net_utility'],net_delta=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],per_date=[dict(date=x['date'],net_delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
 dates=[dict(date=pairs[0]['per_date'][i]['date'],mean_delta=float(np.mean([x['per_date'][i]['net_delta'] for x in pairs]))) for i in range(4)];delta=family['COST_MAIN']['mean_net_utility']-family['ZERO_MAIN']['mean_net_utility'];save('family_summary.json',family);save('paired_summary.json',dict(pairs=pairs,mean_net_delta=delta,relative_mean_change=delta/family['ZERO_MAIN']['mean_net_utility'],positive_seeds=sum(x['net_delta']>0 for x in pairs),per_date=dates,all_seed_win_not_required_for_research=True));verify_trained();save('review.json',dict(passed=True,checks=dict(checks),maximum_numpy_full_error=maxerr,fits=6,all6published_before_validation=True,all3seeds_all_dates_retained=True,no_new_data=True,new_physics=0,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(family=family,paired=read(O/'paired_summary.json')),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
