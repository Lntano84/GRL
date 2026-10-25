"""Small paired NET-target experiment: explicit known-cost input versus dummy."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,gzip,hashlib,importlib.util,json,math,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid164_explicit_cost';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID164_explicit_cost';ARMS=['ZERO42','COST42']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def digest(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 for f in sorted((O/'models').glob('*')) if (O/'models').exists() else []:
  if f.is_file():q=d/'models';q.mkdir(exist_ok=True);shutil.copy2(f,q/f.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def sealed():
 sp=importlib.util.spec_from_file_location('g164_sealed',R/'work/grid139_global_feasibility/run.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b);b.sealed()
def verify():
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
 sealed()
def configure():
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def build(seed):
 import torch
 from torch import nn
 class Q(nn.Module):
  def __init__(self,n):super().__init__();self.q_net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
  def forward(self,x):return self.q_net(x)
 torch.manual_seed(seed);base=Q(41);basehash=digest(base);net=Q(42)
 with torch.no_grad():
  net.q_net[0].weight[:,:41].copy_(base.q_net[0].weight);net.q_net[0].weight[:,41].zero_();net.q_net[0].bias.copy_(base.q_net[0].bias)
  for j in [2,4]:net.q_net[j].weight.copy_(base.q_net[j].weight);net.q_net[j].bias.copy_(base.q_net[j].bias)
 return net,base,basehash
def data(part):
 source=R/'outputs/grid113_relational_learning';public=pickle.loads((source/f'{part}_public.pkl').read_bytes());meta=read(source/f'{part}_metadata.json');x=np.stack([r['global41'] for r in public]);cost=np.stack([r['known_query_cost'] for r in public]);return x,cost,meta
def full_x(x,cost,arm):return np.concatenate([x,cost if arm=='COST42' else np.zeros_like(cost)],axis=1).astype(np.float32)
def prepare():
 assert not O.exists();O.mkdir(parents=True);sealed();assert read(R/'outputs/grid163_factor_first_decision/complete.json')['passed_audit'];configure();old=read(R/'outputs/grid107_direct_value/design.json');task=read(R/'outputs/grid116_matched_module/design.json');x,cost,meta=data('train');z=np.load(R/'outputs/grid107_direct_value/train.npz');assert np.array_equal(x,z['X']) and x.shape==(3582,41);assert all(r['partition']=='train' for r in meta)
 d=dict(id='GRID164_EXPLICIT_COST_INPUT',authorization='User continuation after notice:two tiny models per3existingseeds,no new data.',
  hypothesis='A direct scalar expression of already-known next-batch cost helps recognize unprofitable continuation,one possible contributor toFACTOR positive savings.',
  task='Old local next-batch NET target [0,(gain-fee*queries)/gain_scale] unchanged;not global action-identity andnot newRL.',
  arms=ARMS,seeds=[0,1,2],fit_order=[dict(seed=s,arm=a) for s in [0,1,2] for a in ARMS],new_fits=6,
  inputs='Original normalized41 plus42nd scalar. ZERO42:0;COST42:unchanged fee*nextqueries/gain_scale,unscaled exactly asFACTOR. It is recoverable from original log-next-query count;no new information.',
  network='42->64->64->2 ReLU,7042 parameters both;copy original RANDOM seed41 initialization into first41 columns and later layers,extra column0. No fitted weights loaded.',
  original_random_hashes=read(R/'outputs/grid107_direct_value/preflight.json')['initial_q_hashes'],
  training=dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip=10.,loss='smooth_l1 beta1,mean32x2',sampler_seed_offset=20261022,trace_every=500,wall_cap_s=600),
  population=dict(train_states=3582,train_starts=2378,train_dates=16,validation_states=856,validation_starts=590,validation_dates=4),
  cost_per_query=task['unchanged']['cost_per_query'],gain_scale=task['unchanged']['gain_scale'],
  readout='Strict q_continue>q_stop,all seeds anddates,no calibration/threshold. Final7250only. Publish all6 models before validation scoring.',
  comparisons='COST42 versus pairedZERO42 isolates explicit input under matched capacity/init/exposure. OldBASE,FACTOR,BATCH1 references retained;ZERO42 not assumed identical trained result to old41 due floating-point/kernel difference.',
  interpretation='Report any small recurring improvements and first-difference mechanisms;do not require all seeds/dates win for research value. No automatic adoption/scale. Exposed dates,not independent confirmation.',
  not_grid126='No gross-gain regression,no analytic cost deduction or new link/loss;NET labels andloss unchanged.',
  prohibited=['extra fits/checkpoint/feature/price/seed/threshold searches','reweight positives','new data/scenarios/physics','final8','paper','automatic scale'])
 save('design.json',d);rng=np.random.default_rng(164);sx=rng.normal(size=(8,41)).astype('f');sc=rng.uniform(.001,.1,size=(8,1)).astype('f');checks=[]
 import torch
 for seed in [0,1,2]:
  a,base,bh=build(seed);b,_,_=build(seed);assert bh==d['original_random_hashes'][str(seed)];assert digest(a)==digest(b);assert sum(p.numel() for p in a.parameters())==7042
  with torch.no_grad():qa=a(torch.tensor(full_x(sx,sc,'ZERO42')));qb=b(torch.tensor(full_x(sx,sc,'COST42')));qbase=base(torch.tensor(sx));assert torch.equal(qa,qb);assert torch.allclose(qa,qbase,atol=1e-6,rtol=1e-6)
  a(torch.tensor(full_x(sx,sc,'ZERO42'))).square().mean().backward();b(torch.tensor(full_x(sx,sc,'COST42'))).square().mean().backward();assert a.q_net[0].weight.grad[:,41].abs().sum()==0;assert b.q_net[0].weight.grad[:,41].abs().sum()>0
  sampler=np.random.default_rng(20261022+seed);h=hashlib.sha256();counts=np.zeros(3582,np.int64)
  for _ in range(7250):ids=sampler.choice(3582,32,replace=True,p=z['sampling_probability']);h.update(ids.astype('<i8').tobytes());np.add.at(counts,ids,1)
  assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws']);assert h.hexdigest()==read(R/'outputs/grid116_matched_module/preflight.json')['sampling_streams'][str(seed)]['index_sha256'];checks.append(dict(seed=seed,initial_hash=digest(a),original41_random_hash=bh,sampling_hash=h.hexdigest()))
 save('preflight.json',dict(passed=True,paired=checks,synthetic_backward_only=True,optimizer_instances=0,new_fits=0,matched_initial_functions=True))
 paths=[R/'outputs/grid107_direct_value'/n for n in ['design.json','preflight.json','train.npz','training_rows.json','validation_trajectories.json','validation_results.json']]
 paths +=[R/'outputs/grid113_relational_learning'/n for n in ['train_public.pkl','validation_public.pkl','train_metadata.json','validation_metadata.json','train_targets.npz','validation_targets.npz']]
 paths +=[R/'outputs/grid116_matched_module'/n for n in ['design.json','validation_results.json','validation_trajectories.json','preflight.json']]
 paths +=[R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid163_factor_first_decision/cost_expression_audit.json',R/'outputs/grid162_positive_change_audit/factor_all1770_paired_sequences.json',O/'design.json',O/'preflight.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();save('state.json',dict(stage='frozen'));archive('frozen');print('GRID164_FROZEN')
def train():
 verify();assert not (O/'training_started.json').exists();configure();import torch
 d=read(O/'design.json');x,cost,meta=data('train');z=np.load(R/'outputs/grid107_direct_value/train.npz');y=np.column_stack([np.zeros_like(z['target']),z['target']]);assert np.array_equal(z['target'],np.load(R/'outputs/grid113_relational_learning/train_targets.npz')['target']);xt={a:torch.tensor(full_x(x,cost,a)) for a in ARMS};yt=torch.tensor(y);prob=z['sampling_probability'];start=time.perf_counter();save('training_started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fits_permitted=6,wall_cap_s=600));(O/'models').mkdir();receipts=[]
 for item in d['fit_order']:
  seed,arm=item['seed'],item['arm'];assert time.perf_counter()-start<600;model,base,bh=build(seed);init=digest(model);assert init==d['initial_hash'] if 'initial_hash' in d else True
  assert init==next(c['initial_hash'] for c in read(O/'preflight.json')['paired'] if c['seed']==seed);opt=torch.optim.Adam(model.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);counts=np.zeros(3582,np.int64);losses=[];indexh=hashlib.sha256();fs=time.perf_counter();model.train()
  for update in range(1,7251):
   if time.perf_counter()-start>600:raise TimeoutError('Preserve partial fixed six-fit experiment;no extension or retry')
   ids=rng.choice(3582,32,replace=True,p=prob);indexh.update(ids.astype('<i8').tobytes());np.add.at(counts,ids,1);q=model(xt[arm][ids]);loss=torch.nn.functional.smooth_l1_loss(q,yt[ids],beta=1.);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
   if update%500==0 or update==7250:
    progress=dict(seed=seed,arm=arm,update=update,completed=len(receipts),fit_wall_s=time.perf_counter()-fs,total_wall_s=time.perf_counter()-start,mean_last_loss=float(np.mean(losses[-500:])));save('fit_progress.json',progress);print(json.dumps(progress),flush=True)
  expected=next(c for c in read(O/'preflight.json')['paired'] if c['seed']==seed);assert indexh.hexdigest()==expected['sampling_hash'];assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws']);name=f'{arm}_seed{seed}';path=O/'models'/f'{name}.pt';torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),seed=seed,arm=arm,updates=7250),path);np.savez_compressed(O/f'exposure_{name}.npz',state_draws=counts,losses=np.asarray(losses));receipt=dict(seed=seed,arm=arm,updates=7250,initial_hash=init,final_hash=digest(model),checkpoint_sha256=sha(path),sampling_hash=indexh.hexdigest(),state_draws=int(counts.sum()),parameters=sum(p.numel() for p in model.parameters()),fit_wall_s=time.perf_counter()-fs);receipts.append(receipt);save('fit_receipts.json',receipts)
 assert len(receipts)==6;save('fits.json',dict(passed=True,new_fits=6,records=receipts,total_wall_s=time.perf_counter()-start,validation_used_for_fit=False));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*O.glob('exposure_*.npz'),O/'fits.json',O/'fit_receipts.json']});save('state.json',dict(stage='trained_before_validation'));archive('trained');print('GRID164_SIX_FITS_COMPLETE')
def verify_trained():
 verify()
 for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def publication():
 verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote;rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json');save('prevalidation_publication.json',dict(passed=True,local_sha=local,remote_sha=remote));print('GRID164_WEIGHTS_PUBLISHED')
def trajectory(c,go,name,fee,scale):
 p=0;seq=[]
 while p<len(c['states']):
  act=int(go[p]);seq.append(act)
  if not act:break
  p+=1
 gain=float(c['gains'][:p].sum());q=int(c['queries'][:p].sum());return dict(sample_id=c['sample_id'],date=c['date'],scenario=c['scenario'],physical_state_group=c['physical_state_group'],method=name,actions=seq,batches=p,gain=gain,queries=q,full_gain=float(c['gains'].sum()),full_queries=int(c['queries'].sum()),chosen_pool_id=c['chosen'][p],net_utility=(gain-fee*q)/scale)
def summarize(rows):
 per=[]
 for date in sorted({r['date'] for r in rows}):
  rr=[r for r in rows if r['date']==date];gs=sorted({r['physical_state_group'] for r in rr});totgain=sum(r['full_gain'] for r in rr);totq=sum(r['full_queries'] for r in rr);per.append(dict(date=date,cases=len(rr),mean_net_utility=float(np.mean([r['net_utility'] for r in rr])),mean_gain=float(np.mean([r['gain'] for r in rr])),mean_queries=float(np.mean([r['queries'] for r in rr])),query_fraction=sum(r['queries'] for r in rr)/totq,gain_capture=sum(r['gain'] for r in rr)/totgain if totgain else None,physical_group_mean_utility=float(np.mean([np.mean([r['net_utility'] for r in rr if r['physical_state_group']==g]) for g in gs]))))
 return dict(cases=len(rows),per_date=per,macro={k:float(np.mean([r[k] for r in per if r[k] is not None])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']})
def evaluate():
 verify_trained();assert read(O/'prevalidation_publication.json')['passed'];assert not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all_six_models_published=True));configure();import torch
 d=read(O/'design.json');fee=d['cost_per_query'];scale=d['gain_scale'];cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());data_all={p:data(p) for p in ['train','validation']};lookup={p:{(r['sample_id'],r['position']):i for i,r in enumerate(data_all[p][2])} for p in data_all};scores={};alltr={p:[] for p in data_all};summary={};checks=collections.Counter();states=[]
 for rec in read(O/'fits.json')['records']:
  arm,seed=rec['arm'],rec['seed'];name=f'{arm}_seed{seed}';model,_,_=build(seed);cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu');assert cp['updates']==7250;assert all(int(x['step'])==7250 for x in cp['optimizer']['state'].values());model.load_state_dict(cp['state_dict']);assert digest(model)==rec['final_hash'];model.eval();weights={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()}
  for part,(x,cost,meta) in data_all.items():
   xx=full_x(x,cost,arm)
   with torch.no_grad():q=model(torch.tensor(xx)).numpy()
   manual=xx.astype(float)
   for j in [0,2,4]:
    manual=manual@weights[f'q_net.{j}.weight'].T+weights[f'q_net.{j}.bias']
    if j!=4:manual=np.maximum(manual,0)
   assert np.allclose(manual,q,atol=3e-5,rtol=3e-5);assert np.array_equal(manual[:,1]>manual[:,0],q[:,1]>q[:,0]);checks['independent_score_rows']+=len(q);scores[name+'__'+part]=q
   target=np.load(R/f'outputs/grid113_relational_learning/{part}_targets.npz')['target'];pos=target>0;go=q[:,1]>q[:,0];states.append(dict(method=name,partition=part,positive=int(pos.sum()),false_stop=int(np.sum(pos&~go)),unprofitable_continue=int(np.sum(~pos&go)),positive_recall=float(go[pos].mean()),mean_sign_regret=float(np.mean(np.maximum(target,0)-go*target))))
   rr=[]
   for c in cases:
    if c['partition']!=part:continue
    ii=[lookup[part][c['sample_id'],j] for j in range(len(c['states']))];rr.append(trajectory(c,go[ii],name,fee,scale))
   alltr[part]+=rr;summary[name+'__'+part]=summarize(rr)
 # Prespecified references are saved decisions re-accounted under the identical old task.
 old=read(R/'outputs/grid116_matched_module/validation_trajectories.json');refs=['DIRECT0','DIRECT1','DIRECT2','BASE_FACTOR_seed0','BASE_FACTOR_seed1','BASE_FACTOR_seed2','BATCH1'];cm={c['sample_id']:c for c in cases}
 for name in refs:
  rr=[]
  for r in old:
   if r['method']!=name:continue
   t=trajectory(cm[r['sample_id']],r['actions'],name,fee,scale)
   for k in ['batches','gain','queries','chosen_pool_id']:assert t[k]==r[k]
   assert abs(t['net_utility']-r['net_utility'])<1e-12;rr.append(t)
  assert len(rr)==590;alltr['validation']+=rr;summary[name+'__validation']=summarize(rr)
 np.savez_compressed(O/'state_scores.npz',**scores);save('state_metrics.json',states);save('summary.json',summary)
 for part,rr in alltr.items():save(part+'_trajectories.json',rr)
 # All 1770 treatment/control trajectories, not just previously favorable records.
 v={(r['method'],r['sample_id']):r for r in alltr['validation']};events=[];prior={(r['seed'],r['sample_id']):r['category'] for r in read(R/'outputs/grid162_positive_change_audit/factor_all1770_paired_sequences.json')}
 for seed in [0,1,2]:
  for c in cases:
   if c['partition']!='validation':continue
   a=v[f'COST42_seed{seed}',c['sample_id']];b=v[f'ZERO42_seed{seed}',c['sample_id']];dg=a['gain']-b['gain'];savedq=b['queries']-a['queries'];net=a['net_utility']-b['net_utility'];assert abs(net-(dg+fee*savedq)/scale)<1e-12
   cat='UNCHANGED' if dg==0 and savedq==0 else ('LESS_QUERY_NO_GAIN_LOSS' if savedq>0 and dg>=0 else ('MORE_QUERY_COST_NOT_REPAID' if savedq<0 and net<0 else ('NET_POSITIVE_OTHER' if net>0 else 'NET_NONPOSITIVE_OTHER')))
   events.append(dict(seed=seed,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],category=cat,old_factor_event=prior[seed,c['sample_id']],gain_delta=dg,saved_queries=savedq,net_delta=net,same_action=a['chosen_pool_id']==b['chosen_pool_id'],control_batches=b['batches'],cost_batches=a['batches']))
 save('all1770_paired_events.json',events);family={}
 for mode,fmt in [('ZERO42','ZERO42_seed{}'),('COST42','COST42_seed{}'),('OLD_BASE','DIRECT{}'),('OLD_FACTOR','BASE_FACTOR_seed{}')]:family[mode]={k:float(np.mean([summary[fmt.format(s)+'__validation']['macro'][k] for s in [0,1,2]])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']}
 pairs=[]
 for seed in [0,1,2]:
  a=summary[f'COST42_seed{seed}__validation'];b=summary[f'ZERO42_seed{seed}__validation'];pairs.append(dict(seed=seed,net_delta=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],cost_utility=a['macro']['mean_net_utility'],control_utility=b['macro']['mean_net_utility'],date_differences=[dict(date=x['date'],net_delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
 famdiff=family['COST42']['mean_net_utility']-family['ZERO42']['mean_net_utility'];date_delta=[dict(date=x['date'],mean_delta=float(np.mean([p['date_differences'][i]['net_delta'] for p in pairs]))) for i,x in enumerate(summary['COST42_seed0__validation']['per_date'])];save('family_summary.json',family);save('paired_summary.json',dict(pairs=pairs,mean_net_delta=famdiff,relative_mean_change=famdiff/family['ZERO42']['mean_net_utility'],positive_seeds=sum(p['net_delta']>0 for p in pairs),per_date=date_delta,not_an_adoption_gate=True));verify_trained();save('review.json',dict(passed=True,checks=dict(checks),fits=6,all_final_weights_before_dev=True,all3seeds_all_dates=True,new_physics=0,no_threshold_search=True,all1770_events_accounted=True,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(family=family,paired=read(O/'paired_summary.json')),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
