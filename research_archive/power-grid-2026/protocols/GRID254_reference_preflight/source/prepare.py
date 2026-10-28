"""Generate only the six fixed fit8 packages and eighteen label-free sample streams."""
from pathlib import Path
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,shutil
import numpy as np
import runtime254 as C
PLAN=C.R/'outputs/grid253_stop_source_plan';DATA=C.R/'outputs/grid121_date_transfer'
def archive(phase):
 d=C.A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*C.W.glob('*.py'),*C.O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 for name in ['pairs','expected','lifecycle','worker_probes']:
  if (C.O/name).exists():shutil.copytree(C.O/name,d/name)
 C.save(d/'archive_manifest.json',{p.relative_to(d).as_posix():C.sha(p) for p in d.rglob('*') if p.is_file()})
def main():
 assert not C.O.exists() and C.read(PLAN/'complete.json')['passed'];C.O.mkdir();(C.R/'tmp/grid254').mkdir(parents=True,exist_ok=True);(C.O/'pairs').mkdir();(C.O/'expected').mkdir();C.save(C.O/'design.json',dict(id='GRID254',scope='Zerooptimizer executionqualificationONLY. Sixfit8packages/18streaminitialhashes,6workerprivatecacheandlabel-free syntheticgradient memory probes,originalparentfit-onlyinterface. No training/referenceheldscores/newpolicy.',future='18FACTORreferencejobs only,6singlethreadworkers,7250updatesbatch32,originalAdam/SmoothL1/clip10,600sglobalinclusiveofworkerstartup/no extension. Cannottrain withoutqualifiedfuturestage255authorization.',resource='6workers heldsimultaneously withmaxpublicshape32batch syntheticforward/backward,zerooptimizer/unchangedinitialweights. Require1GiBavailableRAM beyond6MiBAdam allowance atall6ready. Thisis apreflightmargin,not learnedpolicy threshold.',guards='Atomic singlejobclaim;strictglobaldeadline;stopflag;terminateonlyownedchildren;partialfilesretained;no repeatcompletejob. Fakeprocesslifecycle testsdo notcreateNNmodels.',boundaries='Dataset/teacherprefix/taskfee/labels/scenarios/ranking/controllerunchanged. ReferenceNNnotdeployed. No final8/physical/grid/paper/FRAME/install. Completefuturepipeline18+<=22max40 declared253;notparametersearch.'))
 nodes=C.read(PLAN/'reference_nodes.json');meta=C.read(C.R/'outputs/grid113_relational_learning/train_metadata.json');src=np.load(DATA/'train_arrays.npz');raw,target,prob=src['raw'],src['target'],src['original_probability'];assert np.array_equal(target,np.load(C.R/'outputs/grid113_relational_learning/train_targets.npz')['target']);initials={r['seed']:r['initial_hashes']['FACTOR'] for r in C.read(C.R/'outputs/grid168_one_sided_transfer/preflight.json')['paired']};pairs={};jobs=[]
 for node in nodes:
  pairname=node['node'].split('_')[0];fi=np.array(node['fit_ids'],dtype=np.int64);hi=np.array(node['score_ids'],dtype=np.int64);assert set(fi).isdisjoint(hi) and len(fi)+len(hi)==3582;assert {meta[int(i)]['date'] for i in fi}==set(node['fit_dates']);assert len(node['fit_dates'])==8
  if pairname not in pairs:
   dates=[meta[int(i)]['date'] for i in fi];counts={d:dates.count(d) for d in set(dates)};w=np.array([1/counts[d] for d in dates]);w/=w.mean();mean=np.average(raw[fi].astype(float),axis=0,weights=w);scale=np.sqrt(np.average((raw[fi].astype(float)-mean)**2,axis=0,weights=w));scale[scale<1e-8]=1.;norm=np.clip((raw[fi]-mean)/scale,-20,20).astype('f');p=prob[fi]/prob[fi].sum();name=f'pairs/{pairname}.npz';np.savez_compressed(C.O/name,fit_ids=fi,score_ids=hi,mean=mean,scale=scale,normalized_fit=norm,fit_probability=p,target_fit=target[fi]);pairs[pairname]=name
  z=np.load(C.O/pairs[pairname]);assert np.array_equal(z['fit_ids'],fi);rng=np.random.default_rng(20261022+node['seed']);draws=np.zeros(3582,np.int64);h=hashlib.sha256()
  for _ in range(7250):
   ix=fi[rng.choice(len(fi),32,replace=True,p=z['fit_probability'])];np.add.at(draws,ix,1);h.update(ix.astype('<i8').tobytes())
  assert int(draws.sum())==232000 and draws[hi].sum()==0;exp=f"expected/{node['node']}.npz";np.savez_compressed(C.O/exp,state_draws=draws);jobs.append(dict(node=node['node'],pair=node['pair'],seed=node['seed'],package=pairs[pairname],expected_exposure=exp,sampling_hash=h.hexdigest(),initial_hash=initials[node['seed']],fit_dates=node['fit_dates'],score_dates=node['reference_score_dates'],updates=7250,batch=32))
 assert len(pairs)==6 and len(jobs)==18;C.save(C.O/'jobs.json',jobs);C.save(C.O/'package_sources.json',dict(global_target_file='outputs/grid121_date_transfer/train_arrays.npz',export_explicitly_parsed_full_oldtarget_array=True,each_export_contains_only_declared_8fitdate_labels=True,optimizer_datasets_exclude_other8dates=True));paths=[*C.W.glob('*.py'),C.O/'design.json',C.O/'jobs.json',C.O/'package_sources.json',PLAN/'fixed_pipeline_proposal.json',PLAN/'reference_nodes.json',PLAN/'reference_uses.json',DATA/'train_arrays.npz',C.R/'outputs/grid113_relational_learning/train_public.pkl',C.R/'outputs/grid113_relational_learning/train_metadata.json',C.R/'outputs/grid113_relational_learning/train_targets.npz',C.R/'outputs/grid168_one_sided_transfer/preflight.json']+list((C.O/'pairs').iterdir())+list((C.O/'expected').iterdir())
 paths+=[C.R/p for p in C.read(C.R/'outputs/grid168_one_sided_transfer/code_freeze.json')]
 # Frozen original parent interfaces are only inspected on own fit dates.
 for root in [C.R/'outputs/grid168_one_sided_transfer']+[C.R/f'outputs/grid170_remaining_date_folds/fold{f}' for f in [1,2,3]]:
  paths+=[root/'fold_arrays.npz',root/'state_scores.npz',root/'fit_receipts.json']+list((root/'models').glob('FACTOR_seed*.pt'))
 for f in range(4):
  for seed in range(3):paths.append(C.R/f'outputs/grid192_gate_support/fit_features/f{f}_s{seed}.npz')
 paths.append(C.R/'outputs/grid241_parent_prior_plan/qualified_pools.json');C.save(C.O/'source_freeze.json',{p.relative_to(C.R).as_posix():C.sha(p) for p in set(paths)});C.verify();archive('source');print('GRID254_INPUTS_FROZEN',6,18)
if __name__=='__main__':main()
