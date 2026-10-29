"""GRID275: fixed saved-margin algebra and ownfit tree structure only."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,pickle,collections,ast
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid275_forest_direction';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID275_forest_direction';Q=R/'outputs/grid274_stop_context';OLD=R/'outputs/grid271_union_guard';F=R/'outputs/grid273_stop_context_plan';ORIG=R/'outputs/grid267_conditional_forest_plan';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def path_constant(file,key):
 n=next(n.value for n in ast.parse(file.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets));assert isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div) and isinstance(n.left,ast.Name) and isinstance(n.right,ast.Constant);return {'R':R,'REPO':REPO}[n.left.id]/n.right.value

def verify():
 for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')

def published(phase):
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for rel,s in rd(A/phase/'archive_manifest.json').items():assert hashlib.sha256(git('show',h+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==s
 return h

def prepare():
 assert not O.exists() and rd(Q/'NEXT_PROTOCOL_PLAN.json')['id']=='GRID275';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h;ar=path_constant(R/'work/grid274_stop_context/run.py','A')
 for n in ['NEXT_PROTOCOL_PLAN.json','matched271_changes.json','matched271_summary.json','complete.json','independent_audit.json']:
  assert git('show',h+':'+(ar/'audited'/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 for out,root,ph,freeze in [(Q,ar,'trained','trained_freeze.json'),(Q,ar,'scored','scored_freeze.json'),(OLD,REPO/'research_archive/power-grid-2026/protocols/GRID271_union_guard','calibrated','calibrated_freeze.json'),(OLD,REPO/'research_archive/power-grid-2026/protocols/GRID271_union_guard','scored','scored_freeze.json')]:
  for rel,s in rd(out/freeze).items():assert sha(out/rel)==s and hashlib.sha256(git('show',h+':'+(root/ph/rel).relative_to(REPO).as_posix())).hexdigest()==s
 for rel,s in rd(F/'qualified_freeze.json').items():assert sha(F/rel)==s and hashlib.sha256(git('show',h+':research_archive/power-grid-2026/protocols/GRID273_stop_context_plan/qualified/'+rel)).hexdigest()==s
 O.mkdir();save('input_publication.json',dict(passed=True,git_sha=h,actual274archive=str(ar),all_models_scores_ownfit_inputs_specificblobs=True))
 save('design.json',dict(id='GRID275',scope='Readonly26fixedpublishedchanges algebra and16ownfitcopies/oldnewtree structure.0fit/cutoff/forestNNforward/alternativepolicy.',public='ALL26changes: delta_margin=delta_forestscore-delta_tau using publishedold271/new274 arrays. Compareabsolutecomponents/directions without computing hybrid decisions/returns. All12pool old/newtau/limitingties fromsavedcal arrays;structuraltreefeature usage/nodes only,noapply/predict.',train='Afterpublicalgebra published,onlyown273fit y/weight/NET/provenance. Verifyoriginalprefix +16sourcecontextcopies/positive risk additions;describe weightedSTOP/CONT supports. Selector bit labels inherited fromsameid physicalNET andoriginalpublicNETformula,notindependentcausalproof. No data/weights orfields changed in275.',next_basis='Onlyowntraining/publicsemantics tofix ONEfinite nextqualification,notouterdateIDseed or margin-sweep selection. No newstrategy orfeaturetransform now.',count='Current124;actual0newfits. Historical6continuations8285updates/604.703actual/678.156cap separate.',limits='Same543guard/parentown12/internalcal4/exposed16dates/relatedprefixseed,queryproxy notwholeweek/time. No final8/physics/grid/paper/FRAME/install;feeNETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
 paths=[W/'run.py',O/'design.json',O/'input_publication.json',Q/'NEXT_PROTOCOL_PLAN.json',Q/'matched271_changes.json',Q/'matched271_summary.json',Q/'fit_receipts.json',Q/'search_metrics.json',META,F/'augmentation_provenance.json',F/'model_references.json',F/'qualified_freeze.json',*list((Q/'models').glob('*.pkl')),*list((Q/'calibration_predictions').glob('*.npz')),*list((Q/'outer_predictions').glob('*.npz')),*list((OLD/'cutoffs').glob('*.json')),*list((OLD/'calibration_predictions').glob('*.npz')),*list((OLD/'outer_predictions').glob('*.npz')),*list((F/'qualified_inputs').glob('*_fit.npz')),*list((ORIG/'qualified_inputs').glob('*_fit.npz'))]
 for ref in rd(F/'model_references.json'):paths.append(R/ref['model_relative_to_R'])
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID275_SOURCE_FROZEN')

def public():
 verify();h=published('source');assert not (O/'public_started.json').exists();save('public_started.json',dict(source_published=h));meta=rd(META);idx={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};changes=rd(Q/'matched271_changes.json');decomp=[];pools=[]
 for ref in rd(F/'model_references.json'):
  p=ref['pool'];fold,seed=[int(v[1:]) for v in p.split('_')];new=pickle.loads((Q/'models'/f'{p}.pkl').read_bytes());old=pickle.loads((R/ref['model_relative_to_R']).read_bytes());oldt=rd(OLD/'cutoffs'/f'{p}.json')['tau'];newt=new['tau'];oldp=np.load(OLD/'outer_predictions'/f'{p}.npz');newp=np.load(Q/'outer_predictions'/f'{p}.npz');oldc=np.load(OLD/'calibration_predictions'/f'{p}.npz');newc=np.load(Q/'calibration_predictions'/f'{p}.npz');assert np.array_equal(oldp['ids'],newp['ids']) and np.array_equal(oldc['ids'],newc['ids']);where={int(v):j for j,v in enumerate(newp['ids'])};structures=[]
  for label,b in [('old',old),('new',new)]:
   count=collections.Counter();treesbit=nodes=leaves=0
   if not b['KEEP']:
    for t in b['model'].estimators_:
     ff=t.tree_.feature;count.update(map(int,ff[ff>=0]));treesbit+=int(np.any(ff==39));nodes+=int(t.tree_.node_count);leaves+=int((ff<0).sum())
   structures.append(dict(version=label,trees_using_context=treesbit,nodes=nodes,leaves=leaves,feature_split_counts={str(k):v for k,v in sorted(count.items())}))
  pools.append(dict(pool=p,KEEP=new['KEEP'],changed_model=ref['changed'],old_tau=oldt,new_tau=newt,tau_delta=newt-oldt,old_limiting_ids=[] if new['KEEP'] else list(map(int,oldc['ids'][oldc['score']==oldt])),new_limiting_ids=[] if new['KEEP'] else list(map(int,newc['ids'][newc['score']==newt])),structure=structures))
  for r in changes:
   if r['fold']!=fold or r['seed']!=seed:continue
   mid=idx[r['sample_id'],r['position']];j=where[mid];ds=float(newp['forest_score'][j]-oldp['forest_score'][j]);dt=float(newt-oldt);oldm=float(oldp['forest_score'][j]-oldt);newm=float(newp['forest_score'][j]-newt);assert abs((newm-oldm)-(ds-dt))<1e-14
   decomp.append(dict(pool=p,id=mid,seed=seed,date=r['date'],call=r['call'],position=r['position'],old_continue=r['old_continue'],new_continue=r['new_continue'],old_score=float(oldp['forest_score'][j]),new_score=float(newp['forest_score'][j]),old_tau=oldt,new_tau=newt,score_component=ds,cutoff_component=-dt,old_margin=oldm,new_margin=newm,margin_delta=newm-oldm,dominant='score' if abs(ds)>abs(dt) else 'cutoff' if abs(dt)>abs(ds) else 'tie'))
 assert len(decomp)==26 and len(pools)==12;save('public_margin_decomposition.json',decomp);save('public_pool_structure.json',pools);save('public_freeze.json',{n:sha(O/n) for n in ['public_margin_decomposition.json','public_pool_structure.json']});archive('public');print('GRID275_PUBLIC26_ALGEBRA_NO_HYBRID_POLICY')

def analyze():
 verify();h=published('public');assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(public_published=h));prov=rd(F/'augmentation_provenance.json');risk=[]
 for p in rd(F/'model_references.json'):
  pool=p['pool'];z=np.load(F/'qualified_inputs'/f'{pool}_fit.npz');old=np.load(ORIG/'qualified_inputs'/f'{pool}_fit.npz');n=len(old['ids']);assert len(z['ids'])==n+int(z['is_augmented'].sum());parts=[]
  for stop in [True,False]:
   for positive in [True,False]:
    mm=(z['domain_STOP']==stop)&((z['y']>0)==positive);parts.append(dict(domain_STOP=stop,positive=positive,rows=int(mm.sum()),risk=float(z['weight'][mm].sum())))
  for a in [r for r in prov if r['pool']==pool]:
   j,k=a['original_index'],a['new_index'];assert z['ids'][k]==old['ids'][j] and z['y'][k]==old['y'][j]==1 and z['actual_NET'][k]==old['actual_NET'][j] and z['weight'][k]==old['weight'][j] and np.array_equal(z['X32'][k,:39],old['X32'][j,:39]) and z['X32'][k,39]==1 and old['X32'][j,39]==0
  risk.append(dict(pool=pool,KEEP=p['KEEP'],added=int(z['is_augmented'].sum()),old_total_risk=float(old['weight'].sum()),new_total_risk=float(z['weight'].sum()),added_risk=float(z['weight'][n:].sum()),new_positive_fraction=float(z['weight'][z['y']>0].sum()/z['weight'].sum()),partitions=parts))
 decomp=rd(O/'public_margin_decomposition.json');structure=rd(O/'public_pool_structure.json');byseed=[];changes=rd(Q/'matched271_changes.json')
 for seed in range(3):
  rows=[r for r in decomp if r['seed']==seed];cr=[r for r in changes if r['seed']==seed];byseed.append(dict(seed=seed,records=len(rows),score_dominant=sum(r['dominant']=='score' for r in rows),cutoff_dominant=sum(r['dominant']=='cutoff' for r in rows),restored=sum(r['new_continue'] for r in rows),cancelled=sum(not r['new_continue'] for r in rows),raw_query_delta=sum(r['query_delta'] for r in cr)))
 summary=dict(changes26=len(decomp),score_dominant=sum(r['dominant']=='score' for r in decomp),cutoff_dominant=sum(r['dominant']=='cutoff' for r in decomp),changed_cutoffs=sum(r['old_tau']!=r['new_tau'] for r in structure),old_context_trees=sum(r['structure'][0]['trees_using_context'] for r in structure),new_context_trees=sum(r['structure'][1]['trees_using_context'] for r in structure),copies16=sum(r['added'] for r in risk),added_risk=sum(r['added_risk'] for r in risk),per_seed=byseed,newfits0=True,newforward0=True,newpolicy0=True)
 save('ownfit_risk_support.json',risk);save('diagnosis_summary.json',summary);verify();archive('analyzed');print('GRID275_READONLY_DIAGNOSED',json.dumps(summary))
if __name__=='__main__':globals()[sys.argv[1]]()
