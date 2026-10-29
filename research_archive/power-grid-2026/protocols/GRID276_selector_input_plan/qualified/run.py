"""GRID276 qualification only: fixed selector-bit omission, no fit/predict."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,collections,ast,importlib.util
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid276_selector_input_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID276_selector_input_plan';Q=R/'outputs/grid275_forest_direction';F=R/'outputs/grid273_stop_context_plan';N=R/'outputs/grid274_stop_context';META=R/'outputs/grid113_relational_learning/train_metadata.json';ROLES=R/'outputs/grid241_parent_prior_plan/qualified_pools.json';RUNTIME=W/'runtime276.py'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p
def path_constant(p,key):
 n=next(n.value for n in ast.parse(p.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets));assert isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div) and isinstance(n.left,ast.Name) and isinstance(n.right,ast.Constant);return {'R':R,'REPO':REPO}[n.left.id]/n.right.value

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 if phase=='qualified':shutil.copytree(O/'qualified_inputs',d/'qualified_inputs')
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')

def prepare():
 assert not O.exists() and rd(Q/'NEXT_PROTOCOL_PLAN.json')['id']=='GRID276';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 arQ=path_constant(R/'work/grid275_forest_direction/run.py','A');arF=path_constant(R/'work/grid273_stop_context_plan/run.py','A');arN=path_constant(R/'work/grid274_stop_context/run.py','A')
 for n in ['NEXT_PROTOCOL_PLAN.json','public_semantics_addendum.json','complete.json','decision.json']:assert git('show',h+':'+(arQ/'audited'/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 for rel,s in rd(F/'qualified_freeze.json').items():assert sha(F/rel)==s and hashlib.sha256(git('show',h+':'+(arF/'qualified'/rel).relative_to(REPO).as_posix())).hexdigest()==s
 for ph,fr in [('trained','trained_freeze.json'),('scored','scored_freeze.json')]:
  for rel,s in rd(N/fr).items():assert sha(N/rel)==s and hashlib.sha256(git('show',h+':'+(arN/ph/rel).relative_to(REPO).as_posix())).hexdigest()==s
 O.mkdir();save('input_publication.json',dict(passed=True,git_sha=h,qualified273_and_fixed274weights_scores_published=True,actual273_archive=str(arF),actual274_archive=str(arN)))
 save('design.json',dict(id='GRID276',scope='Onlyqualification ONE39DconditionalNETforest;0model/scalarfit/realprobability/newouterstrategy.',basis=rd(Q/'NEXT_PROTOCOL_PLAN.json')['basis'],input='Useexact273fit/cal rows incl16copies;X=X40[:,:39],X32=X32_40[:,:39].Keep original40D arrays asprovenance,NET/y/weights/ids/roles/source_mass/KEEP unchanged,38tanhpublic+b unchanged. No scaler/reweight/dedup/sampledeletion.',collision='Report float64andfloat32 exactgroups/duplicate rowcounts/positivepaircollapse anduniquephysicalIDs. Any opposite-label exactgroup halts qualification withallrecords preserved;norescue byaddbackbit/deletingrecords.',runtime='Onlyas_inputshapeassert39;other267make_classifier/positive_score/protect_cutoff/apply_gate ASTmath unchanged,sklearn1.7.2. Constructoronly notfit/predict. Old40Dforest MUSTNOTreceive39D.',future='Afterqualificationpublished,separate277all11newforests+11same543positivecutoffs=22/one600s,1KEEP,704newtrees/0NNtheta. All11activeinputdimschanged;noincompatibleweightreuse. Same64depth4leaf10etcandseeds0/1/2.',comparison='Mainmatched274(same273samples/weights/16copyexposure),271historicalforest/261qualityanchor/GATE retained. Future249outerX first38tanh cut39,entry261 unchanged. Allmodelsfirstpublished/allpredictionspublishedbeforeoldoutcomes,all3seeds16dates/adversecases.',counts='Current124/276zero/future277+22=146;16copiescorrelated/500->543guardexposure/prior6continuations8285updates/NN604.703actual/678.156cap separate.',limits='Selector maycarrypredictiveselectiondomaininformation;removalnotguaranteedeffective/causalproof. Functionclass/capacity/randomfeaturechoiceschange. Parentown12/internalcal4/exposedcorrelateddata,notOOFblind/fullweek/time. No final8/physics/grid/paper/FRAME/install;feeNETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
 paths=[*W.glob('*.py'),O/'design.json',O/'input_publication.json',Q/'NEXT_PROTOCOL_PLAN.json',Q/'public_semantics_addendum.json',F/'qualified_freeze.json',F/'augmentation_provenance.json',F/'runtime267.py',*list((F/'qualified_inputs').glob('*.npz')),META,ROLES,*list((N/'models').glob('*.pkl')),*list((N/'outer_predictions').glob('*.npz'))]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID276_SOURCE_FROZEN')

def collision(z,key):
 groups=collections.defaultdict(list)
 for j,x in enumerate(z[key]):groups[x.tobytes()].append(j)
 dup=[v for v in groups.values() if len(v)>1];mixed=[v for v in dup if len(set(map(int,z['y'][v])))>1]
 return dict(exact_duplicate_groups=len(dup),extra_duplicate_rows=sum(len(v)-1 for v in dup),duplicate_rows=sum(len(v) for v in dup),opposite_label_groups=len(mixed),opposite_label_indices=mixed)

def qualify():
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h and git('show',h+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();assert not (O/'qualification_started.json').exists();save('qualification_started.json',dict(source_published=h,fit0=True));(O/'qualified_inputs').mkdir();meta=rd(META);roles={r['pool']:r for r in rd(ROLES)};records=[];total=active=guard=pairs=conflicts=0
 for file in sorted((F/'qualified_inputs').glob('*_fit.npz')):
  pool=file.name[:-8];keep=bool(np.load(F/'qualified_inputs'/f'{pool}_base.npz')['fallback_KEEP'][0]);entry=dict(pool=pool,KEEP=keep)
  for role in ['fit','cal']:
   z=np.load(F/'qualified_inputs'/f'{pool}_{role}.npz');out={k:z[k].copy() for k in z.files};out['X40_original']=z['X'].copy();out['X32_40_original']=z['X32'].copy();out['X']=z['X'][:,:39].copy();out['X32']=z['X32'][:,:39].copy();assert out['X'].shape==(len(z['ids']),39) and np.isfinite(out['X']).all() and np.isfinite(out['X32']).all() and np.array_equal(out['X32'],out['X'].astype(np.float32))
   for k in z.files:
    if k not in ['X','X32']:assert out[k].dtype==z[k].dtype and out[k].tobytes()==z[k].tobytes()
   assert np.array_equal(out['X'],out['X40_original'][:,:39]) and np.array_equal(out['X32'],out['X32_40_original'][:,:39]);dates={meta[int(i)]['date'] for i in out['ids']}
   if role=='fit':assert dates<=set(roles[pool]['fit_dates']) and not dates&set(roles[pool]['cal4_dates']+roles[pool]['outer_dates']) and (out['weight']>0).all() and np.isfinite(out['weight']).all()
   else:assert dates<=set(roles[pool]['fit_dates']+roles[pool]['cal4_dates']) and not dates&set(roles[pool]['outer_dates']) and (out['y']==1).all()
   c64=collision(out,'X');c32=collision(out,'X32');entry[role]=dict(rows=len(out['ids']),unique_metadata_ids=len(set(map(int,out['ids']))),unique_calls=len({meta[int(i)]['physical_state_group'] for i in out['ids']}),float64=c64,float32=c32);conflicts+=c64['opposite_label_groups']+c32['opposite_label_groups'];np.savez_compressed(O/'qualified_inputs'/f'{pool}_{role}.npz',**out)
   if role=='fit':total+=len(out['ids']);active+=0 if keep else len(out['ids'])
   else:guard+=len(out['ids'])
  shutil.copy2(F/'qualified_inputs'/f'{pool}_base.npz',O/'qualified_inputs'/f'{pool}_base.npz');records.append(entry)
 for r in rd(F/'augmentation_provenance.json'):
  z=np.load(O/'qualified_inputs'/f"{r['pool']}_fit.npz");a,b=r['original_index'],r['new_index'];assert np.array_equal(z['X'][a],z['X'][b]) and np.array_equal(z['X32'][a],z['X32'][b]) and z['y'][a]==z['y'][b]==1 and z['weight'][a]==z['weight'][b];pairs+=1
 assert total==2875 and active==2741 and guard==543 and pairs==16 and len(records)==12
 def fnodes(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef)}
 old,new=fnodes(F/'runtime267.py'),fnodes(RUNTIME)
 for key in ['make_classifier','positive_score','protect_cutoff','apply_gate']:assert old[key]==new[key]
 spec=importlib.util.spec_from_file_location('grid276_runtime',RUNTIME);RT=importlib.util.module_from_spec(spec);spec.loader.exec_module(RT);import sklearn;assert sklearn.__version__=='1.7.2'
 for seed in range(3):clf=RT.make_classifier(seed);assert all(clf.get_params()[k]==v for k,v in RT.PARAMS.items()) and clf.get_params()['random_state']==seed
 for file in (O/'qualified_inputs').glob('*_fit.npz'):
  z=np.load(file);assert np.array_equal(RT.as_input(z['X']),z['X32'])
 shutil.copy2(RUNTIME,O/'runtime276.py');save('runtime_freeze.json',dict(sha256=sha(RUNTIME),original_functions_unchanged=True,new_shape_only39=True,sklearn_version=sklearn.__version__,fit_calls=0,predict_calls=0))
 arN=Path(rd(O/'input_publication.json')['actual274_archive']);refs=[]
 for e in records:
  p=e['pool'];refs.append(dict(pool=p,KEEP=e['KEEP'],future_newfit=not e['KEEP'],old40_model_relative_to_R=f'outputs/grid274_stop_context/models/{p}.pkl',old40_model_sha256=sha(N/'models'/f'{p}.pkl'),old40_model_git_path=(arN/'trained/models'/f'{p}.pkl').relative_to(REPO).as_posix(),old40_prediction_relative_to_R=f'outputs/grid274_stop_context/outer_predictions/{p}.npz',old40_prediction_sha256=sha(N/'outer_predictions'/f'{p}.npz'),old40_prediction_git_path=(arN/'scored/outer_predictions'/f'{p}.npz').relative_to(REPO).as_posix()))
 save('model_references.json',refs);save('qualified_pools.json',records);save('qualification.json',dict(passed=conflicts==0,fit_rows=total,active_rows=active,guard_rows=guard,context_pairs_collapsed=pairs,opposite_label_groups=conflicts,new_fits=0,new_probabilities=0,new_outer_policy=0,allfields_exceptXshape_byteunchanged=True,no_dedup_or_weightmerge=True,forest_recipe_unchanged=True,future_forests=11,future_scalars=11,future_cap_s=600,sklearn_version=sklearn.__version__))
 save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'qualified_inputs').glob('*.npz')),O/'runtime276.py',O/'runtime_freeze.json',O/'qualification.json',O/'model_references.json',O/'qualified_pools.json']});archive('qualified');print('GRID276_QUALIFICATION',json.dumps(rd(O/'qualification.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
