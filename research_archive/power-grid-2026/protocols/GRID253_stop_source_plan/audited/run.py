"""Metadata/ancestor and complete fixed source-pipeline feasibility, zero fits."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,itertools,collections
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid253_stop_source_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID253_stop_source_plan';FOLD=R/'outputs/grid121_date_transfer';META=R/'outputs/grid113_relational_learning/train_metadata.json';SOURCE=[R/'outputs/grid168_one_sided_transfer']+[R/f'outputs/grid170_remaining_date_folds/fold{i}' for i in [1,2,3]];PLAN=R/'outputs/grid241_parent_prior_plan';INPUT=R/'outputs/grid251_qgap_plan';BASE=R/'outputs/grid208_public_amount'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
def prepare():
 assert not O.exists() and read(R/'outputs/grid252_qgap_residual/complete.json')['passed'];O.mkdir();save('design.json',dict(id='GRID253',scope='Read-onlycurrentSTOPmodeldateexposure/catalog/completeonefixedsourcepipeline countsandfeasibility;0newfit/modelbuild/forward/optimizer/heldscore.',
 reuse='191selectiondiagnosisand235correctedancestry areoldfindings;onlyverifycurrentlineage/availableweights/newfiniteoperations. Notclaimsourcechangealreadyworks.',
 reference='Frozen4outergroup partition/all3seeds. Forgateouterf andrecorddateingroupk!=f use referenceFACTOR trainedoncomplementofgroups{f,k}:8dates. Sixunorderedpairs x3seeds=18uniqueweights,36directeduses. Other12dateoutermodelmayincludeouterf,notallowed.',
 next_unique_probe='Data-sourceprobe usesretained25041paramcorrection class,NOT252newgapinput. TrainSTOP/CONT bitfromunseen-datereferenceNN8,canonical38andparentb fromoriginalf pipeline;deployoriginalfNN12/GATE unchanged. Primarycached250,allseedsno refittingcontrol.',
 limits='RecordwiseSTOPreferenceNNcanexcludeownlabel/outerf;frozenoriginalfparentNNand208headpriorstillown12inclrecord. Innercal4notwholepipeline independent. Explicitconditional-sourceprobe,notfullynested/blind/pureonefactor. No parentheadretraining.',
 counts='Onefixedpipeline:18referenceNNfits,dataqualificationthenatmost11linear+11positive scalar=atmost40total supervised/calibration operations. Two logicallydependent fitcomponents18and<=22,eachglobal600s,no candidate/parametergrid. Allcounts/windowsexplicit;no hiding splitbigsweep.',
 feasibility='18serialold7250updateNNs exceed600s. Fixedproposal6singlethreadworkers/oneorchestrator/600sglobaldeadline,needsseparatezerooptimizerresource/interface/init/data preflight beforefit;historyruntimeonlyestimate,notguarantee. No update/seed/date changes iftimeout.',
 boundaries='No newphysical/data/final8/grid/paper/FRAME/install;deployedcontroller/order/aliases/station128/64/stagnation2/.9/fee/NETlabelsfixed. Allsource/models/predictionspublishbarriers,alladverse retained.'))
 roots=[R/'outputs/grid113_relational_learning',*SOURCE,R/'outputs/grid182_deployment_bundle',R/'outputs/grid165_factor_cost_routing'];candidates=[]
 for root in roots:
  for pattern in ['FACTOR_seed*.pt','ZERO_MAIN_seed*.pt']:candidates+=list((root/'models').glob(pattern))
 paths=[W/'run.py',O/'design.json',META,FOLD/'design.json',R/'outputs/grid191_gate_diagnosis/summary.json',R/'outputs/grid235_calibration_ancestry/qualification.json',R/'outputs/grid235_calibration_ancestry/count_correction.json',R/'outputs/grid252_qgap_residual/NEXT_PROTOCOL_PLAN.md',R/'outputs/grid250_context_residual/complete.json',PLAN/'qualified_pools.json',R/'work/grid168_one_sided_transfer/run.py',R/'work/grid192_gate_support/run.py',R/'work/grid190_extension_gate/run.py',R/'outputs/grid113_relational_learning/train_public.pkl']+candidates
 for root in SOURCE:paths+=[root/'fold_arrays.npz',root/'metadata.json',root/'design.json',root/'fit_receipts.json']
 for root in [roots[0],roots[-2],roots[-1]]:paths+=[root/'design.json']
 for r in read(PLAN/'qualified_pools.json'):paths += [INPUT/'fit_inputs'/f"{r['pool']}.npz",INPUT/'outer_parent_public'/f"{r['pool']}.npz",R/'outputs/grid192_gate_support/fit_features'/f"{r['pool']}.npz",R/r['parent_scaler']]
 paths+=[R/'outputs/grid208_public_amount/gate_predictions.json'];save('catalog_paths.json',[p.relative_to(R).as_posix() for p in candidates]);paths+=[O/'catalog_paths.json'];save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)});verify();archive('source');print('GRID253_SOURCE_FROZEN',len(candidates))
def qualify():
 verify();assert not (O/'qualification_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();save('qualification_started.json',dict(sourcepublished=head,newfits=0,newforwards=0));folds=read(FOLD/'design.json')['folds'];meta=read(META);all_dates={m['date'] for m in meta};groups={r['fold']:set(r['held_dates']) for r in folds};assert len(all_dates)==16 and all(len(g)==4 for g in groups.values()) and set.union(*groups.values())==all_dates and sum(map(len,groups.values()))==16
 catalog=[];times=[];orig={}
 for f,root in enumerate(SOURCE):
  z=np.load(root/'fold_arrays.npz');dates={meta[int(i)]['date'] for i in z['fit_ids']};assert dates==all_dates-groups[f];orig[f]=set(z['fit_ids'].tolist());recs=read(root/'fit_receipts.json');recs=recs.get('records',recs) if isinstance(recs,dict) else recs
  for r in recs:
   if r['arm']=='FACTOR':times.append(r['fit_wall_s'])
 for rel in read(O/'catalog_paths.json'):
  p=R/rel;root=p.parent.parent;f=next((i for i,x in enumerate(SOURCE) if x==root),None);dates=(all_dates-groups[f]) if f is not None else all_dates;catalog.append(dict(path=rel,sha256=sha(p),fit_dates=sorted(dates),outer_group=f,date_source='fold_arraysfit_ids' if f is not None else 'publishedfull16populationdesign',eligible_unseen_pair=False))
 assert len(times)==12;pairs=[];uses=[];ref_fit_rows=0;ref_score_rows=0
 for f,k in itertools.combinations(range(4),2):
  fit_dates=all_dates-groups[f]-groups[k];assert len(fit_dates)==8;fi=[i for i,m in enumerate(meta) if m['date'] in fit_dates];hi=[i for i,m in enumerate(meta) if m['date'] not in fit_dates];pair=f'p{f}{k}'
  for seed in [0,1,2]:
   reuse=[r['path'] for r in catalog if f'seed{seed}.pt' in r['path'] and not set(r['fit_dates'])&(groups[f]|groups[k])];assert not reuse;node=f'{pair}_s{seed}';pairs.append(dict(node=node,pair=[f,k],seed=seed,fit_dates=sorted(fit_dates),reference_score_dates=sorted(groups[f]|groups[k]),fit_rows=len(fi),score_rows=len(hi),fit_ids=fi,score_ids=hi,reusable_qualified_weights=reuse,new_fit_required=True,updates=7250,batch=32,optimizer_draws=7250*32));ref_fit_rows+=len(fi);ref_score_rows+=len(hi)
  for outer,record in [(f,k),(k,f)]:
   for seed in [0,1,2]:uses.append(dict(gate_pool=f'f{outer}_s{seed}',record_date_group=record,reference_node=f'{pair}_s{seed}',fit_label_dates=sorted(fit_dates),record_dates=sorted(groups[record]),outer_dates=sorted(groups[outer]),record_dates_disjoint=True,outer_dates_disjoint=True))
 assert len(pairs)==18 and len(uses)==36 and ref_fit_rows==ref_score_rows==32238;current=[];fit_positive=0;fit_stop=0;fit_seen=0;held_unseen=0;raw_parent_rows=0;all_parent_states=0
 for r in read(PLAN/'qualified_pools.json'):
  pool=r['pool'];f=int(pool.split('_')[0][1:]);z=np.load(INPUT/'fit_inputs'/f'{pool}.npz');h=np.load(INPUT/'outer_parent_public'/f'{pool}.npz');raw=np.load(R/'outputs/grid192_gate_support/fit_features'/f'{pool}.npz');pos={int(i):j for j,i in enumerate(raw['ids'])};indices=np.array([pos[int(i)] for i in z['ids']]);native=raw['x'][indices,81]<=raw['x'][indices,80];assert np.array_equal(native.astype(float),z['X'][:,39]);assert set(z['ids'].tolist())<=orig[f] and not set(h['ids'].tolist())&orig[f];fit_seen+=len(z['ids']);held_unseen+=len(h['ids']);fit_stop+=int(z['domain_STOP'].sum());fit_positive+=int((z['domain_STOP']&(z['y']>0)).sum());raw_parent_rows+=len(raw['ids']);all_parent_states+=sum(m['date'] not in groups[f] for m in meta);current.append(dict(pool=pool,native_NN_label_dates=sorted(all_dates-groups[f]),all_fit8_records_date_participates=True,fit_rows=len(z['ids']),original_sourceSTOP=int(z['domain_STOP'].sum()),sourceSTOP_positive=int((z['domain_STOP']&(z['y']>0)).sum()),all_held_records_date_excluded=True,held_rows=len(h['ids']),parentprior_still_own12=True))
 assert fit_seen==2851 and held_unseen==4681 and fit_stop==1394 and fit_positive==41 and raw_parent_rows==22653 and all_parent_states==32238;parentref=CURRENT=None
 parentapp=[r for r in read(R/'outputs/grid208_public_amount/gate_predictions.json') if r['gate_continue']];# labels already published; do not fit using them
 # Current source participation is the qualified fact; old diagnostic metrics are copied as history.
 old191=read(R/'outputs/grid191_gate_diagnosis/summary.json');old235=read(R/'outputs/grid235_calibration_ancestry/qualification.json');timing=dict(original12FACTOR_wall_s=times,min=float(min(times)),median=float(np.median(times)),max=float(max(times)),serial18_median_estimate_s=float(18*np.median(times)),sixworkers_threewaves_oldmax_no_overhead_s=float(3*max(times)),not_a_runtimeguarantee=True)
 counts=dict(reference_NN_fits=18,new_forest_head_fits=0,new_readout_linear_fits_max=11,new_positive_scalar_fits_max=11,total_newfits_max=40,reference_fit_component=dict(models=18,global_cap_s=600,workers_fixed_proposal=6,torch_threads_perworker=1),readout_fit_component=dict(models_andscalars_max=22,global_cap_s=600),public_fitdate_normalization_packages=6,normalization_packages_not_supervised_calibration=True,reference_NNscore_rows=32238,original_parent_feature_rows_already_cached=22653,original_parent_feature_missing_forward_upperbound=9585,old_parent_heads_reused=24,new_parent_head_forward_rows_upperbound=2*32238,physicalqueries=0,final8_materialized=False,stages_are_two_dependent_components_of_one_candidate_not_hyperparameter_search=True)
 save('verified_model_catalog.json',catalog);save('reference_nodes.json',pairs);save('reference_uses.json',uses);save('current_ancestry.json',current);save('operation_counts.json',counts);save('historical_runtime.json',timing);save('historical_findings_reused.json',dict(GRID191=old191,GRID235=old235,not_new_experiments=True));save('qualification.json',dict(passed=True,newfits=0,newforwards=0,verified_catalog_checkpoints=len(catalog),qualified_reference_reuse=0,required_reference_nodes=18,directed_uses=36,current_fit_rows_seen_by_nativeNN=2851,current_held_rows_excluded_fromnativeNN=4681,current_fit8STOP_positive=41,current_fit8STOP_total=1394,all_reference_recorddate_outerdate_disjoint=True,full_pipeline_independence_claim=False,required_next='GRID254zerooptimizerinterface/resource/lineage/publicinputsandexact18jobrunnerqualification;NOtraininguntilqualified',resource_execution_feasibility_pending=True));verify();save('complete.json',dict(passed=True,qualification_only=True,newfits=0,newforwards=0,complete_plan_operations_max=40,training_not_started=True));archive('qualified');print('GRID253_QUALIFIED',json.dumps(dict(catalog=len(catalog),reference=18,directed=36,counts=counts,timing=timing)))
if __name__=='__main__':globals()[sys.argv[1]]()
