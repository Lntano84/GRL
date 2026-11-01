"""Finite saved activation/support diagnosis. No model imports or new predictions."""
from pathlib import Path
import ast,collections,hashlib,json,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid292_activation_support';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID292_activation_support';PREV=R/'outputs/grid291_control_preparation'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*args):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*args])
def archive(phase,paths):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in paths:
  rel=p.relative_to(O) if p.is_relative_to(O) else Path(p.name);dest=d/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 save(d/'archive_manifest.json',{p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()})
def published(phase,manifest):
 head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 for rel,h in manifest.items():
  p=O/rel;assert sha(p)==h
  assert hashlib.sha256(git('show',head+':'+(A/phase/rel).relative_to(I).as_posix())).hexdigest()==h
 return head
def verify():
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
def prepare():
 assert not O.exists();assert read(PREV/'runtime_complete.json')['passed_delivery'] and not read(PREV/'runtime_complete.json')['benchmark_scope_complete']
 # Resolve original input directory from the numerical source constant without importing it.
 source=R/'work/grid261_bounded_basis/run.py';tree=ast.parse(source.read_text(encoding='utf8'));q=None
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='Q' for t in node.targets):
   assert isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Div) and isinstance(node.value.left,ast.Name) and node.value.left.id=='R';q=R/ast.literal_eval(node.value.right)
 assert q and q.name=='grid260_bounded_basis_plan'
 O.mkdir();review=read(PREV/'saved_partial_review.json');assert review['passed_saved_data'] and len(review['matched_pairs'])==93
 design=dict(id='GRID292',scope='One saved activation and ownfit public-support diagnosis; zero fitting/model scoring/NNheadforward/physics/alternative returns.',upstream_plan=read(PREV/'NEXT_PROTOCOL_PLAN.json'),train_directory=q.relative_to(R).as_posix(),original_Q_constant_source=source.relative_to(R).as_posix(),all_pools=sorted({r['pool'] for r in review['records']}),public_fields='All saved native invocation stages, all original decision rows and fixed matched parent rows; first38 actual residual coordinates tanh ONCE; train260 already tanh. Preserve b/bit and all repeat multiplicities. Per-pool min/max ALL first39 coordinates for ownfit ALL, domain_STOP, domain_CONT, no fitted distance/cutoff.',labels='Only after all public membership/descriptors publication: ownfit y/actual_NET/weight/source_mass, by fixed domain and public role. No actual teacher or outer targets, no train rescore.',stopping='One complete explanation then close this diagnosis. No automatic learning; if no actionable new information, stop branch rather than additional interface or tree repairs.',operation_caps=dict(new_fits=0,new_model_predictions=0,new_outer_strategy_scores=0,new_physics=0))
 save(O/'design.json',design)
 paths=[W/'run.py',O/'design.json',PREV/'NEXT_PROTOCOL_PLAN.json',PREV/'runtime_complete.json',PREV/'saved_partial_review.json',PREV/'model_references.json',PREV/'raw_records_manifest.json',source,R/'outputs/grid113_relational_learning/train_metadata.json',q/'qualified_freeze.json']
 for folder in ['fit_inputs','calibration_inputs']:paths+=sorted((q/folder).glob('*.npz'))
 refs=read(PREV/'model_references.json')
 paths +=[R/ref['residual_model'] for ref in refs]
 for row in review['records']:
  folder=PREV/'runs'/f"r{row['repeat']}__{row['week']}__{row['arm']}__{row['pool']}"
  paths+=sorted(folder.glob('call[0-9][0-9][0-9].json'))
 partial=PREV/'runs/r1__2037-03-02_100__BOUNDED261__f0_s2';paths+=sorted(partial.glob('call[0-9][0-9][0-9].json'))
 paths+=[partial/'failure.json'];save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(set(paths))});archive('source',[W/'run.py',O/'design.json',O/'source_freeze.json']);print('GRID292_SOURCE_FROZEN',len(set(paths)),flush=True)
def ranges(x):return dict(rows=len(x),min=x[:,:39].min(axis=0).tolist() if len(x) else None,max=x[:,:39].max(axis=0).tolist() if len(x) else None)
def public():
 assert not (O/'public_started.json').exists();head=published('source',{'source_freeze.json':sha(O/'source_freeze.json'),'design.json':sha(O/'design.json')});verify();save(O/'public_started.json',dict(source_published=head,new_model_predictions=0))
 D=read(O/'design.json');Q=R/D['train_directory'];review=read(PREV/'saved_partial_review.json');records=review['records']; refs={r['pool']:r for r in read(PREV/'model_references.json')};own={};meta=read(R/'outputs/grid113_relational_learning/train_metadata.json')
 # Explicitly whitelist public NPZ fields; no y/NET/weight/source_mass reads here.
 for pool in D['all_pools']:
  with np.load(Q/'fit_inputs'/f'{pool}.npz') as z:
   X=z['X'].copy();ids=z['ids'].copy();dom=z['domain_STOP'].copy();b=z['b'].copy();keep=bool(z['fallback_KEEP'][0])
  assert X.shape==(len(ids),40) and np.isfinite(X).all() and np.array_equal(X[:,38],b) and (abs(X[:,:38])<=1).all()
  own[pool]=dict(KEEP=keep,rows=len(ids),STOP_rows=int(dom.sum()),CONT_rows=int((~dom).sum()),ids=ids.tolist(),dates=sorted({meta[int(k)]['date'] for k in ids}),physical_calls=sorted({meta[int(k)]['physical_state_group'] for k in ids}),ranges_ALL=ranges(X),ranges_STOP=ranges(X[dom]),ranges_CONT=ranges(X[~dom]))
  (O/'public_fit').mkdir(exist_ok=True);np.savez_compressed(O/'public_fit'/f'{pool}.npz',X=X,ids=ids,domain_STOP=dom,b=b,fallback_KEEP=np.array([keep]))
 actual=[];native=[];parent=[];partial=[]
 sources=[(row,False) for row in records]+[(dict(index=186,arm='BOUNDED261',pool='f0_s2',repeat=1,week='2037-03-02_100'),True)]
 for row,ispartial in sources:
  folder=PREV/'runs'/f"r{row['repeat']}__{row['week']}__{row['arm']}__{row['pool']}"
  for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
   c=read(path)
   for d in c['learned_decisions']:
    item=dict(index=row['index'],arm=row['arm'],pool=row['pool'],repeat=row['repeat'],week=row['week'],step=c['step'],state_hash=c['state_hash'],input_hash=c['input_hash'],batch=d['batch'],action=d['action'],optional_extension=d['optional_extension'],next_ids=d['next_ids'],next_queries=len(d['next_ids']),source_call=path.relative_to(R).as_posix(),partial=ispartial)
    if 'native_q' not in d: item['engineering_fallback']=d.get('engineering_fallback')
    else:item['native_q']=d['native_q']
    native.append(item)
    if not d['optional_extension']:continue
    assert d['native_q'][1]<=d['native_q'][0]
    item.update(parent_b=d['parent_b'],parent_CONT=d['parent_CONT'],residual_KEEP=d['residual_KEEP'])
    if row['arm']=='PARENT208':parent.append(item);continue
    X0=np.array(d['residual_input40'],dtype='float64');assert X0.shape==(40,) and X0[38]==d['parent_b'] and X0[39]==1
    X=X0.copy();X[:38]=np.tanh(X[:38]);logit=d['residual_logit'];assert bool(d['action'])==bool(d['parent_CONT'] and logit>=0)
    item.update(X40_pre_tanh=X0.tolist(),X40_bounded=X.tolist(),saved_logit=logit,saved_correction=logit-d['parent_b'],alpha=d['alpha'],extra_cancel=bool(d['parent_CONT'] and not d['action']),public_support={})
    for group in ['ALL','STOP','CONT']:
     r=own[row['pool']]['ranges_'+group]
     item['public_support'][group]=None if not r['rows'] else [k for k in range(39) if X[k]<r['min'][k] or X[k]>r['max'][k]]
    item['public_signature']=hashlib.sha256((row['pool']+'|'+c['state_hash']+'|'+str(d['batch'])+'|'+str(d['next_ids'])).encode()+X.tobytes()).hexdigest()
    (partial if ispartial else actual).append(item)
 assert len(actual)==len(parent)==329 and len({r['pool'] for r in actual})==12
 for r in actual:
  match=[p for p in parent if (p['pool'],p['repeat'],p['week'],p['step'],p['batch'])==(r['pool'],r['repeat'],r['week'],r['step'],r['batch'])]
  assert len(match)==1 and match[0]['state_hash']==r['state_hash'] and match[0]['parent_b']==r['parent_b'] and match[0]['parent_CONT']==r['parent_CONT']
 stats=dict(accepted_BOUNDED_runs=93,all_pools=D['all_pools'],native_decision_rows=len([r for r in native if not r['partial']]),bounded_native_decision_rows=sum(r['arm']=='BOUNDED261' and not r['partial'] for r in native),extension_invocations=len(actual),parent_positive=sum(r['parent_CONT'] for r in actual),parent_nonpositive=sum(not r['parent_CONT'] for r in actual),residual_continues=sum(bool(r['action']) for r in actual),extra_cancellations=sum(r['extra_cancel'] for r in actual),cancelled_known_queries=sum(r['next_queries'] for r in actual if r['extra_cancel']),unique_public_invocations=len({r['public_signature'] for r in actual}),unique_cancelled_public_states=len({r['public_signature'] for r in actual if r['extra_cancel']}),known_batch_lengths=dict(sorted(collections.Counter(r['next_queries'] for r in actual).items())),outside_any_fit_coordinate={group:sum(bool(r['public_support'][group]) for r in actual) for group in ['ALL','STOP','CONT']},per_pool={p:dict(rows=sum(r['pool']==p for r in actual),parent_positive=sum(r['pool']==p and r['parent_CONT'] for r in actual),extra_cancel=sum(r['pool']==p and r['extra_cancel'] for r in actual),KEEP=own[p]['KEEP']) for p in D['all_pools']},partial_rows_separate=len(partial))
 save(O/'public_actual_rows.json',actual);save(O/'public_parent_rows.json',parent);save(O/'public_native_rows.json',native);save(O/'public_partial_rows.json',partial);save(O/'ownfit_public_support.json',own);save(O/'activation_public_summary.json',stats)
 paths=[*sorted((O/'public_fit').glob('*.npz')),*[O/n for n in ['public_actual_rows.json','public_parent_rows.json','public_native_rows.json','public_partial_rows.json','ownfit_public_support.json','activation_public_summary.json','public_started.json']]]
 save(O/'public_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in paths});archive('public',paths+[O/'public_freeze.json']);print('GRID292_PUBLIC_PASSED',json.dumps(stats,ensure_ascii=False),flush=True)
def join():
 assert not (O/'ownfit_join_started.json').exists();head=published('public',read(O/'public_freeze.json'));verify();save(O/'ownfit_join_started.json',dict(all_public_published=head,only_ownfit_labels=True))
 D=read(O/'design.json');Q=R/D['train_directory'];actual=read(O/'public_actual_rows.json');own=read(O/'ownfit_public_support.json');meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');joined=[]
 for pool in D['all_pools']:
  with np.load(Q/'fit_inputs'/f'{pool}.npz') as z:
   ids=z['ids'].copy();y=z['y'].copy();net=z['actual_NET'].copy();weight=z['weight'].copy();mass=z['source_mass'].copy();dom=z['domain_STOP'].copy()
  pub=np.load(O/'public_fit'/f'{pool}.npz');assert ids.tolist()==own[pool]['ids'] and np.array_equal(dom,pub['domain_STOP']);assert np.array_equal(y,net>0)
  for domain,mask in [('STOP',dom),('CONT',~dom)]:
   total=float(weight[mask].sum());positive=mask&(y>0);negative=mask&~(y>0)
   joined.append(dict(pool=pool,KEEP=own[pool]['KEEP'],domain=domain,rows=int(mask.sum()),positive=int(positive.sum()),negative=int(negative.sum()),positive_risk=float(weight[positive].sum()),negative_risk=float(weight[negative].sum()),positive_risk_fraction=float(weight[positive].sum()/total) if total else None,positive_physical_calls=len({meta[int(i)]['physical_state_group'] for i in ids[positive]}),positive_dates=sorted({meta[int(i)]['date'] for i in ids[positive]}),sum_source_mass=float(mass[mask].sum())))
 summary=read(O/'activation_public_summary.json');margins={p:dict(min_saved_logit=min(r['saved_logit'] for r in actual if r['pool']==p),max_saved_logit=max(r['saved_logit'] for r in actual if r['pool']==p),min_saved_parent_b=min(r['parent_b'] for r in actual if r['pool']==p),max_saved_parent_b=max(r['parent_b'] for r in actual if r['pool']==p),positive_saved_correction=sum(r['pool']==p and r['saved_correction']>0 for r in actual),negative_saved_correction=sum(r['pool']==p and r['saved_correction']<0 for r in actual),fit_dates=own[p]['dates']) for p in D['all_pools']}
 save(O/'ownfit_label_support.json',joined);save(O/'saved_margin_support.json',margins)
 save(O/'review.json',dict(passed=True,all_public_before_ownfit_labels=head,actual_invocations=len(actual),matched_parent_invocations=329,all12pools=True,KEEP_retained=True,partial_separate=True,actual_first38_tanh_once=True,train260_already_tanh=True,new_model_predictions=0,new_outer_scores_or_returns=0,new_fits=0,new_physics=0,public_fields_only_before_join=True,source_freeze_verified=True))
 print('GRID292_OWNFIT_JOINED',json.dumps(dict(activation=summary,ownfit=joined),ensure_ascii=False),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
