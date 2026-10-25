"""Read-only population coverage: existing FRAME paths versus FULL teacher paths."""
from pathlib import Path
import collections, gzip, hashlib, importlib.util, json, shutil, sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid153_path_coverage'
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID153_path_coverage'
spec=importlib.util.spec_from_file_location('g153_reference',R/'work/grid140_global_compression/common.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(np.asarray(x,np.float32).tobytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def steps(p):
 with gzip.open(p/'steps.jsonl.gz','rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def signature(rs):return hashlib.sha256(json.dumps([(s['before_hash'],s['action_hash'],s['after_hash'],s['raw_cost'],s['done'],s['complete']) for s in rs],separators=(',',':')).encode()).hexdigest()
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def folders():
 d=read(R/'outputs/grid104_learning_closedloop/design.json');out=[]
 for x in d['selected']:
  scenario=x['scenario'];teacher=R/'outputs/grid100_data_foundation/bulk_runtime/runs'/f'r0__{scenario}__FULL_TEACHER'
  for repeat in [0,1]:out.append(dict(scenario=scenario,date=x['date'],repeat=repeat,frame=R/'outputs/grid104_learning_closedloop/runs'/f'r{repeat}__{scenario}__FRAME_GUARD',teacher=teacher))
 assert len(out)==8;return out
def verify():
 B.verify_trained()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
def prepare():
 assert not O.exists();O.mkdir(parents=True);B.verify_trained()
 assert read(R/'outputs/grid104_learning_closedloop/complete.json')['passed']
 save('design.json',dict(id='GRID153_PATH_COVERAGE',purpose='Coverage of existing FRAME deployment paths by FULL teacher records; no new trajectories, labels, fit or model inference.',
  population='All four previously exposed GRID104 validation scenario_id100 dates; both repeats checked but r0 only counted as four distinct trajectories. All305 original teacher calls used for exact context lookup.',
  comparison='At same scenario/step compare before-state, action, after-state, raw cost. At search calls compare exact float32 observation/base pair and public context against same-step and all existing teacher calls.',
  compatibility='Library hashes and vector layouts checked; shared context comparisons only on exact observation/base pairs. Compare observed shared feedback without requesting missing outcomes.',
  limits='Exact-byte nonoverlap is empirical coverage, not statistical OOD or causal explanation of held-out FULL-teacher errors. Repeats are not new scenarios. No final8, fit, generation, forecast, physics, threshold selection, paper edits or scale.'))
 paths=[R/'outputs/grid104_learning_closedloop'/n for n in ['design.json','review.json','complete.json']]
 paths += [R/'outputs/grid139_global_feasibility/call_ledger.json',R/'outputs/grid28/offline.json']
 for x in folders():
  for f in [x['frame'],x['teacher']]:
   paths += [f/n for n in ['summary.json','metadata.json','steps.jsonl.gz','vectors.npz']]
   for p in sorted(f.glob('call[0-9][0-9][0-9].json')):paths.extend([p,p.with_name(p.stem+'_state.npz')])
 for ref in B.refs():
  p=R/ref['source'];paths.extend([p,p.with_name(p.stem+'_state.npz')])
 paths.append(O/'design.json');save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(set(paths))})
 save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen'));archive('frozen');print('GRID153_FROZEN')
def call_record(p):
 c=read(p)
 with np.load(p.with_name(p.stem+'_state.npz')) as z:
  assert set(z.files)=={'observation','base_action'};assert z['observation'].shape==(4460,) and z['base_action'].shape==(1605,)
  state=digest(z['observation']);base=digest(z['base_action'])
 assert state==c['state_hash'];return c,(state,base)
def check_run(f,checks):
 rs=steps(f);s=read(f/'summary.json');assert len(rs)==s['steps'];assert sum(r['raw_cost'] for r in rs)==s['cost']
 with np.load(f/'vectors.npz') as z:
  assert z['action'].shape==(len(rs),1605) and z['observation'].shape==(len(rs),4460)
  aa,oo=z['action'],z['observation']
  for j,r in enumerate(rs):
   assert r['step']==j+1 and digest(aa[j])==r['action_hash'] and digest(oo[j])==r['after_hash']
   if j:assert r['before_hash']==rs[j-1]['after_hash']
   checks['step_vectors']+=2
 for p in f.glob('call[0-9][0-9][0-9].json'):
  c,k=call_record(p);assert c['state_hash']==rs[c['step']-1]['before_hash'];checks['call_vectors']+=1
 return rs,s
CONTEXT=['order','aliases','representatives','masked_ids','rho_limit','order_mode']
FEEDBACK=['pool_id','rho','reward','strict_admissible','done','illegal','ambiguous','exceptions','combined_hash']
def analyse():
 verify();assert not (O/'review.json').exists();checks=collections.Counter();teacher_lookup=collections.defaultdict(list);by_scenario={}
 for ref in B.refs():
  p=R/ref['source'];c,k=call_record(p);teacher_lookup[k].append((ref,c));by_scenario.setdefault(ref['scenario'],{})[c['step']]=(ref,c,k)
 checks['teacher_calls_indexed']=sum(map(len,teacher_lookup.values()));assert checks['teacher_calls_indexed']==305
 cache={};per=[];callrows=[];repeats={}
 for x in folders():
  f,t=x['frame'],x['teacher'];fm,tm=read(f/'metadata.json'),read(t/'metadata.json');assert fm['library_hashes']==tm['library_hashes'];checks['library_compatibility']+=1
  if t not in cache:cache[t]=check_run(t,checks)
  ts,summary_t=cache[t];fs,summary_f=check_run(f,checks);repeats[(x['scenario'],x['repeat'])]=signature(fs)
  n=min(len(fs),len(ts));diffs={k:next((j+1 for j in range(n) if fs[j][k]!=ts[j][k]),None) for k in ['before_hash','action_hash','after_hash']}
  same_before=sum(fs[j]['before_hash']==ts[j]['before_hash'] for j in range(n));same_action=sum(fs[j]['action_hash']==ts[j]['action_hash'] for j in range(n))
  calls=sorted(f.glob('call[0-9][0-9][0-9].json'));matched=0;ctx=0;shared=0
  for p in calls:
   c,key=call_record(p);hits=teacher_lookup.get(key,[]);other=by_scenario.get(x['scenario'],{}).get(c['step']);same_step=bool(other and key==other[2]);matched+=bool(hits)
   context_ok=False;shared_count=0;choice_equal=None;feedback_ok=None
   if hits:
    context_ok=all(all(c.get(k)==tc.get(k) for k in CONTEXT) for ref,tc in hits)
    assert context_ok;ctx+=1
    truth={r['pool_id']:r for r in hits[0][1]['outcomes']};feedback_ok=True
    for a in c['outcomes']:
     b=truth[a['pool_id']];assert all(a.get(k)==b.get(k) for k in FEEDBACK),(p,a['pool_id']);shared_count+=1
    shared+=shared_count;ref=hits[0][0];choice_equal=c['chosen_pool_id']==ref['frame_choice'];assert choice_equal
   callrows.append(dict(scenario=x['scenario'],date=x['date'],repeat=x['repeat'],source=p.relative_to(R).as_posix(),step=c['step'],state_hash=key[0],base_hash=key[1],teacher_exact_hits=[a['call_id'] for a,b in hits],same_step_teacher_exists=other is not None,same_step_exact_context=same_step,public_context_matches=context_ok,shared_feedback_checked=shared_count,feedback_matches=feedback_ok,frame_choice_reproduced=choice_equal,paid_queries=len(c['outcomes']),M=len(c['order'])))
  per.append(dict(scenario=x['scenario'],date=x['date'],repeat=x['repeat'],frame_steps=len(fs),teacher_steps=len(ts),frame_complete=summary_f['complete'],teacher_complete=summary_t['complete'],frame_cost=summary_f['cost'],teacher_cost=summary_t['cost'],common_horizon=n,same_before_steps=same_before,same_action_steps=same_action,first_difference=diffs,frame_calls=len(calls),teacher_calls=len(by_scenario.get(x['scenario'],{})),exact_teacher_call_matches=matched,context_matches=ctx,shared_feedback_checks=shared))
 for scenario in sorted({x['scenario'] for x in per}):assert repeats[(scenario,0)]==repeats[(scenario,1)];checks['repeat_pairs']+=1
 for scenario in sorted({x['scenario'] for x in per}):
  a=[r for r in callrows if r['scenario']==scenario and r['repeat']==0];b=[r for r in callrows if r['scenario']==scenario and r['repeat']==1]
  assert [{k:v for k,v in r.items() if k not in ['repeat','source']} for r in a]==[{k:v for k,v in r.items() if k not in ['repeat','source']} for r in b];checks['repeat_call_pairs']+=len(a)
 verify();save('per_run.json',per);save('call_coverage.json',callrows)
 primary=[x for x in per if x['repeat']==0];save('coverage_summary.json',dict(distinct_scenarios=4,repeat_runs=8,frame_calls=sum(x['frame_calls'] for x in primary),exact_teacher_call_matches=sum(x['exact_teacher_call_matches'] for x in primary),same_before_steps=sum(x['same_before_steps'] for x in primary),common_horizon_steps=sum(x['common_horizon'] for x in primary),all_teacher_calls=305,all_teacher_states=len(teacher_lookup),no_new_labels=True))
 save('review.json',dict(passed=True,checks=dict(checks),new_fits=0,new_inferences=0,new_physics=0,new_labels=0,source_hashes_unchanged=True,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(read(O/'coverage_summary.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
