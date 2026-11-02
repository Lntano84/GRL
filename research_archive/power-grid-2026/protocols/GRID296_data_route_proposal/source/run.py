"""One bounded static source/metadata proposal; no arrays, labels or forecasts."""
from pathlib import Path
import collections, hashlib, json, shutil, subprocess, sys
R=Path('D:/Codex/GRL研究'); W=Path(__file__).parent; I=R/'work/GRL-results-integration'
O=R/'outputs/grid296_data_route_proposal'; A=I/'research_archive/power-grid-2026/protocols/GRID296_data_route_proposal'
DOCS=[
 ('teacher_schema','work/grid102_search_rl/prepare.py'),
 ('public_schema','work/grid113_relational_learning/prepare.py'),
 ('own_members','outputs/grid100_data_foundation/bulk_dataset_manifest.json'),
 ('collector_binding','work/grid100_data_foundation/bulk_teacher_worker.py'),
 ('saved_call_schema','work/grid89_review/adapter.py'),
 ('delivered_ledger_schema','work/grid55/worker.py'),
 ('current_support','outputs/grid257_reference_labels/qualification.json'),
 ('actual_runtime','outputs/grid291_control_preparation/GRID291_report.md'),
 ('own_training_roles','outputs/grid113_relational_learning/train_metadata.json')]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*args): return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*args])
def prepare():
 assert not O.exists(); assert read(R/'outputs/grid295_target_label_contract/complete.json')['passed']
 O.mkdir(); docs=[]
 for ident,rel in DOCS:
  p=R/rel; q=O/'source_documents'/(ident+p.suffix); q.parent.mkdir(exist_ok=True); shutil.copy2(p,q)
  docs.append(dict(id=ident,source=rel,snapshot=q.relative_to(O).as_posix(),sha256=sha(p)))
 save(O/'design.json',dict(id='GRID296',documents=docs,document_cap=10,actual_documents=len(docs),scope='Static source/schema and metadata-only membership counts. No trajectory/call contents, pickle/NPZ/model arrays, target values, forecasts, fits or new labels.',prior_plan=read(R/'outputs/grid295_target_label_contract/NEXT_PROTOCOL_PLAN.json'),outcome='One reuse-or-collect proposal and explicit stopping decision; no indefinite qualification chain.',new_fits=0,new_labels=0,new_predictions=0,new_physics=0))
 save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [W/'run.py',O/'design.json',*[O/d['snapshot'] for d in docs]]})
 S=A/'source'; S.mkdir(parents=True,exist_ok=False); shutil.copy2(W/'run.py',S/'run.py')
 for p in O.rglob('*'):
  if p.is_file(): q=S/p.relative_to(O); q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
 save(S/'archive_manifest.json',{p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file()})
 print('GRID296_NINE_DOCUMENTS_FROZEN')
def inspect():
 assert not (O/'inspection_started.json').exists()
 head=git('rev-parse','HEAD').decode().strip(); assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(I).as_posix())==(O/'source_freeze.json').read_bytes()
 for rel,h in read(O/'source_freeze.json').items(): assert sha(R/rel)==h,rel
 save(O/'inspection_started.json',dict(source_published=head,new_fits=0,new_labels=0,new_predictions=0,new_physics=0))
 D=read(O/'design.json'); texts={}; documents=[]
 for d in D['documents']:
  p=O/d['snapshot']; assert sha(p)==d['sha256']; lines=p.read_text(encoding='utf8').splitlines()
  if d['id'] not in ['own_members','own_training_roles','current_support']:
   hits=[dict(line=n,text=t) for n,t in enumerate(lines,1) if any(k in t.lower() for k in ['cost','reward','outcomes','state','save','storage','redispatch','curtail','action','dispatch','optim','wall','second','运行','秒','time','query','rho','train','controller'])]
   documents.append(dict(id=d['id'],source=d['source'],sha256=d['sha256'],excerpts=hits))
   print('\nDOCUMENT',d['id']); print('\n'.join(f"{h['line']}: {h['text']}" for h in hits),flush=True)
  texts[d['id']]=p
 members=read(texts['own_members'])['samples']; own=[m for m in members if m['partition']=='train']; other=[m for m in members if m['partition']!='train']
 roles=read(texts['own_training_roles']); assert all(m['partition']=='train' for m in roles)
 assert {m['date'] for m in roles}=={m['calendar_group'] for m in own}
 assert not {m['calendar_group'] for m in own}&{m['calendar_group'] for m in other}
 counts=dict(teacher_all_members=len(members),teacher_train_members=len(own),teacher_nontrain_members=len(other),own_dates=sorted({m['calendar_group'] for m in own}),own_scenarios=sorted({m['scenario'] for m in own}),own_physical_calls=len({m['physical_state_group'] for m in own}),own_source_calls=len({m['source_call'] for m in own}),own_train_prefix_rows=len(roles),own_train_prefix_calls=len({m['physical_state_group'] for m in roles}),own_train_unique_samples=len({m['sample_id'] for m in roles}),per_date_teacher=dict(sorted(collections.Counter(m['calendar_group'] for m in own).items())),per_date_prefix=dict(sorted(collections.Counter(m['date'] for m in roles).items())),per_date_calls={d:len({m['physical_state_group'] for m in own if m['calendar_group']==d}) for d in sorted({m['calendar_group'] for m in own})})
 pools=read(texts['current_support'])['pools']; counts['pool_support']=[{k:v for k,v in p.items() if k in ['pool','fit8_rows','fit8_STOP','fit8_CONT','fit8_STOP_positive','fit8_CONT_positive','fallback_KEEP']} for p in pools]
 assert len(pools)==12; save(O/'membership_counts.json',counts); save(O/'static_excerpts.json',documents)
 save(O/'inspection_complete.json',dict(passed=True,documents=len(D['documents']),metadata_only=True,new_fits=0,new_labels=0,new_predictions=0,new_physics=0))
 print('COUNTS',json.dumps(counts,ensure_ascii=False))
if __name__=='__main__': globals()[sys.argv[1]]()
