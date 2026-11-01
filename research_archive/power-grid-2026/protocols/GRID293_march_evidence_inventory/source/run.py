"""Eight-document static inventory only; no trajectory/array/model reads or imports."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid293_march_evidence_inventory';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID293_march_evidence_inventory';P=I/'research_archive/power-grid-2026/protocols'
DOCS=[
 ('291_quality',P/'GRID291_control_preparation/budget_limited_results/GRID291_report.md','Current2037 owner-f208/261 quality and scope, saved published result only.'),
 ('184_report',P/'GRID184_search_adapter/final_delivery/GRID184_report.md','Prior2037 same validation dates, different models/arms; not automatically causal for current208/261.'),
 ('184_divergences',P/'GRID184_search_adapter/final_delivery/first_divergences.json','Original first physical differences for CONSENSUS versus FACTOR, scope explicitly limited.'),
 ('59_report',P/'GRID59_complete/outputs/grid59/GRID59_report.md','Historical2035 quality/compute evidence; qualify dates/controller before reuse.'),
 ('61_report',P/'GRID61_complete/outputs/grid61/GRID61_report.md','Historical matched completion mechanism evidence, not current owner-f causal diagnosis.'),
 ('291_model_refs',P/'GRID291_control_preparation/source_launch_recovery02/model_references.json','Exact current owner-f checkpoint/parent/residual bindings; paths/hashes only.'),
 ('184_model_source',R/'work/grid184_search_adapter/models.py','Static original checkpoint/source selectors; never execute model code.'),
 ('March_FULL_summary',R/'outputs/grid100_data_foundation/bulk_runtime/runs/r0__2037-03-02_100__FULL_TEACHER/summary.json','Original reference summary already used in291; no new full trajectory, timing or arrays.')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def stage(phase,paths):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in paths:
  rel=p.relative_to(O) if p.is_relative_to(O) else Path(p.name);q=d/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 save(d/'archive_manifest.json',{p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()})
def prepare():
 assert not O.exists();assert len(DOCS)==8 and all(p.is_file() for _,p,_ in DOCS);O.mkdir();members=[]
 for n,(ident,p,scope) in enumerate(DOCS):
  target=O/'evidence_documents'/f'{n+1:02d}_{ident}{p.suffix}';target.parent.mkdir(exist_ok=True);shutil.copy2(p,target);members.append(dict(id=ident,source=p.relative_to(R).as_posix(),snapshot=target.relative_to(O).as_posix(),sha256=sha(p),scope=scope))
 save(O/'design.json',dict(id='GRID293',protocol='Exactly8 fixed already-published reports/static source/reference-summary documents. No new trajectory/NPZ/model tensors, numeric path comparison, forecasts, fitting or alternative policy returns.',documents=members,operation_caps=dict(document_count=8,new_fits=0,new_predictions=0,new_physics=0),decision='Reuse existing explanations only within their dates/model/arm scope; if a precise current saved-data gap remains, preregister one limited follow-up but do not run it in293.'))
 save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [W/'run.py',O/'design.json',*[O/r['snapshot'] for r in members]]});stage('source',[W/'run.py',O/'design.json',O/'source_freeze.json',*[O/r['snapshot'] for r in members]]);print('GRID293_EIGHT_DOCUMENTS_FROZEN')
def inspect():
 assert not (O/'inventory_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(I).as_posix())==(O/'source_freeze.json').read_bytes()
 for rel,h in json.loads((O/'source_freeze.json').read_text()).items():assert sha(R/rel)==h
 save(O/'inventory_started.json',dict(source_published=head,document_count=8,model_or_trajectory_arrays_read=0))
 summaries=[]
 for m in json.loads((O/'design.json').read_text())['documents']:
  text=(O/m['snapshot']).read_text(encoding='utf8');lines=text.splitlines();hit=[]
  for i,l in enumerate(lines,1):
   if any(k.lower() in l.lower() for k in ['2035','2037','March','3月','三月','成本','因果','局部','首次','差异','FULL','checkpoint','manifest','FACTOR','CONSENSUS','GRID190','GRID208','GRID261']):hit.append(dict(line=i,text=l))
  row=dict(**m,line_count=len(lines),excerpts=hit);summaries.append(row);print('DOCUMENT',m['id'],len(lines),flush=True)
  if m['id']=='291_model_refs':
   refs=json.loads(text);print(json.dumps([dict(pool=r['pool'],checkpoint=r['checkpoint'],owner_fold=r['owner_fold'],seed=r['seed'],parent_heads=r['parent_heads'],residual_model=r['residual_model']) for r in refs[:2]],ensure_ascii=False),flush=True)
  elif m['id']=='184_divergences':
   v=json.loads(text);print(json.dumps(dict(scope=v.get('scope'),entry_count=len(v.get('entries',[])),entries=[{k:x.get(k) for k in ['seed','week','physical_signature_equal','cost_delta','first_physical_action_divergence_step']} for x in v.get('entries',[])]),ensure_ascii=False),flush=True)
  else:
   print('\n'.join(f"{x['line']}: {x['text']}" for x in hit),flush=True)
 save(O/'static_source_excerpts.json',summaries);print('GRID293_STATIC_INVENTORY_READ_COMPLETE')
if __name__=='__main__':globals()[sys.argv[1]]()
