"""Eight static documents for target/label identifiability; never load data tensors."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid295_target_label_contract';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID295_target_label_contract';PREV=R/'outputs/grid294_march_saved_readout'
DOCS=[
 ('286_existing_contract','outputs/grid286_cost_objective_contract/GRID286_report.md','Reuse existing queryNET cost-aware objective verdict, not a repeated qualification or fit.'),
 ('query_environment','work/grid102_search_rl/environment.py','Original query continuation reward/NET formula and horizon.'),
 ('teacher_case_schema','work/grid102_search_rl/prepare.py','Existing own-teacher case/feedback field construction, static schema only.'),
 ('relational_training_schema','work/grid113_relational_learning/prepare.py','Existing public and target dataset construction/roles; no pickle or NPZ load.'),
 ('native_reward','work/grid02/ljn_nn/modules/rewards.py','Actual reward definition used by common.make_env.'),
 ('base_agent','work/grid02/ljn_nn/LJNAgent.py','Original base.act action assembly and continuous response.'),
 ('base_factory','work/grid02/ljn_nn/make_agent.py','Bind make_agent_topoNN to actual agent class statically.'),
 ('operating_cost_ledger','work/grid03/score_adapter.py','Reuse original operating-cost fields and units; not redefining labels.')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def stage(phase,paths):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in paths:
  rel=p.relative_to(O) if p.is_relative_to(O) else Path(p.name);q=d/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 save(d/'archive_manifest.json',{p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()})
def prepare():
 assert not O.exists();assert read(PREV/'complete.json')['passed'] and read(PREV/'publication_review.json')['passed'];assert len(DOCS)==8 and all((R/p).is_file() for _,p,_ in DOCS)
 O.mkdir();members=[]
 for n,(ident,rel,role) in enumerate(DOCS):
  p=R/rel;q=O/'source_documents'/f'{n+1:02d}_{ident}{p.suffix}';q.parent.mkdir(exist_ok=True);shutil.copy2(p,q);members.append(dict(id=ident,source=rel,snapshot=q.relative_to(O).as_posix(),sha256=sha(p),role=role))
 save(O/'design.json',dict(id='GRID295',plan=read(PREV/'NEXT_PROTOCOL_PLAN.json'),documents=members,scope='Eight static report/source/schema documents only. No numerical trajectory/model/NPZ/pickle reads; no generated labels, fitting, model predictions or physical observations. Reuse286 result and294 observations, do not repeat checks.',questions=['Exact original feedback and queryNET target/horizon versus operating-cost metric','Original delivered action versus proposed topology and continuous optimizer layers','Whether checked own-training schema already stores candidate-level operating costs or long-horizon paired labels','Which semantic/target/data gap prevents justified new training; no automatic target changes'],stop='Complete one qualification and give evidence-based learning/data decision, no guessed label fields or auto physical generation.'))
 paths=[W/'run.py',O/'design.json',PREV/'NEXT_PROTOCOL_PLAN.json',PREV/'decision.json',PREV/'publication_review.json',*[O/m['snapshot'] for m in members]];save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});stage('source',[W/'run.py',O/'design.json',O/'source_freeze.json',*[O/m['snapshot'] for m in members]]);print('GRID295_SOURCE_FROZEN_8')
def inspect():
 assert not (O/'qualification_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(I).as_posix())==(O/'source_freeze.json').read_bytes()
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h
 save(O/'qualification_started.json',dict(source_published=head,documents=8,new_fits=0,new_predictions=0,new_labels=0,new_physics=0))
 out=[]
 for m in read(O/'design.json')['documents']:
  lines=(O/m['snapshot']).read_text(encoding='utf8').splitlines();hits=[]
  for j,line in enumerate(lines,1):
   if any(k.lower() in line.lower() for k in ['reward','cost','NET','gain','scale','fee','query','queries','target','dispatch','storage','curtail','def act','toponn','topology','simulate','save','npz','pickle','payload','batch']):hits.append(dict(line=j,text=line))
  out.append(dict(**m,line_count=len(lines),excerpts=hits));print('DOCUMENT',m['id'],len(lines),flush=True);print('\n'.join(f"{x['line']}: {x['text']}" for x in hits),flush=True)
 save(O/'static_contract_excerpts.json',out);print('GRID295_STATIC_QUALIFICATION_READ_COMPLETE')
if __name__=='__main__':globals()[sys.argv[1]]()
