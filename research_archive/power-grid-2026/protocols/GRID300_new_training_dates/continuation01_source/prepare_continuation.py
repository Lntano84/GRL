"""Prepare a reviewable single post-generation continuation; never launch it."""
from pathlib import Path
import hashlib,json,shutil,time
R=Path('D:/Codex/GRL研究');P=Path(__file__).parent;W=P/'continuation01';O=R/'outputs/grid300_new_training_dates';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def once(s,a,b):assert s.count(a)==1,a;return s.replace(a,b)
assert not (O/'continuation01_design.json').exists();stopped=read(O/'generation_stop_complete.json');assert stopped['generated_scenarios']==16 and stopped['accepted_baseline_runs']==0 and not (O/'runs').exists()
original=read(O/'collection_started.json');assert time.perf_counter()<original['deadline']
for n in ['state.py','adapter.py','source_builder.py','worker.py','stages.py']:shutil.copy2(P/n,W/n);assert sha(P/n)==sha(W/n)
s=(P/'manager.py').read_text(encoding='utf8')
s=once(s,' start=time.perf_counter();require_authorization();verify_frozen();D=read(O/\'design.json\')'," require_authorization();verify_frozen();original=read(O/'collection_started.json');start=original['monotonic_start'];D=read(O/'design.json');check({'deadline_monotonic':original['deadline']})")
assert s.count("O/'claims'")==2;s=s.replace("O/'claims'","O/'claims_continuation01'")
s=once(s,"with (O/'collection_started.json').open('x',encoding='utf8')", "with (O/'continuation01_started.json').open('x',encoding='utf8')")
s=once(s,'steps=forecasts=0;records=[];generation=[]',"steps=0;records=[];generation=read(O/'generation_manifest.json')['records'];forecasts=sum(x['schema_forecast_calls'] for x in generation)")
begin=s.index(' def publication(name,deadline):');end=s.index(' def replay(item):',begin)
s=s[:begin]+" def publication(name,deadline):\n  assert name=='baseline'\n  from auto_publish import release_baseline\n  release_baseline(ctx,deadline)\n"+s[end:]
begin=s.index('  for index in range(16):');end=s.index("  for item in D['baseline_runtime']['execution_order']",begin);s=s[:begin]+"  verify_stage('generation')\n"+s[end:]
s=s.replace("O/'collection_error.json'","O/'collection_error_continuation01.json'");s=s.replace('folder,finish_generation,finish_baseline','folder,finish_baseline')
(W/'manager.py').write_text(s,encoding='utf8')
pub=(P/'publish_stage.py').read_text(encoding='utf8');a=pub.index('ALLOWED=');b=pub.index('\ndef main():',a);pub=pub[:a]+"ALLOWED={'baseline':{'baseline_manifest.json','baseline_raw_manifest.json','selected_markers.json','runtime_design.json'}}"+pub[b:];(W/'publish_stage.py').write_text(pub,encoding='utf8')
design={'scope_id':'GRID300_POST_GENERATION_CONTINUATION01','status':'REVIEWABLE_NOT_AUTHORIZED_OR_STARTED','original_start_monotonic':original['monotonic_start'],'original_deadline_monotonic':original['deadline'],'original_generation_gate_failed':True,'generation_new':0,'generation_reused':16,'baseline_runs':144,'max_opposite_runs':576,'max_runs':720,'worker_cap_s':1200,'physical_steps_cap':1452240,'forecasts_cap':22500000,'original_authorization_sha256':sha(O/'physical_authorization.json'),'original_error_sha256':sha(O/'collection_error.json'),'reused_generation_freeze_sha256':sha(O/'generated_freeze.json'),'reused_generation_manifest_sha256':sha(O/'generation_manifest.json'),'publication':'Manager synchronously invokes fixed baseline publisher without active worker; exact whitelist/helper exit/Gitblob/remote required before next model call. No heartbeat release dependency. Failed original2h gate is preserved, not released or reset.','approval':'Requires a new explicit one-time post-generation user reply, including reuse of16 scenarios and continuation only until original Nov5 16:03 deadline. Old broad24h reply is not silently treated as continuation approval.','numerical_behavior':'state/adapter/source_builder/worker copied byte-identical to original300; original models/scalers/roles/controller/fees unchanged; zero new generation/training or target derivation in continuation.'}
save(O/'continuation01_design.json',design)
for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
freeze=read(O/'source_freeze.json')
for p in list(W.glob('*.py'))+[O/n for n in ['continuation01_design.json','generated_freeze.json','generation_manifest.json','collection_error.json','collection_started.json']]:freeze[p.relative_to(R).as_posix()]=sha(p)
save(O/'continuation01_source_freeze.json',freeze)
S=A/'continuation01_source';S.mkdir(exist_ok=False)
for p in W.glob('*.py'):shutil.copy2(p,S/p.name)
for n in ['continuation01_design.json','generated_freeze.json','generation_manifest.json','collection_error.json','collection_started.json']:shutil.copy2(O/n,S/n)
shutil.copy2(O/'continuation01_source_freeze.json',S/'source_freeze.json');shutil.copy2(Path(__file__),S/Path(__file__).name);save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
print('GRID300_CONTINUATION_SOURCE_PREPARED_NO_LAUNCH',len(freeze),original['deadline'])
