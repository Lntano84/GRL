"""Prepare static files/bindings only. No environment, model or profile generation."""
from pathlib import Path
import hashlib,json,shutil,collections
from source_builder import build,instrument_worker
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid300_new_training_dates';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates';P=R/'outputs/grid299_public_state_coverage';E=W/'train_env'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not O.exists() and not E.exists() and not A.exists();assert read(P/'publication_review.json')['passed']
O.mkdir();E.mkdir();(E/'chronics').mkdir();static={};original=R/'work/grid100_data_foundation/trainval_env';calendar=read(P/'new_calendar_manifest.json');assert len(calendar['dates'])==16
for p in original.iterdir():
 if p.is_file() and p.suffix in ['.json','.csv','.py']:
  shutil.copy2(p,E/p.name);static[p.relative_to(R).as_posix()]=sha(p)
p=E/'scenario_params.json';params=read(p);params['all_dates']=[i['date'] for i in calendar['dates']];save(p,params)
for p in E.iterdir():
 if p.is_file():static[p.relative_to(R).as_posix()]=sha(p)
# Freeze the old metadata-only schema environment's constructor bootstrap as bytes.
for p in (original/'chronics/2035-01-01_8').iterdir():
 if p.is_file():static[p.relative_to(R).as_posix()]=sha(p)
for src,dst in [('new_calendar_manifest.json','calendar.json'),('new_pool_roles.json','pool_roles.json'),('frozen_model_references.json','model_references.json')]:shutil.copy2(P/src,O/dst)
shutil.copy2(R/'outputs/grid297_paired_collection_preparation/model_freeze.json',O/'model_freeze.json');shutil.copy2(R/'outputs/grid297_paired_collection_preparation/published_model_bindings.json',O/'published_model_bindings.json')
for rel,h in read(O/'model_freeze.json').items():assert sha(R/rel)==h
template=read(R/'outputs/grid297_paired_collection_preparation/design.json')['execution_order'][0];items=[]
for b in read(P/'proposed_new_baseline_manifest.json'):
 item=dict(template);item.update(index=b['index'],pool=b['pool'],date=b['date'],week=b['week'],model_seed=int(b['pool'][-1]),method='BASELINE261__'+b['pool'],arm='BASELINE261',completion_mode='FROZEN261_NO_INTERVENTION',partition='own_training_'+b['role'],KEEP=b['KEEP'],within_pair=0,pair_index=b['index'],target=None);items.append(item)
assert len(items)==144 and collections.Counter(i['partition'] for i in items)=={'own_training_fit8':96,'own_training_cal4':48}
runtime={'environment':E.relative_to(R).as_posix(),'caps':{'episode_s':1200,'wall_s':86400,'physical_steps':1452240,'public_forecasts':22500000},'timing':'Same original ctl.propose full online features/NN/heads/optimizer/forecast costs. Model/env loading outside per-step, inside episode/total budget; generation schema initialization and solver work separately counted inside total window.','execution_order':items}
design={'id':'GRID300_NEW_TRAIN_DATES_SUFFIX_COLLECTION','status':'IMPLEMENTATION_PREPARATION_NO_PHYSICAL_APPROVAL','environment':runtime['environment'],'baseline_runtime':runtime,'calendar_scenarios':16,'baseline_runs':144,'max_opposite_runs':576,'max_runs':720,'caps':{'wall_s':86400,'generation_s':7200,'generation_worker_s':1800,'episode_s':1200,'physical_steps':1452240,'public_forecasts':22500000},'selection':'Four fixed bins1..504,505..1008,1009..1512,1513..2017; earliest(step,call_index,batch) eligible nativeSTOP in each; no replacement/label/completion filtering.','publication':'generation hashes/receipts before any baseline prediction, then144 baseline raw-source hashes and fixed markers before any opposite replay. One manager waits for exact Git publication; clock continues.','prefix':'Before each pre-event physical step compare whole delivered-action/before hashes, after step exact seven-key prefix; target common includes full base-action/order/aliases/visible-feedback/NN input/raw41/nativeq and prefixcost/signature. Different marker -> fail, no intervention.','missing':'Missing bins/no events/failure/partial retained; zero numerical cost targets or fits in manager. No automatic retries, resume or extra sweeps.','resource_qualification':'2038 compatibility and completion not live-tested; historical estimate10.5..19h is not resourcepassed. Initial generation pilot and fixed fit8 replay pilot are paid parts of scope, not preauthorization smoke.'}
save(O/'design.json',design);save(O/'runtime_design.json',runtime);save(O/'static_freeze.json',static)
old=read(R/'outputs/grid297_paired_collection_preparation/source_freeze.json');freeze=dict(old)
for rel,h in old.items():assert sha(R/rel)==h,rel
for p in list(W.glob('*.py'))+[O/n for n in ['design.json','calendar.json','pool_roles.json','model_references.json','model_freeze.json','published_model_bindings.json','static_freeze.json']]:freeze[p.relative_to(R).as_posix()]=sha(p)
save(O/'source_freeze.json',freeze);save(O/'code_freeze.json',freeze)
for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
compile(build(),'generated_runtime.py','exec');compile(instrument_worker((R/'work/grid55/worker.py').read_text(encoding='utf8')),'instrumented_worker.py','exec')
S=A/'source';S.mkdir(parents=True)
for p in list(W.glob('*.py'))+[p for p in O.glob('*.json') if p.name!='runtime_design.json']:shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
msg='GRID300按299固定新日期方案完成可审阅实施源码/16日期与144基线绑定，独立新环境只复制静态网架/参数、不复制chronics，0模型加载/生成/预测/物理。总24h/生成2h/单生成1800s/单回放1200s与双发布屏障已编码；有限离线检查待源发布后首次执行，尚无新物理批准。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID300实施源准备，未启动生成或回放\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if n=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-04 — GRID300可审阅实施\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID300先发布源再有限离线检查\n\n'+msg+'\n\n1. 核具体源Gitblob/远端后首次qualify，不运行真实环境或模型smoke。\n2. 通过后完成报告/D协作/清单核对，再一次询问16生成+最多720回放/24h范围。\n3. 新批准前不写physical_authorization、不启动manager；旧291/297/298/299完整项不重复。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID300实施源已固定，未获新物理批准\n\n'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
(O/'CONTINUOUS_HANDOFF.md').write_text('# 当前：GRID300实施源已固定，待有限离线检查\n\n'+msg+'\n',encoding='utf8')
print('GRID300_SOURCE_FROZEN',len(items),len(static),len(freeze),len(read(O/'model_freeze.json')))
