"""Freeze ONE all-member saved-record review; no outcome readout or live imports."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';W=Path(__file__).parent
P=R/'outputs/grid297_paired_collection_preparation';O=R/'outputs/grid298_saved_suffix_targets'
A=I/'research_archive/power-grid-2026/protocols/GRID298_saved_suffix_targets'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not O.exists() and not A.exists()
assert read(P/'runtime_complete.json')['accepted_episodes']==288 and read(P/'publication_closeout.json')['passed']
d=read(P/'design.json');raw=read(P/'raw_records_manifest.json');assert sha(P/'raw_records_manifest.json')==read(P/'runtime_complete.json')['raw_archive_manifest_sha256']
by={e['path']:e for e in raw['entries']};refs={};members=[]
for item in d['execution_order']:
 folder=P/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
 record=dict(item);record['folder']=folder.relative_to(R).as_posix();members.append(record)
 for name in ['summary.json','metadata.json','intervention.json','steps.jsonl.gz']:
  rel=(folder/name).relative_to(R).as_posix();e=by[rel];refs[rel]={'sha256':e['sha256'],'bytes':e['bytes'],'role':'saved_episode'}
for n in range(144):
 rel=(P/f'pair_{n:03d}.json').relative_to(R).as_posix();e=by[rel];refs[rel]={'sha256':e['sha256'],'bytes':e['bytes'],'role':'saved_pair_receipt'}
for p in [P/n for n in ['design.json','pool_roles.json','runtime_complete.json','publication_closeout.json','finished.json','source_freeze.json','model_freeze.json','raw_records_manifest.json']]+[R/'work/grid55/worker.py',R/'work/grid03/score_adapter.py',R/'work/grid297_paired_collection_preparation/intervention.py',R/'work/grid297_paired_collection_preparation/adapter.py']:
 refs[p.relative_to(R).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size,'role':'frozen_contract'}
protocol=dict(id='GRID298_SAVED_SUFFIX_TARGETS',wall_cap_s=600,episodes=288,pairs=144,roles={'fit8':96,'cal4':48,'KEEP_pairs':12},
 allowed='Only frozen JSON and steps.jsonl.gz bytes, once per episode; no vectors NPZ, model arrays, live imports, forecasts, physics or fits.',
 target='For matched EVENT pairs with both complete==True and exactly2017 steps: sum raw_cost at trigger step through2017 CONT minus STOP. Negative favors CONT. No queryNET/fee/reward change.',
 horizon='Unequal or incomplete horizons retain both costs and completion/safety outcomes, target=null; NO_ELIGIBLE_EVENT target=null. No terminal penalties, imputation or replacement.',
 safety='Record illegal/ambiguous/exceptions and terminal done/complete separately; complete-cost target never itself certifies safety or training qualification.',
 checks=['exact saved byte hashes','metadata original role/seed/library/costscale/dt consistency','all steps contiguous, exact summary sums','common event and state, original prefix digest/cost','delivered full joint action hashes at trigger','old per-row ledger tolerance; missing fields recorded, not invented','fit8/cal4/outer4 isolation, KEEP retained'],
 numeric='Raw official costs accumulated in original row order for summary equality. Suffix fsum and whole-total delta both saved; arithmetic tolerance=64*machineepsilon*max(1,sum_abs_raw)*max(1,nrows), no performance threshold. Component dollars use per-row marginal_cost times each of four original MWh terms. Original curtailment_delta, not level.',
 no_retry=True,partial_receipts=True,new_training_qualified=False,
 interpretation='One pair is an episode-level intervention, not an independent query sample. Correlated dates/pools/seeds and original own12/internalcal4 ancestry remain. No OOD/threshold/seed search or alternative strategy returns. No new physics authorization.')
O.mkdir();save(O/'protocol.json',protocol);save(O/'members.json',members);save(O/'input_references.json',refs)
for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
freeze={p.relative_to(R).as_posix():sha(p) for p in list(W.glob('*.py'))+[O/'protocol.json',O/'members.json',O/'input_references.json']};save(O/'source_freeze.json',freeze)
S=A/'source';S.mkdir(parents=True)
for p in list(W.glob('*.py'))+list(O.glob('*.json')):shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
msg='用户2026-11-03明确继续，grl恢复ACTIVE。GRID297已完成288条且无活动进程，不重跑采集/归档；GRID298固定全部144对保存JSON/账本一次600秒核查后缀C_CONT-C_STOP，完整2017步且共同前缀通过才有成本目标，失败/未完周/无事件不造低费用正例，安全与角色另列，0新拟合/预测/物理。源码输入先发布再首次读出，无新训练资格。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/name;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID298有限保存目标核查\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if name=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-03 — 恢复并固定GRID298\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID298一次有限保存核查\n\n'+msg+'\n\n1. 核源码/输入具体Gitblob和远端后首次运行readout.py，不重复已完成项。\n2. 全144对/fit8cal4KEEP及全部结束安全保留；不读NPZ，不训练或物理。\n3. 据保存支持给明确裁决，不为自动推进扫模型。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID298有限保存核查已固定\n\n'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
hand='# 当前：用户已继续，GRID298有限保存核查\n\n'+msg+'\n\n下一起点 outputs/grid298_saved_suffix_targets；先核新源发布，再一次读出。297历史关机收据保留不改。\n'
(O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8');(P/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8')
save(O/'NEXT_PROTOCOL_PLAN.json',{'status':'SOURCE_PUBLICATION_BEFORE_ONE_READOUT','training_qualified':False,'source_complete_before_targets':True})
print('GRID298_SOURCE_FROZEN',len(members),len(refs),len(freeze))
