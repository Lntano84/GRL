"""Publish one public-only coverage inventory and fixed acquisition design."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';W=Path(__file__).parent;O=R/'outputs/grid299_public_state_coverage';P=R/'outputs/grid297_paired_collection_preparation';A=I/'research_archive/power-grid-2026/protocols/GRID299_public_state_coverage'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not O.exists() and not A.exists()
raw=read(P/'raw_records_manifest.json');assert sha(P/'raw_records_manifest.json')==read(P/'runtime_complete.json')['raw_archive_manifest_sha256']
refs={}
for e in raw['entries']:
 p=Path(e['path']);name=p.name
 if name=='intervention.json' or (name.startswith('call') and name.endswith('.json')):
  refs[e['path']]={'sha256':e['sha256'],'bytes':e['bytes'],'kind':'public_call_or_intervention'}
for p in [P/'design.json',P/'pool_roles.json',P/'raw_records_manifest.json',P/'runtime_complete.json',R/'work/grid297_paired_collection_preparation/intervention.py',R/'work/grid291_control_preparation/policy.py']:
 refs[p.relative_to(R).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size,'kind':'fixed_contract'}
protocol=dict(id='GRID299_PUBLIC_STATE_COVERAGE',status='SAVED_PUBLIC_INVENTORY_AND_PROPOSAL_ONLY',wall_cap_s=600,no_retry=True,
 baseline='For each of144 fixed original pool-ownweek pairs, read only event.common public identities and event.ordinary261.continue_. Select archived CONT_NEXT if ordinary261 continued, else STOP_NEXT. No event -> fixed STOP_NEXT. This matches original261 intervention action, and afterwards original261; no arm picked by suffix cost/completion/gain.',
 allowed_root=['step','state_hash','model_pool','engineering_fallback','learned_decisions'],allowed_decision=['optional_extension','native_q','owner_pool','next_ids','batch','NN_input','raw_features','action'],
 excluded_fields=['outcomes','local_reward_gain','raw83','parent_mu','residual_logit','cost','rewards after reveal','summary completion for selection','new or298cost targets'],
 public_container_disclosure='JSON is parsed as a container; only whitelisted fields accessed/exported, not a claim of unread bytes/physical blindness. Existing paid feedback within NN_input is public before the pending batch.',
 candidate='Every saved optional_extension=True native_q1<=native_q0 with1..64 nonempty next_ids on the one original261 baseline per pool/week. No new NN/head calls.',
 selection='Fixed4 physical-time bins[1,504],[505,1008],[1009,1512],[1513,2017]. In each choose earliest(step,callordinal,decisionindex), independent of labels, margin, costs, date names or seed success. Do not change bins/K or replace missing bins after inventory.',
 reuse='If selected marker matches already collected297 first event by step,state,NN_input hash,batch and next_ids, reuse the existing pair; never replay it. Other markers propose exactly one NEW opposite-to-ordinary261 episode, paired with existing archived original261 baseline. Shared baseline means correlated target noise, not independent pairs. No vector restore or simulator snapshot assumption.',
 proposal_caps={'total_wall_s':28800,'per_episode_s':1200,'forecasts':9000000,'physical_steps':'2017*number_new_episodes'},
 integrity='Future live opposite episode must reproduce entire archived original261 prefix and exact selected public marker/model/norm/controller identities before one intervention. Saved full joint action hashes before event, not topology-only identities. Prefix mismatch invalidates; no automatic retry, replacement or protocol change.',
 future_target='Originalraw_cost C_CONT-C_STOP from one intervention through fixed2017 step end; bothfull and safe separately. Existing baseline failures/noevent/missingbins remain recorded, no cost label imputation, survivor/related-date limitations explicit. No target derived by299.',
 roles='All12 original pools, ownfit8/cal4/KEEP identities unchanged. No outer4/final8/newdate. More states on existing dates do not add independent dates. No future fit prequalified; no calibration-label borrowing.',
 resource='Use only actual297 total25309.4708868s/288 episodes and1717785forecasts/288 as rough extrapolation. Not resource_passed, not guarantee. If8h looks infeasible, keep full fixed manifest and seek a concrete scope decision later; do not silently prune seeds/dates/bins.',
 new_physics=0,new_predictions=0,new_fits=0,new_cost_labels=0,new_physical_authorization=False)
O.mkdir();save(O/'protocol.json',protocol);save(O/'input_references.json',refs)
for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in list(W.glob('*.py'))+[O/'protocol.json',O/'input_references.json']})
S=A/'source';S.mkdir(parents=True)
for p in list(W.glob('*.py'))+list(O.glob('*.json')):shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
msg='用户2026-11-04继续。GRID298费用头不资格保留；GRID299一次公开状态覆盖盘点，固定全部12池own12原261保存路径，按四个物理时间段最早nativeSTOP选点，只读公开字段和引用hash，0费用标签/预测/训练/物理。评估复用原261基线、每新点只补一条相反动作轨迹的具体范围与8h资源，不拿重复seed作独立状态；新采集尚未授权。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID299固定公开覆盖与复用基线方案\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if n=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-04 — GRID299固定覆盖方案\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID299一次公开覆盖盘点\n\n'+msg+'\n\n1. 先核源码和输入具体Gitblob/远端，再首次600秒inventory，不读费用目标。\n2. 盘点后固定具体状态/复用与新增条数/原角色/资源；不足则裁决，不调整bins追结果。\n3. 新物理必须有完整可审阅实施与单独范围批准，当前不执行。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID299公开覆盖方案\n\n'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
(O/'CONTINUOUS_HANDOFF.md').write_text('# 当前：GRID299已封存，先发布后公开盘点\n\n'+msg+'\n',encoding='utf8')
print('GRID299_SOURCE_FROZEN',len(refs))
