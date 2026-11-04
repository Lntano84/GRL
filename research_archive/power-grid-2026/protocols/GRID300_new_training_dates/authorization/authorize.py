"""Record the actual explicit user reply; no model/environment imports."""
import datetime,json,shutil
from guard import R,O,I,A,W,read,sha,save,validate_receipt,SCOPE
assert not (O/'physical_authorization.json').exists() and not (O/'collection_started.json').exists() and not (O/'claims').exists() and not (O/'runs').exists()
assert not any((W/'train_env/chronics').iterdir())
q=read(O/'preparation_review.json');assert q['passed'] and q['queue_and_clock_boundaries_passed'] and read(O/'publication_review.json')['passed']
auth={'scope_id':SCOPE,'approved_by_user':True,'user_instruction':'批准：16场景、最多720条，24小时上限','user_message_reference':'request_user_input_async/call_cb668b8cbfa94a9bb95465e2a31b73b0/0','question_scope':'Generate16 new training scenarios,144 baseline plus up to576 opposite episodes, max720/24h; generation, loads, startup, failures and publication waits included; no extension/retry; zero training.','generation_scenarios':16,'baseline_runs':144,'max_opposite_runs':576,'max_runs':720,'wall_cap_s':86400,'generation_cap_s':7200,'generation_worker_cap_s':1800,'worker_cap_s':1200,'physical_steps_cap':1452240,'forecasts_cap':22500000,'design_sha256':sha(O/'design.json'),'source_freeze_sha256':sha(O/'source_freeze.json'),'calendar_sha256':sha(O/'calendar.json'),'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_fits':0,'labels_derived_during_collection':0}
validate_receipt(auth,sha(O/'design.json'),sha(O/'source_freeze.json'),sha(O/'calendar.json'));save(O/'physical_authorization.json',auth)
S=A/'authorization';S.mkdir(exist_ok=False)
for p in [O/'physical_authorization.json',W/'authorize.py']:shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
msg='用户2026-11-04已明确批准“16场景、最多720条，24小时上限”。GRID300新授权绑定实际问答引用及design/source/calendar hash，原16生成/144基线/至多576相反动作、24h总/2h生成/1800s生成/1200s回放/1452240步/22500000forecast不变；先发布授权核具体Gitblob及远端，再唯一manager。0新训练或标签，旧291/297预算不继承。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID300已获新范围批准，待首次唯一启动\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if n=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-04 — GRID300新范围批准\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID300批准后首次启动\n\n'+msg+'\n\n1. 发布授权，核精确blob/全部冻结与无活动任务，再首次启动manager。\n2. 原总24h和生成2h时钟连续计两个发布等待，不重置；不补跑或重复资格。\n3. 健康时只轻量跟进；generation/baseline发布等待且无worker时运行固定publish_stage，helper必须退出后才下一屏障；运行worker时不并发Git。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID300新范围已获用户批准\n\n'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
(O/'CONTINUOUS_HANDOFF.md').write_text('# 当前：GRID300新范围明确获批，先发授权再首次启动\n\n'+msg+'\n\n当前尚未调用manager；源码b9ac57f6、准备49488829、收尾025f60c6及有限检查都已完成，不重复。启动后以collection_started/pipeline_state/active_worker和实际Python父子进程为准。\n',encoding='utf8')
save(O/'decision.json',{'status':'USER_SCOPE_APPROVED_AUTH_PUBLICATION_BEFORE_FIRST_START','new_generation_approved':True,'new_physics_approved':True,'new_training_qualified':False,'authorization_sha256':sha(O/'physical_authorization.json')})
print('GRID300_EXPLICIT_SCOPE_AUTHORIZATION_RECORDED',auth['user_message_reference'])
