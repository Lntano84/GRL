"""Byte-only closure of an externally interrupted, non-resumable collection."""
from pathlib import Path
import collections, datetime, hashlib, json, shutil, subprocess, time, zipfile

R=Path('D:/Codex/GRL研究'); I=R/'work/GRL-results-integration'
O=R/'outputs/grid300_new_training_dates'
A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates/continuation01_interrupted'
B=R/'archives/GRID300_continuation01_interrupted_20261105/raw_records'
C=R/'论文协作资料/FRAME_新日期采集实施_GRID300/continuation01_interrupted'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def ps(s): return subprocess.check_output(['powershell','-NoProfile','-Command',s]).decode('utf-8-sig').strip()

assert not B.exists() and not A.exists() and not (O/'continuation01_interruption_complete.json').exists()
assert not subprocess.check_output(['git','-C',str(I),'status','--porcelain']).strip()
probe=json.loads(ps("$x=Get-CimInstance Win32_OperatingSystem; $p=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'grid300_new_training_dates[/\\\\]continuation01[/\\\\](manager|worker|publish_stage)\\.py'}); @{boot=$x.LastBootUpTime.ToString('o');observed=$x.LocalDateTime.ToString('o');active_count=$p.Count} | ConvertTo-Json -Compress"))
assert probe['active_count']==0
m=read(O/'manifest.json'); state=read(O/'pipeline_state.json'); active=read(O/'active_worker.json')
assert m['completed']==35 and len(m['records'])==35 and state['index']==35 and active['index']==35
assert not (O/'finished.json').exists() and not (O/'baseline_publication.json').exists()
assert not (O/'collection_error_continuation01.json').exists()
item=read(O/'design.json')['baseline_runtime']['execution_order'][35]
partial=O/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
assert not (partial/'summary.json').exists()
lines=(partial/'partial_steps.jsonl').read_text(encoding='utf8').splitlines()
rows=[json.loads(x) for x in lines if x.strip()]
assert [x['step'] for x in rows]==list(range(1,len(rows)+1))
assert all(r['index']==j and r['summary']['arm']=='BASELINE261' for j,r in enumerate(m['records']))
start=time.perf_counter(); B.mkdir(parents=True)
files=set(p for p in (O/'runs').rglob('*') if p.is_file())
for pat in ['replay_*.log','worker_result_*.json','worker_setup_*.json','claims_continuation01/*','continuation01_stdout.log','continuation01_stderr.log']:
 files.update(p for p in O.glob(pat) if p.is_file())
snapshots=['manifest.json','pipeline_state.json','active_worker.json','run_context.json','continuation01_authorization.json','continuation01_launch_verification.json','continuation01_launch_process.json','continuation01_started.json','continuation01_runtime_launch_status.json','continuation01_runtime_doc_queue.json','runtime_git_policy_continuation01.md','CONTINUOUS_HANDOFF.md']
files.update(O/n for n in snapshots if (O/n).is_file())
entries=[]; objects={}
with zipfile.ZipFile(B/'raw_objects.zip','x',allowZip64=True) as z:
 for p in sorted(files):
  if time.perf_counter()-start>600: raise TimeoutError('single600s byte preservation window; retain partial')
  assert p.resolve().is_relative_to(R.resolve())
  data=p.read_bytes(); h=hashlib.sha256(data).hexdigest(); obj='objects/'+h[:2]+'/'+h
  entries.append({'path':p.relative_to(R).as_posix(),'sha256':h,'bytes':len(data),'object':obj})
  if h not in objects:
   z.writestr(obj,data,compress_type=zipfile.ZIP_STORED if p.suffix in ['.npz','.gz','.bz2','.zip'] else zipfile.ZIP_DEFLATED,compresslevel=1)
   objects[h]={'object':obj,'bytes':len(data)}
with zipfile.ZipFile(B/'raw_objects.zip') as z:
 for h,v in objects.items():
  if time.perf_counter()-start>600: raise TimeoutError('single600s archive readback window')
  data=z.read(v['object']); assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==h
raw={'passed':True,'local_archive':str(B),'logical_files':len(entries),'logical_bytes':sum(x['bytes'] for x in entries),'unique_objects':len(objects),'zip_bytes':(B/'raw_objects.zip').stat().st_size,'zip_sha256':sha(B/'raw_objects.zip'),'independent_object_readback':True,'raw_contents_uploaded':False,'entries':entries}
save(B/'raw_records_manifest.json',raw); save(O/'continuation01_interruption_raw_manifest.json',raw)
full=sum(bool(x['summary']['complete']) for x in m['records'])
closed={'status':'EXTERNAL_INTERRUPTION_CLOSED_NO_RETRY','observed':probe,'previous_manager_pid':16828,'previous_worker_launcher_pid':active['pid'],'last_saved_log_time':datetime.datetime.fromtimestamp((O/'replay_035.log').stat().st_mtime,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'accepted_baseline_receipts':35,'full2017_baselines':full,'early_terminated_baselines':35-full,'accepted_opposite_runs':0,'accepted_physical_steps':m['physical_steps'],'accepted_forecasts_including_generation_schema':m['forecasts'],'partial_index':35,'partial_saved_rows':len(rows),'partial_saved_forecast_calls_lower_bound':sum(x['all_forecast_calls'] for x in rows),'partial_progress_snapshot':read(partial/'progress.json'),'original_deadline_monotonic':96052.4935945,'original_deadline_local':'2026-11-05T16:03:12+08:00','cross_boot_monotonic_comparison_valid':False,'exact_interruption_time_unknown':True,'cause':'Processes absent after verified newer OS boot; exact shutdown cause unavailable. No manager error receipt was saved.','new_fits':0,'new_cost_targets':0,'paired_cost_targets_available':0,'no_automatic_restart':True,'new_physics_during_closure':0,'raw_manifest_sha256':sha(O/'continuation01_interruption_raw_manifest.json'),'archive_wall_s':time.perf_counter()-start}
save(O/'continuation01_interruption_complete.json',closed)
save(B/'archive_complete.json',{'passed':True,'raw_manifest_sha256':sha(B/'raw_records_manifest.json'),'zip_sha256':raw['zip_sha256']})
save(O/'continuation01_interruption_observation.json',{'closed':closed,'prior_pipeline_state':state,'prior_active_worker':active,'original_manager_error_absent':True,'finished_absent':True,'live_numerical_source_unchanged':True})
report=f'''# GRID300 continuation01：系统重启后发现采集中断

2026年11月5日核查时，系统最近启动于{probe['boot']}，当前没有GRID300管理器、回放worker或发布进程。原日志最后写入{closed['last_saved_log_time']}，状态仍是11月4日第36条执行中的旧快照；没有finished或新manager错误收据。只能确认运行被外部中断且电脑后来重启，不能确定具体关机时刻或原因。跨启动的monotonic时钟已失去原比较基准，绝不据此重新获得原剩余24小时。

实际保存35/144条基线收据：{full}条完整2017步，{35-full}条原环境提前结束，不能将35条都称完整周。已接纳68294物理步、219103次forecast（沿原manifest口径含生成schema检查）。仅覆盖f0_s0十二条、f0_s1 KEEP十二条及f0_s2十一条；其余池和日期尚未执行，不能当全部12池的新结果。

index35为f0_s2、2038-12-06_100，第36条保留{len(rows)}行连续步骤和至少{closed['partial_saved_forecast_calls_lower_bound']}次已落盘forecast调用；progress.json仍是较早512步快照。未保存到磁盘的在途工作未知，不算零。部分条无summary，不能接纳为完成。尚未完成144基线，没有基线发布或相反动作：相反动作0条、配对后缀费用标签0条、新训练0次。因此本轮还没有新成本学习效果结论。

本次只做停机记录和字节归档，无新模型加载/预测、环境调用、策略回报或目标派生。原16场景的既有完整归档未重做。新运行数据{raw['logical_files']}逻辑文件、{raw['logical_bytes']}字节、{raw['unique_objects']}去重对象已保留在{B.as_posix()}，逐对象独立SHA读回通过；包含NPZ的文件仅按字节封存，没有解码数组。大轨迹和原始内容仅D本地，Git仅报告、清单、收据及必要来源引用。

一次受控衔接批准已被使用。遵守no_retry，不启动第三个管理器、不补跑部分或完整条、不延长或重置原北京时间11月5日16:03:12截止。自动运行监控到此结束。若用户以后决定继续，须先另定可审阅的范围、重启后时间守卫和新明确授权；当前不准备新采集、不新增资格串、不训练。保留208/261及全部不利、KEEP、缺失记录；这次中断不是已检验的模型失败。
'''
(O/'GRID300_continuation01_interruption_report.md').write_text(report,encoding='utf8')
msg=f'GRID300 continuation01在系统重启后确认中断：35/144基线收据（{full}完整周、{35-full}提前结束），68294已接纳步/219103forecast；index35部分{len(rows)}步另列，相反动作/费用目标/新训练均0。无活动进程，原一次衔接已使用，不第三次自动恢复、不重置跨启动monotonic或原11月5日16:03:12截止。新原始数据已D无损归档并逐对象SHA读回；自动运行监控结束，后续采集须新独立范围与明确授权。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n; old=p.read_text(encoding='utf8')
 p.write_text(('# 当前：GRID300受控衔接外部中断，禁止自动重启\n\n'+msg+'\n\n以下保留历史。\n\n'+old) if n=='RESEARCH_STATE.md' else old+'\n\n## 2026-11-05 — GRID300受控衔接中断\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID300中断收尾，无获授权的新运行\n\n'+msg+'\n\n1. 完成中断报告、运行清单与D副本的具体Gitblob/远端核验；不重做已完成原始归档。\n2. 旧pipeline/active PID快照不能当活动任务；不启动旧manager/publisher或第三次恢复。\n3. 当前无成对相反动作目标，不训练、不补采、不启动开发验证。只有新的用户研究决定及独立范围/预算授权才能继续物理采集。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md'; p.write_text('# 最新：GRID300受控衔接外部中断\n\n[中断报告](protocols/GRID300_new_training_dates/continuation01_interrupted/GRID300_continuation01_interruption_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
(O/'CONTINUOUS_HANDOFF.md').write_text('# 当前：GRID300 continuation01中断，禁止自动恢复\n\n'+msg+'\n\n先读GRID300_continuation01_interruption_report.md、continuation01_interruption_complete.json和publication review。没有finished不代表可恢复；原manager/worker均已退出且系统重启。旧collection_error是生成阶段历史，没有新manager错误不代表健康。原始旧pipeline/active/run_context已归档，当前状态明确关闭。\n',encoding='utf8')
save(O/'pipeline_state.json',{'phase':'external_interruption_closed_no_retry','accepted_baseline_receipts':35,'full2017_baselines':full,'early_terminated_baselines':35-full,'partial_index':35,'partial_rows':len(rows),'opposite_runs':0,'original_deadline':96052.4935945,'cross_boot_clock_invalid':True,'closure_receipt':'continuation01_interruption_complete.json','no_retry':True})
save(O/'active_worker.json',{'active':False,'verified_after_new_boot':probe['boot'],'prior_manager_pid':16828,'prior_worker_launcher_pid':active['pid'],'observation':probe['observed']})
save(O/'NEXT_PROTOCOL_PLAN.json',{'status':'EXTERNAL_INTERRUPTION_CLOSED_NO_RETRY','allowed_now':'delivery verification only; no new collection, labels, models or qualification','new_physical_scope_requires_explicit_approval':True,'no_automatic_third_continuation':True,'cross_boot_monotonic_invalid':True,'new_training_qualified':False})
save(O/'decision.json',{'status':'EXTERNAL_INTERRUPTION_CLOSED_NO_RETRY','accepted_baseline_receipts':35,'paired_cost_targets':0,'new_training_qualified':False,'new_physical_authorization':False})
A.mkdir(); C.mkdir()
names=['GRID300_continuation01_interruption_report.md','continuation01_interruption_complete.json','continuation01_interruption_observation.json','continuation01_interruption_raw_manifest.json','manifest.json','continuation01_authorization.json','continuation01_launch_verification.json','continuation01_launch_process.json','continuation01_started.json','continuation01_runtime_launch_status.json','continuation01_runtime_doc_queue.json','runtime_git_policy_continuation01.md','CONTINUOUS_HANDOFF.md','NEXT_PROTOCOL_PLAN.json','pipeline_state.json','active_worker.json','decision.json']
for n in names: shutil.copy2(O/n,A/n); shutil.copy2(O/n,C/n)
shutil.copy2(Path(__file__),A/Path(__file__).name)
save(A/'archive_manifest.json',{p.name:sha(p) for p in A.iterdir() if p.is_file()})
save(C/'delivery_hashes.json',{n:sha(C/n) for n in names})
print(json.dumps({'closed':closed,'logical_files':raw['logical_files'],'logical_bytes':raw['logical_bytes'],'unique_objects':raw['unique_objects']},ensure_ascii=False))
