"""Preserve a user-requested stop, byte-only; no labels/models/physics/audit reruns."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,time,zipfile
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid301_remaining_collection';I=R/'work/GRL-results-integration'
A=I/'research_archive/power-grid-2026/protocols/GRID301_remaining_collection/user_stop_20261105'
B=R/'archives/GRID301_user_stop_20261105/raw_records';C=R/'论文协作资料/FRAME_剩余采集实施_GRID301/user_stop_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 assert not A.exists() and not B.exists() and not (O/'user_stop_complete.json').exists()
 assert (O/'STOP_REQUEST.json').exists()
 count=subprocess.check_output(['powershell','-NoProfile','-Command',"@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'grid301_remaining_collection' -and $_.CommandLine -match '(manager|worker|load_only|publish_stage)\\.py'}).Count"]).decode().strip();assert count=='0'
 assert not subprocess.check_output(['git','-C',str(I),'status','--porcelain']).strip()
 manifest=read(O/'manifest.json');state=read(O/'pipeline_state.json');error=read(O/'collection_error_recovery03.json');runtime=read(O/'runtime_design.json');selected=read(O/'selected_markers.json')
 assert manifest['new_accepted']==error['new_accepted']==300 and state['index']==336 and 'stop requested' in error['traceback']
 item=runtime['execution_order'][336];folder=O/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}";assert not (folder/'summary.json').exists()
 rows=[json.loads(x) for x in (folder/'partial_steps.jsonl').read_text(encoding='utf8').splitlines() if x.strip()];assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
 partial=read(folder/'partial_intervention.json');assert partial['steps']==len(rows)==1079
 full=sum(bool(r['summary']['complete']) for r in manifest['records']);assert full==242
 assert all(r['summary']['steps']==2017 for r in manifest['records'] if r['summary']['complete'])
 # Enumerate all new output files once; old300/297 archives are referenced, not reread.
 files={p for p in O.rglob('*') if p.is_file()};files.add(Path(__file__))
 start=time.perf_counter();B.mkdir(parents=True);entries=[];objects={}
 with zipfile.ZipFile(B/'raw_objects.zip','x',allowZip64=True) as z:
  for ordinal,p in enumerate(sorted(files)):
   if time.perf_counter()-start>600:raise TimeoutError('single600s byte archive window; retain partial, do not auto rerun')
   assert p.resolve().is_relative_to(R.resolve());before=p.stat();data=p.read_bytes();after=p.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
   h=hashlib.sha256(data).hexdigest();obj='objects/'+h[:2]+'/'+h
   entries.append({'path':p.relative_to(R).as_posix(),'sha256':h,'bytes':len(data),'object':obj})
   if h not in objects:
    z.writestr(obj,data,compress_type=zipfile.ZIP_STORED if p.suffix in ['.npz','.gz','.bz2','.zip'] else zipfile.ZIP_DEFLATED,compresslevel=1);objects[h]={'bytes':len(data),'object':obj}
   if ordinal and ordinal%2000==0:print('PRESERVED_FILES',ordinal,flush=True)
 with zipfile.ZipFile(B/'raw_objects.zip') as z:
  for h,v in objects.items():
   if time.perf_counter()-start>600:raise TimeoutError('single600s byte archive readback window')
   data=z.read(v['object']);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==h
 raw={'passed':True,'local_archive':str(B),'logical_files':len(entries),'logical_bytes':sum(e['bytes'] for e in entries),'unique_objects':len(objects),'zip_bytes':(B/'raw_objects.zip').stat().st_size,'zip_sha256':sha(B/'raw_objects.zip'),'independent_object_readback':True,'NPZ_parsed':False,'raw_contents_uploaded':False,'reused_old_archive_not_repeated':'archives/GRID300_continuation01_interrupted_20261105/raw_records','entries':entries}
 save(B/'raw_records_manifest.json',raw);save(O/'user_stop_raw_manifest.json',raw);save(B/'archive_complete.json',{'passed':True,'raw_manifest_sha256':sha(B/'raw_records_manifest.json'),'zip_sha256':raw['zip_sha256']})
 complete={'status':'USER_STOPPED_PAUSED_NO_AUTO_RESUME','user_instruction':'我要准备关机了，准备停下来','user_stop_requested':read(O/'STOP_REQUEST.json'),'all_collection_processes_exited':True,'accepted_new_receipts':300,'new_baselines':108,'new_opposite_receipts':192,'planned_new_receipts':108+len(selected['new_opposite_items']),'full2017_new_receipts':full,'early_terminated_new_receipts':300-full,'reused_old_baseline_receipts':35,'accepted_physical_steps_new':manifest['physical_steps_new'],'accepted_forecasts_new':manifest['forecasts_new'],'partial_index':336,'partial_pool':item['pool'],'partial_week':item['week'],'partial_rows_saved':len(rows),'partial_forecasts_saved_counter':partial['forecasts'],'partial_forecasts_from_rows_lower_bound':sum(r['all_forecast_calls'] for r in rows),'remaining_not_started':40,'partial_not_replayed':True,'original_scope_elapsed_s_at_stop':error['wall_s'],'original_deadline_utc_epoch':read(O/'launch_context.json')['deadline_utc_epoch'],'original_clock_not_reset':True,'new_fits':0,'cost_targets_derived':0,'archive_wall_s':time.perf_counter()-start,'raw_manifest_sha256':sha(O/'user_stop_raw_manifest.json'),'automation':'grl already absent, left disabled','future_resume_requires_user_instruction_and_valid_scope':True,'not_method_failure':True}
 save(O/'user_stop_complete.json',complete)
 report=f'''# GRID301：用户准备关机，已停止并保存

用户明确要求“我要准备关机了，准备停下来”。STOP_REQUEST在{read(O/'STOP_REQUEST.json')['requested_local']}写入，管理器和worker按守卫退出，当前无采集/加载/发布进程。collection_error_recovery03中的stop requested是本次人为停止收据，不是模型或方法失败。没有自动恢复，旧grl周期任务此前已删除，保持关闭。

本轮实际计划341条新记录（108基线+固定233相反动作，339空段不替补），已接纳300条：108基线+192相反动作，其中242条完整2017步、58条提前结束。不能把300都称完整周。已接纳592946新物理步、1874873新forecast；复用的旧35条基线另列、不重跑。index336/f3_s0/2038-11-15_100另保存1079连续步及2938次已保存forecast计数，没有summary，不接纳为完整条；未落盘在途工作未知，不算零。另40条未启动。

全部已接受与部分记录、12组模型加载证据、源码引用、角色、baseline publication/选点、logs/claims/原错误及本次停机快照已按字节在D保留：{B.as_posix()}。{raw['logical_files']}逻辑文件、{raw['logical_bytes']}字节、{raw['unique_objects']}去重对象，逐对象独立SHA读回通过。NPZ仅原字节封存，没有解析模型/向量、重新评分、重求解或派生标签。旧300/297完整归档不重做。大原始内容仅D本地，Git只必要报告、收据、清单和源引用。

原总窗累计到停机为{error['wall_s']:.6f}秒，含此前工程失败、等待、加载及发布；不能把它当单纯300条运行耗时。12组模型加载证据通过（首组旧证据+11直接加载），实际回放已发生原NN预测，但新训练0、后缀费用标签0。本轮是采集源数据，尚未有新学习效果结论。

当前用户停止指令优先于此前纯工程启动修复授权。不得自行删除STOP_REQUEST、启动任何旧/新manager、补跑index336或旧部分35、延长或重新使用时钟。后续继续必须有用户新指令；若涉及过期窗口或新执行范围，先固定可审阅范围及预算。未完成/不利/KEEP与全部pool日期记录保留，不能拼成功子集。成本目标只可在全部源发布后另固定保存协议，双方同完整2017步才有数值目标；当前不做该分析或训练。
'''
 (O/'GRID301_user_stop_report.md').write_text(report,encoding='utf8')
 msg='GRID301已按用户关机要求停止，所有采集进程退出：341计划中接纳300（108基线/192相反动作，242完整周/58提前结束），592946新步/1874873forecast；index336另1079步/2938保存forecast，40未启动。新原始数据D无损去重归档并逐对象SHA读回，0新训练/费用标签。STOP_REQUEST保留，不自动恢复/补跑/延窗；grl此前已删除，保持关闭。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=I/'docs'/n;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID301用户关机暂停，已保存并退出\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if n=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-05 — GRID301用户关机暂停\n\n'+msg+'\n',encoding='utf8')
 (I/'docs/NEXT_STEPS.md').write_text('# 当前：用户关机暂停，无自动新工作\n\n'+msg+'\n\n1. 完成本次停机必要报告/收据/清单、D协作副本和具体Gitblob/远端核验，不重复原始ZIP读回或旧归档。\n2. 停止指令优先：不删除STOP_REQUEST、不恢复旧grl或任何manager，不重跑完整或部分物理项。\n3. 用户以后继续时先读当前收据/源清单/原时钟；若窗口过期，须新可审阅范围和预算。当前不派生费用目标、不训练、不新物理。\n',encoding='utf8')
 p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID301用户关机暂停\n\n[停机报告](protocols/GRID301_remaining_collection/user_stop_20261105/GRID301_user_stop_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
 (O/'CONTINUOUS_HANDOFF.md').write_text('# 当前：GRID301用户关机暂停，全部运行进程已退出\n\n'+msg+'\n\n先读GRID301_user_stop_report/user_stop_complete/user_stop_raw_manifest/STOP_REQUEST和发布核验。旧pipeline运行快照、旧PID和启动准备complete不代表正在运行或341全部完成。collection_error_recovery03的stop requested是人为停止，不当新工程故障。原24h截止仍11月6日14:20:46，但当前用户停止优先，不自动恢复。原始数据已D归档，无需重复。\n',encoding='utf8')
 save(O/'pipeline_state.json',{'phase':'user_stopped_paused','new_accepted':300,'planned_new':341,'partial_index':336,'partial_saved_steps':1079,'not_started':40,'no_auto_resume':True})
 save(O/'decision.json',{'status':'USER_STOPPED_PAUSED','new_accepted':300,'new_training_qualified':False,'new_training':0,'cost_targets_derived':0,'no_automatic_resume':True})
 save(O/'NEXT_PROTOCOL_PLAN.json',{'status':'USER_STOPPED_PAUSED','allowed_now':'finish stop delivery verification only; no model/prediction/physics/labels/training','new_user_instruction_required':True,'STOP_REQUEST_must_remain':True,'no_paid_record_repeat':True,'no_new_clock':True})
 q=read(O/'recovery03_runtime_doc_queue.json');q.update(status='READY_FOR_DELIVERY_AFTER_USER_STOP',stop_receipt='user_stop_complete.json',all_processes_exited=True);save(O/'recovery03_runtime_doc_queue.json',q)
 A.mkdir();C.mkdir()
 names=['GRID301_user_stop_report.md','user_stop_complete.json','user_stop_raw_manifest.json','STOP_REQUEST.json','manifest.json','selected_markers.json','baseline_publication.json','baseline_ready.json','load_only_review_recovery03.json','physical_authorization.json','recovery03_authorization.json','recovery03_launch_verification.json','recovery03_launch_process.json','started_recovery03.json','launch_context.json','recovery03_runtime_launch_status.json','recovery03_first_live_progress.json','recovery03_runtime_doc_queue.json','runtime_git_policy_recovery03.md','collection_error_recovery03.json','active_worker.json','pipeline_state.json','decision.json','NEXT_PROTOCOL_PLAN.json','CONTINUOUS_HANDOFF.md']
 for n in names:shutil.copy2(O/n,A/n);shutil.copy2(O/n,C/n)
 shutil.copy2(Path(__file__),A/Path(__file__).name)
 save(A/'archive_manifest.json',{p.name:sha(p) for p in A.iterdir() if p.is_file()})
 save(C/'delivery_hashes.json',{n:sha(C/n) for n in names})
 print('USER_STOP_ARCHIVE_PASSED',len(entries),raw['logical_bytes'],len(objects),complete['archive_wall_s'],flush=True)
if __name__=='__main__':main()
