"""One byte-only preservation of completed generation after publication timeout."""
from pathlib import Path
import datetime,hashlib,json,shutil,time,zipfile
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid300_new_training_dates';A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates';B=R/'archives/GRID300_generation_stop_20261104/raw_records';C=R/'论文协作资料/FRAME_新日期采集实施_GRID300/stopped_generation'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
start=time.perf_counter();assert not B.exists() and not (O/'generation_stop_complete.json').exists();error=read(O/'collection_error.json');assert error['completed_generation']==16 and error['completed_runs']==0
assert not (O/'runs').exists() and not (O/'generation_publication.json').exists()
generation=read(O/'generation_manifest.json');freeze=read(O/'generated_freeze.json');assert generation['completed']==16 and len(freeze)==208
# All originals remain in place. No profiles/arrays are parsed or recomputed.
files={R/p for p in freeze}
for pattern in ['generation_*.log','manager_*log','generation_receipts/*.json','claims/*']:
 files.update(p for p in O.glob(pattern) if p.is_file())
snapshots=['physical_authorization.json','launch_verification.json','launch_process.json','collection_started.json','runtime_launch_status.json','runtime_doc_queue.json','runtime_git_policy.md','pipeline_state.json','active_worker.json','run_context.json','collection_error.json','generation_progress.json','generation_manifest.json','generation_ready.json','generated_freeze.json','CONTINUOUS_HANDOFF.md']
files.update(O/n for n in snapshots if (O/n).is_file());B.mkdir(parents=True);entries=[];objects={}
with zipfile.ZipFile(B/'raw_objects.zip','x',allowZip64=True) as z:
 for p in sorted(files):
  assert p.resolve().is_relative_to(R.resolve())
  if time.perf_counter()-start>600:raise TimeoutError('one600s stop archive window; preserve partial')
  data=p.read_bytes();h=hashlib.sha256(data).hexdigest();rel=p.relative_to(R).as_posix()
  if rel in freeze:assert h==freeze[rel],rel
  name='objects/'+h[:2]+'/'+h;entries.append({'path':rel,'sha256':h,'bytes':len(data),'object':name})
  if h not in objects:z.writestr(name,data,compress_type=zipfile.ZIP_STORED if p.suffix in ['.bz2','.gz','.npz','.zip'] else zipfile.ZIP_DEFLATED,compresslevel=1);objects[h]={'bytes':len(data),'object':name}
with zipfile.ZipFile(B/'raw_objects.zip') as z:
 for h,v in objects.items():
  if time.perf_counter()-start>600:raise TimeoutError('stop archive readback window')
  data=z.read(v['object']);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==h
raw={'passed':True,'local_archive':str(B),'logical_files':len(entries),'logical_bytes':sum(e['bytes'] for e in entries),'unique_objects':len(objects),'zip_bytes':(B/'raw_objects.zip').stat().st_size,'zip_sha256':sha(B/'raw_objects.zip'),'independent_object_readback':True,'entries':entries,'raw_contents_uploaded':False}
save(B/'raw_records_manifest.json',raw);save(O/'generation_stop_raw_manifest.json',raw)
closed={'status':'GENERATION_PUBLICATION_TIMEOUT_CLOSED','generated_scenarios':16,'generated_files':208,'accepted_baseline_runs':0,'accepted_opposite_runs':0,'accepted_replay_steps':0,'accepted_env_forecasts':0,'generation_wrapper_wall_s_sum':sum(r['wall_s'] for r in generation['records']),'recorded_manager_elapsed_s':error['wall_s'],'original_generation_cap_s':7200,'deadline_check_lateness_s':error['wall_s']-7200,'original_total_start':read(O/'collection_started.json')['monotonic_start'],'original_total_deadline':read(O/'collection_started.json')['deadline'],'new_fits':0,'new_labels':0,'no_automatic_restart':True,'archive_manifest_sha256':sha(O/'generation_stop_raw_manifest.json'),'archive_wall_s':time.perf_counter()-start,'not_method_failure':True}
save(O/'generation_stop_complete.json',closed)
tz=datetime.timezone(datetime.timedelta(hours=8));times={n:datetime.datetime.fromtimestamp((O/n).stat().st_mtime,tz).isoformat() for n in ['generation_ready.json','collection_error.json']}
report=f'''# GRID300：16场景已生成，发布等待超时停机

实际完成16/16个2038场景、208文件，原冻结生成质量检查均通过；生成wrapper累计{closed['generation_wrapper_wall_s_sum']:.6f}秒。原24h新范围已批准且授权60b8b542先发布，source/model/date/seed不变。基线0/144、相反动作0/最多576，未进入原NN/头预测或新费用模型训练，尚无配对后缀费用标签。

生成数据在{times['generation_ready.json']}保存就绪，管理器等待发布；原生成子窗截止为11月4日18:03:12（北京时间）。外部定时跟进未及时执行放行。原no_retry守卫随后记录TimeoutError并退出，错误文件时间{times['collection_error.json']}，记录的manager总elapsed={error['wall_s']:.6f}秒，比7200秒子窗晚{closed['deadline_check_lateness_s']:.6f}秒。不能声称恰在7200秒执行退出；该迟检期间处于发布等待，无生成或回放worker。现已核对所有GRID300 Python进程退出。

这是外部跟进与管理器发布等待衔接的流程问题，未检验新模型优劣。原始pipeline_state停留pending是退出前快照，当前以collection_error/generation_stop_complete为准。不重跑任何场景、不安装依赖、不改模型/标签/费用或重置原时钟。

全部原始新场景、16生成收据、日志、claims和原控制快照已在D字节保留：{B.as_posix()}。{raw['logical_files']}逻辑文件/{raw['logical_bytes']}字节，{raw['unique_objects']}内容对象，ZIP{raw['zip_bytes']}字节，逐对象独立读回SHA通过。208新场景文件逐一匹配管理器已保存generated_freeze；没有解压数组做新数值审计、重运行生成器/OPF或读取模型。大原始内容仅D，Git仅必要元数据/引用/停机收据与报告。旧297等完整归档未重复。

16份已生成数据可保留复用。若拟继续，必须另行固定并审阅一次“生成后受控衔接”：原生成2h子窗失败历史保留；不再生成，只跑尚未开始的144基线及最多576相反动作；仍沿原总start9652.4935945/deadline96052.4935945（11月5日16:03:12）剩余时间，绝不重新获得24h。应把无worker阶段发布改为管理器自行同步执行并核helper退出/Gitblob/远端，消除等下一次heartbeat的必要性。原no_retry不允许自动重启，需要用户对该一次衔接明确批准；当前没有启动任何恢复进程。
'''
(O/'GRID300_generation_stop_report.md').write_text(report,encoding='utf8')
msg='GRID300已生成16场景/208文件，wrapper5051.926秒；generation_ready17:29:29后发布未及时放行，原2h子窗超时退出，记录elapsed7802.862秒（迟检602.862秒仅等待）。基线/相反动作均0，0NN预测/训练/标签；全部进程已退出，不自动重启或重生成，原总deadline仍11月5日16:03:12。新数据已D无损归档并对象SHA读回，生成后衔接须新明确批准。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID300发布等待超时停机，16场景已保留\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if n=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-04 — GRID300生成后发布等待停机\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID300停机，无自动重启\n\n'+msg+'\n\n1. 完成停机报告/生成源清单/本地归档与Git核对，禁止执行旧publish_stage放行已过期子窗。\n2. 仅准备一次复用既有16场景、沿原总deadline的受控衔接，改自动同步发布，不重复资格或物理条目。\n3. 新衔接源码/守卫可审阅并核对后，须用户明确批准才启动；未答或原总截止过期则保持停止。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID300生成后发布等待超时停机\n\n[报告](protocols/GRID300_new_training_dates/stopped_generation/GRID300_generation_stop_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
(O/'CONTINUOUS_HANDOFF.md').write_text('# 当前：GRID300生成后发布超时已停机\n\n'+msg+'\n\n先读GRID300_generation_stop_report/generation_stop_complete/collection_error和原collection_started。旧pipeline_state pending是退出前历史，不执行旧publisher/manager，不重试或重新生成。原24h总钟不停，受控衔接需另行源码/数据绑定与真实新批准。\n',encoding='utf8')
save(O/'pipeline_state.json',{'phase':'generation_publication_timeout_closed','generated_scenarios':16,'completed_replays':0,'original_deadline':closed['original_total_deadline'],'error_receipt':'collection_error.json','no_retry':True})
save(O/'NEXT_PROTOCOL_PLAN.json',{'status':'GENERATION_PUBLICATION_TIMEOUT_CLOSED','new_runtime_authorization_required':True,'original_deadline':closed['original_total_deadline'],'allowed_now':'byte preservation and a reviewable post-generation continuation within original total window; no restart or regeneration','new_training_qualified':False})
S=A/'stopped_generation';S.mkdir(exist_ok=False);C.mkdir(exist_ok=False)
names=['GRID300_generation_stop_report.md','generation_stop_complete.json','generation_stop_raw_manifest.json','generated_freeze.json','generation_manifest.json','generation_ready.json','collection_error.json','collection_started.json','physical_authorization.json','launch_verification.json','launch_process.json','runtime_launch_status.json','runtime_doc_queue.json','runtime_git_policy.md','CONTINUOUS_HANDOFF.md','NEXT_PROTOCOL_PLAN.json','pipeline_state.json']
for n in names:shutil.copy2(O/n,S/n);shutil.copy2(O/n,C/n)
shutil.copy2(Path(__file__),S/'close_generation_stop.py');shutil.copy2(O/'manager_stderr.log',S/'manager_stderr.txt');save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()});save(C/'delivery_hashes.json',{n:sha(C/n) for n in names});save(B/'archive_complete.json',{'passed':True,'raw_manifest_sha256':sha(B/'raw_records_manifest.json'),'zip_sha256':raw['zip_sha256']})
print('GRID300_GENERATION_STOP_ARCHIVED',raw['logical_files'],raw['logical_bytes'],raw['unique_objects'],time.perf_counter()-start)
