"""One byte-preserving shutdown archive. No model/policy/cost-label computation."""
from pathlib import Path
import hashlib,json,shutil,sys,time,zipfile,collections
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid297_paired_collection_preparation';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID297_paired_collection_preparation';C=R/'论文协作资料/FRAME_配对采集准备_GRID297';B=R/'archives/GRID297_paired_collection_20261102/raw_records'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 start=time.perf_counter();assert not (O/'runtime_complete.json').exists();assert not B.exists(),'Never repeat an existing full archive'
 finish=read(O/'finished.json');manifest=read(O/'manifest.json');design=read(O/'design.json');auth=read(O/'physical_authorization.json')
 assert finish['runs']==288 and finish['pairs']==144 and len(manifest['records'])==288 and len(design['execution_order'])==288
 assert finish['wall_s']<=28800 and manifest['total_steps']<=580896 and manifest['total_forecasts']<=9000000 and not (O/'collection_error.json').exists()
 assert sha(O/'source_freeze.json')==auth['source_freeze_sha256'] and sha(O/'design.json')==auth['design_sha256']
 for rel,h in read(O/'model_freeze.json').items():assert sha(R/rel)==h,rel
 coverage=[]
 for n,(record,item) in enumerate(zip(manifest['records'],design['execution_order'])):
  assert n==record['index']==item['index'];s=record['summary'];assert s['week']==item['week'] and s['pool']==item['pool'] and s['arm']==item['arm']
  p=O/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
  for name in ['summary.json','intervention.json','metadata.json','steps.jsonl.gz','vectors.npz']:assert (p/name).is_file(),(n,name)
  coverage.append(dict(index=n,pair=item['pair_index'],pool=item['pool'],date=item['date'],role=item['partition'],arm=item['arm'],KEEP=item['KEEP'],steps=s['steps'],complete_week=s['complete'],forecasts=record['intervention']['forecasts'],event_status=record['intervention']['status'],source_folder=p.relative_to(R).as_posix()))
 for n in range(144):
  receipt=read(O/f'pair_{n:03d}.json');assert receipt['pair_index']==n and receipt['prefix_matched'] and receipt['no_labels_derived']
 save(O/'runtime_coverage.json',coverage)
 # All raw bytes stay on D. Only this logical path/hash/size catalogue goes to Git.
 files=list((O/'runs').rglob('*'));files=[p for p in files if p.is_file()]
 for pattern in ['worker_*.log','worker_setup_*.json','worker_result_*.json','pair_*.json','manager_*log']:
  files.extend(p for p in O.glob(pattern) if p.is_file())
 if (O/'claims').exists():files.extend(p for p in (O/'claims').rglob('*') if p.is_file())
 files=sorted(set(files));B.mkdir(parents=True);logical=[];objects={};total=0
 with zipfile.ZipFile(B/'raw_objects.zip','x',allowZip64=True) as archive:
  for j,p in enumerate(files):
   if time.perf_counter()-start>600:raise TimeoutError('shutdown archival window reached; retain partial, do not repeat automatically')
   data=p.read_bytes();h=hashlib.sha256(data).hexdigest();member='objects/'+h[:2]+'/'+h
   logical.append(dict(path=p.relative_to(R).as_posix(),sha256=h,bytes=len(data),object=member));total+=len(data)
   if h not in objects:
    archive.writestr(member,data,compress_type=zipfile.ZIP_STORED if p.suffix in ['.npz','.gz','.zip'] else zipfile.ZIP_DEFLATED,compresslevel=1)
    objects[h]=dict(member=member,bytes=len(data))
   if j%1000==0:print('ARCHIVE_BYTES',j,len(files),total,flush=True)
 # Independent object readback validates stored bytes, without parsing arrays.
 with zipfile.ZipFile(B/'raw_objects.zip') as archive:
  for j,(h,obj) in enumerate(objects.items()):
   if time.perf_counter()-start>600:raise TimeoutError('shutdown archive verification window reached')
   data=archive.read(obj['member']);assert len(data)==obj['bytes'] and hashlib.sha256(data).hexdigest()==h
   if j%1000==0:print('ARCHIVE_READBACK',j,len(objects),flush=True)
 raw=dict(passed=True,local_archive=str(B),logical_files=len(logical),logical_bytes=total,unique_objects=len(objects),zip_bytes=(B/'raw_objects.zip').stat().st_size,zip_sha256=sha(B/'raw_objects.zip'),objects_independently_readback=True,entries=logical,contents_uploaded_to_Git=False)
 save(B/'raw_records_manifest.json',raw);save(O/'raw_records_manifest.json',raw)
 receipt=dict(passed_delivery=True,collection_complete=True,accepted_episodes=288,pairs=144,manager_finished_wall_s=finish['wall_s'],original_cap_s=28800,original_monotonic_start=read(O/'collection_started.json')['monotonic_start'],original_deadline=read(O/'collection_started.json')['deadline'],physical_steps=manifest['total_steps'],forecasts=manifest['total_forecasts'],new_fits=0,new_labels_derived=0,new_policy_predictions_this_closeout=0,paired_receipts144_verified=True,model_payload_hashes84_unchanged=True,raw_archive_manifest_sha256=sha(O/'raw_records_manifest.json'),raw_archive=str(B),archive_wall_s=time.perf_counter()-start,user_shutdown_pause=True,automation_status='PAUSED',note='288 episode execution/byte delivery complete, not all episodes full-week/quality success or a new learned result. No cost-target joins, trajectory numerical audit or new NN/physical calls. All raw records D only.')
 save(O/'runtime_complete.json',receipt)
 report=f'''# GRID297：288条采集结束，关机暂停

固定288/288条、144/144配对已全部执行并有管理器收据，原总窗口{finish['wall_s']:.3f}秒（{finish['wall_s']/3600:.3f}小时），未延长8小时。保存物理步{manifest['total_steps']}、总forecast {manifest['total_forecasts']}。原12池/全日期/KEEP与所有结束状态保留，零训练、未派生费用目标。本报告不是新模型收益或整周质量/提速结论；288条处理完成不代表每条都完成整周。

用户准备关机、最多等半小时，已暂停grl，不新增分析、训练或回放。全部144原配对收据显示共同前缀核对通过；本轮仅保存收据/字节与角色覆盖，不重做完整数值审计。后续C_CONT-C_STOP目标、完成/安全口径和支持分析必须在源记录先发布后另固定一次有限保存只读协议，待用户继续后执行。

全部原始轨迹/调用/日志仅在D本地无损归档：{B.as_posix()}。{len(logical)}逻辑文件、{total}原字节，按内容去重为{len(objects)}对象，压缩归档逐对象独立读回SHA通过。Git只发布必要报告、执行/授权收据、角色覆盖和路径/hash清单，不上传原始向量、调用轨迹或模型权重。源和模型引用保持，84模型载荷哈希未变。

首轮准备上传审批和日志忽略均为已解决历史，不是运行方法失败；本次未重试/补跑、未改阈值费用模型或安装依赖。原291186/192和最终8不动，own12父/内部cal4及已暴露日期和相关前缀/seed/折复用限制保留，非全链OOF/盲测。原queryNET、控制器排序别名锁站/128/64/停滞2/.9不变。历史190拟合与6续训8285updates、604.703实际/用户678.156cap仍另列。

可关机交接：用户明确继续后先读最新四docs、本目录runtime_complete/publication_closeout/CONTINUOUS_HANDOFF与原finished/manifest，不启动采集manager或重做本次完整归档。只讨论新的有限保存结果核查/标签协议，不能把原8小时授权当新物理预算。
'''
 (O/'GRID297_runtime_report.md').write_text(report,encoding='utf8')
 msg=f'GRID297固定288/288条及144配对执行完成，原{finish["wall_s"]:.3f}秒/28800上限，{manifest["total_steps"]}物理步/{manifest["total_forecasts"]}forecast，0训练/标签派生。全部原始{len(logical)}文件D无损归档并逐对象SHA读回，Git只必要报告/收据/清单。用户关机暂停，grl PAUSED；无新实验，完整数值审计/成本标签待继续后另固定有限协议，不重做采集或旧全归档。'
 hand='# 当前：采集288条完成；用户关机暂停，grl PAUSED\n\n'+msg+'\n\n先读integration AGENTS/四docs、本目录GRID297_runtime_report/runtime_complete/publication_closeout（若有）、finished/CONTINUOUS_HANDOFF。用户明确继续前不恢复grl、不开始新任务，不能重复manager/worker/归档/准备资格。原始轨迹仅D归档，必要报告与hash清单发布。当前只完成执行收据/字节交付，未派生新费用目标、做完整数值轨迹审计或训练。恢复后先形成一次有限保存结果核查和标签协议，源记录先发布后连结果，原祖先/暴露/KEEP/反例/最终8边界不变。\n'
 (O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8')
 save(O/'NEXT_PROTOCOL_PLAN.json',dict(status='USER_SHUTDOWN_PAUSED_COLLECTION_COMPLETE',scope='No further work until explicit user continue. Then one bounded saved-data review/target plan after exact source publication, not another physical run or model sweep.',do_not_repeat=['manager','worker','complete archive','prepare','qualify','old291audit'],labels_derived=False,new_training_qualified=False))
 for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=I/'docs'/name;t=p.read_text(encoding='utf8');p.write_text('# 当前：GRID297采集完成，用户关机暂停\n\n'+msg+'\n\n以下保留历史。\n\n'+t if name=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-02 — GRID297采集完成/关机暂停\n\n'+msg+'\n',encoding='utf8')
 (I/'docs/NEXT_STEPS.md').write_text('# 当前 — 等待用户明确继续，grl PAUSED\n\n'+msg+'\n\n1. 当前只完成必要保存/发布与交接，不新增分析、标签或模型。\n2. 用户继续后先读runtime_complete/publication_closeout/finished及最新交接，再固定一次有限保存审计/标签协议；不重做采集/完整归档或旧291。\n3. 全原日期seedKEEP及失败/缺失/结束状态保留，原模型/费用/控制器/最终8不动。\n',encoding='utf8')
 p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID297采集完成，关机暂停\n\n[结果](protocols/GRID297_paired_collection_preparation/runtime_collection/GRID297_runtime_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
 p=O/'GRID297_report.md';p.write_text('# 当前：288条采集已结束，关机暂停\n\n[采集交接结果](GRID297_runtime_report.md)。'+msg+'\n\n以下为历史准备/启动记录。\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
 S=A/'runtime_collection';S.mkdir(exist_ok=False)
 names=['GRID297_runtime_report.md','runtime_complete.json','runtime_coverage.json','raw_records_manifest.json','finished.json','physical_authorization.json','launch_verification.json','collection_started.json','runtime_launch_status.json','CONTINUOUS_HANDOFF.md','NEXT_PROTOCOL_PLAN.json']
 for name in names:shutil.copy2(O/name,S/name);shutil.copy2(O/name,C/name)
 shutil.copy2(Path(__file__),S/'shutdown_closeout.py');save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
 save(C/'runtime_delivery_hashes.json',{n:sha(C/n) for n in names});save(B/'archive_complete.json',dict(passed=True,logical_files=len(logical),unique_objects=len(objects),manifest_sha256=sha(B/'raw_records_manifest.json'),archive_sha256=raw['zip_sha256']))
 print('GRID297_SHUTDOWN_ARCHIVE_COMPLETE',len(logical),total,len(objects),time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
