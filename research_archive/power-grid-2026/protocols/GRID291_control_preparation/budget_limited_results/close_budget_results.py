"""Lossless D: raw archive and report delivery; no model or environment imports."""
from pathlib import Path
import collections, datetime, hashlib, json, shutil, zipfile
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid291_control_preparation';I=R/'work/GRL-results-integration'
# Read the already-used stage constant, do not invent an archive alias.
import ast
tree=ast.parse((W/'guard.py').read_text(encoding='utf8'))
namespace={};exec(compile('\n'.join((W/'guard.py').read_text(encoding='utf8').splitlines()[:4]),str(W/'guard.py'),'exec'),namespace)
A=namespace['A'];L=R/'archives/GRID291_control_replay_20261101';C=R/'论文协作资料/FRAME_真实控制结果_GRID291'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def archive_raw():
 v=read(O/'saved_partial_review.json');assert v['passed_saved_data'] and v['accepted_runs']==186
 dest=L/'raw_records';dest.mkdir(parents=True,exist_ok=False);objects=dest/'objects';objects.mkdir()
 files=sorted(p for p in (O/'runs').rglob('*') if p.is_file())
 names=['manifest.json','pipeline_state.json','active_worker.json','pipeline_error.json','timeout.json','benchmark_started.json','launch_resume01_started.json','launch_resume01.json','physical_authorization.json','design.json','source_freeze.json','model_references.json','budget_stop_plan.json']
 files+=sorted({p for pattern in ['resume01_worker_*.log','pipeline_resume01_*.log','worker_result_*.json','worker_setup_*.json','manager_receipt_*.json'] for p in O.glob(pattern)})
 files+=[O/n for n in names];assert len(files)==len(set(files))
 entries={};known={};large={}
 with zipfile.ZipFile(dest/'small_files.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for j,p in enumerate(files):
   h=sha(p);n=p.stat().st_size
   if h not in known:
    if n>=1048576:
     q=objects/(h+'.blob');shutil.copyfile(p,q);known[h]='objects/'+q.name;large[h]=n
    else:z.write(p,h);known[h]='small_files.zip!'+h
   entries[p.relative_to(O).as_posix()]=dict(sha256=h,bytes=n,location=known[h])
   if j%1000==0:print('ARCHIVED_FILE',j,len(files),flush=True)
 with zipfile.ZipFile(dest/'small_files.zip') as z:
  for h,loc in known.items():
   actual=hashlib.sha256(z.read(h)).hexdigest() if loc.startswith('small_files') else sha(dest/loc)
   assert actual==h
 manifest=dict(files=entries,logical_files=len(entries),logical_bytes=sum(x['bytes'] for x in entries.values()),unique_objects=len(known),large_objects=len(large),large_unique_bytes=sum(large.values()),small_zip_sha256=sha(dest/'small_files.zip'),format='Exact original bytes deduplicated only by full file SHA256; every logical path preserved. Large raw vectors/calls remain in D local archive; Git receives this manifest, aggregate saved evidence, source and receipts.')
 save(dest/'records_manifest.json',manifest);shutil.copy2(dest/'records_manifest.json',O/'raw_records_manifest.json')
 save(O/'raw_archive_review.json',dict(passed=True,archive_directory=str(dest),logical_files=len(entries),logical_bytes=manifest['logical_bytes'],unique_objects=len(known),large_unique_bytes=sum(large.values()),small_zip_bytes=(dest/'small_files.zip').stat().st_size,every_unique_object_readback=True,records_manifest_sha256=sha(dest/'records_manifest.json'),raw_files_published_to_git=False,reason='Preserve complete/partial raw data locally on D; avoid large vectors/checkpoints/dependencies in repository, publish reproducible hashes and necessary evidence.'))
 print('RAW_ARCHIVE_PASSED',json.dumps(read(O/'raw_archive_review.json')),flush=True)
def report():
 v=read(O/'saved_partial_review.json');assert v['passed_saved_data'];rs=v['records'];ps=v['matched_pairs'];m=v['macros'];pct=lambda x:f'{100*x:+.6f}%'
 assert len(ps)==93 and all(p['same_horizon'] and p['same_trajectory'] for p in ps)
 original=O/'GRID291_report.md';old=O/'GRID291_preparation_report_before_results.md'
 if not old.exists():shutil.copy2(original,old)
 a=m['all_available_pairs'];first=m['repeat0_complete_design'];last=[p for p in ps if p['repeat']==1]
 mean=lambda rows,k:sum(r[k] for r in rows)/len(rows)
 lines=['# GRID291：208与261真实控制回放，预算到限结果','',
 '结论：本轮保住了与208一致的实际动作轨迹和成本，但未建立261稳定的控制提速。原4小时上限严格停止，接受186/192条；另1条仅192步，最后5条未启动。不是192条完成，也不是方法失败。原准备报告另存，不覆盖历史科学记录。','',
 '## 范围与完整覆盖','',
 '固定4个原已暴露validation日期scenario100、环境seed0、4fold×3seed共12pool（含f0_s1 KEEP）、PARENT208/BOUNDED261、两次严格逆序计划192条。第1重复96条全部有收据；第2重复接受90条。186条组成93个完整的双臂配对（第1重复48对、第2重复45对），两臂各93条；并非根据结果删去条目。第二重复缺三组March配对：f0_s0/f0_s1/f0_s2。',
 '原monotonic_start=5185.1399816，deadline=19585.1399816；第185项保存后仅剩21.070638秒。index186在原窗口到限停止，其failure保存192步/1608次forecast/10.456099秒控制；最后5条未启动。原manager报TimeoutExpired属于总预算耗尽，不当模型或方法失效。运行进程均退出，无延长、补跑或第二运行器。',
 f"186条接受轨迹共{sum(r['steps'] for r in rs):,}物理步、{sum(r['total_queries'] for r in rs):,}次forecast；部分条另计，不并入完整配对。原6000000查询和387264步上限未超。零训练/新日期/新FRAME/最终8读取。",'',
 '## 完成、成本和质量','',
 '93对的观测、动作、逐步成本和结束位置签名逐项一致，完整周成本差为0；可用90个同臂跨重复核对的轨迹也全部一致。两臂各69条完成整周、24条May在1390步结束。质量按FULL完整且本方法完成并成本≤FULL的100.5%，未完FULL只标REFERENCE_LIMITED；没有混用旧FULL时延。','',
 '| 日期 | 接受条数（两臂） | 步数 | 质量 | 相对完整FULL成本范围 |',
 '|---|---:|---:|---|---:|']
 for week in sorted({r['week'] for r in rs}):
  rows=[r for r in rs if r['week']==week];rat=[r['full_cost_ratio']-1 for r in rows if r['full_cost_ratio'] is not None]
  lines.append(f"| {week} | {len(rows)} | {rows[0]['steps']} | {','.join(sorted({r['quality'] for r in rows}))} | {(pct(min(rat))+' 至 '+pct(max(rat))) if rat else '参照未完，不判整周质量'} |")
 lines+=['','March两臂所有接受配置均超过100.5%质量门限，不能把“与208成本一致”写成全部质量通过。August/October通过，May参照和两臂都未完周，不能把1390步当整周或提速证据。','',
 '## 实际控制、查询及尾部延迟','',
 '下表先在同pool/日期/重复内计算261/208−1，再对可用配对等权；负号为减少。预算导致第二重复缺项，因此同时报告完整设计的首重复，不能把93对称为完整双重复均衡估计。所有93对仍完整保留。','',
 '| 指标 | 首重复48对（设计完整） | 全部93对（预算缺项） |','|---|---:|---:|']
 for k,label in [('controller_change','完整ctl.propose控制时间'),('candidate_query_change','候选查询'),('total_query_change','总forecast'),('all_p95_change','所有步骤P95'),('all_p99_change','所有步骤P99'),('search_p95_change','搜索步骤P95'),('search_p99_change','搜索步骤P99')]:lines.append(f'| {label} | {pct(first[k])} | {pct(a[k])} |')
 lines+=['',f"第二逆序重复45对控制时间宏变化{pct(mean(last,'controller_change'))}；全93对43个更快、50个更慢。三seed宏控制变化为"+'、'.join(f"seed{s} {pct(v['perseed'][str(s)]['controller_change'])}" for s in [0,1,2])+'，不据此择seed。',
 '两臂原始候选查询总差仅−2、总forecast差也−2：只在f2_s1的March同一状态，两次计时各取消1条最终候选，动作/后续轨迹不变。该配对控制反而分别增加0.186179%和1.825857%；不能将查询微省直接归因于实际提速，也不是跨seed重复。',
 '原始每条均值（当前可用且匹配的93条/臂，作为描述值）：','',
 '| 方法 | 控制秒 | 候选query | 总query | 全步骤P95毫秒 | 搜索P95毫秒 |','|---|---:|---:|---:|---:|---:|']
 for arm in ['PARENT208','BOUNDED261']:
  rows=[r for r in rs if r['arm']==arm];lines.append(f"| {arm} | {mean(rows,'controller_s'):.6f} | {mean(rows,'candidate_queries'):.6f} | {mean(rows,'total_queries'):.6f} | {1000*mean(rows,'all_p95'):.6f} | {1000*mean(rows,'search_p95'):.6f} |")
 lines+=['','P95/P99为每条轨迹自身分位数后宏比较，并非把全部步骤混成独立样本。优化器、所有forecast、公开特征、NN/208/261推理、搜索其余和控制其余互斥分解通过，逐条time_parts保留；加载和启动计入原4小时但在ctl.propose外单列。没有把单状态推理时间或早停当完整加速。','',
 '## 已保存记录核查与反例','',
 f"核对{v['checks']['saved_vector_rows']:,}条动作/观测向量、{v['checks']['candidate_feedback_rows']:,}条已付费候选反馈、全部逐步成本账本、查询计数、候选别名和最佳已观察合法动作、实际模型公开输入/原nativeSTOP/父与261决定、可选一次补批及互斥计时。658次原NN STOP（两臂各329次）中，261额外取消仅2次；实际原NN决策1356次。无新增模型预测或独立AC重求解。工程输入回退0，非法/歧义步骤0；48个异常结束步骤对应May两臂原轨迹，不能隐去。",
 '首轮只读核对错误：exceptions字段为列表和returned动作/含base动作哈希语义在执行前修正留源；实际第一次只读脚本因NPZ成员在逐步循环重复解压而在首条核对完成前停止。修复仅将每个NPZ成员读入一次，断言不变，后续首次完成186条保存核对。未重跑物理或完整审计；saved_review.py的192完成入口没有执行，saved_partial_review.json明确budget limited。','',
 '## 研究裁决与后续边界','',
 '保留261作为本轮与208质量/轨迹一致的开发候选，以及唯一状态两次重复省1查询的小线索；本轮不支持稳定实际提速，亦未修复March质量失败。不把它升级为整周质量胜出或独立泛化。208/261继续冻结，不回开265—284森林逐项修补、不换名重训成本分类器、不为完成192而追加时间。',
 '下一项若继续，只能先固定一个有限的已保存记录诊断：解释实际329次父入口中261只额外取消1个唯一状态的激活范围及与原自身训练支持的差距。使用已发表输入/决定，不训练、不新评分/回报/物理、不按已看结果挑阈值/seed。必须先有具体输入和停止条件，不能为了自动推进继续接口核查或盲试模型。新的物理回放/日期/预算需要另外确定范围，不自动启动。',
 '原父own12/内部cal4祖先和已暴露日期、相关前缀/seed/fold非独立限制保持。f0_s1 KEEP不当新学习覆盖；不拼成功子集，不设全赢或5%新门槛。历史190拟合、6续训8285updates、参考NN累计604.703秒/用户授权678.156秒仍另列，本轮0拟合。','',
 '## 文件与归档','',
 '- saved_partial_review.json：全部186逐条结果、93配对、90重复核对、所有缺项、宏口径和P95/P99。',
 '- run_metrics.csv / saved_decisions.json：逐条指标与已保存模型决定清单。',
 '- 原186完整及1部分轨迹、调用原输入/反馈/向量/日志位于outputs/.../runs；D盘lossless内容寻址归档与raw_records_manifest.json保留全部路径和原字节hash。',
 '- 大体积原始向量/调用留D本地完整归档，Git发布源码、原超时、必要结果与完整hash清单；不无差别上传依赖或覆盖旧完整归档。',
 f'- D完整归档：{L}/raw_records；D结果协作副本：{C}。']
 original.write_text('\n'.join(lines)+'\n',encoding='utf8')
 decision=dict(status='BUDGET_LIMITED_CLOSED',accepted_runs=186,planned_runs=192,partial_index=186,unstarted=5,scientific='Same actual trajectories/cost as208 on93 observed matched pairs; only two queries saved in one repeated state; stable end-to-end speedup not established; March quality failure remains.',resume_authorized=False,additional_training_or_physics_started=False,next_scope='Only preregistered finite saved activation/support diagnosis, no new forecasts or outcome scores.',main_anchors=['208','261'],closed_forest_branch='265-284')
 save(O/'runtime_results_decision.json',decision)
 print('REPORT_WRITTEN',flush=True)
def deliver():
 assert read(O/'raw_archive_review.json')['passed'];assert read(O/'saved_partial_review.json')['passed_saved_data']
 names=['GRID291_report.md','GRID291_preparation_report_before_results.md','saved_partial_review.json','saved_decisions.json','run_metrics.csv','raw_records_manifest.json','raw_archive_review.json','postrun_model_hashes.json','runtime_results_decision.json','budget_stop_plan.json','readout_source_note01.json','readout_io_note03.json','timeout.json','pipeline_error.json','manifest.json','saved_partial_review_resumed_io03.log']
 stage=A/'budget_limited_results';stage.mkdir(exist_ok=False);C.mkdir(exist_ok=False)
 for n in names:
  shutil.copy2(O/n,stage/n);shutil.copy2(O/n,C/n);assert sha(O/n)==sha(stage/n)==sha(C/n)
 for p in [W/'saved_partial_review.py',W/'close_budget_results.py']:shutil.copy2(p,stage/p.name)
 save(stage/'archive_manifest.json',{p.name:sha(p) for p in stage.iterdir() if p.is_file()})
 save(C/'delivery_hashes.json',{n:sha(C/n) for n in names})
 msg='GRID291在原14400秒上限停止，186/192接受，另192步部分条/5未启动；93匹配轨迹及成本全同、90可用重复全同，0新拟合。261仅同一March f2_s1状态两重复各省1query，完整首重复48对控制-0.328563%、第二重复45对+0.037920%、全93对-0.151232%且搜索P95+0.751516%，不建立稳定实际提速。March两臂42条均FAIL_COST，August/October96条PASS、May48条REFERENCE_LIMITED；完整/部分与全部反例留D lossless归档，必要结果及清单发布。原窗口/协议不变，不补跑，208/261冻结，关闭森林修补，最终8不动。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=I/'docs'/n;t=p.read_text(encoding='utf8')
  entry='\n\n## 2026-11-01 — GRID291预算到限真实控制结果\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':t='# 当前：GRID291原4小时到限，保存结果收尾完成\n\n'+msg+'\n\n以下保留历史记录，当前以本段和NEXT_STEPS为准。\n\n'+t
  else:t+=entry
  p.write_text(t,encoding='utf8')
 nexttext='# 当前 — GRID291预算到限，不补跑\n\n1. 186完整收据及1条192步部分输出已首次保存核对；不调用192入口saved_review，不再manager/worker，不重做审计/完整归档或预测。\n2. 阅读outputs/grid291_control_preparation/GRID291_report.md、saved_partial_review.json与runtime_results_decision.json：93匹配轨迹/成本一致、查询仅少2，不支持稳定控制提速，March质量仍失败；原4小时/全部缺项保留。\n3. 上传与D协作/清单/具体Gitblob/远端SHA收尾核对后，更新runtime_complete与持续交接。\n4. 下一步仅允许先固定一项有限已保存激活/自身训练支持诊断，定位实际父入口中的残差取消范围；先写清输入和终止条件，0拟合/新评分回报/物理。不为自动推进无穷追加接口核查/森林修补。\n5. 新物理/日期/预算不自动启动，最终8/费用标签/控制器/原种子及历史190拟合/6续训边界不动。\n'
 (I/'docs/NEXT_STEPS.md').write_text(nexttext,encoding='utf8')
 p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID291原4小时到限，186/192保存结果\n\n[真实控制报告](protocols/GRID291_control_preparation/budget_limited_results/GRID291_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
 hand='# GRID291真实回放已预算到限，禁止补跑\n\n'+msg+'\n\n先读integration/docs/RESEARCH_STATE和NEXT_STEPS、GRID291_report.md、saved_partial_review.json、runtime_results_decision.json。benchmark进程均已退出，当前pipeline_state原benchmark是历史快照，active_worker也是历史PID，不重启。原pipeline_error/timeout属于总窗耗尽，finished不存在；complete是旧准备收据，不冒充192完成。saved_partial_review已passed不重做；只读NPZ解压修复原源/日志已留并完成首次核对。D raw_records已全部hash读回，Git只必要清单/结果/源码。报告和结果新副本FRAME_真实控制结果_GRID291，旧准备副本保留。\n\n下一步只能先资格固定有限保存激活/自身训练支持诊断，不训练/新模型评分/回报/物理，不调阈值seed或调用例外，不新日期/FRAME/电网/论文/最终8。主锚208/261，森林修补关闭。只有发现新信息缺口且方案有限才推进，否则裁决结束支线。先完成本轮发布SHA/具体blob收尾再另阶段，已有完整项不重复。\n'
 (O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8');shutil.copy2(O/'CONTINUOUS_HANDOFF.md',stage/'CONTINUOUS_HANDOFF.md');shutil.copy2(O/'CONTINUOUS_HANDOFF.md',C/'CONTINUOUS_HANDOFF.md')
 save(stage/'archive_manifest.json',{p.name:sha(p) for p in stage.iterdir() if p.is_file() and p.name!='archive_manifest.json'})
 print('DELIVERY_WRITTEN',flush=True)
if __name__=='__main__':
 import sys
 globals()[sys.argv[1]]()
