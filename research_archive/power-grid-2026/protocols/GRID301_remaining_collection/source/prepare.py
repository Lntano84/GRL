"""Prepare a reviewable remaining-work scope; never authorize or simulate."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';W=Path(__file__).parent
OLD=R/'outputs/grid300_new_training_dates';O=R/'outputs/grid301_remaining_collection'
A=I/'research_archive/power-grid-2026/protocols/GRID301_remaining_collection'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 assert not O.exists() and not A.exists();O.mkdir(); A.mkdir()
 original=R/'work/grid300_new_training_dates/continuation01'
 for n in ['state.py','adapter.py','worker.py','auto_publish.py']:
  shutil.copy2(original/n,W/n)
 # Only the output namespace changes in static assembly; no NN/math changes.
 s=(original/'source_builder.py').read_text(encoding='utf8').replace("OUT=ROOT/'outputs/grid300_new_training_dates'","OUT=ROOT/'outputs/grid301_remaining_collection'")
 assert s!=(original/'source_builder.py').read_text(encoding='utf8');(W/'source_builder.py').write_text(s,encoding='utf8')
 s=(original/'publish_stage.py').read_text(encoding='utf8').replace("started=read(O/'collection_started.json');ctx={'deadline_monotonic':started['deadline']}","started=read(O/'launch_context.json');ctx=started.copy()")
 s=s.replace("started['deadline']-__import__('time').perf_counter()","started['deadline_monotonic']-__import__('time').perf_counter()")
 assert "collection_started.json" not in s and "started['deadline']" not in s
 (W/'publish_stage.py').write_text(s,encoding='utf8')
 d=read(OLD/'design.json');m=read(OLD/'manifest.json');assert [r['index'] for r in m['records']]==list(range(35))
 assert read(OLD/'continuation01_interruption_complete.json')['partial_index']==35
 d.update(id='GRID301_REMAINING_BASELINES_AND_OPPOSITES',status='PREPARATION_NOT_AUTHORIZED',reuse_indices=list(range(35)),skip_partial_indices=[35],new_baseline_indices=list(range(36,144)),max_new_opposites=572,max_new_runs=680,wall_cap_s=86400,generation_new=0,generation_reused=16,old_scope_closed=True)
 d['baseline_runtime']['caps'].update(wall_s=86400,physical_steps=1371560,public_forecasts=22500000)
 save(O/'design.json',d);save(O/'runtime_design.json',d['baseline_runtime'])
 shutil.copy2(OLD/'pool_roles.json',O/'pool_roles.json')
 bindings=[]
 for r in m['records']:
  item=d['baseline_runtime']['execution_order'][r['index']];folder=OLD/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
  bindings.append({'index':r['index'],'pool':item['pool'],'date':item['date'],'folder':folder.relative_to(R).as_posix(),'summary_sha256':sha(folder/'summary.json'),'intervention_sha256':sha(folder/'intervention.json'),'original_record_retained':True})
 save(O/'reuse_bindings.json',{'records':bindings,'old_manifest_sha256':sha(OLD/'manifest.json'),'old_raw_manifest_sha256':sha(OLD/'continuation01_interruption_raw_manifest.json'),'skip_partial_index':35,'partial_saved_steps':731,'no_repeat':True})
 # Resource estimate uses saved episode wall summaries, not fresh timing.
 walls=[r['summary']['wall_s'] for r in m['records']]
 estimate={'saved_episode_count':35,'saved_mean_episode_wall_s':sum(walls)/35,'new_max_runs':680,'naive_new_episode_wall_estimate_s':sum(walls)/35*680,'new_proposed_total_wall_cap_s':86400,'new_generation':0,'old_sunk_accepted_steps':68294,'old_partial_saved_steps':731,'old_accepted_forecasts':219103,'old_partial_saved_forecasts_lower_bound':2834,'old_unlogged_work_unknown':True,'resource_passed':False,'not_a_guarantee':'Prior35 include10 early-ended episodes; remaining pools/relative scheduling and new head/solver behavior may differ. All startup/loading/publication/failures count in new budget.'}
 save(O/'resource_estimates.json',estimate)
 plan={'status':'SOURCE_PREPARATION_NO_NEW_PHYSICAL_AUTHORIZATION','scope_id':d['id'],'new_generation':0,'reused_generation':16,'reused_baselines':35,'new_baselines':108,'skip_partial_index':35,'max_new_opposite_runs':572,'max_new_runs':680,'proposed_wall_cap_s':86400,'worker_cap_s':1200,'physical_steps_cap':1371560,'forecasts_cap':22500000,'new_budget_requires_explicit_user_reply':True,'old_deadline_not_restarted_or_extended':True,'new_cost_targets':0,'new_fits':0,'fit8_cal4_outer4_and_KEEP_preserved':True,'missing_pool_date_row':d['baseline_runtime']['execution_order'][35],'new_boot_stops_no_auto_resume':True}
 save(O/'collection_proposal.json',plan)
 (O/'GRID301_report.md').write_text('''# GRID301：复用已保存进度的剩余采集准备

用户2026年11月5日要求继续，当前只准备剩余工作的实施和审阅，不把“继续”解释为新增24小时物理预算批准。旧GRID300中断与原截止保持关闭，旧管理器永不重启。

复用16个2038场景和35条已接纳基线（25完整周、10提前结束）。原index35的731步部分轨迹保留，但不重跑；其余index36..143共108条基线尚未启动。所有可用基线143条（35旧+108新）均按原四个物理时间段最早nativeSTOP规则选择，最多572条一次相反动作，共最多680条新回放。不会按成本、完成、日期或种子挑选成功子集。f0_s2的2038-12-06缺一条基线及其最多四个干预点，必须保留此缺失；全部12池及KEEP仍按原角色清单处理，不声称144条齐全。

新提议总预算24小时、单条1200秒、新物理步上限1371560、新forecast上限22500000；复用旧生成，不重新给旧24小时。旧已接纳68294步/219103forecast和731步/至少2834forecast部分耗费另列，在途未知不算零。旧运行平均耗时粗估见resource_estimates.json：含10条早停，不能保证新范围能完成。新窗口含启动、加载、失败、基线封存及同步发布等待；任一上限先到即停，部分不补跑，不自动恢复或延窗。

实现采用独立GRID301目录，避免改写旧运行目录。复用原state/adapter/worker及原NN12到208/261数值逻辑；静态装配仅改输出目录。新管理器持久单次认领、缺失index35显式排除、已保存35条仅引用；每条完成即时保存新manifest。基线来源先同步发布并核Git helper退出/具体blob/远端，才允许相反动作。只反转一次原STOP/CONT后恢复原261，共同公共marker、before/after及完整联合动作前缀必须匹配，观测向量不冒充可恢复环境快照。

新时间守卫绑定明确的新授权、开机标识、UTC绝对截止和本次monotonic截止；任一时间到限即停。电脑重启后开机标识改变，或启动记录已存在，均拒绝旧任务恢复。没有新授权时launch/manager/worker/publisher都必须在真实模型和环境导入前拒绝。新授权发布后才有唯一启动；当前没有写physical_authorization或launch_context。

仅采集原始来源，不在采集中派生成本标签或训练。将来只有双方完整同2017步且共同前缀通过，另一个固定保存协议才能读出C_CONT-C_STOP；提前结束的安全完成信息另列。共享基线、own12/cal4祖先及同源合成日期仍相关，非全链OOF/盲测，query非真实时延。最终8、新电网、FRAME、论文和依赖不动，原费用NET奖励、排序别名、锁站128/64、停滞2/.9不变。

下一步是源发布后一次有限管理/路由索引/缺授权静态合成检查，零真实模型加载预测、标签或模拟。通过后才询问用户是否批准上述最多680条/新24小时范围；上传或继续整理不等采集授权。无授权不运行环境smoke。未来训练和192条开发验证均另行资格与预算。
''',encoding='utf8')
 save(O/'NEXT_PROTOCOL_PLAN.json',{'allowed_now':'source publication then one finite control-only preparation check; no physics','physical_scope_pending':True,'new_training_qualified':False,'report':'GRID301_report.md'})
 (O/'CONTINUOUS_HANDOFF.md').write_text('# GRID301：准备剩余采集，未批准未启动\n\n先读GRID301_report/design/collection_proposal/reuse_bindings/resource_estimates。旧GRID300已关闭，不重跑旧35或index35部分。新108基线+最多572相反动作/680条/提议新24h须独立明确批准。源先发后一次控制资格；0真实模型预测/物理/标签/训练。\n',encoding='utf8')
 # Existing scientific dependencies retain original hashes; only new files added.
 freeze=read(OLD/'continuation01_source_freeze.json')
 for p in [*W.glob('*.py'),*[O/n for n in ['design.json','pool_roles.json','reuse_bindings.json','collection_proposal.json']]]:
  freeze[p.relative_to(R).as_posix()]=sha(p)
 for n in ['manifest.json','continuation01_interruption_complete.json','continuation01_interruption_raw_manifest.json','model_freeze.json','static_freeze.json']:
  freeze[(OLD/n).relative_to(R).as_posix()]=sha(OLD/n)
 save(O/'source_freeze.json',freeze)
 S=A/'source';S.mkdir()
 for p in [*W.glob('*.py'),*[O/n for n in ['design.json','pool_roles.json','reuse_bindings.json','collection_proposal.json','resource_estimates.json','source_freeze.json','GRID301_report.md','NEXT_PROTOCOL_PLAN.json','CONTINUOUS_HANDOFF.md']]]:shutil.copy2(p,S/p.name)
 save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
 msg='GRID301已准备独立剩余采集源码：复用16场景/35基线，跳过index35部分，108未启动基线+最多572相反动作=最多680新回放，拟新24h/1200s单条。用户“继续”仅用于准备；未写新物理授权，0预测/模拟/训练。新UTC+monotonic+开机标识、单次认领和同步发布守卫；先源后一次控制资格，通过后才询问新预算。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=I/'docs'/n;old=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID301剩余采集实施准备，未获新预算\n\n'+msg+'\n\n以下保留历史。\n\n'+old) if n=='RESEARCH_STATE.md' else old+'\n\n## 2026-11-05 — GRID301剩余采集实施准备\n\n'+msg+'\n',encoding='utf8')
 (I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID301源准备，不启动物理\n\n'+msg+'\n\n1. 精确白名单发布新源、核远端/具体blob后一次有限管理资格，不重做旧资格/归档或84模型核验。\n2. 通过后才询问最多680条/新24h范围；旧300授权、原截止和grl旧监控均不恢复。\n3. 未获明确答复不写physical_authorization，不运行环境或真实预测。未来目标派生/训练/开发验证另定范围。\n',encoding='utf8')
 p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID301剩余采集实施准备\n\n[方案](protocols/GRID301_remaining_collection/source/GRID301_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
 print('SOURCE_PREPARED',len(freeze),len(bindings),estimate['naive_new_episode_wall_estimate_s'])
if __name__=='__main__':main()
