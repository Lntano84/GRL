"""Deliver one reviewable remedy, no automatic resumption or new generation."""
from pathlib import Path
import datetime,hashlib,json,shutil,time
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid300_new_training_dates';W=Path(__file__).parent;A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates';C=R/'论文协作资料/FRAME_新日期采集实施_GRID300/continuation01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'continuation01_decision.json').exists();q=read(O/'continuation01_preparation_review.json');assert q['passed'] and not (O/'continuation01_authorization.json').exists() and not (O/'continuation01_started.json').exists() and not (O/'runs').exists()
original=read(O/'collection_started.json');remaining=original['deadline']-time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc)
report=f'''# GRID300：生成后受控衔接一次方案，尚未启动

原采集已停止：16/16场景及208文件生成完成，生成wrapper5051.926秒；17:29:29进入发布等待。外部定时跟进未及时放行，原2h子窗失败，管理器下一次守卫检查在elapsed7802.862秒记录超时并退出（迟检602.862秒，无活动生成/回放worker）。这不是模型效果失败，基线0/144、相反动作0/最多576，尚无配对费用标签、原NN/头预测或新训练。

原始数据和错误已完整保留并发布必要停机清单04870cf0；D字节归档290文件/104946434字节、230对象已逐对象独立读回，208场景文件匹配原generated_freeze。没有重做OPF、场景、旧资格或旧297等归档。当前仍没有恢复进程。

原设计把发布交给外部跟进，不能保证在生成2h内及时执行。这一流程设计需要修正。已准备独立continuation01版本，只改变一次失败后的控制衔接：

1. **复用既有16场景，新生成0。** 场景源清单、原失败记录和原起点已在48caca85的continuation01_source独立发表。原2h发布门失败历史保留，不补做“按时通过”收据、不释放旧门、不重新设生成窗口。
2. **原总截止不变。** 使用原monotonic_start9652.4935945和deadline96052.4935945，仍于北京时间11月5日16:03:12到限。报告时点UTC{utc.isoformat()}剩余约{remaining/3600:.3f}小时，等待用户期间继续消耗原窗口；不会再获得24小时。原单回放1200秒/1452240步/22500000forecast等上限不变。
3. **只执行尚未开始的回放。** 144基线及最多576相反动作仍同原日期/12模型池/角色/KEEP/四段公开最早事件规则，0训练与费用标签派生。不存在任何已完成或部分回放可重试；旧16生成claims和启动/错误文件原样保留，新manager用continuation01_started及claims_continuation01独立认领，仍禁止重复启动。
4. **发布由管理器同步执行。** 在144基线后没有活动worker的阶段，管理器自行调用固定publisher，等git_push.cmd退出、具体Gitblob和远端一致才继续相反动作。仍是同一绝对截止，失败保留部分停止；不等下次heartbeat，不改成后台另一路网络，也不与worker的HEAD/remote检查并发。

数值部分state/adapter/source_builder/worker/stages共5模块与原已资格版本逐字节相同，原NN12/208/261模型、归一化、权重、费用标签、控制器排序别名锁站128/64停滞2/.9均不变。只改manager、授权/时钟guard与自动发布控制；旧源码不覆盖，初始静态依赖绑定快照也保留。没有新增模型、参数或查询策略。

针对新控制逻辑已做一次有限离线检查，原模型和9项数值流程资格没有重跑：

- 5模块字节相同；编译通过，AST确认manager只有replay调用，没有generation入口，start绑定原collection_started，不新取总起点。
- 18项新授权字段逐项损坏全部拒绝，包括原起止计时与0新生成；需要新的真实用户消息引用及新design/source绑定，旧一般批准不能自动满足。
- 5个同步发布/截止检查通过：无worker时先启动publisher后验证；有worker拒绝发布；helper失败不放行；过期子截止拒绝；增加原绝对截止拒绝。均用合成子进程替身，没有实际Git发布或模型/环境调用。
- 新manager、worker、publisher三个实际入口均在缺continuation01授权时拒绝，未导入真实模型/环境。没有新生成、预测、物理步、标签或训练。

当前请求只是一项明确的恢复决定：是否允许**一次生成后受控衔接，复用现有16场景，在原11月5日16:03:12总截止前完成尽可能多的尚未开始回放**。这是对原失败后no_retry约定的一次新例外，必须用户明确批准；未答、拒绝或到原总截止后均不启动，不自动补跑/续窗。新授权须绑定原批准、失败记录、已生成数据及本次源hash，先发布核验再启动唯一新manager。

生成质量检查通过不保证模型或回放效果；新2038同合成来源、own12/internalcal4祖先和相关日期/seed限制仍在。新训练仍未资格，后续费用标签只有双方完整同2017步才定义，安全完成另列，未来192验证另预算，最终8不读标签/不生成，不扩电网论文或依赖。
'''
(O/'GRID300_continuation01_report.md').write_text(report,encoding='utf8')
save(O/'continuation01_decision.json',{'status':'WAITING_EXPLICIT_ONE_TIME_POST_GENERATION_CONTINUATION','new_generation':0,'reuse_scenarios':16,'source_sha':q['source_sha'],'original_start_monotonic':original['monotonic_start'],'original_deadline_monotonic':original['deadline'],'remaining_s_at_report':remaining,'observed_utc':utc.isoformat(),'new_runtime_authorized':False,'new_training_qualified':False,'approved_scope_must_preserve_failed_generation_gate':True})
msg='GRID30016场景已生成并归档，原发布等待2h子窗失败后所有进程退出，基线/相反动作均0；no_retry不自动重启。continuation01源48caca85已准备并通过一次控制检查，5数值模块字节相同、18新授权字段/5自动发布和截止检查/3缺新授权入口拒绝通过，0新生成模型预测物理训练。仅待用户允许一次复用16场景、沿原11月5日16:03:12绝对截止的受控衔接，不再给24h。'
hand='# 当前：GRID300已停，受控衔接准备通过但未获批\n\n'+msg+'\n\n先读GRID300_generation_stop_report/generation_stop_complete/collection_error、GRID300_continuation01_report/continuation01_design/continuation01_preparation_review/continuation01_decision/continuation01_publication_review（交付后）。原3184/15560已退出，旧pipeline pending是历史；当前pipeline_closed。不执行旧manager或旧publisher，不重生成。新版本guard只有新continuation01_authorization发布且原deadline未到才允许启动；未答前保持PAUSED，不重复问题、资格或归档。\n'
(O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8')
save(O/'NEXT_PROTOCOL_PLAN.json',{'status':'STOPPED_WAITING_ONE_TIME_POST_GENERATION_CONTINUATION_SCOPE','old_generation_gate_failed':True,'new_generation':0,'original_deadline_monotonic':original['deadline'],'allowed_now':'new control source/results delivery only; no new process or prediction','on_approval':'Bind true user reply to continuation01_design/source/old auth/error/generated data + original start/deadline, publish exact new authorization, verify then sole continuation01 manager. Not original manager.','new_training_qualified':False})
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID300停机，受控衔接待批准\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if n=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-04 — GRID300一次受控衔接准备\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：停止，待一次受控衔接新批准\n\n'+msg+'\n\n1. 完成受控衔接源码/收据发布核验后，只询问一次原截止内的恢复，不给新24h。\n2. 未答前不写continuation01_authorization、不执行新旧manager，不重做生成/资格/归档。\n3. 原绝对deadline后不能按本方案启动。所有数据保留，不当方法失败。\n',encoding='utf8')
S=A/'continuation01_results';S.mkdir(exist_ok=False);C.mkdir(exist_ok=False)
names=['GRID300_generation_stop_report.md','GRID300_continuation01_report.md','continuation01_design.json','continuation01_decision.json','generation_stop_complete.json','CONTINUOUS_HANDOFF.md','NEXT_PROTOCOL_PLAN.json']
for n in names:shutil.copy2(O/n,S/n);shutil.copy2(O/n,C/n)
shutil.copy2(O/'continuation01_preparation_review.json',S/'preparation_review.json');shutil.copy2(O/'continuation01_preparation_review.json',C/'continuation01_preparation_review.json')
for p in [Path(__file__)]+list(O.glob('*.continuation01_denied.txt')):shutil.copy2(p,S/p.name)
for n in ['continuation01_control_stdout.log','continuation01_control_stderr.log']:shutil.copy2(O/n,S/n.replace('.log','.txt'))
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()});save(C/'delivery_hashes.json',{n:sha(C/n) for n in names+['continuation01_preparation_review.json']})
print('GRID300_CONTINUATION_READY_NOT_STARTED',remaining/3600)
