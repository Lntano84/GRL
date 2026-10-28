from pathlib import Path
import sys,json,hashlib,shutil,subprocess,datetime
R=Path('D:/Codex/GRL研究'); W=R/'work/grid254_reference_preflight'; O=R/'outputs/grid254_reference_preflight'; REPO=R/'work/GRL-results-integration'; A=REPO/'research_archive/power-grid-2026/protocols/GRID254_reference_preflight'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def verify_manifest(base,p):
 m=read(p)
 for rel,h in m.items():assert sha(base/rel)==h,str(base/rel)
 return len(m)
assert not (O/'independent_audit.json').exists(), 'Do not repeat completed audit'
assert not (R/'outputs/grid255_reference_stop_nn').exists()
counts={'active_source':verify_manifest(R,O/'source_freeze_recovery01.json'),'qualified':verify_manifest(O,O/'qualified_freeze.json')}
for phase in ['source','qualified']:
 counts['archive_'+phase]=verify_manifest(A/phase,A/phase/'archive_manifest.json')
jobs=read(O/'jobs.json');workers=read(O/'worker_checks.json');resources=read(O/'resource_checks.json');parents=read(O/'parent_interface_checks.json');inputs=read(O/'input_checks.json');life=read(O/'lifecycle/tests.json');q=read(O/'qualification.json');complete=read(O/'complete.json')
assert len(jobs)==len(workers)==len(inputs)==18 and len(parents)==12 and len(resources)==3
assert {j['node'] for j in jobs}=={j['node'] for j in workers} and len({j['package'] for j in jobs})==6
for j in jobs:
 r=next(r for r in workers if r['node']==j['node'])
 assert r['passed'] and r['parameters']==12980 and r['optimizer_instances']==r['optimizer_steps']==0
 assert r['initial_hash']==r['post_probe_hash']==j['initial_hash'] and r['weights_unchanged'] and r['torch_threads']==1 and not r['training_labels_loaded']
 assert read(O/'worker_probes'/j['node']/'complete.json')==r
for r in inputs:assert all(v is True for k,v in r.items() if k!='node')
for r in resources:assert len(r['nodes'])==6 and r['all6resident']['available_physical']>=r['reserve_required']
for r in parents:
 assert r['original_model_unchanged'] and r['new_reference_hidden_never_used'] and r['only_fit8_public_rows'] and r['oldcached_rows_reproduced']==2 and r['missing_parent_interface_tested']
 assert sha(R/r['model_path'])==r['original_model_hash']
assert life['passed'] and life['atomic_duplicate_rejected'] and life['expected_training_denial_exit']!=0
assert all(life[n]['all_workers_exited'] and not life[n]['passed'] for n in ['expired_deadline','live_deadline','user_stop'])
# Query the actual process table; no new fitting or model forward.
cmd="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'grid254_reference_preflight[\\/](worker|manager|qualify|resume01)\.py|grid255_reference_stop_nn[\\/](worker|manager)\.py' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
p=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True)
assert p.returncode==0,p.stderr
assert not p.stdout.strip(),p.stdout
assert q['passed'] and q['all_workers_exited'] and complete['passed'] and complete['NN_reference18_not_started']
minimum=min(r['all6resident']['available_physical'] for r in resources)
audit=dict(passed=True,receipt_audit_only=True,new_fit=0,new_forward=0,source_manifest_counts=counts,workers=18,unchanged_initialweights=18,all_workers_exited=True,process_table_checked=True,parent_interfaces=12,parent_fit_forward_rows=sum(r['rows'] for r in parents),minimum_available_bytes=minimum,minimum_available_GiB=minimum/1024**3,reserve_bytes=resources[0]['reserve_required'],initial_memory_API_failure_preserved=True,complete_checks_not_rerun=True,actual_training_not_started=True)
save(O/'independent_audit.json',audit)
save(O/'decision.json',dict(passed=True,execution_ready=True,scientific_effect_not_evaluated=True,next='GRID255 fixed 18-reference data foundation only; publish its source/input authorization before starting one six-worker 600s manager.',training_this_stage=0,full_pipeline_fit_count_bound=40,no_budget_or_seed_change=True,preflight_not_timing_guarantee=True))
report=f'''# GRID254：参考模型执行预检

执行资格通过，18个参考模型尚未训练。本阶段优化器实例、优化器更新、新训练模型、新策略评分均为0；这不是学习效果结果。

## 为什么做

GRID253核实门控训练行的停止点来自见过这些日期的原网络，而外层记录未参与该网络训练。下一步用同时排除记录日期组和当前外层组的参考网络重新产生训练停止点，检验这项来源差异。差异是否造成退化仍需后续比较，不能预先认定。

## 已核验

- 6组固定8天数据包、18个固定采样流；全3种子，每模型7250更新、batch32，架构12980参数及原优化设置保留。原标签总数组在准备阶段已解析，各导出优化数据只含指定8天，不声称准备过程完全未读取其他旧标签。
- 六进程同时驻留的三轮检查均通过，18份初始/检查后权重哈希完全一致，每进程单线程。使用32行公开拟合输入做合成梯度检查，不创建优化器或用标签训练；成功合成检查18次，首次中断另有至多6次，全部披露。
- 六进程同时驻留时最小剩余内存{minimum/1024**3:.3f} GiB，高于固定1 GiB+6 MiB预留。该测量不保证真实18模型一定在600秒内结束。
- 12个原父网络接口共36条自身拟合日期公开行核验，包含24条旧缓存重现和12条缺失隐藏表示接口；原权重不变。后续父收益头必须接原网络隐藏表示，不能接新参考网络隐藏表示。
- 去重、截止、停止信号与子进程退出/部分输出保留测试通过。故意触发的拒绝/超时不是正式训练失败。当前无预检/参考训练进程。

## 工程恢复

首次预检因Windows内存计数接口缺少64位HANDLE参数声明而停止，0优化器/0训练。原冻结源和日志保留在interrupted_attempts/memory_api01；仅修正内存计数与活动清单验证，并新增恢复入口。已完成的数据与生命周期核验按哈希复用，未重跑；只补完资源和父接口检查。数值训练worker、采样与数据未改。

## 下一阶段固定范围

GRID255先封存发布授权、源码和输入，再由唯一管理器执行18参考网络、6单线程worker、一个600秒总窗口（含启动），不延长/减更新/换种子/重做完整项。全部18权重发布后才能做32238行参考公开评分。其后新停止点及输入先发布，再连接原教师NET标签；后续同GRID250类40输入41参数读出至多11线性+11保护步长，另一600秒窗口。两组件有实际数据依赖，全探针至多40拟合，非参数搜索。

部署原控制器、原父NN/24收益头、费用、日期、种子、排序、128/64/停滞2/0.9均不变，不接GRID252新qgap，不生成最终8或新物理数据。父先验仍有自身12天标签祖先，内部cal4仍非全链独立，只检验参考停止点来源；开发代理效用不等完整周质量或真实控制时延。
'''
(O/'GRID254_report.md').write_text(report,encoding='utf8')
handoff='''# GRID254后持续交接

先读integration/docs/RESEARCH_STATE.md和NEXT_STEPS.md。GRID254执行资格/独立收据审计已完成，不重做；complete/qualification/independent_audit/decision均passed，训练0。六并发三轮最小可用RAM1.889GiB，18初始权重不变、12原父接口36行通过、无活动worker。首次Win32 HANDLE参数错误原源/日志已保留，recovery01只补未完成资源接口；active manifest为source_freeze_recovery01.json，不把旧错误当新故障。

下一GRID255只执行253固定18参考NN数据基础：必须复用work/grid254_reference_preflight/worker.py、manager.py、runtime254.py和6包18stream的已资格哈希，不改数值代码。先独立封存设计/授权/源码及输入并推送核对远端，再创建outputs/grid255_reference_stop_nn/authorization.json（资格SHA、qualified_freeze SHA、jobs SHA、models18、fit_cap_s600、all_sources_inputs_published true）。首次manager.py train以独占training_started/run_context防重复；6单线程worker/18jobs，一个600秒窗口含启动，每模型原7250updates/batch32/FACTOR12980/原Adam/SmoothL1/clip10，全3种子。不得在254补做训练。任一工程失败/超时保留完整项和部分权重/暴露，不自动重跑或续窗，不减更新/改种子。

完成18后核对固定采样哈希/232000draws每模型/fit8外draw0/初始hash/7250步/严格全局截止与子进程退出，全部模型先发布后做参考公开评分。仅训练本数据基础，不顺带提前读新STOP对应teacher标签。参考总32238公开行；旧父缓存22653字节复用、<=9585缺失必须原ownerf NN隐藏，不能新NN8隐藏塞旧208头；24父头0拟合。原参考排除记录日期组及outer组，18无序对x全3种子/36用途固定。

之后新公共选择/输入先发后接旧teacher标签，样本/原风险公式数值和全部selectedSTOP正例保护集合重算，不再固定85。fit8/cal4日期不变，f0_s1KEEP/其他空域或无fitSTOP正例KEEP。后续读出同250 canonical38+原b+nativeSTOP位=40输入41参，不加252gap，<=11linear+11原保护步长/另一600s，全探针<=40真实依赖操作不隐去。父NN/收益头仍own12含自身标签，内部cal4仍可能进其他参考祖先，不称全链OOF/盲测或整周/真实时延提升。保留230/250/252和全部反例，不按外层调参/择种子。

原费用/标签/全日期种子/控制器/排序别名/锁站/128/64/停滞2/.9保持；最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。各学习组件<=22/600s，不延长/盲扫。源码输入、全部模型、全部预测按屏障先发后接结果。Git helper必须等session退出才下一helper/屏障。按AGENTS更新docs/归档首页、读取源码实际A归档推送核对协作/清单/远端。持续自主推进至用户停，更新grl起点，不恢复旧自动化、不归档聊天。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
C=R/'论文协作资料/FRAME_参考模型执行预检_GRID254';C.mkdir(parents=True,exist_ok=False)
for name in ['GRID254_report.md','qualification.json','independent_audit.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md']:
 shutil.copy2(O/name,C/name)
 assert sha(O/name)==sha(C/name)
save(O/'delivery_check.json',dict(passed=True,collaboration_path=str(C),files={p.name:sha(p) for p in C.iterdir()},archive=str(A),training_not_started=True))
D=REPO/'docs'; state=D/'RESEARCH_STATE.md'; s=state.read_text(encoding='utf8'); title,rest=s.split('\n',1); state.write_text(title+'\n\n## 2026-10-28 — GRID254零优化器执行资格\n\n18固定stream/6数据包、六进程三轮与12父接口36行通过，18初始权重不变，最小可用1.889GiB高于1GiB+6MiB预留；独立收据/冻结审计通过、当前无worker。优化器/更新/训练/新策略评分均0。首次Win32内存HANDLE声明错误已保留，只补未完检查，未重做数据/生命周期。下一255先封存发布后执行固定18参考NN/六单线程/一次600s，保持7250更新及全种子，不自动续窗。预检不保证实际耗时，仍非全链OOF。\n'+rest,encoding='utf8')
(D/'NEXT_STEPS.md').write_text('''# Next steps — GRID254后

1. GRID254资格及收据审计完成不重做；读取outputs/grid254_reference_preflight/CONTINUOUS_HANDOFF.md与qualification/qualified_freeze/active source_freeze_recovery01。
2. GRID255先发布固定设计/源码输入及哈希授权，再唯一manager训练18参考NN，6单线程worker，7250updates/batch32/原FACTOR设置，一个600s含启动。不重复完整项、不续窗/减更新/换种子。失败保留部分与完整收据。
3. 全部18权重核验/发布后才参考公开评分32238行；原父NN隐藏/24头复用、旧缓存22653/缺失<=9585，禁止新隐藏接旧头。公共选择输入先发布后接原teacher NET。读出同250不接252gap，后续<=11linear+11alpha/600s；全探针<=40依赖拟合。
4. 保留原祖先局限/开发属性/全部种子反例，不称全链OOF/完整周或实时时延胜出。最终8/新物理/电网/论文/FRAME不动，不安装。Git helper严格串行，收尾更新docs/归档/协作并核对远端。
''',encoding='utf8')
for name,txt in [('DECISIONS.md','GRID254：资源/接口预检通过后才允许下一255固定18参考数据基础。一次600秒含启动，6单线程，不以预检时间保证真实训练成功。工程恢复仅Win32内存参数与活动清单，原数值源/输入不改。'),('EXPERIMENT_LOG.md','GRID254：6包18stream/18零优化器合成检查/12原父接口36行完成，最小剩余1.889GiB，0训练/评分。首次内存API工程错误保留，复用已完成检查；仅收据审计无新forward。')]:
 p=D/name;p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-10-28 — '+txt+'\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID254参考网络执行预检\n\n[报告](protocols/GRID254_reference_preflight/audited/GRID254_report.md)；六并发资源/接口资格完成，训练尚未开始。\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
AD=A/'audited';AD.mkdir(exist_ok=False)
for name in ['GRID254_report.md','independent_audit.json','decision.json','delivery_check.json','CONTINUOUS_HANDOFF.md']:shutil.copy2(O/name,AD/name)
shutil.copy2(Path(__file__),AD/'deliver.py')
save(AD/'archive_manifest.json',{p.relative_to(AD).as_posix():sha(p) for p in AD.rglob('*') if p.is_file()})
print('GRID254_DELIVERED',json.dumps(audit,ensure_ascii=False))

