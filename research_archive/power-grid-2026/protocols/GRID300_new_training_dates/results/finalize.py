"""Archive the actual finite preparation receipts and reviewable implementation."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';W=Path(__file__).parent;O=R/'outputs/grid300_new_training_dates';A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates';C=R/'论文协作资料/FRAME_新日期采集实施_GRID300'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'decision.json').exists();q=read(O/'preparation_review.json');b=read(O/'stage_boundary_review.json');assert q['passed'] and b['passed'] and q['queue_and_clock_boundary_review_sha256']==sha(O/'stage_boundary_review.json')
assert not (O/'physical_authorization.json').exists() and not (O/'collection_started.json').exists();assert not any((W/'train_env/chronics').iterdir())
report=f'''# GRID300：新日期采集程序已准备，等待实际范围批准

2026-11-04用户要求按GRID299方案继续实施。源b9ac57f6先发布，实际有限准备检查通过；当前0真实模型加载/预测、0场景生成/物理回放、0费用标签或训练。新环境目录只复制静态网架与配置，chronics为空。没有写入physical_authorization或启动manager，旧291/297回放与298/299完整分析未重复。

已实现固定2038年16个训练日期家族、各scenario100/envseed0及原冻结生成器三seed；原12owner-f模型池各新own12日期，144条基线中96fit8/48cal4、KEEP12。新角色沿原月份第一/第三周一槽位继承，原最终8场景不生成或读取标签。新year/profile兼容性尚未做真实环境smoke，生成器失败即留原文件停止，不换seed/日期救结果。

流程只有以下三段，贯穿同一24小时管理器窗口：

1. 生成16份新场景。复用原Chronix2Grid wrapper、既有依赖和原质量/有限性检查；旧训练环境只提供原有name_gen/gen_type排序信息，初始化可能发生的schema计算单列且计总预算，不运行旧控制器/step/simulate、不把旧chronics复制为新数据。新环境所有静态参数字节不变，只有all_dates改为固定2038清单。生成OPF/solver工作另记，不冒充0物理计算。
2. 新场景全部文件path/hash与生成收据发布后，跑144条原261基线。原NN12归一化/64+16隐藏/q/已知费用到208两头再到261，全部权重和浮点次序复用，不使用参考NN8隐藏、不训练或换scaler。基线实时保存所有nativeSTOP的公开身份，完毕后按四段[1..504]/[505..1008]/[1009..1512]/[1513..2017]最早事件固定至多576个干预点，不看费用或完成状态筛选，空段不替补。
3. 全部基线源文件哈希、成员和干预点发布后，每点只跑一条与原261决定相反的轨迹，另一臂复用保存基线。干预前每步先核完整联合action_hash及before_hash，再在步后核after_hash/账本前缀；到选定点核全base_action/order/aliases/可见反馈/NN输入/raw41/nativeq/批次等身份，再只反转一次STOP或CONT。之后恢复原261。同一基线被多个目标共享，目标相关，不能增加独立样本数；观测向量不会用来恢复环境/controller状态。

两个发布等待都要求没有活动worker，使用原git_push.cmd严格白名单、退出码和具体Gitblob/远端校验；管理器原时钟继续走，等待时间也计入总上限，生成发布还计入原2小时子窗。不会在每worker核HEAD时并发Githelper。发布失败/未及时完成时留源和部分，不能续窗启动下一段。

| 固定范围/上限 | 数量 |
| --- | ---: |
| 新日期家族及生成场景 | 16 |
| 原261新基线 | 144 |
| 相反动作回放 | 最多576 |
| 回放合计 | 最多720 |
| 单一总窗口（含加载/启动/生成/发布等待/失败） | 86400秒（24h） |
| 生成总子窗 / 每场景 | 7200秒 / 1800秒 |
| 每条回放 | 1200秒 |
| 回放物理步 / 所有env forecast | 1452240 / 22500000 |
| 新训练 | 0 |

管理器、生成worker和回放worker在任何真实模型/环境导入前检查新scope授权，绑定设计/源码冻结/日期清单hash、上述全部计数与预算和真实用户消息引用，并要求授权/源/准备收据已发布。工作进程还需精确manager父进程（或一层原Windows venv redirector）、同token及持久claim；旧291或297授权无效。不可重复的started/claim文件防止补跑完整或部分项目。用户停止、首次到限、前缀不符或工程失败都会停新增并保留记录，不自动重试或恢复。

实际准备验证如下，均不接真实模型/环境：

- 84份既有NN/scaler/头/残差等载荷文件的本地hash及具体已发布Gitblob核对，未反序列化。
- 144唯一基线、16新日期、8fit/4cal/排除outer角色及KEEP核对，生成后的动态队列只按已冻结规则构造。
- 9个合成状态流程：基线行为不变、两向一次反转、随后恢复、公共标记和完整动作不符拒绝、缺事件拒绝，以及选择后续nativeSTOP时之前事件仍按原261执行。
- 17项授权字段逐项破坏均拒绝；manager/生成worker/回放worker缺授权的三个真实进程入口都在live导入前退出。
- 首次资格检查{q['wall_s']:.6f}秒通过。其后只针对尚未覆盖的队列边界追加一次独立合成检查：0个和576个目标均正确生成连续索引与空段记录；过期总窗口、未重置时钟及用户停止三个检查通过。原资格未重做，初始收据保留，补充收据另存。
- 原生成运行源码和插入before-step检查后的worker均只编译通过，未执行真实NN、forecast、env.step、OPF或generator。没有安装任何依赖。

历史资源粗估仍约10.5至19小时，不是通过资源资格或24h完成保证。特别是2038生成兼容性、真实跨进程重放确定性、全部后缀是否完成尚未知。第一个生成pilot和f0_s0固定fit8回放pilot都是批准后本范围内的有偿工作，失败也消耗同一窗口，不做额外未计费smoke。

本程序只采原始源记录。待源全部发表后才另固定一次保存C_CONT-C_STOP后缀费用派生；双方完整同2017步才比较费用，完成、安全异常独立保留，早停不当低成本成功。即使采集完成，也没有预先取得新费用头的训练资格，不借cal4正例到fit8，不按日期/调用/seed设例外。新候选未来192开发验证需另外预算，最终8不动。

原208/261主锚、230历史保留，森林/残差阈值alpha风险特征seed修补与换名成本头关闭。原费用NET/奖励/排序别名/锁站/128/64/停滞2/.9不变；新日历同合成生成机制、旧own12/internalcal4祖先与暴露相关性仍在，不称全链OOF/盲测/真实时延优势。历史190拟合及6续训8285updates、604.703实际/用户678.156cap另列，本轮新增拟合0。

当前只需一个具体范围决定：是否批准本次16新场景生成+144基线及最多576相反动作回放，最多720条、总24小时且到限停止保留部分。此前“只设计”的答复以及“继续实施”均不写成该物理批准；批准后先发布真实新授权并核blob/远端，才首次唯一manager。
'''
(O/'GRID300_report.md').write_text(report,encoding='utf8')
save(O/'decision.json',{'status':'WAITING_USER_GENERATION_AND_PHYSICAL_SCOPE','implementation_reviewable':True,'finite_offline_preparation_passed':True,'live_compatibility_passed':False,'resource_passed':False,'new_generation_approved':False,'new_physics_approved':False,'new_training_qualified':False,'new_fits':0,'new_predictions':0,'new_physics':0})
save(O/'complete.json',{'preparation_complete':True,'passed':True,'runtime_complete':False,'generated_scenarios':0,'physical_episodes':0,'new_model_loads':0,'new_predictions':0,'new_fits':0,'labels_derived':0})
msg='GRID300实施完成并通过有限离线检查：源b9ac57f6先发，84既有载荷hash/blob、144成员/16日期、9合成流程/17错误授权字段/3缺授权入口拒绝、0或576目标及3时钟停止边界通过，0真实模型加载/预测/生成/物理/标签/训练。新16生成+144基线+至多576相反动作=至多720条/单总24h须新范围批准，原预算不继承，未启动任何manager。'
hand='# 当前：GRID300准备通过，等待一次新物理范围决定\n\n'+msg+'\n\n先读GRID300_report/design/preparation_review/stage_boundary_review/decision/complete/publication_review与NEXT_PROTOCOL_PLAN。用户尚未答复新16生成/最多720条24h范围，不创建physical_authorization或执行manager/generator/worker；不重复已完成资格、模型hash、旧采集或归档。批准后按guard.py精确字段记录真实用户问答，先发布授权核具体Gitblob/远端，再唯一manager。原时钟包括两个公开发布等待；所有原始模型/轨迹D本地，Git仅必要源码/引用/清单/收据。\n'
(O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8')
save(O/'NEXT_PROTOCOL_PLAN.json',{'status':'WAITING_USER_16_GENERATION_MAX720_REPLAYS_24H','already_done':['implementation','84payload_hash_and_blob','finite_preparation','queue_clock_boundaries'],'scope_id':'GRID300_NEW_TRAIN_DATES_SUFFIX_COLLECTION','authorization_required':{'generation_scenarios':16,'baseline_runs':144,'max_opposite_runs':576,'max_runs':720,'wall_cap_s':86400,'generation_cap_s':7200,'generation_worker_cap_s':1800,'worker_cap_s':1200,'physical_steps_cap':1452240,'forecasts_cap':22500000},'before_start':'Explicit new user scope reply -> exact design/source/calendar-bound authorization receipt -> publish and verify -> sole manager. No repeat source qualification or unauthorized live smoke.','training_qualified':False})
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID300采集实施准备通过，待新范围批准\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if n=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-04 — GRID300离线准备通过\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID300等待新物理范围批准\n\n'+msg+'\n\n1. 完成必要准备结果/D协作/具体Gitblob核对后，只询问一次16生成+最多720条/24h新范围。\n2. 未答前不写授权、不运行模型/环境smoke、不重复资格。\n3. 明确批准后按scope/design/source/calendar和精确预算发新授权，核远端后唯一manager，原24h不重置，不用旧291/297授权。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID300采集实施已准备，待新范围批准\n\n[报告](protocols/GRID300_new_training_dates/results/GRID300_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
shutil.copy2(O/'CONTINUOUS_HANDOFF.md',R/'outputs/grid299_public_state_coverage/CONTINUOUS_HANDOFF.md')
S=A/'results';S.mkdir(exist_ok=False);C.mkdir(parents=True,exist_ok=False)
names=['GRID300_report.md','design.json','calendar.json','pool_roles.json','model_references.json','source_freeze.json','preparation_review.json','preparation_review_initial.json','stage_boundary_review.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md','NEXT_PROTOCOL_PLAN.json']
for name in names:shutil.copy2(O/name,S/name);shutil.copy2(O/name,C/name)
for p in [W/'stage_boundary_check.py',Path(__file__)]+list(O.glob('*.missing_auth.txt')):shutil.copy2(p,S/p.name)
for n in ['qualification_stdout.log','qualification_stderr.log']:shutil.copy2(O/n,S/n.replace('.log','.txt'))
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()});save(C/'delivery_hashes.json',{n:sha(C/n) for n in names})
print('GRID300_PREPARATION_DELIVERED_WAITING_SCOPE')
