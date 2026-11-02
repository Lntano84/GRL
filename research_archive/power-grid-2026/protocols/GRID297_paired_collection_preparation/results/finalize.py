"""Deliver saved preparation receipts; no predictions/physics or repeated checks."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid297_paired_collection_preparation';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID297_paired_collection_preparation';C=R/'论文协作资料/FRAME_配对采集准备_GRID297'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'complete.json').exists();q=read(O/'preparation_review.json');assert q['passed'] and q['new_physics']==q['new_model_loads']==0
D=read(O/'design.json');assert len(D['pairs'])==144 and len(D['execution_order'])==288
save(O/'decision.json',dict(status='PREPARATION_PASSED_WAITING_NEW_PHYSICAL_SCOPE',source_published=q['source_published'],user_approved_specific_git_upload=True,upload_block_resolved=True,preparation_passed=True,physical_authorized=False,physical_started=False,resource_passed=False,real_environment_smoke_test=False,new_fits=0,new_labels=0,new_predictions=0,new_physics=0,scope='Only new288episodes/144pairs/28800s physical collection needs a separate explicit decision; original291 approval not inherited.',stop='Do not launch manager/worker, create physical authorization, rerun qualification/prepare or add engineering/model stages while waiting. No funded collection means close the operating-cost learning route.'))
report='''# GRID297：有限配对采集准备通过，等待物理范围决定

已完成可审阅实施、精确引用和有限静态/合成检查。用户明确允许上传至Lntano84/GRL后，冻结源码先发布e4b6fe24，再首次资格通过；先前两次上传审批拒绝及未执行命令记录留在pending_review，没有绕过。此次0模型加载、0真实预测、0拟合、0标签和0物理。

## 为什么这次值得做

296已确认原queryNET包含安全反馈收益与查询代理费用，但不是完整控制动作的后续运行费用。原222调用展开出的3582前缀，没有同一状态下两种决定的配对物理后缀。当前准备只为补这个明确缺口，不再改森林、阈值、风险、特征、种子或换名训练成本头；208/261和原NET仍冻结。

## 固定可审阅范围

| 项目 | 唯一固定方案 |
|---|---|
| 成员 | 全12原pool各自own12日期、scenario100、环境seed0；逐池排除其outer4与原validation4/final8 |
| 数量/角色 | 144配对、288条独立从周初运行的episode；fit8角色96对、cal4角色48对，其中KEEP12对只保留不新拟合 |
| 触发 | 冻结261共同前缀上首次原nativeNN STOP且待买批次非空；只按公开资格，不看费用或结果选调用；不存在则NO_ELIGIBLE_EVENT，不替换 |
| 两臂 | STOP立刻停，CONT仅买原下一批至多64候选；该次之后均恢复同一原261控制器，不持续强制CONT |
| 配对 | 同一pool/日期两臂相邻，预定交替先后；核共同观测/base动作/顺序/别名/可见反馈/native输入/完整交付前缀，相同触发前缀不足即拒绝该配对 |
| 交付 | 记录原连续优化后的实际完整动作和原成本分项，不能把搜索拓扑候选当交付动作 |
| 总资源硬限 | 一次28800秒即8小时，含加载启动/失败工作；单条1200秒，最多580896物理步、9000000总forecast，串行，先到任一限制即停止 |
| 未完/失败 | 保存全部原始完整/部分/无触发/工程失败；不续窗、不补跑、不换日期seed、无自动恢复 |
| 新目标 | 配对源记录全部先发布，之后再固定只读派生C_CONT−C_STOP物理后缀费用；完成/安全分列，失败或不同时域不当低费用赢家。采集程序本身不生成目标/拟合 |

8小时不是完成保证：296按旧291粗估约6.19或6.73小时，日期/查询/干预分支都可能不同；旧29114400秒只完成186/192，这次不会把旧4小时授权续用。未进行真实环境smoke或载入模型试预测，不能称运行资源通过。若首个真实调用有工程/版本问题，也计入获批窗口并保留，不承诺免费重试。

## 本次确实检查了什么

- 原84模型/scaler/checkpoint文件只做字节哈希与已有Git具体blob核对，无反序列化模型或NN/头评分；owner-f绑定保持，不能把参考NN8隐藏送原NN12父头。
- 298源码/输入引用、242环境/日期文件引用与固定设计已封存。日期生成来源为既有自身train场景，不新生成日期。288条唯一成员和144两臂配对、fit/cal/KEEP范围检查通过。
- 生成runner及worker源码编译通过；6组合成分支覆盖STOP/单批CONT、后续回原261、nativeCONT无触发、共同前缀匹配及不匹配/首臂无事件拒绝。仅人工桩反馈，不是真实模型或物理预测。
- 6种无效授权字段被拒；manager/worker在缺少新physical_authorization时于模型/环境导入前拒绝。新授权需绑定本design和source_freeze、确切288/144/8h/1200/步数及forecast上限，并核具体Git blob和远端发布，不能只信布尔。
- 单管理器持久claim禁止重启/重做；运行中截止/逐forecast/物理步守卫和部分日志已写成。静态检查不证明未来跨进程共同前缀或求解器版本一定一致，差异必须保留并停止，不能改数值救结果。

## 后续仅一个必要决定

是否批准这份288条、8小时硬上限的真实物理配对采集？批准上传与恢复30分钟跟进均不等于批准物理。明确批准后，先发布具体source/design/hash授权和首次启动核验，才允许唯一manager首次运行；未批准不写physical_authorization、不运行模拟器。此后不追加另一轮工程核查维持推进。

没有新学习资格。只有标签真正可识别并有跨原训练日期支持，才单独固定唯一模型/匹配对照/误停代价，至多22拟合、一次600秒；不承诺此数据必有改进。若无合格完整配对或无预算，关闭该成本学习路线。未来新候选验证若需原4开发日期×12pool×2臂×2逆序共192条，另有独立预算，不藏在这次采集内。

全部种子/日期/KEEP/不利结果保留，pool/相关前缀并非独立场景；own12父与内部cal4祖先、已暴露开发限制不变，非全链OOF/盲测/整周质量保证。原费用NET/奖励/控制器/排序别名/锁站/128/64/停滞2/.9保持。旧291186/192及已付费部分绝不重做；最终8、新日期FRAME、电网、论文、依赖不动。历史190拟合及6续训8285updates、NN604.703实际/用户678.156cap另列。

源码：work/grid297_paired_collection_preparation；本目录design、model_references、published_model_bindings、pool_roles、source/model/data_freeze、preparation_review、qualification日志、upload_approval和decision/complete。归档A=GRID297_paired_collection_preparation，D协作副本FRAME_配对采集准备_GRID297。complete.passed只表示本次准备交付，不是288条采集完成。
'''
(O/'GRID297_report.md').write_text(report,encoding='utf8')
save(O/'complete.json',dict(passed=True,preparation_only=True,physical_scope_authorized=False,collection_started=False,episodes_executed=0,new_model_loads=0,new_predictions=0,new_fits=0,new_labels=0,new_physics=0))
plan=dict(status='WAITING_USER_288_EPISODE_8_HOUR_PHYSICAL_SCOPE',source_sha=q['source_published'],scope=dict(episodes=288,pairs=144,wall_cap_s=28800,worker_cap_s=1200,physical_steps_cap=580896,forecasts_cap=9000000),on_approval='Record only a direct user scope answer in physical_authorization binding design/sourcefreeze and message reference; publish exact authorization blob and verify remote; then first unique manager. Do not reuse291 approval.',while_waiting='No repeated checks/prepare/qualification, new model stages, extra forecasts or physics. Already passed finite preparation complete.',on_decline='Close operating-cost data/learning branch, retain208/261 and all adverse evidence; no proxy workaround.')
save(O/'NEXT_PROTOCOL_PLAN.json',plan)
msg='GRID297具体仓库上传经用户批准后源e4b6fe24先发，首次有限准备资格通过：84既有模型具体Gitblob、288唯一成员/144配对，6合成流/6授权字段拒绝/manager和worker缺授权拒绝、生成源码编译；0真实模型加载预测/拟合标签/物理。旧上传拒绝已解决留档。现在仅等待明确288条/8h新物理scope，未批准不写授权或启动；旧291不补跑。资源/真实env smoke未通过或执行，资格不保证8h完成。'
hand='# GRID297有限准备完成：仅等待288条/8h物理范围决定\n\n'+msg+'\n\n先读integration AGENTS/docs当前状态/NEXT以及本目录GRID297_report/design/preparation_review/decision/complete/publication_review/NEXT_PROTOCOL_PLAN。不要重复prepare/资格/旧291审计或新增模型工程串。用户批准上传到Lntano84/GRL已落实，publication_block及pending_review是历史，不再等待上传授权。只有明确批准本次288条/8h后，绑定design/source_freeze/确切预算和用户消息的新physical_authorization先发布核具体blob/远端，才首次唯一manager；不继承291或定时恢复授权。不批准则关该成本采集学习路线。原模型协议/日期seedKEEP/最终8/祖先开发与旧190拟合6续训边界保持。\n'
(O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8')
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/name;t=p.read_text(encoding='utf8');p.write_text('# 当前：GRID297准备通过，等新物理范围批准\n\n'+msg+'\n\n以下保留历史。\n\n'+t if name=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-02 — GRID297准备资格完成\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前 — 等待288条/8h真实配对采集范围决定\n\n'+msg+'\n\n1. 向用户展示已发布的具体报告，请求唯一必要物理scope。等待时不重做资格/盘点或启动任何预测物理。\n2. 明确批准后先新physical_authorization绑定本design/sourcefreeze和288/144/28800/1200/580896/9000000具体值，发布核Git blob及远端，再首次唯一manager，原总窗不续/部分不重做。\n3. 拒绝则关闭本成本学习路线，保留208/261与所有反例，不盲训成本头。旧291和最终8不动。\n4. 完成本轮D协作/新归档清单/具体blob/远端核验，同步交接和自动跟进。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID297准备资格通过，物理尚未批准\n\n[报告](protocols/GRID297_paired_collection_preparation/results/GRID297_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
S=A/'results';S.mkdir(exist_ok=False)
names=['GRID297_report.md','design.json','model_references.json','published_model_bindings.json','pool_roles.json','source_freeze.json','model_freeze.json','data_freeze.json','resource_estimates.json','data_collection_proposal.json','preparation_review.json','qualification_stdout.log','qualification_stderr.log','upload_approval.json','decision.json','complete.json','NEXT_PROTOCOL_PLAN.json','CONTINUOUS_HANDOFF.md']
for name in names:shutil.copy2(O/name,S/name);shutil.copy2(O/name,C/name)
shutil.copy2(W/'finalize.py',S/'finalize.py');save(S/'archive_manifest.json',{p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file()})
save(C/'delivery_hashes.json',{p.relative_to(C).as_posix():sha(p) for p in C.rglob('*') if p.is_file() and p.name!='delivery_hashes.json'})
print('GRID297_PREPARATION_DELIVERY_READY_SCOPE_NOT_AUTHORIZED')
