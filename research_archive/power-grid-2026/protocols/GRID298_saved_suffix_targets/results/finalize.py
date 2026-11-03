"""Summarize ONLY the first saved readout outputs; no source trajectories reread."""
from pathlib import Path
import collections,hashlib,json,math,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';W=Path(__file__).parent;O=R/'outputs/grid298_saved_suffix_targets'
A=I/'research_archive/power-grid-2026/protocols/GRID298_saved_suffix_targets';C=R/'论文协作资料/FRAME_后缀费用目标_GRID298'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'decision.json').exists() and read(O/'complete.json')['passed']
summary=read(O/'summary.json');pairs=read(O/'pairs.json');episodes=read(O/'episodes.json');assert len(pairs)==144 and len(episodes)==288
clusters={}
for p in pairs:
 if p['target_valid'] and p['target_cost_CONT_minus_STOP']!=0:
  k=(p['date'],p['trigger_step'],p['common_before_hash'],p['target_cost_CONT_minus_STOP']);clusters.setdefault(k,[]).append(p)
groups=[dict(date=k[0],trigger_step=k[1],state_hash=k[2],cost_delta_CONT_minus_STOP=k[3],pool_records=len(v),pair_indices=[p['pair_index'] for p in v],roles=dict(collections.Counter(p['role'] for p in v)),KEEP_records=sum(p['KEEP'] for p in v),ordinary261_continue_records=sum(p['ordinary261_continue'] for p in v)) for k,v in clusters.items()]
active_fit=[p for p in pairs if p['role']=='own_training_fit8' and not p['KEEP']];fitgood=[p for p in active_fit if p['target_valid'] and p['target_cost_CONT_minus_STOP']<0]
support=dict(nonzero_groups=groups,active_fit_lower_cost_records=len(fitgood),active_fit_lower_cost_unique_states=len({p['common_before_hash'] for p in fitgood}),active_fit_lower_cost_dates=sorted({p['date'] for p in fitgood}),active_pool_fit_target_counts={pool:summary['by_pool_role'][pool]['own_training_fit8']['valid_complete_targets'] for pool in summary['by_pool_role'] if pool!='f0_s1'},active_fit_cost_gain_abs=-math.fsum(p['target_cost_CONT_minus_STOP'] for p in fitgood),missing_ledger_rows=sum(e['missing_ledger_rows'] for e in episodes),valid_target_missing_ledger_rows=sum(p[a]['missing_ledger_rows'] for p in pairs if p['target_valid'] for a in ['STOP','CONT']),ordinary261_complete_event_continue=sum(p.get('ordinary261_continue',False) for p in pairs if p['target_valid']))
save(O/'support.json',support)
decision=dict(status='FIRST_STOP_COST_HEAD_NOT_QUALIFIED_CLOSE_BRANCH',new_training_qualified=False,new_collection_authorized=False,keep_models=['PARENT208','BOUNDED261'],historical_reference='230',
 evidence={'all_pair_cost_targets':124,'active_fit_targets':71,'active_pool_fit_targets_min':min(support['active_pool_fit_target_counts'].values()),'active_pool_fit_targets_max':max(support['active_pool_fit_target_counts'].values()),'active_fit_lower_cost_records':len(fitgood),'active_fit_lower_cost_unique_states':support['active_fit_lower_cost_unique_states'],'active_fit_lower_cost_abs_gain':support['active_fit_cost_gain_abs']},
 rationale='Actual matched joint physical suffix targets now exist. Yet one earliest nativeSTOP per own-date produces only5..8 complete fit targets per active pool; all8 beneficial active fit records are one shared Sep21 state, and three f3 pools have no beneficial fit example. Cal4 gains cannot be borrowed into fit8. Do not fit41-dimensional head or a tree/threshold/risk sweep on this support. No claim that cost learning is impossible or208/261 optimal.',
 no_new_strategy_returns=True,no_alternative_policy=True,
 next='Close this first-STOP cost-learning branch. Preserve all targets/safety/roles for future research. Reopening needs a separately reviewed data coverage design with more independent decision states, justified information value and new physical budget; no automatic new collection, model search, or192 validation.')
save(O/'decision.json',decision)
grows='\n'.join('| '+g['date']+' | '+str(g['trigger_step'])+' | '+str(g['pool_records'])+' | '+f"{g['cost_delta_CONT_minus_STOP']:+.6f}"+' | '+str(g['roles'].get('own_training_fit8',0))+'/'+str(g['roles'].get('own_training_cal4',0))+' |' for g in groups)
prows='\n'.join('| '+pool+' | '+str(roles['own_training_fit8']['valid_complete_targets'])+' | '+str(roles['own_training_fit8']['CONT_lower_cost'])+' | '+str(roles['own_training_fit8']['CONT_higher_cost'])+' | '+str(roles['own_training_fit8']['zero_cost_delta'])+' |' for pool,roles in summary['by_pool_role'].items())
report=f'''# GRID298：配对后缀费用目标已核查，本批不训练新费用头

2026-11-03用户明确继续。源码844c7fd1及全部固定成员/1308项路径哈希引用先发布，首次保存读出{summary['wall_s']:.6f}秒/600秒完成。288条、144配对、566046保存物理行均核查；0新物理、模型预测或拟合，不重跑297采集/归档。144对的共同前缀或无事件轨迹、原角色/库/费用尺度、原账本容差、实际完整交付动作哈希均通过。未读取NPZ向量或模型权重；动作证明限于原完整联合交付的已保存哈希和冻结来源，不新解码/求解。

| 结果范围 | 配对数 | 费用降低 | 费用升高 | 费用不变 |
| --- | ---: | ---: | ---: | ---: |
| 全部完整且有事件目标 | 124 | 11 | 17 | 96 |
| 活跃11池 fit8 | 71 | 8 | 13 | 50 |
| 活跃11池 cal4 | 43 | 2 | 2 | 39 |
| 固定KEEP池 | 10 | 1 | 2 | 7 |

费用目标固定为触发步骤起至2017步的原运行费用之和 C_CONT-C_STOP，负数表示额外买一批后费用更低；原queryNET/奖励/费用定义未改。每对只在同一原261共同前缀上干预一次，随后仍由冻结261完成。它比较STOP_NEXT与CONT_NEXT两个采集臂，不等同新策略对原261的成绩。124个完整有事件对中原261在11处本就会继续，其余113处停止；没有回测或拼接新的部署策略。

144对中11对无合格事件，留空目标；9对双方都于1192步提前结束（均同一December7日期在不同池复用），也留空目标。270/288条完成2017步，18条提前结束；提前结束不按低费用胜出。全288条账本有18个缺失ledger行，均不进入124个完整目标；完整目标两臂均未记录非法、歧义或异常步骤。完成/安全与费用分别保存，不做终止惩罚/补值、不换样本，也不能据此保证未来安全。

124个完整目标中96对轨迹与费用不变；28对完整交付改变并有非零费用差。这28条池记录集中在以下4组同日期/触发状态/费用结果，不能算28个独立场景，更不能把共享日期的收益累加当新部署收益：

| 日期 | 触发步 | 相关池记录 | 每条原费用差 | fit8/cal4条数 |
| --- | ---: | ---: | ---: | ---: |
{grows}

全124条池记录的费用差描述合计+599246.786407，其中降低费用合计31320.023773、升高费用合计630566.810181；这些量重复了同一日期/状态，不是独立期望收益。Sep21的每条降低仅4.731049原费用单位；较大降低来自Oct5，当前只出现在两个cal4记录，两处原261本已继续。不能借cal4收益填补训练正例、按日期设例外或据此选种子。

训练支持是本次明确裁决的原因：每活跃池只有5..8个完整fit8目标，全部活跃fit8仅8条降低费用、总37.848389，且全来自同一Sep21状态。f3三个池没有降低费用的fit8样本。其余两类非零训练结果为Jun1和Nov2的费用升高；模型选择不能按这些少数调用特设门槛。参考如下，KEEP只保留不训练：

| 池 | fit8完整目标 | 降低 | 升高 | 相同 |
| --- | ---: | ---: | ---: | ---: |
{prows}

原账本仍使用逐步边际价乘网损、再调度、弃电变化、储能通量四项；物理弃电水平另存，不替代原弃电变化成本。原官方raw_cost为目标依据，组件和与官方费用的舍入残差/逐行容差完整保留在episode及pair收据。后续状态不同后的forecast不当同状态反事实，额外查询数不当实际控制时延。

裁决：本批已解决“没有配对物理后缀费用标签”的可定位缺口，但不足以支撑重新训练41维费用头、森林或阈值/风险搜索。关闭本次“每周首次STOP”成本学习支线，保留208/261质量主线与全部反例/角色/KEEP。若将来重开，需要先说明更多独立决策状态的覆盖设计及信息价值，再另定物理预算；本轮不自动采集、验证192条或强行训练。没有宣布成本学习不可能或原模型最优。

所有数据仍为原own12、内部cal4祖先和已暴露16日期，折/模型种子/前缀复用非独立、非全链OOF或盲测。历史190拟合、6续训8285updates/604.703秒实际与用户678.156秒cap另列，本轮新增拟合0。最终8/新日期/FRAME/电网/论文/依赖不动；原控制器/排序别名/锁站/128/64/停滞2/.9及费用NET标签不变。Git仅必要派生报告/收据及引用清单，原轨迹与模型正文仍D本地。
'''
(O/'GRID298_report.md').write_text(report,encoding='utf8')
msg='GRID298首次全部144对保存读出15.555秒/600完成，566046行核验：124完整有事件目标=11降/17升/96相同；11无事件及9同1192步未完对无目标。活跃fit8只有71条/每池5..8，8条降费全同一Sep21状态，cal4大收益不能借入训练。0新拟合预测物理，保留208/261，关闭本次首次STOP费用头支线，不扫模型/补采/启动192验证。'
hand='# 当前：GRID298完成，本批不训练新费用头\n\n'+msg+'\n\n先读本目录GRID298_report/summary/support/decision/complete及publication_review（完成后），不重复readout、297采集或全归档。自动跟进ACTIVE但无获资格的新实验，状态无新信息安静；重开必须有新信息及独立可审阅覆盖/预算，不因自动推进制造阶段。\n'
(O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8')
plan=dict(status='BRANCH_CLOSED_NO_SUPPORTED_NEW_FIT',scope='All144 saved pairs audited and124 cost targets derived once. No next numerical experiment, fit, collection or192 validation authorized by this stage.',closed=['firstSTOP cost head on this dataset','forest/residual/risk/threshold/seed repairs'],requirements_to_reopen=['new identifiable information gap','more independent decision-state coverage design','separate explicit physical budget if new simulation needed'],new_training_qualified=False,automation='ACTIVE_QUIET_IF_UNCHANGED')
save(O/'NEXT_PROTOCOL_PLAN.json',plan)
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/name;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID298完成，成本头不资格\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if name=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-03 — GRID298配对费用目标/支线裁决\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID298完成，无获资格的新实验\n\n'+msg+'\n\n1. 完成必要派生结果上传与具体Gitblob/D协作核验，不重跑readout/采集/旧归档。\n2. 保留208/261；本批不训练、不扩大物理预算、不按已看结果挑种子或调用。\n3. 仅新信息和独立可审阅覆盖/预算支持时才重开成本学习；无变化安静，不制造新阶段。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID298目标核查完成，本批成本头不资格\n\n[报告](protocols/GRID298_saved_suffix_targets/results/GRID298_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
(R/'outputs/grid297_paired_collection_preparation/CONTINUOUS_HANDOFF.md').write_text(hand+'\n最新起点 outputs/grid298_saved_suffix_targets。\n',encoding='utf8')
S=A/'results';S.mkdir(exist_ok=False);C.mkdir(parents=True,exist_ok=False)
names=['protocol.json','source_freeze.json','launch_verification.json','started.json','summary.json','support.json','decision.json','complete.json','pairs.json','episodes.json','GRID298_report.md','CONTINUOUS_HANDOFF.md','NEXT_PROTOCOL_PLAN.json']
for name in names:shutil.copy2(O/name,S/name);shutil.copy2(O/name,C/name)
for sub in ['episode_receipts','pair_receipts']:shutil.copytree(O/sub,S/sub)
shutil.copy2(Path(__file__),S/'finalize.py')
for name in ['readout_stdout.log','readout_stderr.log']:shutil.copy2(O/name,S/(name.replace('.log','.txt')))
save(S/'archive_manifest.json',{p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file()})
save(C/'delivery_hashes.json',{name:sha(C/name) for name in names})
print('GRID298_FINALIZED',len(pairs),len(groups),'NO_NEW_FIT')
if __name__=='__main__':pass
