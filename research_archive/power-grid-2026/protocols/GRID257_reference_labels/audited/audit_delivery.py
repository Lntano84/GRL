"""First independent saved-label/risk qualification audit and delivery; no fitting."""
from pathlib import Path
import json,hashlib,shutil,collections
import numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid257_reference_labels';P=R/'outputs/grid256_reference_public';W=Path(__file__).parent;REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID257_reference_labels';ROLE=R/'outputs/grid241_parent_prior_plan'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'review.json').exists();q=read(O/'qualification.json');assert q['passed'] and q['actual_fits']==0
for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
for rel,h in read(O/'qualified_freeze.json').items():assert sha(O/rel)==h,rel
for rel,h in read(A/'qualified/archive_manifest.json').items():assert sha(A/'qualified'/rel)==h,rel
meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];records=[];sumfit=0;sumcal=0;sumpos=0;activefit=0;activepos=0;activecontpos=0;activecalpos=0;calls=set();dates=set();allcal_calls=set()
for role in read(ROLE/'qualified_pools.json'):
 pool=role['pool'];qual=next(r for r in q['pools'] if r['pool']==pool);z=np.load(O/'fit_inputs'/f'{pool}.npz');c=np.load(O/'calibration_inputs'/f'{pool}.npz');pub=np.load(P/'fit_public'/f'{pool}.npz');pc=np.load(P/'calibration_public'/f'{pool}.npz');labels=np.load(O/'role_labels'/f'{pool}.npz');full=np.load(P/'public_full'/f'{pool}.npz');mass=np.load(O/'public_mass'/f'{pool}.npz');idx={int(i):k for k,i in enumerate(labels['ids'])};assert np.array_equal(labels['ids'],full['ids']) and np.array_equal(labels['actual_NET'],labels['gain_normalized']-labels['known_charge']);assert set(meta[int(i)]['date'] for i in labels['ids'])==set(role['fit_dates'])|set(role['cal4_dates']);assert not set(meta[int(i)]['date'] for i in labels['ids'])&set(role['outer_dates'])
 charge=np.array([fee*meta[int(i)]['next_queries']/scale for i in labels['ids']]);assert np.array_equal(charge,labels['known_charge'])
 for a,b in [(z,pub),(c,pc)]:
  for key in ['ids','X','b']:assert np.array_equal(a[key],b[key])
  pp=[idx[int(i)] for i in a['ids']];assert np.array_equal(a['actual_NET'],labels['actual_NET'][pp]) and np.array_equal(a['y']>0,a['actual_NET']>0)
 assert all(meta[int(i)]['date'] in role['fit_dates'] for i in z['ids']);assert c['X'].shape[1]==z['X'].shape[1]==40 and np.array_equal(z['domain_STOP'],pub['domain_STOP']);assert np.array_equal(c['fit_role'],pc['fit_role'])
 # Public domain masses must sum half before eligibility/risk filtering.
 assert np.array_equal(mass['ids'],full['ids']) and np.array_equal(mass['selected'],full['selected']) and np.array_equal(mass['domain_STOP'],full['domain_STOP']);assert (mass['source_mass'][~full['selected']]==0).all()
 for domain in [True,False]:
  m=full['selected']&(full['domain_STOP']==domain);assert abs(mass['source_mass'][m].sum()-.5)<1e-12
 pp=[idx[int(i)] for i in z['ids']];assert np.array_equal(z['source_mass'],mass['source_mass'][pp]);stop=z['domain_STOP'];pos=z['y']>0;fallback=pool=='f0_s1' or not np.any(pos&stop) or not stop.any() or not (~stop).any();assert bool(z['fallback_KEEP'][0])==fallback==qual['fallback_KEEP']
 if not fallback:
  n=int(stop.sum())
  for mask in [stop,~stop]:
   risk=z['source_mass'][mask]*abs(z['actual_NET'][mask]);expected=.5*n*risk/risk.sum();assert np.allclose(z['weight'][mask],expected,atol=1e-15,rtol=1e-14) and abs(z['weight'][mask].sum()-.5*n)<1e-10
  assert qual['numerical41_gradient_max_error']<1e-6;activefit+=len(z['ids']);activepos+=int((pos&stop).sum());activecontpos+=int((pos&~stop).sum());activecalpos+=int((c['y']>0).sum())
  for i in z['ids'][pos&stop]:calls.add(meta[int(i)]['physical_state_group']);dates.add(meta[int(i)]['date'])
 for i in c['ids'][c['y']>0]:allcal_calls.add(meta[int(i)]['physical_state_group'])
 sumfit+=len(z['ids']);sumcal+=len(c['ids']);sumpos+=int((c['y']>0).sum());records.append(dict(pool=pool,fit_rows=len(z['ids']),fit_STOP_positive=int((pos&stop).sum()),fit_CONT_positive=int((pos&~stop).sum()),constraint_positive=int((c['y']>0).sum()),KEEP=fallback,own12_label_roles_only=True,original_float64NET_preserved=True,new_mass_risk_verified=True))
assert sumfit==2859 and sumcal==2054 and sumpos==500 and activefit==2725 and activepos==222 and activecontpos==296 and activecalpos==464
assert all(r['old_NET_bit_exact'] and r['max_abs_error']==0 for r in read(O/'old_label_alignment.json'));assert sum(r['old214_rows'] for r in read(O/'old_label_alignment.json'))==22653;assert all(r['same_structural_rule_max_abs_error']<1e-15 for r in read(O/'old_structure_alignment.json'));assert q['linear_fits_planned']==q['scalar_fits_planned']==11
review=dict(passed=True,qualification_saved_output_audit_only=True,new_optimizer_or_scalar_fits=0,new_network_forward=0,new_policy_score=0,old214_NET_bit_exact_rows=22653,old192_structure_rule_reproduced=True,candidate_fit_rows=2859,candidate_fit_STOP_positive=248,active_fit_rows=activefit,active_fit_STOP_positive=activepos,active_fit_CONT_positive=activecontpos,active_fit_STOP_positive_distinct_calls=len(calls),active_fit_STOP_positive_dates=len(dates),candidate_constraint_rows=sumcal,candidate_constraint_positive=sumpos,active_constraint_positive=activecalpos,all_constraint_positive_distinct_calls=len(allcal_calls),linear_fits_next=11,scalar_fits_next=11,total_fits_next=22,fit_cap_s_next=600,pools=records,historical_container_parse_disclosed=True,final8_not_generated_or_read=True)
save(O/'review.json',review);save(O/'decision.json',dict(passed=True,ready_for_fixed_training=True,next='GRID258 unique same250-class anchored correction:11linear41params+11original221steps,one600s. Inputs/source publish then allweights/steps before oldouter scores, allpredictions before outcomejoin.',predeclared_f0_s1KEEP_unchanged=True,all_other_pools_supported=True,no_effectiveness_claim=True))
save(O/'complete.json',dict(passed=True,qualification_only=True,actual_fits=0,new_network_forward=0,new_outer_policy_scores=0,future_fits=22,future_fit_cap_s=600))
report='''# GRID257：参考停止点标签与训练资格

新标签连接及资格核验通过，尚未训练。原2859条候选中有248条STOP回本正例、313条CONT回本正例。固定f0_s1继续KEEP，实际用于下一轮的11池共2725行，其中STOP回本222条、CONT回本296条；11个41参数线性模型与11个原正例保护步长具备既定支持，下一轮仍一次600秒、总22拟合，不调参数或新增分支。

## 标签没有更换

沿用原教师gains和queries，以及原费用/收益尺度，按旧196/213/214顺序计算float64的gain/scale - fee*queries/scale。22653条与旧214重叠记录的NET逐位相同，没有用NN的float32训练target替代，没有因新停止点缺少旧中间文件就删样本或重定义标签。

历史cases容器整体反序列化；收益/查询字段只按每个模型池自身own12日期的metadata索引访问，fit8连接拟合风险，own12的selectedSTOP连接经验保护。不声称历史容器的外层存储物理上未被解析，也不称全链独立盲测；最终8未生成或读取。

## 权重按原规则重算

先在新参考选择的全部own12 STOP和CONT上，按日期、物理调用、样本前缀等分结构质量；CONT再在同一样本的停止前位置内等分。两域各半，在原父b>0和fit8资格筛选前计算。该公式先复现旧192存储的结构质量，随后为新样本重新计算；旧mass数值未直接复用。

拟合权重仍为各域mass*absNET分别归一化到一半，并保留原NfitSTOP的总缩放，目标仍是加权平均逻辑损失加完整单位L2。41维梯度资格通过，实际优化器和步长拟合均为0。

## 更多支持，不等于效果提高

own12保护候选2054条中有500条回本正例，固定KEEP池占36条，实际11池将保护464条；不能继续使用旧85的集合。实际222条拟合STOP正例覆盖25个不同物理调用、10个旧开发日期；多种子、外层用途和相关前缀会重复，不是222个独立场景。500条候选保护正例对应43个不同物理调用。

这些结果说明新的训练停止点提供了更多有价值补查的标签，尚不能推出新模型会少误停或改善整周表现。参考NN只修正停止点来源；原父NN/收益头仍own12含自身标签，内部cal4仍可能进入其他参考的祖先，own12保护是经验约束而非独立验证。

## 下一步固定方案

先发布本轮全部合格输入和qualified_freeze，再单独封存GRID258源码。只训练同GRID250的40输入41参数ANCHORED模型：11个线性修正器加11个原221最大正例保护步长，f0_s1 KEEP，原8/4日期角色与全部种子保留；一次600秒，不延长/扫参或择种子。原219加权均值逻辑损失+全41参数单位L2、零初始化、原L-BFGS-B maxiter2000/gtol1e-6/ftol1e-12/maxls50，所有success且原始梯度无穷范数<.001后才能评分。

先发布全部模型/步长，再读取原已发表的部署公共输入计算新外层分数；全部分数发布后才连接结果。主匹配原250与公共GATE，230只保留历史主候选；本轮样本来源/权重数值/保护集合改变，不称单一因素消融。部署控制器、原父b/排序别名/锁站/128/64/停滞2/.9、原费用/标签/全部日期种子不变，不加入252qgap。最终8、新物理/电网/论文/FRAME未动。
'''
(O/'GRID257_report.md').write_text(report,encoding='utf8')
handoff='''# GRID257后持续交接

先读integration/docs/RESEARCH_STATE/NEXT，以最新为准，有运行器跟进；完整拟合/生成/评分/资格审计/归档不重复。GRID256公开69文件先发布63856c8d/3701581c，GRID257源码80211640先发后标签连接。257资格/首轮保存结果核验完成，0优化器/alpha/网络forward/新outer策略。旧214重叠22653条NET逐位一致，旧192结构公式复现，未用NNfloat32target。历史cases容器整体反序列化披露，字段仅按每池own12索引，非物理未解析外层储存或全链独立。

新候选2859=1251STOP+1608CONT，STOP回本248/CONT313；原f0_s1KEEP保留26/17正例不训练。11活跃池2725行，STOP回本222/CONT296；own12 STOP候选2054/回本500，其中KEEP36，实际11池保护464。222拟合STOP正例只25物理调用/10日期，500约束正例43调用；前缀/种子/用途相关，不能算独立或已改善。新结构质量在原b筛选前的全部own12新selectedSTOP/CONT计算，再fit8各域半半mass*absNET，未复用旧mass/85集合。qualified_freeze和fit/calibration_inputs在outputs/grid257_reference_labels。

下一GRID258先核对257合格输入已发布/具体Git blob/远端，再冻结唯一同250类源码输入推送。固定11线性41参+11原221正例步长=22，一次600s，不延长/重做/扫参，f0_s1KEEP，所有其他池已资格支持。40输入=38公共+原父b+nativeSTOP，不加252qgap；原219 meanNLL+单位完整41L2含截距、零41init、scipy原L-BFGS-B maxiter2000/gtol1e-6/ftol1e-12/maxls50。仅fit8风险数据训练theta，单alpha保护该池全部own12 selectedSTOP正例(新集合)，按原221浮点表达式，不换校准源/新增查询约束。先全部模型和步长发布，再计算外层策略分数；所有12池预测发布后才连接结果。原部署公共输入直接复用GRID249 outer_parent_public和旧220base_CONT/父b，输入仍原213inner8scaler、40维nativeSTOP=1，无新NN/头forward。核对old250公共输入相同，不追加分支/网络/特征/费用或新原FRAME比较。可以参考250run训练和score代码，但不能调用旧prepare/train重做旧模型。

主匹配冻结250与公共GATE，230保持历史主候选，全部日期种子反例保留，不从已看outer挑规则/参数/种子。样本来源/风险数值/约束集合都改变，不称纯单因素消融或同样本暴露。原父NN/头own12和内部cal4祖先局限仍在，own12正例保护非独立cal4，不称全链OOF/盲测。局部代理不等整周/实际时延，不把更多query带来质量增量当同预算优势；小而重复改善保留，不设全赢5%新门槛。

原18参考NN不重训，255用户授权额外120s实际46.547/累计604.703s/上限678.156和6续训更新8285全部保留，不冒充原600窗。整个单探针18轨迹+后续22拟合，两有依赖组件和恢复实例/时间不隐去。最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖；部署控制器/原费用标签/排序别名/锁站/128/64/停滞2/.9不变。

按AGENTS更新docs状态/决定/日志/NEXT/归档首页，实际源码A归档，协作/清单/本地远端核验；Githelper严格等退出再下一helper/发布屏障，待提交路径白名单不夹带。每实质阶段简短中文汇报真实报告，无变化安静；同步本交接和grl起点。用户允许自主推进至喊停；停则停新增并暂停grl，不恢复其他旧自动化不归档聊天。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
D=A/'audited';D.mkdir(exist_ok=False)
for name in ['GRID257_report.md','review.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md']:shutil.copy2(O/name,D/name)
shutil.copy2(Path(__file__),D/'audit_delivery.py');save(D/'archive_manifest.json',{p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file()})
C=R/'论文协作资料/FRAME_参考停止点标签资格_GRID257';C.mkdir(exist_ok=False)
for name in ['GRID257_report.md','review.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md']:shutil.copy2(O/name,C/name);assert sha(O/name)==sha(C/name)
save(O/'delivery_check.json',dict(passed=True,collaboration=str(C),files={p.name:sha(p) for p in C.iterdir()},archive=str(A)))
DD=REPO/'docs';summary='GRID257标签/结构质量/风险资格完成，0拟合/新forward/新outer策略。旧214重叠22653条NET逐位一致、旧192结构公式复现；新fit候选2859，含STOP回本248/CONT313，固定KEEP池26/17正例仍不训练。活跃11池2725行/STOP正222/CONT正296，own12候选正500(KEEP36)实际保护464；STOP训练正仅25调用/10日期，非独立记录数。下一258固定11×41参线性+11原221步长/一次600s，先发布输入源码/全权重/全预测各屏障，不改原目标或费用。'
p=DD/'RESEARCH_STATE.md';s=p.read_text(encoding='utf8');title,rest=s.split('\n',1);p.write_text(title+'\n\n## 2026-10-29 — GRID257参考停止点标签资格\n\n'+summary+'\n'+rest,encoding='utf8')
(DD/'NEXT_STEPS.md').write_text('# Next steps — GRID257后\n\n1. '+summary+' 资格/旧完整模型/生成评分不重做。\n2. 核对257qualified_freeze全输入发布，再独立封存258源码，消费fit_inputs/calibration_inputs；同25040输入41参、原219均值NLL+全单位L2、零41init/原L-BFGS-B、11线性+11原正例步长一次600s。所有success/gradinf<.001后才评分，失败保留不救结果；f0_s1KEEP不激活。\n3. 全权重/步长先发布，复用249 outer_parent_public(原b/base_CONT/40维)评分，无新NN/头forward；全预测先发布后连结果。主匹配250/GATE，全种子反例保留，原祖先/开发/整周时延局限保留。最终8/物理/电网/论文/FRAME不动，Githelper串行及路径白名单。\n',encoding='utf8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
 p=DD/name;p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-10-29 — GRID257标签资格\n\n'+summary+'\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID257参考停止点标签资格\n\n[报告](protocols/GRID257_reference_labels/audited/GRID257_report.md)；'+summary+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print('GRID257_AUDITED_DELIVERED',activepos,activecontpos,activecalpos,len(calls),len(dates))
