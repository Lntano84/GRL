"""First saved-public-output audit; no repeat NN/forest prediction or label access."""
from pathlib import Path
import json,hashlib,shutil,collections
import numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid256_reference_public';W=Path(__file__).parent;REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID256_reference_public';ROLE=R/'outputs/grid241_parent_prior_plan';PLAN=R/'outputs/grid253_stop_source_plan'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'review.json').exists();g=read(O/'generation_receipts.json');assert g['passed']
for rel,h in read(O/'generated_freeze.json').items():assert sha(O/rel)==h,rel
for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
for rel,h in read(A/'generated/archive_manifest.json').items():assert sha(A/'generated'/rel)==h,rel
access=read(O/'public_access.json');forbidden={'y','NET','actual_NET','mass','target','target_fit','event_y'};assert all(not forbidden&set(a['fields']) for a in access)
meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');uses=read(PLAN/'reference_uses.json');nodes={n['node']:n for n in read(PLAN/'reference_nodes.json')};refs={};refrows=0
for name,n in nodes.items():
 z=np.load(O/'reference_scores'/f'{name}.npz');assert np.array_equal(z['ids'],n['score_ids']) and z['q'].shape==(len(n['score_ids']),2) and np.isfinite(z['q']).all();assert {meta[int(i)]['date'] for i in z['ids']}==set(n['reference_score_dates']);assert not set(n['fit_dates'])&set(n['reference_score_dates']);refs[name]={int(i):z['q'][k] for k,i in enumerate(z['ids'])};refrows+=len(z['ids'])
assert refrows==32238;raw=np.load(R/'outputs/grid214_inner_head_generation/public_raw41.npz')['values'];records=[];totals=collections.Counter();groups={}
for i,m in enumerate(meta):groups.setdefault(m['sample_id'],[]).append(i)
for ids in groups.values():ids.sort(key=lambda i:meta[i]['position'])
for role in read(ROLE/'qualified_pools.json'):
 pool=role['pool'];z=np.load(O/'public_full'/f'{pool}.npz');par=np.load(O/'original_parent83'/f'{pool}.npz');old=np.load(R/'outputs/grid192_gate_support/fit_features'/f'{pool}.npz');ids=z['ids'].astype(int);pos={int(i):k for k,i in enumerate(ids)};dates=set(role['fit_dates'])|set(role['cal4_dates']);assert {meta[int(i)]['date'] for i in ids}==dates and not dates&set(role['outer_dates']);assert np.array_equal(ids,par['ids'])
 for u in [u for u in uses if u['gate_pool']==pool]:
  for i in ids:
   if meta[int(i)]['date'] in u['record_dates']:assert z['reference_node'][pos[int(i)]]==u['reference_node'] and np.array_equal(z['reference_q'][pos[int(i)]],refs[u['reference_node']][int(i)])
 chosen=np.zeros(len(ids),bool);stop=np.zeros(len(ids),bool)
 for groupids in groups.values():
  if meta[groupids[0]]['date'] not in dates:continue
  ix=[pos[i] for i in groupids];q=z['reference_q'][ix];native=q[:,1]<=q[:,0];s=int(np.flatnonzero(native)[0]) if native.any() else len(ix);chosen[np.array(ix[:s],int)]=True
  if s<len(ix):chosen[ix[s]]=True;stop[ix[s]]=True
 assert np.array_equal(chosen,z['selected']) and np.array_equal(stop,z['domain_STOP']);assert np.array_equal((z['reference_q'][:,1]<=z['reference_q'][:,0])[chosen],stop[chosen]);sc=np.load(R/role['residual_public41_scaler']);keep=[k for k in range(41) if k not in [14,18,19]];X=np.c_[((raw[ids]-sc['mean41'])/sc['scale41'])[:,keep],z['b'],(z['reference_q'][:,1]<=z['reference_q'][:,0]).astype(float)];assert np.array_equal(X,z['X']) and X.shape==(len(ids),40)
 gain=z['parent_event_p']*z['parent_mu'];assert np.array_equal((gain-z['charge'])/(gain+z['charge']),z['b']) and np.array_equal(z['parent_CONT'],gain>z['charge']);fit=np.array([meta[int(i)]['date'] in role['fit_dates'] for i in ids]);assert np.array_equal(fit,z['fit_role']) and np.array_equal(~fit,z['cal_role'])
 oldpos=[pos[int(i)] for i in old['ids']];assert np.array_equal(par['x'][oldpos],old['x']);cached_bytes=hashlib.sha256(par['x'][oldpos].tobytes()).hexdigest();assert cached_bytes==hashlib.sha256(old['x'].tobytes()).hexdigest()
 for folder,mask in [('fit_public',fit&chosen&z['parent_CONT']),('calibration_public',stop&chosen&z['parent_CONT'])]:
  pub=np.load(O/folder/f'{pool}.npz');assert np.array_equal(pub['ids'],ids[mask]) and np.array_equal(pub['X'],X[mask]) and np.array_equal(pub['b'],z['b'][mask]) and np.array_equal(pub['domain_STOP'],stop[mask]);assert not forbidden&set(pub.files)
 fp=np.load(O/'fit_public'/f'{pool}.npz');cal=np.load(O/'calibration_public'/f'{pool}.npz');r=dict(pool=pool,fit8_rows=len(fp['ids']),fit8_STOP=int(fp['domain_STOP'].sum()),fit8_CONT=int((~fp['domain_STOP']).sum()),own12_STOP_constraints_public_rows=len(cal['ids']),cached83_rows=len(old['ids']),missing_original83_rows=len(ids)-len(old['ids']),cached_bytes_identical=True,no_outcomes_in_exports=True);records.append(r)
 for k in ['fit8_rows','fit8_STOP','fit8_CONT','own12_STOP_constraints_public_rows','cached83_rows','missing_original83_rows']:totals[k]+=r[k]
assert totals['cached83_rows']==22653 and totals['missing_original83_rows']==9585 and totals['fit8_rows']==2859 and totals['fit8_STOP']==1251 and totals['fit8_CONT']==1608 and totals['own12_STOP_constraints_public_rows']==2054
review=dict(passed=True,receipt_saved_output_audit_only=True,reference_rows=refrows,pools=records,totals=dict(totals),all_reference_uses_exclude_record_and_outer_dates=True,first_STOP_preSTOP_CONT_selection_exact=True,canonical40_inputs_and_roles_exact=True,original_parent_cache_bytes_reused=True,weights_unchanged=True,NN_or_head_predictions_repeated_in_audit=0,new_teacher_label_join=0,new_fits=0,new_outer_policy_scores=0,actual_generation_wall_s=g['generation_wall_s'])
save(O/'review.json',review);save(O/'decision.json',dict(passed=True,public_generation_ready_for_publication=True,next='GRID257 separately freeze original-teacher NET provenance and structural mass join; only after allGRID256public exports published. No training or new policy in257.',outcomes_not_yet_connected=True,learning_effect_unknown=True))
save(O/'complete.json',dict(passed=True,public_only=True,reference_rows=32238,fit8_candidate_rows=2859,calibration_candidate_rows=2054,new_teacher_label_join=0,new_fits=0,new_outer_policy_scores=0))
report=f'''# GRID256：参考停止点公开输入

已生成并核验18个固定参考网络的32238条公开评分，覆盖原4外层组和全部3种子。模型训练0次，未连接收益标签，未训练新门控或评估新策略。生成耗时{g['generation_wall_s']:.3f}秒，不是在线控制时间。

## 本轮结果

- 按每条既定教师前缀路径的首次q1<=q0选择STOP，之前为辅助CONT；全部继续至末端不虚构STOP样本。合计13286条被选STOP、11217条被选CONT，均是跨池/种子记录，不能当成独立场景。
- 沿用原父收益判断b>0和固定fit8日期后，有2859条训练候选：1251条STOP、1608条CONT。own12的STOP保护候选2054条。它们尚未连标签，正例数、KEEP分支和最终训练模型数要到下一资格阶段才能确定，不能继续沿用旧85个正例。
- 22653条原父网络83维缓存逐字节复用，9585条缺失输入只调用原owner网络补齐；原24个收益头复用、0拟合。旧缓存对应父b与原公开b最大误差小于1e-12。未把新8天参考隐藏向量输入旧收益头。
- 修正器保持GRID250的40维输入：固定inner8 scaler下的38个公共特征、原父b、nativeSTOP位；不加入GRID252连续qgap。参考网络则沿用原GRID121原始41维数据及六组fit8归一化；这两套输入不能混淆。

## 核验边界

18权重先发布并逐项核对Git blob；GRID256源码输入5732e2fc再发布后才生成。只读取公共输入字段，数组访问清单未读取y/actual_NET/target/mass等标签字段，也未打开cases.pkl。首次独立核验只检查保存的分数、首次停止路径、角色、40维输入和缓存字节，不重复网络或树预测。

新参考网络同时排除记录日期组及当前外层日期组；原父网络/收益头仍有own12标签祖先，内部cal4可能进入其他参考的训练祖先，不是全链独立OOF或盲测。16日期均为已暴露开发数据，多个前缀/种子/外层用途不独立。本轮不说明学习质量提高，也不能推断整周质量或实际时延。

## 下一步

先完整发布本轮全部公开分数、选择和输入，再单独封存GRID257标签来源及结构质量计算方案。按原费用/收益尺度连接原教师NET，先核对同一记录与旧已发表标签一致，不因新增记录覆盖改变标签。按原日期/物理调用/前缀结构质量和两域各半mass*absNET公式重算新样本权重；旧正例集合和权重数值不能复用。只做资格，不直接训练。后续仍至多11线性+11保护步长、一次600秒；f0_s1及无fitSTOP正例/空域KEEP不变。

原费用标签/全部种子日期/部署控制器/排序别名/锁站/128/64/停滞2/0.9不变，最终8和新物理/电网/论文/FRAME未动。
'''
(O/'GRID256_report.md').write_text(report,encoding='utf8')
handoff='''# GRID256后持续交接

先读integration/docs/RESEARCH_STATE和NEXT_STEPS，以最新为准；有运行器跟进，完整生成/评分/拟合/审计/归档不重做。GRID255恢复18/18与实际604.703s/授权678.156s已保留，不训练参考模型。GRID256一次公开生成完成32238参考行、22653原父缓存字节复用、9585原owner隐藏补齐、24父头0拟合；首次保存结果审计通过，0标签连接/拟合/新outer策略。公开fit8候选2859(STOP1251/CONT1608)，own12 STOP保护候选2054；未连接标签，正例和活跃拟合数未知，不再沿用85固定。

先完成GRID256全公开输出发布屏障：核对generated_freeze、归档清单、本地远端SHA与具体Git blob，再另阶段GRID257源码/输入封存发布，才连接原教师NET标签。257只标签来源/结构质量/风险和保护集合资格，不训练门控/NN、不新outer策略评分。原同记录NET/费用/scale必须与旧214/242标签核对一致；不能因新STOP覆盖缺失改标签、删记录或假定NN浮点target与原门控NET完全相同。只读取相应自身own12角色的原教师标签，fit8用于拟合风险，全部own12 selectedSTOP正例用于原221经验保护；cal4仍非独立。结构质量按原日期/physical_state_group/sample_id/前缀规则为新STOP/CONT总体重算，再按原两域半半mass*absNET归一化；不得直接复用旧192mass或按外层改比例。若定义/覆盖有歧义先只读资格停，不私自换目标或调参。

固定fit8/cal4和全部3种子/16已暴露日期，f0_s1KEEP，其他fitSTOP正例0或任一域空则KEEP，不补样本/换日期。之后同250 38公共+原父b+nativeSTOP位=40输入41参、不加252qgap；至多11线性+11原221步长，一次600s，全部模型步长先发后外层score、全部预测先发后连结果。18参考轨迹+至多22读出，两真实依赖组件/6恢复更新和实际时间已明确，不隐藏大搜索。

公共256的参考评分使用原GRID121 raw41+六固定fit8scaler/clip20；原父83仍原ownerf网络及原208头，不能新8天隐藏塞旧头；修正器公共38仍原213inner8scaler。所有256文件无收益标签，public_access是具体数组字段清单。不得重跑完整参考评分/父缺失forward来作审计；检查保存输出和哈希即可。

原父NN/头own12及内部cal4祖先局限仍在，不称全链OOF/盲测，旧开发前缀不独立。保留230/250/252线索与全部反例，不择种子/按outer调阈值alpha正则费用、不设全赢5%门槛；代理不等整周或实际时延。原费用/标签/控制器/排序别名/锁站/128/64/停滞2/.9不改；最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。

按AGENTS更新docs状态决定日志NEXT首页，实际源码A归档、协作/清单/hash/远端核验。Githelper严格等退出再下一helper/屏障，待提交路径白名单不夹带。每实质阶段简短中文汇报真实报告，无变化安静；更新本交接/grl起点。用户停则停新增并暂停grl，不恢复其他旧自动化不归档聊天。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
D=A/'audited';D.mkdir(exist_ok=False)
for name in ['GRID256_report.md','review.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md']:shutil.copy2(O/name,D/name)
shutil.copy2(Path(__file__),D/'audit_delivery.py');save(D/'archive_manifest.json',{p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file()})
C=R/'论文协作资料/FRAME_参考停止点公开输入_GRID256';C.mkdir(exist_ok=False)
for name in ['GRID256_report.md','review.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md']:shutil.copy2(O/name,C/name);assert sha(O/name)==sha(C/name)
save(O/'delivery_check.json',dict(passed=True,collaboration=str(C),files={p.name:sha(p) for p in C.iterdir()},archive=str(A)))
DD=REPO/'docs';summary='GRID256固定公开生成/首次保存结果核验完成：18参考32238行，22653原父83缓存字节复用、9585原owner补齐，24头0拟合；fit8候选2859=1251STOP+1608CONT，own12保护候选2054(标签未连，非正例数)。0新训练/标签连接/新outer策略。固定40维/原角色与父b重现，下一257先发布屏障后仅标签来源/新结构质量与风险资格，不能旧85/mass照搬。'
p=DD/'RESEARCH_STATE.md';s=p.read_text(encoding='utf8');title,rest=s.split('\n',1);p.write_text(title+'\n\n## 2026-10-29 — GRID256参考停止点公开输入\n\n'+summary+'\n'+rest,encoding='utf8')
(DD/'NEXT_STEPS.md').write_text('# Next steps — GRID256后\n\n1. '+summary+' 不重做完整模型/评分/forward/审计归档。\n2. 先核对256全公开输出已Git发布，再独立封存257源码输入后连接原teacher NET；同记录核对旧标签与费用scale一致，不能把NNtarget精度默认为原门控标签、改标签/删除缺失。按原公共结构质量规则重算新STOP/CONT风险与own12 STOP正例保护，fit8/cal4固定。257只资格不训练，歧义先定位。\n3. 后续同250 40input41param、不加252gap，f0_s1/空域/无fitSTOP正例KEEP，<=11linear+11alpha/600s，两依赖组件计数披露；原祖先/开发/时延局限不变，最终8/物理/电网/论文/FRAME不动。Githelper串行及路径白名单，核对协作归档远端。\n',encoding='utf8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
 p=DD/name;p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-10-29 — GRID256公开生成\n\n'+summary+'\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID256参考停止点公开输入\n\n[报告](protocols/GRID256_reference_public/audited/GRID256_report.md)；'+summary+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print('GRID256_AUDITED_DELIVERED',json.dumps(dict(totals),ensure_ascii=False))
