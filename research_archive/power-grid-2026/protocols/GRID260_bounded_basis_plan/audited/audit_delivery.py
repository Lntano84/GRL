"""First saved qualification audit and delivery; no fitting or outer input export."""
from pathlib import Path
import json,hashlib,shutil,numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid260_bounded_basis_plan';Q=R/'outputs/grid257_reference_labels';W=Path(__file__).parent;REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID260_bounded_basis_plan'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'complete.json').exists();qual=read(O/'qualification.json');assert qual['passed'] and qual['actual_fits']==qual['new_alpha_fits']==qual['new_outer_policy_scores']==0
for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h
for rel,h in read(O/'qualified_freeze.json').items():assert sha(O/rel)==h
for rel,h in read(A/'qualified/archive_manifest.json').items():assert sha(A/'qualified'/rel)==h
fitrows=0;calrows=0;pos=0;active=0;rows=[]
for p in qual['pools']:
 pool=p['pool'];assert p['parameters']==41 and p['finite_difference_gradient_error']<1e-6
 for folder in ['fit_inputs','calibration_inputs']:
  z=np.load(O/folder/f'{pool}.npz');old=np.load(Q/folder/f'{pool}.npz');assert set(z.files)==set(old.files);expected=old['X'].copy();expected[:,:38]=np.tanh(expected[:,:38]);assert np.array_equal(z['X'],expected)
  for key in z.files:
   if key!='X':assert z[key].dtype==old[key].dtype and z[key].tobytes()==old[key].tobytes()
  assert np.isfinite(z['X']).all() and abs(z['X']).max()<=1 and np.linalg.norm(np.c_[z['X'],np.ones(len(z['X']))],axis=1).max()<=np.sqrt(41)+1e-12
  if folder=='fit_inputs':
   fitrows+=len(z['ids']);assert bool(z['fallback_KEEP'][0])==p['KEEP']
   if not p['KEEP']:active+=len(z['ids'])
  else:calrows+=len(z['ids']);pos+=int(z['y'].sum());assert (z['X'][:,39]==1).all()
 rows.append(dict(pool=pool,all_nonX_fields_identical=True,transform_exact=True,b_bit_and_offset_preserved=True,gradient_qualification_retained=True))
assert fitrows==2859 and active==2725 and calrows==2054 and pos==500 and sum(p['future_linear_fit'] for p in qual['pools'])==11
sat=read(O/'saturation_checks.json');coll=read(O/'collision_checks.json');stats=dict(exact_unit_public_coordinates=sum(r['exact_unit_coordinates'] for r in sat),total_public_coordinates=sum(r['total_public_coordinates'] for r in sat),matrix_rows_with_exact_unit=sum(r['rows_with_exact_unit'] for r in sat),original_duplicate_groups=sum(len(r['original_duplicate_groups']) for r in coll),transformed_duplicate_groups=sum(len(r['transformed_duplicate_groups']) for r in coll),introduced_opposite_label_collision_groups=sum(len(r['newly_mixed_due_to_mapping']) for r in coll),old_fit_augmented_norm_max=max(p['fit']['original_augmented_norm_max'] for p in qual['pools']),new_fit_augmented_norm_max=max(p['fit']['bounded_augmented_norm_max'] for p in qual['pools']),largest41gradient_error=max(p['finite_difference_gradient_error'] for p in qual['pools']))
assert stats['introduced_opposite_label_collision_groups']==0
save(O/'independent_audit.json',dict(passed=True,saved_qualification_audit_only=True,actual_fits=0,new_alpha_fits=0,new_network_forward=0,new_outer_inputs_or_scores=0,rows=rows,statistics=stats,all_rows_labels_weights_roles_unchanged=True,fit8_rows=fitrows,active_rows=active,protected_positive=pos,extra_scale_or_feature_choice=False))
save(O/'decision.json',dict(passed=True,ready_for_one_fixed_training=True,next='GRID261 one11linear+11original221positivealpha/one600s; same41paramobjective andallrows/weights/500positives,oneKEEP. Primary258; transformALL38deploymentpublicX only afterallweights published.',qualification_not_effectiveness=True,do_not_tune_scale_from_outer=True))
save(O/'complete.json',dict(passed=True,qualification_only=True,actual_fits=0,new_outer_policy_scores=0,future_linear_fits=11,future_scalar_fits=11,future_total_fits=22,future_fit_window_s=600))
report=f'''# GRID260：有界公共输入资格

固定输入映射已通过资格检查，尚未训练。对全部38个已标准化公共坐标使用tanh，父b、停止标记、偏置b，以及所有行、标签、权重、日期角色和KEEP均逐字段保持。

## 已验证

- 原2859条拟合候选、2725条活跃拟合行、2054条own12 STOP约束候选和500个正例保留；活跃11池仍保护464条，f0_s1继续KEEP。
- 24份自身fit8/cal4矩阵合计{stats['total_public_coordinates']}个公共坐标，其中{stats['exact_unit_public_coordinates']}个在浮点计算下精确达到±1，涉及{stats['matrix_rows_with_exact_unit']}个矩阵行；矩阵间存在角色/种子复用，不能当独立样本数。
- 每份矩阵中，原始与变换后的完全重复输入组数均为0，未新增相反标签的精确输入碰撞。它不证明细微区分能力没有下降。
- 拟合矩阵的最大增广行L2范数从{stats['old_fit_augmented_norm_max']:.6f}降至{stats['new_fit_augmented_norm_max']:.6f}；全部41个增广坐标绝对值≤1。原41维目标最大有限差分梯度误差{stats['largest41gradient_error']:.3g}，没有调用优化器或拟合alpha。

## 数学含义和局限

线性修正为theta与有界增广输入的内积，其绝对值≤||theta||1≤sqrt(41)||theta||2。这个界随学习到的参数变化，不是固定末端±2硬截断。参数仍41个，但函数类及单位L2对应的功能先验改变，不能声称严格同容量或预设性能提高。

本轮只检查自身fit8/cal4，未读取新的外层结果、未导出变换后的外层输入或新策略分数。映射可能压缩有用差异；没有精确碰撞不代表能保住回本或减少查询，也不能保证整周质量/真实控制时间。

## 下一步固定一次训练

先发布全部资格输入和qualified_freeze，再独立封存GRID261：11线性+11原221正例保护步长、一次600秒，f0_s1 KEEP，40输入41参数、原219均值NLL+完整单位L2、零41初始化、原L-BFGS-B设置及success/gradinf<.001资格保留。主匹配258，250低查询/GATE和230历史结果复用，不扫描变换尺度、正则、阈值或种子。

全部权重和步长发布后，才在原249部署公共输入上施加相同的38维映射并评分；b和nativeSTOP列保留，不能沿用未变换的原258评分入口。全部预测发布后才连接结果。原标签/费用/风险/500保护/原父控制器和128/64/停滞2/.9不变，原祖先和暴露开发局限保持，最终8/新物理/第二电网/论文/FRAME未动。
'''
(O/'GRID260_report.md').write_text(report,encoding='utf8')
handoff='''# GRID260后持续交接

先读integration/docs/RESEARCH_STATE/NEXT，以最新为准。260仅固定整体38公共坐标tanh资格完成，源码743bb294先发；实际0优化器/alpha/NN头forward/新outer输入或策略分数，不重做资格/归档。全部257行/NET/权重/源质量/角色/KEEP/b/nativeSTOP/offsetb逐字段不变：2859候选/2725活跃、2054保护候选/500正例(活跃464)、11活跃池+f0_s1KEEP。24矩阵186694公共坐标22个精确饱和±1/20矩阵行，旧新完全重复输入组均0、无新增相反标签碰撞；非无信息损失证明。最大拟合增广行范数38.634718→4.966364，41梯度最大误差1.23e-10。固定尺度1来自旧标准化，不拟合温度/扫阈值/选列或按outer定变换。

下一GRID261只一个新候选：先核对260qualified_freeze/全部fit_inputs与calibration_inputs已发布且Git blob/远端一致，再独立封存261源码输入发布。消费260数组，仅fit8风险训练theta、own12全部500正例单alpha保护(活跃464)，固定11线性41参+11原221正例步长/一次600秒，不延长/重做，f0_s1KEEP不激活。原219 meanNLL+全41单位L2含截距，零41init，scipyL-BFGS-B maxiter2000 gtol1e-6 ftol1e-12 maxls50；全部solver success且原始gradinf<.001后才评分，失败保留不救结果。可以继承258数值训练循环并做一致性检查，不调用旧train重做旧模型。参数个数仍41但函数类和L2功能先验变了，明确披露。

部署评分必须同步变换：全部12权重/步长先发布后，读取原GRID249 outer_parent_public中的X，用Z=X.copy();Z[:,:38]=np.tanh(X[:,:38]);Z[:,38:40]仍原父b/nativeSTOP，原offsetb/base_CONT/ids保持，才按原221浮点表达式计算新分数。不把原258未变换X直接交新theta；无需新NN/收益头forward、新scaler或标签。全部12预测/4681分数先发布后才连原旧开发结果/14268序列。主匹配冻结258，250低query/GATE/230历史全部复用，不追加变换/alpha/正则/费用/阈值/种子搜索。

公共有界输入使线性修正<=||theta||1<=sqrt41||theta||2，这是依赖theta的界，不是固定末端±2输出上限；旧48硬界结果不当本方案有效证据。资格无碰撞和梯度通过不等质量改善。保留258回本48/55及其种子反例，259四类贡献/方向步长诊断不重跑，不给Jan5/调用/种子例外；小而重复改善保留，不设全赢5%新门槛，不把多query质量提升冒充等预算。

原父NN/头own12及内部cal4祖先局限不变；旧开发和相关前缀/种子/折复用非独立，不称全链OOF/盲测，局部代理不等整周或真实控制时间。原费用标签/风险权重/全部日期种子/控制器/排序别名/锁站/128/64/停滞2/.9保持，最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。18参考轨迹/6恢复8285update/原604.703s与用户678.156cap和后续读出操作数全部保留，不隐藏大搜索。

每实质阶段简短中文汇报真实报告，无变化或纯准备安静。已有运行器跟进，完整拟合评分审计归档不重复；按AGENTS实际源码A归档，更新docs状态决定日志NEXT/首页，协作/清单/本地远端核验。Githelper严格等session退出再下一helper/屏障，并待提交路径白名单。用户允许持续自主推进训练至喊停；停则停新增并暂停grl，不恢复其他旧自动化不归档聊天。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
D=A/'audited';D.mkdir(exist_ok=False)
for name in ['GRID260_report.md','independent_audit.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md']:shutil.copy2(O/name,D/name)
shutil.copy2(Path(__file__),D/'audit_delivery.py');save(D/'archive_manifest.json',{p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file()})
C=R/'论文协作资料/FRAME_有界公共输入资格_GRID260';C.mkdir(exist_ok=False)
for name in ['GRID260_report.md','independent_audit.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md']:shutil.copy2(O/name,C/name);assert sha(O/name)==sha(C/name)
save(O/'delivery_check.json',dict(passed=True,collaboration=str(C),files={p.name:sha(p) for p in C.iterdir()},actual_archive=str(A)))
msg='GRID260固定全部38公共坐标tanh资格完成，0优化器/alpha/新NN头forward/新outer输入或策略。257行/标签/权重/角色/b/bit/offsetb/KEEP全不变，2725活跃行/500正例(活跃464)。24矩阵186694公共坐标22精确饱和/20矩阵行，旧新重复输入组均0，无相反标签精确碰撞；最大拟合增广行范数38.634718→4.966364，41梯度max1.23e-10。仅资格不证明有效，41参数但函数类/L2功能先验变更。下一261固定11线性+11原221步长/600s，与258主匹配，部署也须同38维tanh；全模型/预测发布屏障保持。'
DD=REPO/'docs'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=DD/name;s=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID260有界公共输入资格\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':title,rest=s.split('\n',1);s=title+'\n'+entry+rest
 else:s+=entry
 p.write_text(s,encoding='utf8')
(DD/'NEXT_STEPS.md').write_text('# Next steps — GRID260后\n\n1. '+msg+' 资格和归档不重做。\n2. 核对260qualified_freeze输入发布后，独立封存261单候选源；消费260fit/cal数组、11theta41参+11原正例步长一次600s，原219meanNLL+全单位L2/零init/原L-BFGS-B及数值资格，KEEP不变。\n3. 全权重步长先发布后在249原部署X做同样Z[:,:38]=tanh(X[:,:38])，b/bit/offset/base_CONT保持，0新NN/头forward；全预测先发布后连结果。主匹配258及250/GATE历史，所有种子反例保留，不扫尺度/正则/alpha/阈值/费用。原祖先/开发/整周时延局限和最终8/物理/电网/论文/FRAME边界保留。Githelper串行与路径白名单。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID260有界公共输入资格\n\n[报告](protocols/GRID260_bounded_basis_plan/audited/GRID260_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print('GRID260_AUDITED_DELIVERED',json.dumps(stats,ensure_ascii=False))
