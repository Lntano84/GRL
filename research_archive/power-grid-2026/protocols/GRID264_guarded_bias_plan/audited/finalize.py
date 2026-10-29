from pathlib import Path
import sys,json,shutil
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
O,R,A,REPO=C.O,C.R,C.A,C.REPO
assert not (O/'independent_audit.json').exists();C.verify();q=C.rd(O/'qualification.json');assert q['passed'] and q['actual_fits']==q['actual_scalar_minimum_calls']==0
assert sum(p['positives'] for p in q['pools'])==500 and sum(p['positives'] for p in q['pools'] if not p['KEEP'])==464
assert all(p['old_positive_scores_finite_nonnegative'] and p['same_batch_and_row_expression'] and p['outer4excluded'] and p['real_delta_not_computed'] for p in q['pools'])
for rel,h in C.rd(O/'qualified_freeze.json').items():assert C.sha(O/rel)==h
assert C.rd(O/'synthetic_checks.json')['passed'] and C.rd(O/'runtime_freeze.json')['work/grid264_guarded_bias_plan/runtime264.py']==C.sha(C.W/'runtime264.py')
C.save('independent_audit.json',dict(passed=True,saved_receipts_hashes_roles500and464_verified=True,real_scalar_fit0=True,syntheticonly_fitted_deltas=True,no_new_outer_policy_or_outcomes=True))
C.save('decision.json',dict(passed=True,ready_for_single265_scalar_component=True,actual264fits0=True,future_newscalar11_KEEP1_cap600=True,explicit_changed_stopping_threshold_and_empirical_objective=True,no_theta_or_NN_refit=True,no_outer_guarantee=True))
(O/'GRID264_report.md').write_text('''# GRID264：停止强度校准执行前核查

本阶段资格通过，真实拟合0、真实偏置求解0、新外层策略分数/结果0。12池原权重及校准输入的发布Git blob已核对；原500条回本STOP经验记录保持，其中活跃464条、KEEP36条。固定8/4内部角色和外层4日期排除关系核对，原父own12及内部cal4祖先局限没有因此消失。

检查仅确认现有正例分数有限且非负、整批与逐行使用同一浮点表达式，没有求真实最小分数或偏置。合成数据覆盖并列、零裕度、极小数、一ULP、空保护集、KEEP及非法负保护分数。相等时继续，空集和KEEP保持原策略；非法负保护分数必须失败，不能偷偷钳制救结果。

下一独立GRID265若执行：冻结GRID261全部网络、线性参数、alpha和tanh38表示，仅校准11个向下偏置，1个KEEP，一次600秒。每池唯一规则是取全部原own12回本STOP记录允许的最大可行偏置，不扫描多个强度、不按外层选择。原分数减偏置后仍以零判断，但数学上等价提高原分数阈值；它明确改变停止工作点和经验校准目标，不冒充阈值或原NLL目标不变。

所有12组标量/固定模型先发布，才计算新的4681公共外层分数；所有12份预测先发布，才连接旧开发结果。可复用GRID261已发布公开分数再减偏置，避免不必要重算网络。主要匹配固定GRID261，全部日期/种子/回本损失反例保留，不追调偏置。经验保护不保证新外层质量，不能声称泛化或整周/真实时延改善。

264新增拟合0，分支仍62；265作为新假设若执行另11，总计73。6次旧续训8285更新、原NN604.703秒/用户批准678.156秒上限另列。最终8未生成/读取，无新物理/电网/论文/FRAME/安装，原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保持。
''',encoding='utf8')
C.archive('audited');D=R/'论文协作资料/FRAME_停止强度资格_GRID264';D.mkdir(exist_ok=False)
for name in ['GRID264_report.md','qualification.json','synthetic_checks.json','independent_audit.json','decision.json','complete.json']:shutil.copy2(O/name,D/name)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID264执行资格通过、0真实拟合/偏置极值/新outer分数结果；12固定模型/校准输入发布blob核对，原500正例(活跃464/KEEP36)、8/4角色及outer4排除关系、整批/逐行表达式一致，合成并列/零/极小数/一ULP/空集/KEEP/非法负保护分数通过。下一独立265仅冻结261校准11最大可行向下偏置+1KEEP/600s，全标量先发、全预测先发后连结果；显式等价提高阈值与新经验目标，不称原NLL/阈值不变，不扫描力度。资格无外层质量保证，全部反例保留。当前分支62、265若执行另11至73，原祖先/暴露开发/整周/时延边界不变。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/name;t=p.read_text(encoding='utf8');e='\n\n## 2026-10-29 — GRID264停止强度资格\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+e+rest
 else:t+=e
 p.write_text(t,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID264后\n\n1. '+msg+' 完整资格不重做。\n2. 下一265先核对264qualified_freeze/runtime_freeze与具体发布blob，独立冻结源码输入后才11标量/一次600s；不重训theta/NN。使用已资格runtime的base_score/guarded_shift/apply_shift，KEEP不变；全模型标量发布后复用261公开分数减delta，再全预测发布后连旧结果。\n3. 主匹配261及GATE，所有seed日期/反例保留，不按结果改力度或选种子。新假设计数另11，最终8/物理/电网/论文/FRAME不动，不安装；原费用标签控制器排序别名锁站128/64/停滞2/.9保持。Githelper串行及路径白名单、实际A归档协作远端核验。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID264停止强度资格\n\n[报告](protocols/'+A.name+'/audited/GRID264_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print('GRID264_AUDITED_ZERO_FITS')
