from pathlib import Path
import sys,shutil,json,numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,P=C.O,C.A,C.REPO
assert not (O/'complete.json').exists();C.verify();q=C.rd(O/'qualification.json');assert q['passed'];rr=C.rd(O/'qualified_pools.json');extra=0
for r in rr:
 p=r['pool']
 for role in ['fit','cal']:
  old=np.load(C.F/'qualified_inputs'/f'{p}_{role}.npz');new=np.load(O/'qualified_inputs'/f'{p}_{role}.npz')
  assert new['X'].shape[1]==39 and np.array_equal(old['X'][:,:39],new['X']) and np.array_equal(old['X32'][:,:39],new['X32']) and np.array_equal(old['X'],new['X40_original'])
  for k in old.files:
   if k not in ['X','X32']:assert old[k].dtype==new[k].dtype and old[k].tobytes()==new[k].tobytes()
  assert r[role]['float32']['opposite_label_groups']==r[role]['float64']['opposite_label_groups']==0
 extra+=r['fit']['float32']['extra_duplicate_rows']
assert extra==16
C.save('independent_audit.json',dict(passed=True,all2875fitrows2741active543guardpreserved=True,allNonXfields_byteunchanged=True,all39prefix_exact=True,all16positivepairs_collapse_without_dedup=True,float32_duplicate_extra_rows16=True,no_opposite_label_conflicts=True,originalruntime_math_ASTsame_except_shape=True,newfits=0,newpredict=0))
plan=C.rd(C.Q/'NEXT_PROTOCOL_PLAN.json');plan['id']='GRID277';plan['qualified_source']='Only276qualified_inputs/qualified_freeze/runtime276/model_references;new277sourcepublishedbeforefirstfit. All11activeforestsfresh39D,1KEEP. One600s supervisor includesstartup/loading/fit/calibration/checks;no continuation.';C.save('EXECUTION_PLAN.json',plan);C.save('decision.json',dict(passed=True,ready_for_one_fixed277=True,newfits=0,future11forests11cutoffs=True));C.save('complete.json',dict(passed=True,newfits=0,newpredict=0,rows2875_active2741_guard543=True,duplicates16_no_conflict=True))
(O/'GRID276_report.md').write_text('# GRID276：附加森林39维输入资格\n\n资格通过，本阶段模型/截点拟合、真实概率和新策略均为0。只省略附加NET森林输入的最后selector位，保留38个有界公共坐标和父b，旧40维输入另存来源；原261控制器/入口不改。\n\n273全部2875训练行（活跃2741）、标签、NET、权重、角色、ids、source_mass及543条保护行逐字段保持。仅X/X32改为原前39列，保留所有负例及16条正例副本，不合并权重或去重。16对副本输入重合，恰有16条额外同标签重复，无相反标签精确冲突；它们仍是相关旧数据。训练日期排除各池cal4/outer4。\n\nruntime仅39维shape检查改变，其余固定ExtraTrees配方及概率/保护截点/门控函数AST相同；sklearn1.7.2构造接口通过，未fit或predict。旧40维森林不能直接使用新输入。\n\n下一独立277重新训练11个39维森林、校准11个同543经验正例截点，1KEEP，一次600秒/22操作，原64树/depth4/leaf10/种子等配方固定。全模型截点先发，所有预测先发后连接旧开发结果。主要匹配274同样本/风险暴露，保留271和261质量锚/GATE及全部反例。输入函数类和随机特征选择改变，不称严格同容量或因果消融。\n\n当前分支124，未来277独立22至146；原6续训8285更新/NN604.703实际与用户678.156上限另列。父own12/内部cal4祖先、暴露16日期/相关前缀种子限制保持，经验保护非独立外层保证，不称全链OOF/盲测/整周质量或实际时延。最终8、新物理/电网/论文/FRAME未动，无安装，原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保持。\n',encoding='utf8')
C.save('previous_approval_failure_note.json',dict(previous_attempt_not_executed=True,reason='Automatic approval service usage limit prevented command dispatch.',newfiles_in_previous_attempt=False,newfits_in_previous_attempt=0,recovered_after_later_approved_command=True,protocol_unchanged=True));C.archive('audited')
D=C.R/'论文协作资料/FRAME_39维输入资格_GRID276';D.mkdir(exist_ok=False)
for n in ['GRID276_report.md','qualification.json','independent_audit.json','complete.json','decision.json','EXECUTION_PLAN.json','qualified_pools.json']:shutil.copy2(O/n,D/n)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID276唯一39维输入资格通过，0拟合/概率/策略。273原2875训练行/2741活跃/543guard及标签NET权重角色ids逐字段保持，X仅取前39；16副本对坍缩为同标签重复，无相反标签冲突，不去重或改权重。runtime仅shape39改变，原配方/数值函数AST保持/sklearn1.7.2构造通过。下一独立27711新森林+11同543正例截点/一次600秒/1KEEP，旧40维权重不能接39维；全模型先发、全预测先发后结果，主匹配274同暴露，全部日期seed反例保留。当前124未来146、原祖先开发/整周时延界限保持。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=P/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID276选择器位省略输入资格\n\n'+msg+'\n'
 if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+rest
 else:t+=entry
 p.write_text(t,encoding='utf8')
(P/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID276后\n\n1. '+msg+' 完整资格/核验/归档不重做。\n2. 下一277先核对276qualified_freeze/runtime/modelrefs/EXECUTION_PLAN具体Gitblob与远端，再封存源码输入推送，11森林+11截点一次600秒不延长/扫参。0NNtheta重训，1KEEP。\n3. 原249公共X先tanh前38再切39，全模型截点先发后评分、全12预测先发后旧结果，主匹配274/271/261/GATE，保留全部反例；最终8/新物理电网论文FRAME不动，不安装，实际A归档协作远端核验和Git串行白名单保持。\n',encoding='utf8')
p=P/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID276选择器位省略输入资格\n\n[报告](protocols/'+A.name+'/audited/GRID276_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID276_AUDITED_ZERO_FITS')
