from pathlib import Path
import json,hashlib,shutil
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid255_reference_stop_nn';W=R/'work/grid255_reference_stop_nn';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID255_reference_stop_nn'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert not (O/'delivery_check.json').exists();a=read(O/'audit.json');ok=a['component_complete'];n=a['complete_models'];wall=a['manager_wall_s'];assert a['receipt_and_checkpoint_checks_passed'] and a['all_workers_exited']
save(O/'complete.json',dict(passed=ok,receipt_audit_passed=True,reference_models_complete=n,expected=18,new_reference_forward=0,new_teacher_label_join=0,new_policy_scores=0,all_workers_exited=True,window_s=600,manager_wall_s=wall))
summary=f'GRID255固定18参考组件完整{n}/18，管理器{wall:.3f}s；完整项7250更新/232000draw，excluded0、源/采样/权重/Adam步数/截止核验通过，0参考评分/新标签连接/策略评估。'
if ok:
 next_step='GRID256仅公开参考评分和原父输入构造：18权重及trained_freeze先封存推送核对远端/具体Git blob，才能32238行公开参考score。复用原teacher路径公开向量；原父缓存22653字节复用，<=9585缺失行必须原ownerf NN隐藏，24收益头0拟合，不能新参考NN8隐藏接旧208头。公开选择/新输入全部先发，不能提前接新STOP对应teacher NET标签。256不训练读出、不评分新outer策略。'
else:
 next_step='GRID255固定600秒组件未完整，保留全部完整模型/部分检查点和暴露。不能通过全18发布屏障，禁止参考评分/新标签连接/读出训练。仅允许下一阶段只读资源/调度/时间分解，不自动续窗/重跑完整项/减更新/换种子。若需新计算窗口或协议更改，列具体决定请求用户，不把工程截止当方法失败。'
handoff=f'''# GRID255后持续交接

先读work/GRL-results-integration/docs/RESEARCH_STATE.md、NEXT_STEPS.md，以最新为准；当前GRID255训练/只读收据权重审计完成，不重跑。{summary} 结果{'通过' if ok else '未完整'}。complete.json的passed表示18模型组件是否完成，audit.receipt_and_checkpoint_checks_passed表示现有收据核验是否通过，两者分开。

{next_step}

当前原253唯一源探针完整计数18NN数据基础+后续至多11线性/11原221正例保护步长，两个真实数据依赖组件各<=22/一次600秒，总<=40拟合，不是超参搜索。18参考仅离线停止点来源，不替换部署NN。后续固定主匹配250，canonical38+原父b+nativeSTOP位=40输入41参，不加252qgap；fit8/cal4固定，f0_s1KEEP，其他空域/无fitSTOP正例KEEP，不换日期或补样本。新参考会改变样本、按原各域半半mass*absNET重算风险数值和own12全部selectedSTOP正例保护集合，不再85固定。公开新选择输入先发布后接原teacher标签；全部模型标量先发后外层score；全部预测先发后连结果。

参考NN训练同时排除记录日期组和当前outer组，但父原NN/208头仍own12含自身标签，内部cal4仍可能进入其他参考祖先，不称全链OOF/盲测。所有开发历史已暴露、相关前缀不独立，训练loss不等补查效果，代理效用不等整周质量/真实控制时延。保留230主候选、250低query两种子/252恢复和全部反例，不拼策略择种子，不按已看outer调阈值/alpha/费用/正则，不设全赢5%新门槛。

原费用标签/全原日期种子/部署控制器/排序别名/锁站/128/64/停滞2/.9不变。最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。已有运行器跟进不启动第二个，完整拟合评分审计归档不重复；工程失败先保留冻结源与部分输出，不改协议/窗口且不重复完整项才修复。Git严格等helper退出后下一helper/屏障。按AGENTS更新docs状态决定日志NEXT首页，归档读实际源码A并核对报告/协作/清单/远端SHA。每实质阶段简短中文汇报真实报告，无变化安静；同步本交接和grl起点。用户停则停新增并暂停grl，不恢复旧自动化不归档聊天。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
C=R/'论文协作资料/FRAME_参考停止点数据基础_GRID255';C.mkdir(exist_ok=False)
for name in ['GRID255_report.md','audit.json','decision.json','complete.json','CONTINUOUS_HANDOFF.md']:shutil.copy2(O/name,C/name);assert sha(O/name)==sha(C/name)
save(O/'delivery_check.json',dict(passed=True,collaboration=str(C),files={p.name:sha(p) for p in C.iterdir()},archive=str(A),no_reference_forward=True))
D=REPO/'docs';p=D/'RESEARCH_STATE.md';s=p.read_text(encoding='utf8');title,rest=s.split('\n',1);p.write_text(title+f'\n\n## 2026-10-28 — GRID255参考停止点数据基础\n\n{summary} '+('全部18权重发布后下一256仅公开参考/原父输入构造，不提前连接teacher标签，不作学习效果结论。' if ok else '600秒组件未完整，保留部分不续窗，不能进入新评分标签屏障；先只读定位资源与时间，需要改窗口另列决定。')+'\n'+rest,encoding='utf8')
(D/'NEXT_STEPS.md').write_text(f'''# Next steps — GRID255后

1. {summary} 训练与审计/归档不重做。
2. {next_step}
3. 完整单源探针18NN+后续<=11linear+11alpha，两真实依赖组件各600秒不延长；同250类40输入41参、不加252gap。公共选择输入先发后连teacher，全部模型/预测按屏障发布，重新计算风险及保护集合而非85固定。
4. 原父网络/头祖先仍own12，内部cal4非全链独立；开发代理不等完整周/时延。保留全部种子反例，不按外层择优/调参。最终8/物理/电网/论文/FRAME不动，不安装。Git helper严格串行，核对归档协作远端。
''',encoding='utf8')
for name,text in [('DECISIONS.md',summary+' '+next_step),('EXPERIMENT_LOG.md',summary+' 源eb6fdb65/授权877fa65c严格先发后训；原数值源不改，未执行新参考forward。')]:
 p=D/name;p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-10-28 — '+text+'\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID255参考停止点数据基础\n\n[报告](protocols/GRID255_reference_stop_nn/trained/GRID255_report.md)；'+summary+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
AD=A/'delivery';AD.mkdir(exist_ok=False)
for name in ['CONTINUOUS_HANDOFF.md','complete.json','delivery_check.json']:shutil.copy2(O/name,AD/name)
shutil.copy2(Path(__file__),AD/'deliver.py');save(AD/'archive_manifest.json',{p.relative_to(AD).as_posix():sha(p) for p in AD.rglob('*') if p.is_file()})
print('GRID255_DELIVERED',ok,n,wall)
