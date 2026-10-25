import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];assert D.read(O/'tree_path_receipt.json')['passed'];event=D.read(O/'factor_event_receipt.json');actions=D.read(O/'factor_equal_gain_action_check.json');assert event['passed'] and actions['passed'];f=D.read(O/'factor_vs_base_summary.json');s=D.read(O/'current_change_summary.json');ordering=D.read(O/'ordering_explanation.json');categories=D.read(O/'factor_event_summary.json')
 lines=['# GRID162：从已经出现的改善入手','',
 '本轮按照用户的新判断调整研究顺序：先找到具体改善、解释怎样发生，再围绕它设计小步改进；不要求一条线索先在所有日期和种子稳定获胜才值得研究。原采用判据和历史结果不改，但不再把“整体未过筛”直接当作结束该路线的理由。','',
 '## 更值得优先追的旧线索：FACTOR确实少做了一部分查询','',
 '旧补查任务中，FACTOR相对BASE三种子平均净效用提高2.73%，其中约70.2%的平均净增量来自查询费用减少，其余约29.8%来自收益增量。不是先前猜测的“动作—反馈配对优势”，但新增模块本身的正向变化值得单独解释。两者参数量和部分输入不同，尚不能把增益归因于某一个因素。','',
 '逐条配对全部590个补查起点、三个种子，共1770对记录：260对查询更少，补查收益与原模型完全相同，最终候选pool_id也完全相同；其中231对双方的补查增益均为0。这些记录涉及218个起点、43个物理状态组，覆盖全部四个开发日期。重复起点和种子不当作独立新场景。这里参考的是旧BASE，不是原FRAME，不能直接说保持FRAME的整周质量。',
 '', '| 变化类别 | 种子×起点对数 | 对平均净效用变化的贡献 |','|---|---:|---:|']
 labels={'LESS_QUERY_NO_GAIN_LOSS':'少查且不损失补查收益','LESS_QUERY_SMALL_GAIN_SACRIFICE_NET_POSITIVE':'少查、牺牲少量收益后仍净赚','MORE_QUERY_GAIN_PAYS':'多查且收益覆盖费用','LESS_QUERY_GAIN_LOSS_TOO_LARGE':'少查但漏掉收益过多','MORE_QUERY_COST_NOT_REPAID':'多查但收益不足抵费用','UNCHANGED':'不变'}
 for x in categories:lines.append(f"| {labels.get(x['category'],x['category'])} | {x['seed_prefix_pairs']} | {x['net_macro_contribution']:+.6f} |")
 lines += ['', '最大正贡献来自“少查但收益不变”，最大负贡献来自“多查却没赚回费用”。这给出了可以放大的明确对象：扩大前者覆盖，同时找出后者为什么还在继续。不能只把模型整体换得更保守，因为另有7对多查带来了足以覆盖费用的收益。',
 '', '按日期看，三月主要靠少查，虽牺牲少量收益仍净赚；八月收益和查询费用两边同时改善；五月增加查询没有赚回费用。先研究三月和八月出现的具体变化，并把五月作为反向对照，后续仍评估全部日期。不是把有利日期选成最终测试集。',
 '', '## 当前192模型：正向变化也不是空的','',
 '与原RF140相比，RF158在三个开发日期新增4次正确早停，共少查256个候选；丢掉2次原本正确的早停，多查128个；另一次少查216个却改变了动作。这三部分必须分开，不能把344个总查询差全部叫有效收益。',
 '', '进一步把两份已有分数与两条已有阈值交叉读取，没有选择新阈值：旧模型换成新较低阈值，也会在新增4个正确案例停，但同时会在校准集增加4次误停。新模型在同一较低阈值下压住了这4次校准误停。因而有价值的线索不只是“允许多停”，还包括压低了部分不该停的状态分数，为更多正确早停腾出空间。当前仍有开发误停，不能宣称这种区分已经普遍成立。',
 '', '把这4个新增正确案例与4个被压住的校准反例交叉看，旧分数在16个组合中有6个排序正确，新分数16个都正确。这是选出案例后的解释，不是16个独立样本或新的泛化指标；不能用它替代全部62次搜索。',
 '', '已把11个改变案例的旧、新随机森林分数逐树精确拆成根概率加路径贡献。新模型的根概率统一提高约0.03349，所以“整体更倾向停止”本身不能解释为什么有些危险状态反而被压低。可接纳比例、前两名奖励差、负载率与阈值等路径贡献有差异，下一步应核对具体叶子支持和分支，而不是立即增加输入。相关字段彼此相关，树路径记账不是电网因果证明。',
 '', '## 后续先做什么','',
 '优先沿两条已有正向变化推进：第一，解释FACTOR那260对少查且同动作的停止，尤其是为何原BASE还继续；同时检查301对多查不划算的反向记录，找共同的公开输入条件。第二，解释RF158压低校准误停与保留新增正确停止的区别，保留两次退步和一个新误停作为对照。全局跨站新网络先降为备选，不再因为旧方案整体未过筛就马上换结构。',
 '', '如果原因是更清楚的费用表达、聚合方式或容量，就分别做最小对照，不执着证明注意力或配对一定有用。新训练另定固定方案并提前通知。探索时允许局部改善；最后确认效果时再核对覆盖范围、代价、实际控制质量与泛化。当前两条线索属于不同任务，数值不混作一个总成绩，原标签和质量要求也不改。',
 '', '本轮只读复核旧分数、搜索序列和费用，新增训练0、模型预测0、物理采集0、最终8读取0。两条已有阈值交叉只作解释，未选为部署策略。全部正向和反向记录、配对分解与复算收据均保留。']
 (O/'GRID162_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('decision.json',dict(user_priority='Investigate existing positive changes and amplify their mechanism before demanding stable broad gains.',exploration_is_not_adoption=True,priority1='FACTOR fewer-query same-action events versus unnecessary continuation',priority2='RF158 selective suppression of wrong exits and new correct exits',defer_new_global_structure=True,old_gates_not_retroactively_changed=True,no_model_adopted=True,new_fits=0,ready_to_scale=False));D.save('complete.json',dict(passed_accounting=True,new_fits=0,new_predictions=0,new_physics=0,not_independent_confirmation=True));D.archive('complete')
 c=R/'论文协作资料/FRAME_正向改善追踪_GRID162';c.mkdir(exist_ok=False)
 for n in ['GRID162_report.md','current_change_summary.json','factor_vs_base_summary.json','factor_event_summary.json','factor_equal_gain_action_check.json','ordering_explanation.json','decision.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='用户调整探索优先级：从已出现的改善及原因入手，不要求先稳定获胜才研究；采用标准和旧结果不改。GRID162原RF158新增4正确省256、丢2多128、错1少216拆开。旧FACTOR比BASE净效用+2.73%，约70.2%来自查询费用减少；260种子起点对更少查询同收益同动作（218起点/43物理组/四日期），另301对多查不划算。优先追已有正向变化，反例共同分析，暂缓新全局结构。无训练/预测/物理/最终8；事后解释非新确认。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID162正向变化优先\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 用户优先正向改善，GRID162后\n\n1. 主线先追已有改善，不再把整体未过筛当作立即终止探索的门槛。旧采用判据/不利结果保留；局部线索不宣称泛化或采用。\n2. 优先解释旧FACTOR260对少查同收益同动作（218起点/43物理组），对照301对多查不划算。核公开输入、已知费用表达、模型继续优势及首次决策分歧，全部种子/日期保留。旧局部净效用与当前保FRAME动作目标分开，不混成绩。\n3. 次线追RF158新增4正确/压低4校准错停，与丢2旧正确/新增1错停对照；先看既有树叶支持/排序变化，不立即添特征或调阈值。\n4. 跨站新结构打包降为备选。找到可解释正向变化后做小而固定的因素对照；新拟合前通知。扩大数据/物理前给具体方案等用户决定，最终8不动，不改论文。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID162正向变化优先\r\n\r\n[报告](protocols/GRID162_positive_change_audit/complete/GRID162_report.md)；优先解释并放大已有改善，正反变化同时保留。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=9));q=D.A/'verification';q.mkdir();shutil.copy2(O/'publication_verification.json',q/'publication_verification.json');(q/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(q/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID162_REPORT_COMPLETE')
if __name__=='__main__':main()
