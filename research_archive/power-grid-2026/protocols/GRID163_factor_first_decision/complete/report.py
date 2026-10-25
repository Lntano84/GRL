import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];trace=D.read(O/'first_difference_summary.json');effects=D.read(O/'component_effect_summary.json');results=D.read(O/'main_only_summary.json');val=[r for r in results if r['partition']=='validation'];fam={m:sum(r['mean_net_utility'] for r in val if r['mode']==m)/3 for m in ['ORIGINAL_BASE','FACTOR_TOTAL','FACTOR_MAIN_ONLY']};D.save('family_summary.json',dict(validation=fam,all_seeds_retained=True))
 old117=R/'outputs/grid117_module_diagnosis/GRID117_report.md';old126=R/'outputs/grid126_known_cost_gain/GRID126_report.md';D.save('prior_work_overlap.json',dict(old117=dict(source=str(old117.relative_to(R)),sha256=D.sha(old117),already_studied='Frozen main/correction and attention diagnostics forJOINT versusFACTOR,including additive nonidentifiability.',new_here='AlignFACTOR versus originalBASE to all260 positive same-action savings and301 costly continuations.'),old126=dict(source=str(old126.relative_to(R)),sha256=D.sha(old126),already_studied='Learn gross gain then deduct known cost,nonnegative links.',not_to_repeat='Do not rename analytical-cost deduction as a new experiment. A future explicit-cost INPUT versus dummy under identical NET target is different.')))
 lines=['# GRID163：FACTOR的小优势具体来自哪里','',
 '本轮把已有改善落实到第一次决定和实际输出分支，没有训练。最值得保留的线索是：FACTOR已经在一批状态里识别出了可以不再查询的情况，而不是只有一个好看的平均数。','',
 '## 260条正确节省的第一步','',
 '260条少查且保留相同收益/动作的记录，全部在下一批收益为0时选择了停止。其中215条发生在补查入口，45条发生在查询过一部分批次后；覆盖218个起点、43个物理状态组、四个开发日期。模型在其中有重复出现的正确判断，值得继续研究。',
 '', '反向的301条多查却没赚回费用的记录，在第一次分歧时也都面对下一批非正净收益，其中285条下一批奖励增益为0。正确停止的下一批候选数中位数14.5，额外无利继续的中位数5。这提示“小批次很便宜，所以容易乐观继续”是可检验的解释，不是已经确定的原因。',
 '', '不能把全部继续压掉：另7条新增继续面对的下一批净收益全部为正，最终也赚回了费用；6条过早停止的下一批则全部有正净收益。需要扩大正确停止覆盖，同时保留对真实机会的识别。',
 '', '## 已保存主干和修正分支的分工','',
 '| 变化类别 | 记录数 | 修正分支改变主干原决定 |','|---|---:|---:|']
 labels={'LESS_QUERY_NO_GAIN_LOSS':'少查且收益不变','MORE_QUERY_COST_NOT_REPAID':'多查未赚回费用','LESS_QUERY_SMALL_GAIN_SACRIFICE_NET_POSITIVE':'少查、轻微损失后仍净赚','MORE_QUERY_GAIN_PAYS':'多查且赚回费用','LESS_QUERY_GAIN_LOSS_TOO_LARGE':'漏掉收益过多'}
 for x in effects:lines.append(f"| {labels[x['category']]} | {x['events']} | {x['correction_flips_main']} |")
 lines += ['', '在260条正确节省里，141条是按这份保存权重的分工，修正项把主干的“继续”翻成“停止”；另外119条主干本身就已与FACTOR最终决定相同。额外无利继续中106条由修正项翻为继续，195条主干已继续。因此修正分支既有值得放大的正向作用，也有明确的改进空间。',
 '', '| 固定权重读法 | 验证三种子平均净效用 |','|---|---:|']
 for m,v in fam.items():lines.append(f'| {m} | {v:.8f} |')
 lines += ['', '仅保留共训练主干的三个种子效用为0.168777、0.158491、0.137568；完整FACTOR为0.167280、0.161758、0.173091。修正项在两个种子帮助、一种子反向，不要求全部同向才把它当作研究线索。主干单独并不是另一份独立训练BASE，不能拿这个删分支结果证明训练时的唯一因果来源。',
 '', 'GRID117已经做过主干/修正的诊断并指出两者存在加性分工不唯一；本轮不将这件事包装为新发现。新增的是把FACTOR相对原BASE的260条正向节省、301条无利继续，与第一次决定和分支输出逐条对应。',
 '', '## 下一项小对照已经明确','',
 '在全部4438份旧公开输入中，新增分支接收的精确下一批费用，都能从原41项中的log_next_batch_queries还原。因而直接给网络费用标量没有增加信息，但可能让网络更容易学习查询成本。先用小对照分清“费用表达更直接”能解释多少改善，比立刻增加关系网络更具体。',
 '', '注意GRID126已经试过回归总改进再扣除已知费用，不能重复它。拟做的区别是：保持原净收益监督目标、损失、抽样和网络规模，比较额外一个精确费用输入与同形状零输入，不改费用或标签。原FACTOR和BASE保留作参照；新训练前明确通知。',
 '', '全部旧模型完整输出按原批次32重执行，13314个状态分数与旧值完全一致；主干另用NumPy复算，最大误差约8.96e-7。没有拟合、参数更新、环境运行或最终测试。输入只取已揭示反馈；下一批收益仅在事后审计中读取。本轮仍是同网、旧16/4日期的局部净效用任务，不把它直接当全局FRAME动作一致性或整周质量结论。']
 (O/'GRID163_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('decision.json',dict(positive_signal_to_explore=True,does_not_require_all_seeds_win=True,next_hypothesis='Explicit known-cost input versus equivalent log count,under unchanged net target.',do_not_repeat_grid126=True,adopted=False,new_fits=0,ready_to_scale=False));D.save('complete.json',dict(passed_audit=True,new_fits=0,new_physics=0,positive_events_traced=260));D.archive('complete')
 c=R/'论文协作资料/FRAME_FACTOR改善来源_GRID163';c.mkdir(exist_ok=False)
 for n in ['GRID163_report.md','first_difference_summary.json','component_effect_summary.json','family_summary.json','cost_expression_audit.json','prior_work_overlap.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID163追FACTOR已有改善：260正确少查下一批增益全0，215入口/45后续；以保存权重分工141由修正翻继续为停。301无利继续下一批净收益全非正，285增益0，106修正翻为继续。正确停止nextq中位14.5、无利继续5，值得检验费用表达而不盲加网络。完整FACTOR均值.167376，共训练主干单独.154945，2/3种子分支帮助；尊重117加性不可识别，不当训练因果证明。已核4438费用可从旧logn精确恢复；拟只改费用输入表达的配对小对照，不重做126费用外扣。无拟合/物理/最终8。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID163_FACTOR已有改善来源\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID163后，优先放大已有改善\n\n1. 训练前通知后，做一个小而固定的“显式下一批费用输入 vs 零输入”配对对照。仍原局部净收益标签、旧16/4拆分、费用、7250更新、原抽样和三个种子；不改目标为gross gain、不重复GRID126。相同42维网络/初始化区分输入表达，不择种子。\n2. 既看原590验证起点总体，也看260正向停止和301无利继续的变化；它们是已暴露事后解释组，不按它们选阈值、丢数据或宣称独立确认。小改善有重复性就继续，不用全日期全种子同向作探索门槛。\n3. 无新物理/数据扩大/最终8，旧局部效用与当前全局保FRAME动作目标分开，不动论文。新数据/物理扩大前先给用户范围。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID163_FACTOR改善来源\r\n\r\n[报告](protocols/GRID163_factor_first_decision/complete/GRID163_report.md)；260正确停止逐条核对，转向费用表达的小对照。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=8));q=D.A/'verification';q.mkdir();shutil.copy2(O/'publication_verification.json',q/'publication_verification.json');(q/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(q/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID163_REPORT_COMPLETE')
if __name__=='__main__':main()
