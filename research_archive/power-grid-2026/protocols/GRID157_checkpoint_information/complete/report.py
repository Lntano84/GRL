import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];ms=D.read(O/'depth_ranking.json');op=D.read(O/'opportunity_summary.json');neg=D.read(O/'negative192_summary.json');profiles=D.read(O/'feature_profiles.json')
 constant=[k for k,v in profiles[0]['features'].items() if v['constant']]
 lines=['# GRID157：128检查点到底提供了什么信息','',
 '原18项输入在128处并不等于18种独立证据。153个可决策入口中，有9项恒定；其余多项相互可推导。保留候选数、最佳奖励、最佳负载率、阈值、已观测可接纳比例和前两名奖励差这6项，就能逐项精确重建18项。这个核查没有减少模型输入，也没有更改方法。',
 '', '9项恒定字段：'+', '.join(constant)+'。其中奖励改进、停滞和改进批次比例都还没有后续批次可以统计。因此在128处要求模型停止，主要是在用有限的静态前缀摘要推断尚未看到的后续机会。',
 '', '| 查询深度 | 记录/继续标签 | 日期留一继续AP | AUC | 继续比例 |','|---|---:|---:|---:|---:|']
 for x in ms:lines.append(f"| {x['group']} | {x['rows']}/{x['continue_rows']} | {x['continue_AP']:.3f} | {x['continue_AUC']:.3f} | {x['continue_prevalence']:.3f} |")
 a=op['ALL_ELIGIBLE']['date_macro_query_reduction'];b=op['EARLIEST192']['date_macro_query_reduction']
 lines += ['', '192及以后有了后续反馈，已有日期留一分数的区分表现也更好，但各深度的标签与样本不同，不能据此宣称多查一批必然造成学习性能提升。12折实际9次停止均发生在192。',
 '',f"在完全相同的保原动作且少查询目标下，12拟合日期的事后压缩机会从允许128时的{a:.2%}，降到最早192时的{b:.2%}。等待一批仍保留约{b/a:.1%}的宏平均机会，同时牺牲{(a-b)*100:.2f}个百分点。因而“先多看一批”有可检验的依据，也有明确计算代价，不能只强调它更容易学。",'',
 '192处10个继续标签全部保留，来自6个日期；5个上一批有改进、5个无改进，不能把“上一批是否改善”直接当新的触发规则。9个停止动作的一步奖励低于原FRAME，1个更高；后者依旧按原动作一致性目标判继续，未改标签。一步奖励更高也不能推出完整周更好。该反例分类是分析后的描述性核查，未来信息只在审计文件中，未进入任何在线特征。',
 '', '这些证据支持的下一项是一个小而明确的可证伪对照：固定原模型/参数，只在192检查点训练和判断，同时比较“原RF140也仅在192判断”，区分重训与改变决策位置的效果。保留原18项、原标签、原日期、完整搜索回放和固定预算对照；开发日期已暴露，不称独立验证，不据结果继续扫描其它检查点。训练前告知。',
 '', '本轮153次6字段重建通过，原386留出分数直接复用，无新增模型分数、拟合、标签、物理、数据或最终8。事后机会不是在线算法收益，当前没有模型通过扩大门槛。']
 (O/'GRID157_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('complete.json',dict(passed=True,new_fits=0,new_scores=0,new_labels=0,new_physics=0,adopted=False));D.archive('complete')
 c=R/'论文协作资料/FRAME_检查点信息诊断_GRID157';c.mkdir(exist_ok=False)
 for n in ['GRID157_report.md','depth_ranking.json','opportunity_summary.json','negative192_summary.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID157零拟合信息诊断：128入口18项中9恒定，6基础数可精确重建18；153入口留一AUC .541，192 .674、更后 .798，但标签/人群变化非因果。原12拟合日期保动作事后机会39.68%→最早192的24.41%（留61.5%，非学习收益）。192继续10项分布6日期、停滞0/1各5，不选新触发规则。拟下一项唯一192固定配置拟合+原RF同位置控制，不扫描其它点；训练前通知，不扩大。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID157检查点信息\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID157后\n\n1. 训练前通知后，仅一项192检查点RF对照，原参数/标签/18输入/12-4-4日期；原RF140同样限192作控制，CAP192/256/FRAME完整保留。\n2. 不因结果扫描256等新检查点、调参或择日期；开发集已暴露，不独立确认。保原动作+省查询门槛不变。\n3. 无新物理/数据/最终8/论文/自动扩大；扩大前通知并等用户决定。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID157检查点信息诊断\r\n\r\n[报告](protocols/GRID157_checkpoint_information/complete/GRID157_report.md)；入口摘要缺少批次趋势，先看一批仍有机会但有代价。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=6));d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID157_REPORT_COMPLETE')
if __name__=='__main__':main()
