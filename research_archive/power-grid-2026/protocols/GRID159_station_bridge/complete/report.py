import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];s=D.read(O/'summary.json');per=D.read(O/'per_date.json')
 lines=['# GRID159：继续全局搜索如何改变补查价值','',
 '在原12个拟合日期里，这种作用并非只有十月一个案例，但也不能解释所有误停风险。55个继续标签中，11个（20%）符合“后续全局搜索改变最好动作所属站，再通过该站补查得到最终动作”的描述，涉及8次独立搜索、5个日期。全部55个继续前缀来自37次搜索、11个日期；同一调用的多个前缀没有冒充独立样本。','',
 '| 日期 | 继续前缀/搜索 | 发现新站后补查的前缀/搜索 |','|---|---:|---:|']
 for x in per:lines.append(f"| {x['date']} | {x['continue_prefixes']}/{x['continue_calls']} | {x['bridge_prefixes']}/{x['bridge_calls']} |")
 lines+=['', '这11项的后续全局奖励改进中位数0.00627、最小0.000105；随后局部奖励改进中位数0.000186。数值小并不代表完整周价值小或大；它们能确认的是：全局阶段有时承担发现补查位置的作用。不能只用当前奖励是否足够好，代替判断再搜索是否有价值。',
 '', '其余44个继续前缀也完整保留。当前结果不足以说站级机制是学习失败的唯一或主要原因，更不足以说加入站级特征就能解决问题。这个分类是对原反馈路径的解释，没有改变任何训练标签、动作或阈值。',
 '', '## 停止前能看到哪些站级信息','',
 '已为全部386个原拟合前缀整理公开站级记录：每站对应多少候选代表、已经查了多少、还剩多少、本站当前最好已观测动作的奖励和负载率。候选归属严格使用公开别名映射，含跨站别名的代表可能计入多个站，计数不宣称互斥。完全未查询的站只记录剩余候选，不填任何预测反馈。',
 '', '这些记录只接收候选顺序、公开别名和已揭示前缀。386次隐藏后缀扰动均不改变结果；未来最终动作、奖励增益和“发现新站后补查”标记单独保存在审计文件，未进入站级输入。尚未训练或评估这个表示，也没有提取校准/开发站级特征。',
 '', '下一项有价值的问题已收窄：当前18项把反馈压成一个全局最好值，是否丢掉了不同变电站之间的覆盖和质量差异？先检查新记录是否提供跨日期稳定的区分信息，不能因为机制解释成立就直接扩网络。原停止标签、固定预算对照、完整搜索回放和质量要求继续保留。',
 '', '本轮零拟合、零模型评分、零新标签、零物理采集；最终8场景未生成。完成前核对冻结源、公开输入和归档，所有反例保留。']
 (O/'GRID159_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('complete.json',dict(passed=True,new_fits=0,new_scores=0,new_labels=0,new_physics=0,new_cal_dev_profiles=0,ready_to_scale=False));D.archive('complete')
 c=R/'论文协作资料/FRAME_站发现与补查_GRID159';c.mkdir(exist_ok=False)
 for n in ['GRID159_report.md','summary.json','per_date.json','design.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID159原拟合数据机制核查：55继续前缀/37搜索/11日期中，11前缀/8搜索/5日期属于后续全局发现站再局部选最终动作（20%，非全部原因）。386公开站级覆盖/最佳已观测反馈档案与后缀扰动通过，未来机制标记分离；未提取新校准/开发站特征，零拟合/新评分/新标签/物理/最终8。不直接扩网络，下一步查站级记录是否有稳定增量信息。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID159站发现与补查机制\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID159后\n\n1. 优先回答：站级覆盖与已观测反馈的分布，是否给原18项提供跨日期稳定增量信息。先冻结检查方法，再提取原校准记录；不要碰新开发表示/最终8来选特征。\n2. 公开站级输入与未来机制审计严格分离；未查询站奖励保持缺失。20%站发现路径不是全部原因，不将该机制标记直接变训练标签。\n3. 不再扫描检查点/组合权重/阈值，不扩网络。新拟合前通知；若需扩大数据或物理评估先通知用户决定。原模型不采用，不改论文。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID159站发现与补查\r\n\r\n[报告](protocols/GRID159_station_bridge/complete/GRID159_report.md)；20%原拟合继续前缀涉及站发现→局部补查，公开站级信息待资格核查。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=6));d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID159_REPORT_COMPLETE')
if __name__=='__main__':main()
