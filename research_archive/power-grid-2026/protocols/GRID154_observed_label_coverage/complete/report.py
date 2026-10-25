import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];s=D.read(O/'summary.json');dates=D.read(O/'per_date.json');rows=D.read(O/'observed_targets.json')
 unmatched=[r for r in rows if not r['teacher_context_matched']];D.save('nonteacher_prefix_summary.json',dict(prefixes=len(unmatched),stop=sum(x['stop_label'] is True for x in unmatched),continue_=sum(x['stop_label'] is False for x in unmatched),unknown=sum(x['stop_label'] is None for x in unmatched)))
 lines=['# GRID154：已有FRAME反馈能识别多少停止标签','',
 '原学习目标不变：在原全局搜索检查点提前停止，随后执行原同站补查，最终动作必须与原FRAME相同，且总候选查询更少。只核查已有30次实际FRAME搜索，不新采集或训练。',
 '',f"78个可决策前缀中，62个可停止、13个必须继续、3个未知；75/78的标签可由当次已经查询的反馈确定。未知不是负例，也没有从其它状态借预测结果补齐。",'',
 '| 日期 | 搜索次数 | 可停/继续/未知前缀 | 原查询 | 已观测事后最少查询 |','|---|---:|---:|---:|---:|']
 for x in dates:lines.append(f"| {x['date']} | {x['calls']} | {x['stop']}/{x['continue_']}/{x['unknown']} | {x['original_queries']} | {x['observed_hindsight_queries']} |")
 lines+=['',f"每次搜索只在已能证明动作一致且节省查询的前缀中，事后选择总查询最少者；没有这样的前缀就保留原搜索。四日期查询减少宏平均为{s['date_macro_observed_hindsight_reduction']:.2%}。这是未来已揭示反馈支持的可实现机会下界，不是在线策略、学习结果或实际控制时间提速。也没有证明未知前缀不能节省。",'',
 '识别规则：所需补查范围由当时已知的最好动作、原顺序和公开别名确定。总查询已不节省时可直接判继续；查询更少但缺任一所需候选结果时标未知；结果齐全才复算最终动作。原FRAME选择的正确性由官方规则与独立实现逐次复算。',
 '', '三个未知前缀恰好都能在同状态教师记录中查到负标签。教师结果只用于独立审计，没有填入观测标签表。这说明丢掉缺失记录可能优先丢掉重要反例，不能拿75个可识别前缀直接训练后宣称数据更干净。三月可识别前缀没有负例，但仍有一个未知，不能声称这个日期总能停止。',
 '', '30次官方/独立原搜索重执行、11450项合法反馈、74项独立选择、46次主动遮蔽补查反馈后保持未知，以及67项精确教师标签交叉核对均通过。无物理预测、环境步进、拟合或模型评分，最终8场景不变。',
 '', '下一步对固定RF140、LR150、等权组合151和固定256，在这些保存状态上按原阈值作一次回放。缺失反事实结果继续保留未知，原标签和质量门槛不变。即使保存状态表现改善，也不能直接宣称闭环整周质量或控制时间改善。']
 (O/'GRID154_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('complete.json',dict(passed=True,known_labels=75,unknown_labels=3,new_fits=0,new_physics=0,new_cohort=0));D.archive('complete')
 c=R/'论文协作资料/FRAME_现有反馈标签_GRID154';c.mkdir(exist_ok=False)
 for n in ['GRID154_report.md','summary.json','per_date.json','nonteacher_prefix_summary.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID154已保存实际FRAME反馈标签识别：30搜索78前缀，62可停/13继续/3未知；75标签可识别。已观测事后保动作少查询机会宏平均34.10%，不是在线策略收益。3未知在精确教师审计中均负，不填标签，不能丢未知后训练。官方独立复算30、合法反馈11450、遮蔽46、交叉核对67通过。下一步只读冻结RF140/LR150/等权151/CAP256保存状态回放，禁止拟合/阈值救结果/最终8/扩大。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID154现有反馈标签\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID154后\n\n1. 固定旧RF140/LR150/MEAN151/CAP256，对全部30保存实际FRAME状态回放；原阈值，缺失补查反馈保留未知，不隐性回退原搜索后称学会。\n2. 仅保存状态比较，不能代替完整周质量或实测控制时间；不丢3未知，不拟合选择性观测标签。\n3. 无新仿真/采集/最终8/论文/自动扩大；训练前通知，扩大前等用户决定。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID154现有反馈标签\r\n\r\n[报告](protocols/GRID154_observed_label_coverage/complete/GRID154_report.md)；75/78标签可由已查询反馈识别，3未知保留。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=6));d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID154_REPORT_COMPLETE')
if __name__=='__main__':main()
