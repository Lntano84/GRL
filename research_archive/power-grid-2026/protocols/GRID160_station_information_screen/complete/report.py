import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];dec=D.read(O/'decision.json');summ=D.read(O/'neighbor_summary.json')
 lines=['# GRID160：站级信息的第一轮资格核查','',
 '这组8项站级摘要没有通过事先规定的筛查，因此不训练26项变体、不更换子集或距离救结果。它对部分继续记录的邻域有改善，但没有表现出稳定增量。','',
 '| 表示/分区 | 继续记录有5近邻 | 其中无继续邻居 | 停止记录有5近邻 | 其中混入继续标签 |','|---|---:|---:|---:|---:|']
 for x in summ:lines.append(f"| {x['view']}/{x['partition']} | {x['continue_full5']} | {x['continue_empty']} | {x['stop_full5']} | {x['stop_contaminated']} |")
 lines+=['', '校准集继续记录的空白邻域从11/45降到10/45，但停止记录混杂数由55增加到60；七月继续空白数由2增到3。因此没有满足“继续侧改善、停止侧不恶化、每个日期不恶化”的完整门槛。这个邻域规则是筛查约定，不是“站级信息无法学习”的证明。',
 '', '固定8项为当前最好站覆盖率、已查站比例、完全未查站的剩余候选归属质量、最好站之外的剩余归属质量比例、已观测可接纳站比例、各站最好已观测奖励的均值/标准差/前两名差。使用完整公开别名映射，未查询反馈不填值，未来机制标记不进入这些摘要。',
 '', '584个原拟合/校准前缀的4672项标量由独立实现核对，584次隐藏后缀扰动与未来字段拒绝检查通过；原18项几何诊断精确复现。没有提取新开发站级特征，没有拟合、新模型评分或物理采集，最终8仍未生成。',
 '', '## 这一阶段对后续路线的约束','',
 '目前证据不支持继续通过换检查点、微调阈值或追加几个标量来推进。原18项在入口缺少趋势，单192拟合仍误停；站发现确实影响补查位置，但这组全局站级摘要还不能可靠识别相应风险。机制存在与模型能识别机制，是两件不同的事。',
 '', '下一步需要先把表示层的问题说清楚：把“每个站查到什么”保留为结构化输入，与压成全局均值的输入有何可检验差异？同时区分直接全局动作改进和发现补查位置两条路径，最终监督目标仍是原动作一致且省查询。先查相关表示方法和已有数据支持，再设计一次固定比较；不从这四个反复暴露的校准日期继续选择特征，不把阴性结果当作必须扩大规模的理由。',
 '', '当前没有通过采用或扩大门槛的学习方案。需要新拟合时先通知，需要新数据/物理扩大时先提交具体范围，不改变原FRAME或论文结论。所有当前记录、模型与不利结果都保留。']
 (O/'GRID160_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('complete.json',dict(passed_audit=True,information_gate_passed=dec['passed_prespecified_information_screen'],new_fits=0,new_scores=0,new_dev_features=0,adopted=False,ready_to_scale=False));D.archive('complete')
 c=R/'论文协作资料/FRAME_站级信息资格_GRID160';c.mkdir(exist_ok=False)
 for n in ['GRID160_report.md','decision.json','neighbor_summary.json','per_date.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID160原386fit/198cal固定8站级摘要资格未过：继续空白11→10有改善，但停止混杂55→60，七月空白2→3。不给26项拟合资格、不换子集/度量救结果，不证明站级信息不可学。584公开前缀/4672独立标量/后缀拒未来核验通过，未提新开发站特征/拟合/评分/物理/最终8。下一步先研究结构化站反馈表示与全局标量压缩的差异，避免继续用重复暴露四校准日期选字段。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID160站级信息资格未过\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID160后\n\n1. 不再换检查点/阈值/标量组合。先读原始结构化集合/关系表示论文与已有GRID113等失败记录，固定“站级完整观测结构vs全局标量”的科学比较，避免仅因新模型更复杂而训练。\n2. 原目标仍保动作且省查询；未来机制审计不进输入，不更改标签救结果。当前四校准日期重复暴露，不拿来继续选字段或声称独立确认。\n3. 新拟合前通知，任何数据/物理扩大先提交具体方案等用户决定；最终8未生成，不动论文/原控制器，不恢复旧自动化。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID160站级信息资格\r\n\r\n[报告](protocols/GRID160_station_information_screen/complete/GRID160_report.md)；固定8摘要未过预定筛查，未启动26项训练。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=6));d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID160_REPORT_COMPLETE')
if __name__=='__main__':main()
