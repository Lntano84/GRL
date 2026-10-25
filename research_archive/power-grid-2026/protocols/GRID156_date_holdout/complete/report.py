import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];m=D.read(O/'ranking_metrics.json');s=D.read(O/'search_summary.json')
 lines=['# GRID156：原随机森林是否学到跨日期关系','',
 '只用原12个拟合日期做逐日期留一诊断，每次完整留出该日的两个场景及全部搜索。12次拟合均用原RF140参数/种子，标签和18项输入不变；填补中位数及样本权重只用另外11个日期计算。不是挑选12个模型，不是新开发集或最终测试。','',
 '| 读数 | 继续类AP | 继续类AUC | 继续类比例 |','|---|---:|---:|---:|']
 for key,label in [('pooled_in_sample','原模型拟合内'),('pooled_heldout','留一拼接（描述性）'),('pooled128','留一128检查点')]:
  a=m[key];lines.append(f"| {label} | {a['continue_AP']:.3f} | {a['continue_AUC']:.3f} | {a['continue_prevalence']:.3f} |")
 a=m['within_date_macro'];lines+=['',f"11个同时含正负例的留出日期，日期内AUC宏平均{a['mean_within_date_continue_AUC']:.3f}，AP宏平均{a['mean_within_date_continue_AP']:.3f}。不同折的分数标度可能不同，拼接指标不能代替日期内指标；不同查询深度也会影响任务难度。不能把这些结果概括为完全没有信号。",'',
 '| 策略/读法 | 动作差异 | 提前停止次数 | 查询总数 | 日期查询减少宏平均 |','|---|---:|---:|---:|---:|']
 for k in ['FRAME','CAP256','RF140_IN_SAMPLE_REFERENCE','RF156_DATE_HELDOUT']:
  x=s[k];lines.append(f"| {k} | {x['action_differences']}/{x['calls']} | {x['model_stops']} | {x['queries']} | {x['date_macro_query_reduction']:.2%} |")
 lines+=['', '原阈值0.996307保持不变，留一模型在161次搜索中动作全部一致，但只停9次、每次均在192候选处，各节省64，总计576，日期宏平均1.12%。CAP256节省更多，但在这批拟合日期中改变了4次原动作，不能拿它的速度直接当同质量优势。原模型拟合内读数仅作落差参考。',
 '', '这次诊断更具体地说明：模型能在部分状态排序风险，但保持原动作的高阈值只留下少数退出。128入口的跨日期AUC为0.541，较难区分；把所有深度混在一起看0.715会遮住这个问题。接下来核查入口处哪些输入实际上恒定，以及等待一批反馈后还剩多少同目标压缩机会。',
 '', '12次总拟合耗时0.654秒，386个留出分数由独立导出树复算，644个完整保存搜索回放通过。没有重新校准、开发日期评分、新物理、最终8或模型采用；没有把12折模型挑一个用于部署。']
 (O/'GRID156_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('complete.json',dict(passed_audit=True,fits=12,adopted=False,ready_to_scale=False));D.archive('complete')
 c=R/'论文协作资料/FRAME_按日期留一诊断_GRID156';c.mkdir(exist_ok=False)
 for n in ['GRID156_report.md','ranking_metrics.json','search_summary.json','per_date_metrics.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID156固定原RF12拟合日期整日留一：原配置/种子/标签，折内中位数权重，12拟合总0.654秒。拼接继续AP .239/AUC .715、128 AUC .541；日期内11日期AUC宏.761，不能说全无信号。原阈值161/161原动作、9停均192、查询宏少1.12%；CAP256有4动作差。无重校准/新开发评分/物理/最终8/择折采用。下一步零拟合核查入口信息及等待一批后的同目标机会。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID156整日留一诊断\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID156后\n\n1. 零拟合核查原12拟合日期128/192/更后检查点的输入有效信息及分层排序；不把按深度标签变化当因果证明。\n2. 用原动作相同且省查询标签，量化等待一批后仍可保留多少事后机会；只分析，不选择新退出规则或重训。\n3. 不堆网络/再调阈值；新训练先通知，扩大等用户决定。无新物理/最终8/论文。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID156整日留一诊断\r\n\r\n[报告](protocols/GRID156_date_holdout/complete/GRID156_report.md)；入口128区分弱，保动作阈值只剩少量192退出。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=6));d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID156_REPORT_COMPLETE')
if __name__=='__main__':main()
