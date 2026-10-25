"""Document fixed mean results without selecting an alternative combination."""
import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'state.json')['stage']=='audited'
 review=D.read(O/'review.json');assert review['passed'];decision=D.read(O/'decision.json');sums=D.read(O/'summary.json')
 get=lambda part,n:next(x for x in sums if x['evaluation_partition']==part and x['method']==n)
 p=get('validation','MEAN151_CALIBRATED');cut=D.read(O/'cutoff.json')['primary_threshold'];metrics=D.read(O/'ranking_metrics.json')
 lines=['# GRID151：两份冻结评分的固定等权组合','',
  '不训练新模型，只在相同已查询前缀计算0.5×RF140停止分数＋0.5×LR150停止分数。两者用相同原18项和样本权重训练，模型家族不同；本轮固定等权，不尝试其它权重、logit平均或专门针对某个错误的规则。','',
  '这一动机来自不同模型评分的互补可能，不表示其错误独立或组合一定有效。两个成分的开发结果已暴露，组合公式和校准阈值虽在组合开发评分前发布，本轮仍是反复开发，不能包装为独立确认。','',
  f"原校准程序得到阈值{cut:.17g}：从0.5开始，用当前真正到达的错误停止点提高阈值，并重查完整校准搜索。固定0.5读数保留。原动作保持/严格少查标签、日期、强制停止、别名与条件同站补查均不改变，没有新风险容忍。",'',
  '等权分数平均与常用[软投票](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.VotingClassifier.html)形式相同，但本轮没有调用VotingClassifier.fit重训成分，也没有证明两份分数已校准为自然事件概率。在线需要两份模型的计算，查询节省不代表真实控制时间节省。','',
  '## 开发完整搜索','',
  '| 方法 | 原动作差异 | 不合格学习停止 | 候选查询 | 查询节省日期宏平均 | 学习提前停止 |','|---|---:|---:|---:|---:|---:|']
 for name in ['FRAME','CAP128','CAP192','CAP256','CAP512','RF140_CALIBRATED','LR150_CALIBRATED','MEAN151_FIXED_0.5','MEAN151_CALIBRATED']:
  r=get('validation',name);lines.append(f"| {name} | {r['action_differences']}/62 | {r['joint_failures']} | {r['queries']} | {100*r['date_macro_query_reduction']:.3f}% | {r['model_stops']} |")
 lines += ['',
  f"主组合动作差异{p['action_differences']}/62、查询节省宏平均{100*p['date_macro_query_reduction']:.3f}%，原开发筛查通过={decision['passes_development_screen']}。所有19种旧/新读数、每日期和所有反例均保存，不在结果出来后挑一项作为主模型。",'',
  '| 日期 | 组合动作差异 | 组合查询 | 固定256查询 |','|---|---:|---:|---:|']
 for r in decision['per_date']:lines.append(f"| {r['date']} | {r['action_differences']} | {r['queries']} | {r['cap256_queries']} |")
 lines += ['',
  '## 排序与互补性','',
  '| 分区 | 模型 | 继续类AP | AUC |','|---|---|---:|---:|']
 for r in metrics:lines.append(f"| {r['partition']} | {r['model']} | {r['continue_ap']:.4f} | {r['auc']:.4f} |")
 lines += ['',
  'same_depth_complementarity.json按相同查询深度和入口分支，保留全部正停止/继续配对的两者都正确、仅森林正确、仅线性正确、都错及并列计数。这些配对相关，不是独立实验或错误独立性证明。平均指标改善也不能代替高分端误停和完整搜索筛查。','',
  '## 判读与范围','']
 if decision['passes_development_screen']:lines += ['组合通过原开发筛查，只允许提出实际FRAME保存状态覆盖检查。先查质量/停止机会分布和在线额外推理开销，不自动部署、物理回放或扩大数据。']
 else:lines += ['组合未通过原开发筛查，不采纳，也不继续扫描另一权重。接下来只读检查真正支配高分停止的共同难例和下一批反馈，保留原FRAME与全部强简单对照。']
 lines += ['',
  f"726份逐前缀均值及原校准复算通过；{review['checks']['complete_searches']}次完整搜索的已查集合/补查/最佳已观察动作复核，{review['checks']['old_rows']}条旧记录不变。成分模型、权重、标签、日期和阈值均未修改。新拟合0、物理调用0，最终8测试未生成。",'',
  '本轮是FULL教师保存状态上的开发压缩结果，不是实际FRAME闭环轨迹、整周成本、独立测试或实测控制提速。']
 (O/'GRID151_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
 D.save('complete.json',dict(passed_audit=True,new_fits=0,new_physical=0,development_screen_passed=decision['passes_development_screen'],adopted=False,ready_to_scale=False,final8_absent=True))
 D.save('state.json',dict(stage='complete'));D.archive('complete')
 c=R/'论文协作资料/FRAME_固定分数组合_GRID151';c.mkdir(exist_ok=False)
 names=['GRID151_report.md','design.json','cutoff.json','prevalidation_publication.json','summary.json','ranking_metrics.json','same_depth_complementarity.json','review.json','decision.json','complete.json']
 for n in names:shutil.copy2(O/n,c/n)
 msg=f"两冻结模型仅固定0.5/0.5分数平均，零重训/权重扫描。组合开发动作差异{p['action_differences']}/62、查询节省宏平均{100*p['date_macro_query_reduction']:.3f}%，原筛查={decision['passes_development_screen']}；公式数组阈值在组合开发评分前发布，但所有成分开发已暴露，不作独立确认。5795搜索/5185旧记录复核通过；无物理/最终8/自动采用或扩大。报告research_archive/power-grid-2026/protocols/GRID151_fixed_score_mean/complete/GRID151_report.md"
 D.note('complete',msg)
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID151固定分数组合\r\n\r\n[报告](protocols/GRID151_fixed_score_mean/complete/GRID151_report.md)；仅固定各半均值，零重训，原门槛完整保留。\r\n\r\n'.encode('utf8')+p.read_bytes())
 count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_verified=count,collaboration_files_verified=len(names),constituents_unchanged=True,new_fits=0))
 q=D.A/'verification';q.mkdir();shutil.copy2(O/'publication_verification.json',q/'publication_verification.json')
 (q/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(q/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n')
 print('GRID151_REPORT_COMPLETE')
if __name__=='__main__':main()
