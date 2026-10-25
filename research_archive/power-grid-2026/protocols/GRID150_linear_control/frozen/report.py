"""Publish the one-fit linear control with full existing anchors and scope."""
import json,shutil
import run as G
def main():
 G.verify_trained();O,R,REPO=G.O,G.R,G.REPO
 assert G.read(O/'state.json')['stage']=='audited';review=G.read(O/'review.json');assert review['passed']
 sums=G.read(O/'summary.json');cut=G.read(O/'cutoff.json');d=G.read(O/'decision.json')
 get=lambda part,name:next(s for s in sums if s['evaluation_partition']==part and s['method']==name)
 primary=get('validation','LR150_CALIBRATED');metrics=G.read(O/'fit_calibration_metrics.json')+[dict(partition='validation',**r) for r in G.read(O/'validation_metrics.json')]
 methods=['FRAME','CAP128','CAP192','CAP256','CAP512','RF140_CALIBRATED','RF142_CALIBRATED','RF144_CALIBRATED','RF146_ENTRY128_CALIBRATED','LR150_FIXED_0.5','LR150_CALIBRATED']
 assert all(any(s['method']==n for s in sums) for n in methods)
 lines=['# GRID150：相同全局动作保持任务的逻辑回归对照','',
  '只训练一份L2正则逻辑回归，沿用旧GRID101固定C=1、lbfgs、最多2000次、种子20261022、无类别平衡。旧逻辑回归是局部净收益任务，不能视为本任务已有结果。当前只检验低容量线性分类器是否优于既有森林，不添加物理12项或其它新特征。','',
  '原18项与18个缺失标记、386拟合/198校准/142开发前缀、12/4/4日期、原RF140分层样本权重、标签、候选顺序、全局128/64检查点与原条件补查全部保持。原拟合中位数不变，在386拟合行上做标准化后训练；校准与开发仅使用冻结变换。','',
  '这改变了分类器家族及其必要预处理，不是同参数量或孤立网络结构消融。模型分数不当作自然风险概率。目标仍为返回原FRAME动作且严格少查，没有换质量容忍或把不同动作改成成功。','',
  f"唯一拟合用了{G.read(O/'fit_receipt.json')['iterations']}次优化迭代，收敛={G.read(O/'fit_receipt.json')['converged']}，耗时{G.read(O/'fit_receipt.json')['elapsed_s']:.4f}秒；这不是控制时间。经验阈值{cut['primary_threshold']:.17g}，按原0.5起点和最高错误停止分数规则校准，模型/标准化/阈值推送后才评分开发验证。",'',
  '## 完整搜索的开发读数','',
  '| 方法 | 与原FRAME动作不同 | 不合格学习停止 | 候选查询 | 查询节省日期宏平均 | 学习提前停止 |','|---|---:|---:|---:|---:|---:|']
 for name in methods:
  r=get('validation',name);lines.append(f"| {name} | {r['action_differences']}/62 | {r['joint_failures']} | {r['queries']} | {100*r['date_macro_query_reduction']:.3f}% | {r['model_stops']} |")
 lines += ['',
  '全部17种旧参照/新读数都在summary和逐调用文件保留；表中列出事先确定的主要参照。CAP带原强制停止、无可接纳动作继续和相同补查，不是无条件截断。日期宏平均先场景、再同日期副本、最后日期平均。固定0.5完整保留，不按结果替换校准主读数。','',
  f"主LR150_CALIBRATED动作差异{primary['action_differences']}/62、不合格停止{primary['joint_failures']}，查询节省宏平均{100*primary['date_macro_query_reduction']:.3f}%。原开发筛查通过={d['passes_development_screen']}。门槛仍是动作/查询不合格为0，宏平均严格超过CAP256，每个日期查询不多且至少一个日期更少。",'',
  '| 日期 | 主模型动作差异 | 主模型查询 | CAP256查询 |','|---|---:|---:|---:|']
 for r in d['per_date']:lines.append(f"| {r['date']} | {r['action_differences']} | {r['queries']} | {r['cap256_queries']} |")
 lines += ['', '## 继续少数类与排序','',
  '| 分区/读数 | 继续AP | AUC | 继续召回 | 有效停止召回 |','|---|---:|---:|---:|---:|']
 for r in metrics:lines.append(f"| {r['partition']}/{r['reading']} | {r['continue_ap']:.4f} | {r['auc']:.4f} | {100*r['continue_recall']:.2f}% | {100*r['valid_stop_recall']:.2f}% |")
 lines += ['',
  '分类排序是辅助指标；只有在相同完整搜索、动作保持和查询口径下才判断计算收益。旧模型和所有不利日期保留，全部新错误见all_new_model_failures.json。','',
  '## 核验与判读','',
  f"726个公开输入/隐藏后缀检查、726个手工sigmoid分数复算通过，最大误差{review['max_score_error']:.3g}；拟合均值/方差、原分层权重和L2目标一阶梯度独立复核。归一化梯度无穷范数{review['fit_optimality_gradient_inf']:.3g}。{review['checks']['complete_searches']}次完整搜索和{review['checks']['old_reference_rows_unchanged']}条旧参照核对通过，模型阈值评分前发布。",'',
  '标准化和逻辑回归含义依据安装的scikit-learn1.7.2与[官方StandardScaler](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html)、[LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)核对。只有一次新拟合，没有重训旧模型或参数扫描。','']
 if d['passes_development_screen']:lines+=['该基线通过原开发搜索筛查，但还不是闭环或泛化结论。下一步先审查其有效停止和错误尾部，并提出实际FRAME路径保存状态覆盖核查，不自动部署/扩数据。']
 else:lines+=['该基线未通过原开发搜索筛查，不采纳，不按开发结果换C、阈值或标签。下一步先只读核查错误尾部与已有森林重合程度，区分单一模型问题与多个不同模型共享的困难状态，再决定是否存在有依据的下一实验。']
 lines += ['',
  '原开发日期已反复暴露，不能当新独立测试；数据来自FULL教师，实际FRAME物理路径不同。动作保持不等同整周成本或安全，查询减少不等同实测控制提速。最终8测试未生成，无物理仿真、新数据、RL/神经网络、论文或图表修改。']
 (O/'GRID150_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
 G.save('complete.json',dict(passed_audit=True,new_fits=1,old_refits=0,new_physical=0,development_screen_passed=d['passes_development_screen'],adopted=False,ready_to_scale=False,final8_absent=True))
 G.save('state.json',dict(stage='complete'));G.verify_trained();G.archive('complete')
 c=R/'论文协作资料/FRAME_同任务线性对照_GRID150';c.mkdir(exist_ok=False)
 names=['GRID150_report.md','design.json','scaler.json','fit_receipt.json','cutoff.json','prevalidation_publication.json','summary.json','fit_calibration_metrics.json','validation_metrics.json','calibration_history.json','review.json','decision.json','complete.json']
 for n in names:shutil.copy2(O/n,c/n)
 msg=f"原18项、386权重与12/4/4日期的逻辑回归一次拟合完成；固定C1/lbfgs/2000继承旧任务配置，无扫描。主开发动作差异{primary['action_differences']}/62、查询节省宏平均{100*primary['date_macro_query_reduction']:.3f}%，原筛查={d['passes_development_screen']}。模型/标准化/阈值评分前发布，726独立sigmoid/公开输入、5185搜索与4575旧参照复核通过，全部旧参照/坏结果保留。无新物理/最终8/采纳/扩大。报告research_archive/power-grid-2026/protocols/GRID150_linear_control/complete/GRID150_report.md"
 G.note('complete',msg)
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID150同任务低容量对照\r\n\r\n[报告](protocols/GRID150_linear_control/complete/GRID150_report.md)；原18项逻辑回归一次拟合，模型阈值评分前发布。\r\n\r\n'.encode('utf8')+p.read_bytes())
 count=0
 for phase in G.A.iterdir():
  for n,h in G.read(phase/'archive_manifest.json').items():assert G.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert G.sha(p)==G.sha(O/p.name)
 G.save('publication_verification.json',dict(passed=True,archive_files_verified=count,collaboration_files_verified=len(names),original_sources_models_unchanged=True,one_fit_only=True))
 d=G.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json')
 (d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':G.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n')
 print('GRID150_REPORT_COMPLETE',count)
if __name__=='__main__':main()
