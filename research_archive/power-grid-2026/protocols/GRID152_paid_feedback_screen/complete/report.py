import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed']
 dec=D.read(O/'decision.json');dominance=D.read(O/'two_score_dominance.json');sums=D.read(O/'neighbor_summary.json')
 lines=['# GRID152：高分误停与已有付费反馈的信息筛查','',
  '本轮不训练。第一项核查解释为什么两份模型的简单组合受限；第二项只用386拟合/198校准记录，筛查固定8项旧投影中已有、当前18项未单独保留的付费反馈信息。新的8项开发验证值未提取。','',
  '## 两份评分的共同困难','',
  '| 分区/可比范围 | 可停止前缀 | 被继续前缀两项分数同时支配 |','|---|---:|---:|']
 for x in dominance:lines.append(f"| {x['partition']}/{x['scope']} | {x['stop_rows']} | {x['dominated_stop_rows']} |")
 lines += ['',
  '支配指某个继续前缀的RF140和LR150停止分数，都不低于某个可停止前缀。任何只依赖这两分数、且分别单调递增的合并函数，都不能在同一阈值下接受后者而拒绝前者。这是逐前缀两维评分的可分性约束，不是整个先到先停搜索策略的不可行性证明，也不约束使用原始输入的其它模型。','',
  '同深度/入口分支的校准153个可停止前缀中，91个存在这样的继续前缀。它说明继续扫描另一组组合权重缺乏依据，不能据此宣布任务不可学习。全部支配关系保存，不选择某一个权重。','',
  '## 固定8项付费反馈字段','',
  '使用旧GRID101公开投影中已定义的terminal_fraction、illegal_fraction、ambiguous_fraction、exception_fraction、admissible_reward_mean、admissible_reward_std、station_admissible_fraction、incumbent_relative_position。它们只看已查询反馈、公开顺序和别名，原18项完整保留。不是使用未来动作标签造特征。','',
  '提取前固定筛查：只在原校准集，继续状态的5近邻空白数必须严格减少、停止侧混杂数不能增加，4个日期都不能变差；不通过就不训练这套8项、不换子集或距离救结果。','',
  '| 视图/分区 | 继续状态有5个近邻 | 其中无继续近邻 | 停止状态有5个近邻 | 其中混入继续标签 |','|---|---:|---:|---:|---:|']
 for s in sums:lines.append(f"| {s['view']}/{s['partition']} | {s['continue_full5']} | {s['continue_empty']} | {s['stop_full5']} | {s['stop_contaminated']} |")
 lines += ['',
  f"预定筛查通过={dec['passed_prespecified_information_screen']}。校准继续空白从11/45增至15/45，停止侧混杂55→58；一月/七月都变差。因此不训练26项变体，不继续调整这8项。几何诊断不是预测性能，也不证明所有丰富反馈表示都无效。",'',
  '非法/歧义比例在这些记录中恒定，terminal与exception分布可能重复；全部字段的取值数公开，不把新增8列说成8种有效独立信息。','',
  '584条严格公开输入的8项独立标量复算、隐藏后缀和未来参数拒绝核对通过。模型、原标签、权重、日期、阈值及所有旧结果不变，未进行新的模型预测、策略回放、物理仿真、数据扩展或最终8测试。','',
  '下一项检查转向数据来源覆盖：用已有FRAME实际运行日志核查FULL教师状态与实际FRAME状态的重合程度。先只读核查，不重跑环境、不为当前失败模型扩规模。']
 (O/'GRID152_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
 D.save('complete.json',dict(passed_audit=True,new_fits=0,new_physical=0,information_screen_passed=dec['passed_prespecified_information_screen'],validation_features_unextracted=True,final8_absent=True));D.archive()
 c=R/'论文协作资料/FRAME_付费反馈筛查_GRID152';c.mkdir(exist_ok=False)
 for n in ['GRID152_report.md','design.json','decision.json','neighbor_summary.json','neighbor_per_date.json','extra_profile.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID152零拟合筛查：同深度/入口校准153可停中91被某继续前缀两评分同时支配，非完整策略不可学证明。预定8项旧付费反馈增量只提取fit/cal，校准继续邻域空白11→15，停止侧混杂55→58，预定门槛未过，禁止训练该26项/换子集救结果。开发新8项未提取，584标量/后缀核验通过。下一步只读检查既有实际FRAME路径覆盖，不重跑/扩大/最终8。报告research_archive/power-grid-2026/protocols/GRID152_paid_feedback_screen/complete/GRID152_report.md'
 entry='\n\n## 2026-10-25 — GRID152付费反馈筛查完成\n\n'+msg+'\n'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID152后\n\n1. 不扫描评分权重，不训练未过筛的26项输入，不以更多特征/模型掩盖无增量。\n2. 只读核查既有实际FRAME与FULL教师的状态/基础动作/公开上下文覆盖；不要重跑或把覆盖不足当成当前教师开发失败的唯一原因。\n3. 保持原标签/对照/日期，未通过模型不部署。无新物理/数据/最终8/RL/论文或扩大。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID152付费反馈资格筛查\r\n\r\n[报告](protocols/GRID152_paid_feedback_screen/complete/GRID152_report.md)；校准门槛未过，不启动26项训练。\r\n\r\n'.encode('utf8')+p.read_bytes())
 count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=len(list(c.iterdir())),new_fits=0))
 q=D.A/'verification';q.mkdir();shutil.copy2(O/'publication_verification.json',q/'publication_verification.json')
 (q/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(q/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n')
 print('GRID152_REPORT_COMPLETE')
if __name__=='__main__':main()
