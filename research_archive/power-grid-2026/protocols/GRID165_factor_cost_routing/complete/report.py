import json,shutil
import run as D
def main():
 D.verify_trained();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'] and D.read(O/'diagnosis_receipt.json')['passed'] and D.read(O/'contrast_receipt.json')['passed'];fam=D.read(O/'family_summary.json');paired=D.read(O/'paired_summary.json');events=D.read(O/'paired_event_summary.json');control=D.read(O/'control_old_factor_reproduction.json')
 lines=['# GRID165：费用路由里的正向变化与代价','',
 '费用也直接进入FACTOR主干后，出现167条少查且保留相同收益/动作的记录，涉及147个起点、41个物理状态组、全部四个开发日期。这部分改善值得保留；同时9条过早停止丢掉较多收益、114条额外查询没有赚回费用，使总体平均净效用低于原FACTOR约2.58%。不把正向记录随整体均值一起丢掉，也不把这167条单列成整体成功。','',
 '| 模型 | 平均净效用 | 平均查询数 | 完整补查查询比例 | 收益保留率 |','|---|---:|---:|---:|---:|']
 for name in ['ZERO_MAIN','COST_MAIN','OLD_FACTOR','OLD_BASE','COST42']:
  r=fam[name];lines.append(f"| {name} | {r['mean_net_utility']:.8f} | {r['mean_queries']:.3f} | {r['query_fraction']:.2%} | {r['gain_capture']:.2%} |")
 lines += ['', '对照ZERO_MAIN在全部13314个原FACTOR状态上，分数和决定逐项完全一致。两组同12980参数、原随机初始化、旧16/4日期、NET目标与7250更新；修正分支都已有费用信息，只改变主干是否直接接收费。六次终点训练总耗时约441.4秒，验证前全部权重已封存推送。','',
 '| 配对种子 | COST_MAIN−ZERO_MAIN净效用 |','|---|---:|']
 for r in paired['pairs']:lines.append(f"| {r['seed']} | {r['net_delta']:+.8f} |")
 lines += ['', '总体负差约91.4%来自收益减少，8.6%来自查询费用增加。八月三个种子均丢失效用较多；五月两个种子改善、种子平均略有改善。当前不能简单地把费用接入更多分支就当成“放大”，但已经定位了可以保留和需要保护的两类状态。','',
 '| 变化类别 | 种子×起点记录 | 对平均效用变化的贡献 |','|---|---:|---:|']
 labels={'LESS_QUERY_NO_GAIN_LOSS':'少查、同收益同动作','LESS_QUERY_GAIN_LOSS_TOO_LARGE':'少查但漏收益过多','MORE_QUERY_COST_NOT_REPAID':'多查未赚回费用','LESS_QUERY_SMALL_GAIN_SACRIFICE_NET_POSITIVE':'少查、牺牲少量收益仍净赚','MORE_QUERY_GAIN_PAYS':'多查且收益足够','UNCHANGED':'不变'}
 for r in events:lines.append(f"| {labels[r['category']]} | {r['seed_start_pairs']} | {r['net_contribution']:+.6f} |")
 lines += ['', '## 正向变化可以怎样继续推进','',
 '167条无损节省在第一次分歧后的下一批增益全部为0，其中150条对应不足64个候选的批次，中位数6个。9条漏掉收益的下一批净收益则全部为正，5条是完整64候选批次、4条是尾批。不能只用“整批/尾批”这个标签选规则，否则仍会漏掉有价值尾批。','',
 '已经训练的小费用模型COST42，在这9条有价值机会中有8条主张继续；但它也会反对167条正确节省中的82条。三者的分歧提供了一个明确、无新训练的下一项：以原FACTOR、费用路由FACTOR和小费用模型的既有二元决定做一次固定多数票检查。它等价于修正建议与小模型一致时才改变原FACTOR决定，不更改原阈值、不选择权重。','',
 '这只是普通多数投票，不包装成新架构或安全保证。规则是在这些已暴露案例的诊断后形成，后续只能作为开发探索；不能假装它是未见数据验证。必须同时报告保护了多少有益继续、放弃多少正确节省、新增哪些错误，并比较全部起点与三个种子，不挑有利记录。','',
 '本轮只用已保存教师路径，167条记录不是167个独立场景。局部补查净效用不等于完整周质量/真实控制时间；三模型的在线推理开销也尚未计入原查询费。无新场景、物理回放或最终8测试。两组完整正反结果与独立数值核验均归档。']
 (O/'GRID165_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('decision.json',dict(positive_subgroups_retained=True,positive_no_gain_loss_pairs=167,overall_relative_change=paired['relative_mean_change'],all3seeds_overall_negative=True,next='One fixed majority vote of originalFACTOR,routedFACTOR,COST42;no weights/threshold tuning,posthocdevelopment only.',small_cost42_gain_preserved=True,adopted=False,ready_to_scale=False));D.save('complete.json',dict(passed_audit=True,fits=6,new_physics=0,positive_subgroups_retained=True));D.archive('complete')
 c=R/'论文协作资料/FRAME_费用路由与正向子群_GRID165';c.mkdir(exist_ok=False)
 for n in ['GRID165_report.md','paired_summary.json','family_summary.json','paired_event_summary.json','existing_expert_summary.json','control_old_factor_reproduction.json','decision.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID165保留正向变化：同形12980参数主干费用路由，ZERO_MAIN逐项复现原FACTOR13314状态；COST_MAIN新增167对少查同收益动作/41物理组/四日期，但9对漏收益、114对无利多查使整体-2.58%，3种子整体负。不是全盘否定费用表达，保留COST42+0.81独立线索。COST42在9漏收益中8要求继续，也反对167正确节省中82；下一步仅一个固定三模型多数票离线开发检查，不调权重/阈值，不假装独立确认。6拟合441.4秒无新物理/数据/最终8。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID165费用路由正向子群保留\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID165保留167条局部改善\n\n1. 一个固定多数票开发检查：原FACTOR、路由FACTOR、小COST42原有二元决定，票多者决定；无训练、无权重/阈值搜索、不变费用。规则来自已暴露诊断，明确事后探索，不独立确认。全部旧train/validation起点/三个种子保留。\n2. 同时计保住有益查询、损失正确节省、新增错误、总净效用/收益/查询；没有全种子赢的探索门槛，正向线索与反例都追。\n3. COST42+0.81独立改善仍保留，不把路由整体负解释为费用表达全无效。推理费用未计入本任务代理query费，不宣称实测控制提速。无新物理/数据/最终8/论文，扩大前给用户范围。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID165费用路由正向子群\r\n\r\n[报告](protocols/GRID165_factor_cost_routing/complete/GRID165_report.md)；167条无损节省保留，追小模型分歧与机会保护。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=9));q=D.A/'verification';q.mkdir();shutil.copy2(O/'publication_verification.json',q/'publication_verification.json');(q/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(q/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID165_REPORT_COMPLETE')
if __name__=='__main__':main()
