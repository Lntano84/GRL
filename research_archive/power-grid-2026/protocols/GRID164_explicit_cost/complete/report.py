import json,shutil
import run as D
def main():
 D.verify_trained();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'] and D.read(O/'diagnosis_receipt.json')['passed'];fam=D.read(O/'family_summary.json');pairs=D.read(O/'paired_summary.json');pay=D.read(O/'payoff_decomposition.json');comp=D.read(O/'complementarity_summary.json')
 lines=['# GRID164：把已知费用直接告诉小模型','',
 '出现了一个值得继续追的小改善：显式费用COST42相对同形状零输入ZERO42，三种子平均补查净效用提高0.81%，2/3种子提高。这个结果保留为正向探索线索，不要求全部日期和种子都赢才值得研究。','',
 '两组都用42→64→64→2、7042参数，配对初始张量、原3582训练状态、7250更新、同一抽样流、原净收益Huber标签和费用。只有最后一维是已知下一批费用，还是0。费用本来能从原41项的log_next_batch_queries还原，因此没有增加信息，检验的是表达方式。',
 '', '| 模型 | 平均净效用 | 平均查询数 | 相对完整补查查询比例 | 收益保留率 |','|---|---:|---:|---:|---:|']
 for k in ['ZERO42','COST42','OLD_BASE','OLD_FACTOR']:
  v=fam[k];lines.append(f"| {k} | {v['mean_net_utility']:.8f} | {v['mean_queries']:.3f} | {v['query_fraction']:.2%} | {v['gain_capture']:.2%} |")
 lines+=['', 'ZERO42在全部13314份旧BASE状态决定上完全复现旧决定，分数最大误差约9.54e-7。对照不是换了一套随机初始权重；两个新组的初始输出一致，新增费用连接从0开始训练。',
 '', '三个种子相对配对对照的净效用差依次为−0.001276、+0.000906、+0.004349。按日期平均，三月+0.001148、五月−0.000110、八月+0.004766、十月−0.000498。改善不是所有日期都成立，但有两个种子、两个日期的正向变化，可以继续查，不直接作否定结论。',
 '', '## 改善具体发生在哪里','',
 '1770个配对起点中，66对少查而收益不变，最终动作也相同；3对通过增加查询抓到值得付费的收益机会；7对牺牲少量收益后省下更多费用。反向156对多查没赚回费用、1对过早停止漏掉收益，也全部保留。',
 '', '这次平均净增量约91.4%来自收益增加，8.6%来自查询费用减少。原猜测偏向更会减少查询，实际更接近帮助模型保留一部分收益机会。平均查询数略降，但按日期相对完整补查的查询比例略升，因为不同日期的完整补查量不同；不挑其中一个数字宣称全面省查询。',
 '', '## 与FACTOR有何关系','',
 '相对原BASE，显式费用组独有18对正向记录，涉及13个物理状态组、四日期；FACTOR独有238对正向记录，涉及44个物理状态组、四日期。两者还有58对共同改善。它们各自有用的地方不完全相同，这比只看谁的平均分更高更值得继续研究。这里的“独有”是事后案例解释，不是在线能直接识别的选择标签，也不能把两边增益相加。',
 '', '下一项最小对照：原FACTOR修正分支已经接收费用，主干只接收原41项。两组都保留原FACTOR修正分支、相同扩展主干和原目标，只比较费用是否也直接进入主干。它检验信息送入的位置，不增加新网络模块，不改价格、标签或候选顺序。提前告知后执行固定三种子配对，不据中间分数改方案。',
 '', '六次固定终点训练总耗时约28.95秒，权重在本次验证评分前完整封存推送。26628状态分数用独立NumPy复算，全部决定一致；1770条贡献核账通过。无新物理、数据生成、最终8或论文修改。仍是旧16/4开发日期、相关补查起点的局部净效用探索，不直接等同整周控制收益。']
 (O/'GRID164_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('decision.json',dict(positive_exploratory_signal=True,relative_mean_gain=pairs['relative_mean_change'],positive_seeds=2,all_seed_win_not_required=True,mechanism='Useful gain preservation is larger component than query savings in this trial.',next_small_comparison='Route already-known cost into FACTOR backbone versus zero,keep correction inputs fixed.',adopted=False,ready_to_scale=False));D.save('complete.json',dict(passed_audit=True,fits=6,positive_exploratory_signal=True,new_physics=0));D.archive('complete')
 c=R/'论文协作资料/FRAME_显式费用小改善_GRID164';c.mkdir(exist_ok=False)
 for n in ['GRID164_report.md','paired_summary.json','family_summary.json','payoff_decomposition.json','complementarity_summary.json','decision.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID164小正向线索：同形42维7042参数/初始化/样本/NET目标，COST42相对ZERO42平均净效用+0.81%，2/3种子、2/4日期正，保留继续研究不高门槛否定。ZERO42复现旧BASE13314状态决定。净增量91.4%收益、8.6%费用；66对少查同动作收益、3对多查值得、156对多查不值等全保留。与FACTOR独有改善18 vs238对，下一步固定费用路由进FACTOR主干的小对照，不把增益相加。6拟合28.95秒，无物理/新数据/最终8。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID164显式费用小改善\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID164已有小改善继续放大\n\n1. 通知后固定小对照：FACTOR主干42维COST_MAIN vs ZERO_MAIN，修正分支均保留原41/结构/已知费用，参数/初始化/旧16-4数据/NET目标/7250更新/三个配对种子一致。只改主干是否直接接收费，不新加结构或改标签。\n2. 依据小改善和正向覆盖继续探索，不设全种子全日期同赢门槛。正反记录全保留，成本模型91.4%收益增量不强说省查询；互补是事后案例，不直接加总增益或在线择优。\n3. 新训练方案封存再训，全部终点封存再评估。无新物理/数据扩大/最终8/论文；扩大前给用户范围。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID164显式费用小改善\r\n\r\n[报告](protocols/GRID164_explicit_cost/complete/GRID164_report.md)；配对平均+0.81%、2/3种子正向，保留推进。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=8));q=D.A/'verification';q.mkdir();shutil.copy2(O/'publication_verification.json',q/'publication_verification.json');(q/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(q/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID164_REPORT_COMPLETE')
if __name__=='__main__':main()
