import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];f=D.read(O/'family_summary.json');p=D.read(O/'paired_summary.json');ev=D.read(O/'paired_event_summary.json');query_reduction=1-f['ONE_SIDED']['mean_queries']/f['FACTOR']['mean_queries'];D.save('key_changes.json',dict(net_utility_relative_to_factor=p['relative_change_vs_factor'],net_utility_relative_to_base=p['relative_change_vs_base'],mean_queries_relative_reduction=query_reduction,gain_capture_percentage_point_change=100*(f['ONE_SIDED']['gain_capture']-f['FACTOR']['gain_capture']),positive_seeds=p['positive_seeds'],positive_seed_date_pairs=sum(x['net_delta']>0 for r in p['pairs'] for x in r['per_date']),unchanged_seed_date_pairs=sum(x['net_delta']==0 for r in p['pairs'] for x in r['per_date'])))
 lines=['# GRID167：把局部节省变成单向的学习修正','',
 '顺着已有改善得到了一项小而方向更一致的开发结果：相对原FACTOR，平均补查净效用提高0.77%，平均每个起点查询数减少8.51%。三个种子分别提高约0.95%、1.08%、0.31%；四个日期的种子平均均提高。12个种子×日期组合中11个提高、1个不变。','',
 '规则很简单：原FACTOR已决定停止就保持停止；原FACTOR要继续时，只有费用路由模型和小费用模型都建议停，才提前停。没有新训练、阈值或权重。它是单向停止修正，不包装成新的神经网络架构。','',
 '| 策略 | 平均净效用 | 平均查询数 | 完整补查查询比例 | 收益保留率 |','|---|---:|---:|---:|---:|']
 for name in ['BASE','FACTOR','COST42','ROUTED','MAJORITY','ONE_SIDED']:
  r=f[name];lines.append(f"| {name} | {r['mean_net_utility']:.8f} | {r['mean_queries']:.3f} | {r['query_fraction']:.2%} | {r['gain_capture']:.2%} |")
 lines += ['', '相对原BASE，最终平均净效用提高3.52%，包含原FACTOR已有改善；不能把独立COST42的0.81%与本轮0.77%直接相加。调用均衡平均效用也从0.140449提高到0.141525。','',
 '## 改善是怎么来的','',
 '原多数票既能提前停止，也能把原STOP改成继续。前一部分节省被后一部分额外无利查询抵消。单向规则保留提前停的部分，恢复原FACTOR的停止决定，解决的是修正方向而非继续堆网络。',
 '', '1770条验证配对中，86条更少查询且收益/最终动作完全相同，涉及78个起点、32个物理状态组、全部四日期；4条牺牲少量收益后仍净赚；1条漏掉收益过多而净亏；1679条不变。正向变化和代价都保留。收益保留率从78.33%略降至78.09%，不能写成奖励或质量完全无损。','',
 '## 有一个明确但有限的结构性质','',
 '每个状态的继续决定为：原FACTOR继续 AND（路由模型继续 OR 小模型继续）。它逐状态不大于原继续决定。对于相同起点、相同候选顺序、每批查询数为正的一次补查，沿共同前缀归纳可知新策略停止位置不晚于原策略，因此候选查询不增加。该性质与模型预测准确率无关；它不保证返回同动作、保住未来收益或完整周质量，也不保证包含三模型推理后的实际时间更短。',
 '', '13314个保存状态的“不重新打开停止”及8904条训练/验证搜索的查询上界均复核通过；全量结果精确对应上轮已有方向分解。因此本轮效果在运行前已可从旧记录推知，设计中也公布了预期值，不把代数复算叫新的盲验证。','',
 '## 下一步怎么继续','',
 '这项小改善值得保留和迁移验证，不因幅度小而停止。下一步只固定当前规则，在旧训练日期中按已存在的日期分组整组留出，再从其余日期重新训练三个组成模型，看看同样的单向修正能否在未参与这些模型拟合的日期仍有改善。不扩网络、不再挑投票规则，原4个开发日期和最终8测试都不用于选择这一轮。',
 '', '本轮仍是同网、已暴露旧开发轨迹上的局部NET任务；它明确了一个可解释改法，尚不等于独立泛化或真实控制提速。数据扩大/新物理验证另行给范围，论文不自动修改。']
 (O/'GRID167_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('decision.json',dict(positive_exploratory_signal=True,all3seeds_positive=True,all4mean_dates_positive=True,local_query_bound_verified=True,next='Fixed rule date-transfer check on one pre-existing whole-date fold;retrain constituents from remaining dates.',no_rule_selection=True,not_independent_confirmation=True,adopted=False,ready_to_scale=False));D.save('complete.json',dict(passed_audit=True,new_fits=0,new_forwards=0,new_physics=0,positive_signal_retained=True));D.archive('complete')
 c=R/'论文协作资料/FRAME_单向停止小改善_GRID167';c.mkdir(exist_ok=False)
 for n in ['GRID167_report.md','key_changes.json','family_summary.json','paired_summary.json','paired_event_summary.json','decision.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID167单向停止开发解释得到可保留小改善：原FACTORSTOP不重开，继续时两帮助模型均STOP才提前停；相对FACTOR净效用+0.77%、均查询-8.51%，3/3种子正，四日期平均均正，11/12种子日期正1同。相对原BASE+3.52%，不加总COST42独立0.81。86对同收益动作少查、4轻微牺牲仍净正、1漏收益负。13314状态/8904搜索查询不增加；规则来自旧方向分解，预期效果已知非盲确认。下一步只固定此规则做一组既有日期留出重训组成模型，不继续规则扫描/扩网络/物理/最终8。'
 for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  path=REPO/'docs'/name;old=path.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID167单向停止小改善\n\n'+msg+'\n'
  if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  path.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID167小改善迁移\n\n1. 保留单向规则不变，不再扫描组合。训练前通知，选既有日期折中的固定第一折，旧训练16日期中整组留4日期，另外12日期拟合组成模型；看这个小改善是否还能迁移。原4开发不参与新比较，最终8不碰。\n2. 折内仅训练日期计算归一化与抽样；旧fee/gain_scale作为已冻结任务单位保留并披露来源，不冒充完全未暴露的新测试。所有3种子/全部留出日期保留。\n3. 不要求全种子全日期赢才值得探索；分别记效用/收益/查询/反例。无新物理或数据扩大，论文不改；数据/物理扩大先给用户范围。\n',encoding='utf8',newline='\n')
 path=REPO/'research_archive/power-grid-2026/README.md';path.write_bytes('# 最新：GRID167单向停止小改善\r\n\r\n[报告](protocols/GRID167_one_sided_stop/complete/GRID167_report.md)；净效用+0.77%、均查询-8.51%，固定规则继续查日期迁移。\r\n\r\n'.encode('utf8')+path.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for file in c.iterdir():assert D.sha(file)==D.sha(O/file.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=8));q=D.A/'verification';q.mkdir();shutil.copy2(O/'publication_verification.json',q/'publication_verification.json');(q/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(q/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID167_REPORT_COMPLETE')
if __name__=='__main__':main()
