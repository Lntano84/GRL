"""Independent summary arithmetic from frozen receipts; no model refit or AC re-solve."""
from pathlib import Path
import csv,hashlib,json,math,shutil,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid65'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D=read(OUT/'design.json');F=read(OUT/'finished.json');R=read(OUT/'review.json');T=read(ROOT/'outputs/grid64/training_finished.json')
 assert read(OUT/'pipeline_finished.json')['passed'] and R['passed_saved_data']
 for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
 assert len(F['summaries'])==70 and len({(s['repeat'],s['week'],s['method']) for s in F['summaries']})==70
 by={(s['repeat'],s['week'],s['method']):s for s in F['summaries']};derived=[];bydate=[]
 for method in ['FULL','FRAME','GLOBAL_MATCH','META_DQN0','META_DQN1']:
  times=[];queries=[];deltas=[];quality=complete=0;control_mean_sum=0.;query_sum=0
  for week in D['weeks']:
   ss=[by[rep,week,method] for rep in [0,1]];ff=[by[rep,week,'FULL'] for rep in [0,1]]
   assert all(ss[0][k]==ss[1][k] for k in ['cost','steps','complete','n1_forecasts','public_forecasts'])
   assert all(a['steps']==b['steps'] and a['complete']==b['complete'] for a,b in zip(ss,ff))
   delta=(ss[0]['cost']-ff[0]['cost'])/ff[0]['cost'] if ss[0]['complete'] else None
   quality+=int(delta is None or delta<=.005);complete+=int(ss[0]['complete'])
   if delta is not None:deltas.append(delta)
   own=statistics.mean(s['controller_s'] for s in ss);ref=statistics.mean(s['controller_s'] for s in ff)
   times.append(1-own/ref);control_mean_sum+=own;query_sum+=ss[0]['n1_forecasts']
   if ff[0]['n1_forecasts']:queries.append(1-ss[0]['n1_forecasts']/ff[0]['n1_forecasts'])
   bydate.append({'method':method,'date':week,'cost_delta_full':delta,'complete':ss[0]['complete'],'steps':ss[0]['steps'],'queries':ss[0]['n1_forecasts'],'control_mean_s':own,'control_std_s':statistics.stdev(s['controller_s'] for s in ss)})
  row={'method':method,'quality_dates':quality,'completed_dates':complete,'macro_time_reduction_full':statistics.mean(times),'macro_query_reduction_full':statistics.mean(queries),'max_complete_cost_delta_full':max(deltas),'controller_mean_sum_s':control_mean_sum,'query_sum_per_repeat':query_sum}
  expected=next(s for s in R['summaries'] if s['method']==method)
  for k in ['quality_dates','completed_dates','macro_time_reduction_full','macro_query_reduction_full','max_complete_cost_delta_full']:
   assert abs(row[k]-expected[k])<1e-12
  derived.append(row)
 # Recompute calibration feasibility and shared choice from the stored per-state predictions.
 options=[]
 for ci in range(3):
  fits=[f for f in T['fits'] if f['costindex']==ci];assert len(fits)==2
  means=[];maxs=[];qs=[];feasible=True
  for f in fits:
   path=ROOT/f"outputs/grid64/models/cost{ci}_seed{f['seed']}"
   assert sha(path/'final.pt')==f['final_sha256'] and sha(path/'initial.pt')==f['initial_sha256']
   rr=f['rows'];m=statistics.mean(v['regret'] for v in rr);mx=max(v['regret'] for v in rr)
   feasible &= m<=statistics.mean(v['frame_regret'] for v in rr)+1e-7 and mx<=max(v['frame_regret'] for v in rr)+1e-7
   means.append(m);maxs.append(mx);qs.append(statistics.mean(v['queries'] for v in rr))
  options.append({'costindex':ci,'feasible':bool(feasible),'mean_queries':statistics.mean(qs),'mean_regret':statistics.mean(means),'max_regret':max(maxs)})
 selected=min([v for v in options if v['feasible']],key=lambda v:(v['mean_queries'],v['mean_regret'],v['costindex']))
 assert selected['costindex']==T['selected']['costindex']==1
 full=derived[0];frame=derived[1]
 for d in derived:
  d['pooled_time_reduction_full']=1-d['controller_mean_sum_s']/full['controller_mean_sum_s']
  d['pooled_query_reduction_full']=1-d['query_sum_per_repeat']/full['query_sum_per_repeat']
 paired=[]
 for seed in [0,1]:
  m=next(d for d in derived if d['method']==f'META_DQN{seed}')
  paired.append({'seed':seed,'frame_pooled_time_reduction_meta':1-frame['controller_mean_sum_s']/m['controller_mean_sum_s'],'frame_pooled_query_reduction_meta':1-frame['query_sum_per_repeat']/m['query_sum_per_repeat']})
 saved={'passed':True,'frozen_files_verified':len(read(OUT/'code_freeze.json')),'summaries_independently_recalculated':derived,'calibration_choice_independently_recalculated':selected,'paired_pooled_diagnostic':paired,'new_rollouts':0,'new_fits':0,'no_independent_ac_solve':True,'aggregation_note':'Primary: average repeats within date, then average date ratios. Pooled sums are supplementary, do not replace primary or imply significance.'}
 (OUT/'postrun_reconciliation.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
 with (OUT/'interpretation_per_date.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(bydate[0]));w.writeheader();w.writerows(bydate)
 report=['# GRID65 结果解读与论文增补','',
 '**完成 70 条闭环回放和保存数据审计。本批 FRAME 在质量与查询量上优于这项 ICAPS 2022 自适应停止方法的电网适配，实际控制时间优势较小。**','',
 '| 方法 | 质量筛查通过日期 | 完整周日期 | 查询减少/FULL | 控制时间减少/FULL | 最坏完整周成本增幅/FULL |','|---|---:|---:|---:|---:|---:|']
 for d in derived:report.append(f"|{d['method']}|{d['quality_dates']}/7|{d['completed_dates']}/7|{100*d['macro_query_reduction_full']:.2f}%|{100*d['macro_time_reduction_full']:.2f}%|{100*d['max_complete_cost_delta_full']:+.3f}%|")
 report+=['','质量标准为不比同批 FULL 更早失败，且共同完成周成本增幅不超过 0.5%。所有方法都完成五个完整周，六月和五月均在第 1901 和 747 步共同结束。七日期质量通过不表示七日期都完成完整周。查询宏平均覆盖六个有搜索日期；控制时间宏平均覆盖七个等时域日期。','',
 '## 与论文来源对照相比，FRAME 获得了什么','',
 'META-DQN 种子 0 在三月成本高于 FULL 2.973%；种子 1 在三月高 3.983%、九月高约 1.186%，因此分别只通过 6/7 和 5/7 日期质量筛查。FRAME 的最坏增幅仍为三月 0.372%。',
 '按日期等权宏平均，FRAME 比两个元策略种子分别少查询 18.67%、27.96%；控制时间分别减少 0.056%、1.743%。其中种子 0 的实际控制时间几乎持平，不能把查询减少的幅度当作同等幅度的控制加速。',
 'FRAME 也并非逐日期更优：三月 FRAME 比两个元策略都更慢；四月两个元策略的完整周成本略低于 FRAME；十一月种子 1 成本也略低。全部逐日期结果保留。','',
 '作为补充口径，将七日期的控制秒数直接汇总后，FRAME 比种子 0 多约 0.63% 控制时间，与种子 1 几乎相同。该汇总与先算日期比例再平均的主口径不同，不能只选有利的一种。本轮的主要增益是质量稳定性与查询压缩，尚无稳定、显著的对元策略整套控制器加速结论。','',
 '## 这个基线如何得到','',
 '源方法：Bhatia 等，ICAPS 2022，作者仓库 Metareasoning.jl。保留 Double DQN、目标网络、回放池及时间效用差奖励，限制动作空间为 STOP/CONTINUE。原文还调整 A*/RRT* 参数，这部分没有移植，故论文应明确写作“ICAPS 2022 元控制方法的 stopping-only 电网适配版”。',
 '排除所有七个评价日期家族后，用 14 个训练家族的 129 个状态训练，4 个校准家族的 22 个状态选时间价格。两个种子、三个预定价格共六次固定训练，总墙钟 202.76 秒，所有初始、末端与未选配置均保留。选择时间价格 0.0025，两个种子均达到预定的一步校准质量要求。没有使用这七日期的成绩选择价格或 checkpoint。',
 '训练使用已付费完整候选反馈生成的元决策环境，新增教师仿真为零；每个模型的 15,000 episode 是重复抽取已有训练状态，不是新增 15,000 个电网场景。校准质量是一步 N1 奖励，闭环结果说明这项代理仍可能遗漏长期成本效应。','',
 '## 等预算全局扩展的结论保持','',
 'GLOBAL_MATCH 在新批次仍通过 7/7 日期质量标准，查询减少 74.09%、实际控制时间减少 13.79%，与 FRAME 接近。相同新增查询量下，同站补全的成本额外收益仍仅为三月 0.04677% FULL、四月 0.00782% FULL；没有建立 0.1% 实用尺度的结构选择增量。',
 '因此，补入 META-DQN 后，论文已有一个明确来源的自适应停止对照，但不能据此恢复“同站补全已被证明独特有效”的主张。','',
 '## 可直接用于论文的实验描述','',
 '“我们增加了基于 Bhatia 等（ICAPS 2022）元控制框架的自适应停止对照。该对照使用 Double DQN，根据已评估候选的反馈决定停止或继续执行下一批全局查询。训练与校准按日期家族排除评价数据，两个模型种子均保留。七个日期的闭环结果显示，FRAME 通过全部质量筛查，而元控制对照的两个种子分别通过六个和五个日期。FRAME 相对完整搜索减少 74.09% 的查询，控制时间减少 13.58%。相对于元控制对照，FRAME 的优势主要体现在质量尾部与查询量；整套控制时间差异较小。等预算全局扩展同样具有竞争力，表明当前结果支持整体计算组织的作用，尚不足以单独归因于同站结构选择。”','',
 '## 审计与范围','',
 f"保存数据审计核对 {R['physical_rows']:,} 个物理步、{R['public_forecasts']:,} 次公开预测，以及 {R['matched_full_feedback']:,} 项与 FULL 共同输入下的反馈。35/35 个日期—方法组的两重复物理路径一致，三个旧方法的 42 项质量/查询/步数检查复现 GRID63。运行区间无电源事件。补充脚本独立重算全部汇总和校准选择，{saved['frozen_files_verified']} 项冻结哈希通过。",'',
 '这仍是一个电网上七个已暴露日期的比较；适配版不代表原论文全部能力或所有先进规划方法。审计核验保存轨迹与费用，没有独立重新求解 AC 潮流。原中文 V4 与用户修改稿没有覆盖。']
 (OUT/'GRID65_结果解读.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 pack=ROOT/'论文协作资料/论文方法对照_GRID65';pack.mkdir(parents=True,exist_ok=True)
 for name in ['GRID65_结果解读.md','GRID65_report.md','review.json','design.json','preflight.json','postrun_reconciliation.json','per_family.csv','per_run.csv','frame_vs_meta.csv','latency_per_run.csv','interpretation_per_date.csv','power_events.json']:
  shutil.copy2(OUT/name,pack/name)
 trainpack=ROOT/'论文协作资料/自适应停止对照_GRID64'
 for name in ['design.json','learning_preflight.json','learning_qualification.json','training_finished.json','data_receipts.json']:
  shutil.copy2(ROOT/'outputs/grid64'/name,trainpack/name)
 print(json.dumps({'passed':True,'summary_rows':len(derived),'frozen_files':saved['frozen_files_verified'],'pooled_diagnostic':paired},ensure_ascii=False))
if __name__=='__main__':main()
