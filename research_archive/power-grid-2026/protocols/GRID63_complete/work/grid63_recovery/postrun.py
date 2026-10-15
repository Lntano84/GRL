"""Independent arithmetic reconciliation and interruption/event qualification; no simulation."""
from pathlib import Path
import csv,datetime as dt,hashlib,json,math,shutil,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid63'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp(s):return dt.datetime.fromisoformat(s.replace('Z','+00:00'))
def main():
 R=read(OUT/'review.json');F=read(OUT/'finished.json');D=read(OUT/'design.json')
 assert R['passed_saved_data'] and read(OUT/'recovery_finished.json')['passed']
 assert len(F['summaries'])==63 and len({(r['week'],r['repeat'],r['method']) for r in F['summaries']})==63
 frozen=read(OUT/'code_freeze.json')
 for p,h in frozen.items():assert sha(ROOT/p)==h,p
 intervals=read(OUT/'successful_run_intervals.json');events=read(OUT/'power_events.json');classified=[]
 assert len(intervals)==63
 for i,row in enumerate(intervals):
  assert row['index']==i and stamp(row['end_utc'])>=stamp(row['start_utc'])
  if i:assert stamp(row['start_utc'])>=stamp(intervals[i-1]['end_utc'])
 for e in events:
  hits=[r['index'] for r in intervals if stamp(r['start_utc'])<=stamp(e['utc'])<=stamp(r['end_utc'])]
  classified.append({**e,'overlapping_successful_workers':hits,'location':'interruption_gap' if not hits and stamp(intervals[43]['end_utc'])<stamp(e['utc'])<stamp(intervals[44]['start_utc']) else 'successful_worker_or_other'})
 assert all(e['location']=='interruption_gap' and not e['overlapping_successful_workers'] for e in classified)
 rows=F['summaries'];by={(r['week'],r['repeat'],r['method']):r for r in rows}
 independent=[];dates=[]
 for method in ['FULL','FRAME','GLOBAL_MATCH']:
  times=[];queries=[];worst=[];quality=completed=0
  for week in D['weeks']:
   own=[by[week,k,method] for k in range(3)];full=[by[week,k,'FULL'] for k in range(3)]
   assert len({(r['cost'],r['steps'],r['n1_forecasts'],r['complete']) for r in own})==1
   assert all(a['steps']==b['steps'] and a['complete']==b['complete'] for a,b in zip(own,full))
   delta=(own[0]['cost']-full[0]['cost'])/full[0]['cost'] if own[0]['complete'] else None
   quality+=int(delta is None or delta<=.005);completed+=int(own[0]['complete'])
   if delta is not None:worst.append(delta)
   times.append(1-statistics.mean(r['controller_s'] for r in own)/statistics.mean(r['controller_s'] for r in full))
   if full[0]['n1_forecasts']:queries.append(1-own[0]['n1_forecasts']/full[0]['n1_forecasts'])
  item={'method':method,'quality_dates':quality,'completed_dates':completed,'macro_time_reduction_full':statistics.mean(times),'macro_query_reduction_full':statistics.mean(queries),'max_complete_cost_delta_full':max(worst)}
  old=next(r for r in R['summaries'] if r['method']==method)
  for k,v in item.items():assert v==old[k] if isinstance(v,(str,int)) else abs(v-old[k])<1e-12,(k,v,old[k])
  independent.append(item)
 for week in D['weeks']:
  a=by[week,0,'FRAME'];b=by[week,0,'GLOBAL_MATCH'];f=by[week,0,'FULL']
  assert a['n1_forecasts']==b['n1_forecasts']
  gain=(b['cost']-a['cost'])/f['cost'] if a['complete'] and b['complete'] and f['complete'] else None
  dates.append({'date':week,'full_cost':f['cost'] if f['complete'] else None,'frame_cost':a['cost'] if a['complete'] else None,'global_cost':b['cost'] if b['complete'] else None,'frame_gain_of_full':gain,'frame_queries':a['n1_forecasts'],'global_queries':b['n1_forecasts'],'steps':a['steps'],'complete':a['complete']})
 complete_gains=[r['frame_gain_of_full'] for r in dates if r['complete']]
 P={'passed':True,'independent_arithmetic':independent,'dates':dates,'mean_gain_over_five_complete_dates':statistics.mean(complete_gains),'frozen_files_verified':len(frozen),'events_classified':classified,'successful_run_timing_qualified':True,'raw_audit_full_interval_timing_qualified':False,'qualification_basis':'All six events lie after persisted successful worker 43 and before resumed worker 44; none overlaps any of 63 original-local log/receipt intervals. Raw audit retained unchanged. No observation or timing value subtracted.','successful_replay_wall_sum_s':math.fsum(r['wall_s'] for r in rows),'elapsed_including_interruption_gap_s':F['wall_s'],'new_simulations':0,'new_fits':0,'scope':'Saved data and independent arithmetic, not independent AC re-solve.'}
 (OUT/'postrun_reconciliation.json').write_text(json.dumps(P,ensure_ascii=False,indent=2),encoding='utf-8')
 report=['# GRID63：FRAME 与等预算全局扩展的闭环比较','',
 '**结论：全局扩展具有相近竞争力，当前矩阵没有建立结构化补全的实质增量。**',
 '', '63 条串行回放全部完成，保存数据审计通过，三次重复的物理路径在 21 个日期—方法组中完全一致。7 个日期的质量筛查均通过；其中 5 个完成完整周，六月与五月分别共同在第 1901、747 步结束。',
 '', '| 方法 | 质量筛查 | 完整周 | 查询减少 / FULL | 控制时间减少 / FULL | 最坏完整周成本增幅 / FULL |','|---|---:|---:|---:|---:|---:|']
 for r in independent:report.append(f"|{r['method']}|{r['quality_dates']}/7|{r['completed_dates']}/7|{100*r['macro_query_reduction_full']:.3f}%|{100*r['macro_time_reduction_full']:.3f}%|{100*r['max_complete_cost_delta_full']:+.5f}%|")
 report+=['','查询比例先按日期计算，再对 6 个有搜索日期等权平均；控制时间比例对全部 7 个等时域日期平均。每个日期先平均三次计时，重复不增加独立场景数量。','',
 '## 等预算补查的直接结果','',
 '| 日期 | 完成情况 | FRAME 查询 | GLOBAL 查询 | GLOBAL−FRAME 成本 / FULL |','|---|---|---:|---:|---:|']
 for r in dates:
  gain=f"{100*r['frame_gain_of_full']:+.5f}%" if r['frame_gain_of_full'] is not None else '不作完整周比较'
  report.append(f"|{r['date']}|{'完整周' if r['complete'] else str(r['steps'])+'步共同结束'}|{r['frame_queries']}|{r['global_queries']}|{gain}|")
 report+=['',f"FRAME 在三月和四月分别减少成本 1386.1803 和 149.1433 单位，约为 FULL 成本的 0.04677% 和 0.00782%；其余三个完整日期成本相同。五个完整日期等权平均增益为 {100*statistics.mean(complete_gains):.5f}% FULL。没有日期达到预设 0.1% 实用增量尺度。",'',
 '因此冻结判读为 `GLOBAL_COMPETITIVE_NO_MATERIAL_STRUCTURE_GAIN`。当前数据支持反馈停止及有限补查节省查询，并没有把收益明确归因于“补查同一变电站”。两种策略在本批的全周查询总量也逐日期相同，但协议匹配的本义仍是同一输入状态下的额外配额。','',
 '## 计时与中断核对','',
 '第 44 条完整记录之后，外部执行会话中断。原第 45 条部分记录、日志、进度及哈希已单独保留；前 44 条未重跑，只重启尚未完成条目。中断片段至少留下 512 步进度、915 次公开预测和 29.516 秒控制时间，这些不是完整中断开销，不能当作零成本。','',
 '原审计在从首次启动到最后结束的整个日历区间发现 6 条电源事件，因此初始 `review.json` 的时间资格为 false。逐条核对 63 个成功 worker 的原始本地日志创建时间与结果落盘时间后，6 条事件全部位于中断空档：在第 44 条成功结果之后、恢复运行之前；没有事件与成功回放区间重叠。`successful_run_intervals.json` 与 `postrun_reconciliation.json` 保存这次定位。原审计、全部原始时间值均保留，没有扣减时间或择取最快重复。','',
 f"成功回放的 worker 墙钟总和为 {P['successful_replay_wall_sum_s']/60:.2f} 分钟。首次启动至最终完成的日历时间为 {F['wall_s']/60:.2f} 分钟，包含中断空档，两种口径分别记录。",'',
 '控制时间、标准差、最小/最大值见 `per_family.csv`；逐次决策 P95/P99、最大延迟与有搜索决策的 P95 见 `latency_per_run.csv`。FRAME 对 GLOBAL 的宏平均控制时间减少约 0.307%，差异小且逐日期方向混合；本轮不主张结构补全本身带来稳定提速。','',
 '## 对论文主张的影响','',
 '1. 保留框架相对 FULL 的质量、查询与实测控制时间结果。','2. 将 GLOBAL_MATCH 加入主要机制对照。','3. 撤去“现有消融已经证明同站结构选择具有实质优势”的表述；原 NO_LOCAL 对照说明增加查询可修复部分质量，不能单独识别同站选择的贡献。','4. 同站补全可作为框架中可替换的补查实现，当前证据不足以使它独立承担核心创新。','5. 当前只完成本次有限比较，不自动训练、扩日期或改参数。旧 V4 与用户修改稿均保留。','',
 '这些是 7 个已暴露日期上的机制检验，未解决外部教师日期覆盖问题，也不是跨电网确认。审计核验保存向量、成本账目和合法反馈，没有重新独立求解 AC 潮流。']
 (OUT/'GRID63_结果解读.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 pack=ROOT/'论文协作资料/等预算闭环比较_GRID63';pack.mkdir(parents=True,exist_ok=True)
 for name in ['GRID63_结果解读.md','GRID63_report.md','design.json','preflight.json','review.json','postrun_reconciliation.json','successful_run_intervals.json','per_run.csv','per_family.csv','frame_vs_global.csv','latency_per_run.csv','call_accounting.csv','recovery_receipt.json','power_events.json']:
  shutil.copy2(OUT/name,pack/name)
 print(json.dumps({'passed':True,'verdict':R['verdict'],'timing_events_in_gap':len(classified),'mean_gain_of_full':statistics.mean(complete_gains),'successful_wall_minutes':P['successful_replay_wall_sum_s']/60},ensure_ascii=False))
if __name__=='__main__':main()
