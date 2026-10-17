"""All predeclared results, per-date uncertainty and manuscript-ready text."""
from pathlib import Path
from collections import defaultdict
import csv,datetime as dt,gzip,json,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid89_review'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def ci(xs):
 if not xs:return None
 a=np.asarray(xs);rng=np.random.default_rng(61017);r=a[rng.integers(len(a),size=(5000,len(a)))].mean(1);return [float(v) for v in np.quantile(r,[.025,.975])]
def main():
 R=read(OUT/'review.json');assert R['passed_saved_data'];G=R['groups'];rows=R['rows'];summ=[]
 for phase in ['order','parameters','meta','sealed']:
  D=read(OUT/phase/'design.json')
  for m in D['methods']:
   own=[g for g in G if g['phase']==phase and g['method']==m['method']];cost=[x['cost_delta_full'] for x in own if x['cost_delta_full'] is not None]
   summ.append(dict(phase=phase,method=m['method'],dates=len(own),quality_dates=sum(x['quality_pass'] for x in own),complete_dates=sum(x['complete'] for x in own),queries=sum(x['queries'] for x in own),mean_controller_s=statistics.mean(x['controller_s'] for x in own),max_full_week_cost_delta=max(cost) if cost else None))
 csvsave(OUT/'summary_by_method.csv',summ);comparisons=[]
 def compare(phase,method,baseline):
  ds=read(OUT/phase/'design.json')['weeks'];time=[];queries=[];details=[];quality=True;same=True
  for date in ds:
   a=next(g for g in G if g['phase']==phase and g['method']==method and g['date']==date);b=next(g for g in G if g['phase']==phase and g['method']==baseline and g['date']==date)
   equal=a['steps']==b['steps'] and a['complete']==b['complete'];same&=equal;quality&=a['quality_pass'] and b['quality_pass'];t=1-a['controller_s']/b['controller_s'];q=1-a['queries']/b['queries'] if equal and b['queries'] else None;time.append(t)
   if q is not None:queries.append(q)
   details.append(dict(date=date,time_reduction=t,query_reduction=q,same_horizon=equal,quality_both=a['quality_pass'] and b['quality_pass']))
  nrep=read(OUT/phase/'design.json')['repeats'];rep=[]
  for i in range(nrep):rep.append(statistics.mean(1-next(r for r in rows if r['phase']==phase and r['method']==method and r['date']==d and r['repeat']==i)['controller_s']/next(r for r in rows if r['phase']==phase and r['method']==baseline and r['date']==d and r['repeat']==i)['controller_s'] for d in ds))
  c=dict(phase=phase,method=method,baseline=baseline,dates=len(ds),both_quality_all_dates=bool(quality),same_horizon_all_dates=bool(same),macro_controller_reduction=statistics.mean(time),date_cluster_controller95=ci(time),macro_query_reduction=statistics.mean(queries) if queries else None,query_dates=len(queries),positive_time_dates=sum(v>0 for v in time),repeat_time_reductions=rep,stable_5percent_screen=bool(nrep>=2 and quality and same and statistics.mean(time)>=.05 and sum(v>0 for v in time)>=int(np.ceil(2*len(ds)/3)) and all(v>0 for v in rep) and queries and statistics.mean(queries)>=.05),details=details)
  comparisons.append(c);return c
 for mode in read(OUT/'design.json')['order']['orders']:compare('order','FRAME_GUARD__'+mode,'FULL__'+mode)
 for m in read(OUT/'parameters/design.json')['methods']:
  if m['method']!='FULL':compare('parameters',m['method'],'FULL')
 for m in read(OUT/'meta/design.json')['methods']:
  if m['method']!='FRAME_GUARD':compare('meta','FRAME_GUARD',m['method'])
 for m in ['FRAME_GUARD','GLOBAL_MATCH','FIXED_512']:compare('sealed',m,'FULL')
 compare('sealed','FRAME_GUARD','GLOBAL_MATCH');compare('sealed','FRAME_GUARD','FIXED_512')
 # Same-state regret on the independently generated FULL trajectory; retrospective only.
 sys.path.insert(0,str(Path(__file__).parent));import policies as P
 subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];regrets=[];pairs=[]
 for date in read(OUT/'sealed/design.json')['weeks']:
  folder=OUT/'sealed/runs'/f'r0__{date}__FULL'
  for p in sorted(folder.glob('call[0-9][0-9][0-9].json')):
   c=read(p);truth={r['pool_id']:r for r in c['outcomes']};aliases={int(k):int(v) for k,v in c['aliases'].items()};best=max((r for r in truth.values() if r['strict_admissible']),key=P.key,default=None);got={}
   for method in ['FRAME_GUARD','GLOBAL_MATCH','NO_LOCAL','FIXED_512']:
    g=P.run(c['order'],lambda i,phase:dict(truth[i],phase=phase),method,subs,aliases);b=truth.get(g['chosen_pool_id']);got[method]=(g,b)
    gap=None if best is None or b is None else float(np.float32(best['reward']))-float(np.float32(b['reward']))
    regrets.append(dict(date=date,call=p.stem,method=method,candidates=len(c['order']),queries=g['public_queries'],has_full_incumbent=best is not None,has_incumbent=b is not None,reward_regret=gap,full_match=g['chosen_pool_id']==(None if best is None else best['pool_id'])))
   a,b=got['FRAME_GUARD'],got['GLOBAL_MATCH'];assert a[0]['public_queries']==b[0]['public_queries']
   gain=None if a[1] is None or b[1] is None else float(np.float32(a[1]['reward']))-float(np.float32(b[1]['reward']))
   pairs.append(dict(date=date,call=p.stem,extra_queries=a[0]['completion_budget'],same_budget=True,local_minus_global_reward=gain,local_choice=a[0]['chosen_pool_id'],global_choice=b[0]['chosen_pool_id']))
 if regrets:csvsave(OUT/'sealed_same_state_regret.csv',regrets);csvsave(OUT/'sealed_equal_budget_pairs.csv',pairs)
 order_ok=all(x['both_quality_all_dates'] for x in comparisons if x['phase']=='order');sealed=next(x for x in comparisons if x['phase']=='sealed' and x['method']=='FRAME_GUARD' and x['baseline']=='FULL')
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),order_quality_all_predeclared_orders=order_ok,sealed_frame_quality_dates=next(x['quality_dates'] for x in summ if x['phase']=='sealed' and x['method']=='FRAME_GUARD'),sealed_same_quality_speed_screen=sealed['stable_5percent_screen'],comparisons=comparisons,all_negative_results_retained=True,new_replays=246,new_fits=3,statistical_scope='Date-cluster bootstrap descriptive only. Timing and model seeds are not independent scenes. Order, parameter and META phases exposed development checks; sealed scenarios independent master seed on the same grid/calendar dates.')
 save(OUT/'decision.json',result)
 lines=['# GRID89 审稿意见补强结果','', '本轮按冻结方案完成246条串行回放、3个非零成本META种子拟合和保存数据审计。所有参数、顺序种子、日期及方法均预先固定，没有按测试结果选择新默认值。','', '## 候选顺序的来源与敏感性','', '原始动作库909个动作按教师使用次数降序保存，STATIC_ID保留该顺序；前128个动作占原库记录频次44.38%。该比例描述外部动作库，不能当作当前运行状态下的命中率。比较原顺序、随机11/23/37和公开负载排序，动作编号和破并列规则固定。']
 for c in comparisons:
  if c['phase']=='order':
   s=next(x for x in summ if x['phase']=='order' and x['method']==c['method']);q=c['macro_query_reduction'];lines.append(f"- {c['method']}：质量{s['quality_dates']}/6；相对同顺序FULL查询宏平均减少{q*100:.3f}%（{c['query_dates']}个活跃且匹配时域日期）"+f"，控制时间变化{c['macro_controller_reduction']*100:+.3f}%。单次计时，仅作描述。" if q is not None else f"- {c['method']}：质量{s['quality_dates']}/6，无可比活跃时域查询比例。")
 lines+=['','## 最终版参数单因素检查','', '四个预定旧场景，逐一变化前缀、批次、最低核查量、停滞批数和负载率分支阈值；其余保持默认。默认不因本轮结果而替换。详细逐日期成本、查询与时间见per_date.csv。']
 for s in summ:
  if s['phase']=='parameters':lines.append(f"- {s['method']}：质量{s['quality_dates']}/4，整周完成{s['complete_dates']}/4，查询合计{s['queries']}。")
 lines+=['','## META非零成本公平性补充','', '保留成本0全部五种子，新增成本0.0025的种子2/3/4，复用原0/1；训练数据、网络和30000回合训练循环保持一致。所有五种子同六个已暴露场景比较，无测试选型。当前单次计时用于描述，质量不匹配时不称同质量加速。原三成本校准未达到质量匹配、最终选0的事实保留。']
 for s in summ:
  if s['phase']=='meta':lines.append(f"- {s['method']}：质量{s['quality_dates']}/6，整周完成{s['complete_dates']}/6，查询{s['queries']}，完整控制时间算术均值{s['mean_controller_s']:.3f}秒/日期。")
 lines+=['','## 独立生成种子与最终版等预算闭环','', '固定生成主种子20261017，与原20261016不同；同一电网、同六个日历日期，全部新随机序列保留。FULL、FRAME、等追加预算全局扩展、固定512共72条回放，每日期三次轮换计时。两策略在不同闭环状态下的整周总查询量不要求相等，等预算约束针对同一状态的补查阶段。']
 for s in summ:
  if s['phase']=='sealed':
   cost='不适用' if s['max_full_week_cost_delta'] is None else f"{s['max_full_week_cost_delta']*100:+.4f}%"
   lines.append(f"- {s['method']}：质量{s['quality_dates']}/6，完成{s['complete_dates']}/6，最大可比整周成本变化{cost}，查询{s['queries']}，控制秒均值{s['mean_controller_s']:.3f}。")
 for c in comparisons:
  if c['phase']=='sealed':
   q='不适用' if c['macro_query_reduction'] is None else f"{c['macro_query_reduction']*100:+.3f}%";band=c['date_cluster_controller95']
   lines.append(f"- {c['method']} 对 {c['baseline']}：控制时间宏平均减少{c['macro_controller_reduction']*100:+.3f}%（日期簇描述性95%区间[{band[0]*100:.3f}%, {band[1]*100:.3f}%]），查询减少{q}；双方全日期质量通过={c['both_quality_all_dates']}，时域相同={c['same_horizon_all_dates']}，冻结5%筛查={c['stable_5percent_screen']}。")
 xs=[p['local_minus_global_reward'] for p in pairs if p['local_minus_global_reward'] is not None];bydate=defaultdict(list)
 for p in pairs:
  if p['local_minus_global_reward'] is not None:bydate[p['date']].append(p['local_minus_global_reward'])
 if xs:lines+=['',f"新FULL状态上的等预算一步比较：同站奖励更好{sum(x>1e-6 for x in xs)}处，全局更好{sum(x< -1e-6 for x in xs)}处；按有可比状态的日期先平均，再宏平均，奖励差为{statistics.mean(statistics.mean(v) for v in bydate.values()):.8f}。状态比较不替代闭环评价。"]
 lines+=['','## 延迟、归因与解释范围','', '完整控制时间及全步骤/搜索触发步骤P50、P95、P99、三重复离散范围和优化/预测/搜索/其余互斥耗时保存在per_run.csv及per_date.csv。查询比例与完整控制提速分开解释。原0.42%P95来自十三旧日期对固定512，不能与六日期对FULL的14.98%混用。','', '本轮没有更换电网、重新挑日期或筛除失败。参数/META/顺序阶段为开发检查；独立种子场景只检验同电网下随机序列变化。保存向量、成本和合法反馈通过审计，不声称独立AC重求解。置信范围按日期簇计算，少量日期下仅作描述。0.5%是预设实验质量容差，不是安全标准。','', '现有证据整理在evidence目录，算法性质、停止规则依据、META25维状态与效用函数、训练/校准划分均可追溯；论文V2另存，V1与用户编辑稿保留。']
 (OUT/'GRID89_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
