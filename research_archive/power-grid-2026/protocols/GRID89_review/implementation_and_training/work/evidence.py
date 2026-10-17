"""Recompute existing evidence by date; no new training or simulator queries."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,hashlib,json,math,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid89_review/evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csvread(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def dump(name,data):
 if not data:return
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
def interval(xs,seed=61017):
 if not xs:return None
 rng=np.random.default_rng(seed);a=np.asarray(xs);b=a[rng.integers(len(a),size=(5000,len(a)))].mean(1);return [float(x) for x in np.quantile(b,[.025,.975])]
def main():
 OUT.mkdir(exist_ok=True);asset=ROOT/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz'
 with np.load(asset) as z:c=z['counts']
 assert len(c)==909 and (c[:-1]>=c[1:]).all()
 provenance=dict(library=asset.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),actions=len(c),nonincreasing_teacher_counts=True,total_teacher_counts=int(c.sum()),first128_count_share=float(c[:128].sum()/c.sum()),evidence=['work/grid02/ljn_nn/README.MD explains teacher collection','modules/utils.py preserves action_space array row order','modules/topology_heuristic.py loads the list without reordering','work/grid36/cheap_rank.py STATIC_ID sorts retained representatives'],limitation='Counts describe external action-library collection, not runtime action success; raw teacher date coverage not independently verified. All comparators share the fixed external library.')
 (OUT/'order_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
 pairs=csvread(ROOT/'outputs/grid85_mechanism/equal_budget_pairs.csv');groups=defaultdict(list);unscored=[]
 for r in pairs:
  if r['local_reward_gain']=='' or r['global_reward_gain']=='':
   r['signed_local_minus_global_gain']=None;unscored.append(r);continue
  r['signed_local_minus_global_gain']=float(r['local_reward_gain'])-float(r['global_reward_gain']);groups[(r['cohort'],r['date'])].append(r)
 dump('equal_budget_unscored.csv',unscored)
 dates=[]
 for (cohort,date),rs in groups.items():
  x=[r['signed_local_minus_global_gain'] for r in rs]
  dates.append(dict(cohort=cohort,date=date,triggered_states=len(rs),local_wins=sum(v>1e-6 for v in x),global_wins=sum(v< -1e-6 for v in x),mean_local_minus_global_gain=statistics.mean(x),min_gain=min(x),max_gain=max(x)))
 dump('equal_budget_by_date.csv',dates);summary=[]
 for cohort in ['old13','new6']:
  ds=[d for d in dates if d['cohort']==cohort];xs=[d['mean_local_minus_global_gain'] for d in ds]
  summary.append(dict(cohort=cohort,dates_with_completion=len(ds),date_macro_gain=statistics.mean(xs),date_cluster_bootstrap95=interval(xs)))
 (OUT/'reward_uncertainty.json').write_text(json.dumps(dict(results=summary,interpretation='Descriptive date-cluster bootstrap; few exposed dates, no significance or independent confirmation claim. No per-state independence assumption.'),ensure_ascii=False,indent=2),encoding='utf-8')
 # Exact cohort/reference labels prevent mixing the six-scene FULL speedup with old13 FIXED512 P95.
 lat=[]
 for stage in [81,84,88]:
  S=read(ROOT/f'outputs/grid{stage}/finished.json')['summaries']
  for s in S:
   folder=ROOT/f'outputs/grid{stage}/runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
   with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(l) for l in f]
   calls=[read(p) for p in folder.glob('call[0-9][0-9][0-9].json')];active={c['step'] for c in calls};a=[x['decision_s'] for x in steps];b=[x['decision_s'] for x in steps if x['step'] in active]
   lat.append(dict(stage=f'GRID{stage}',cohort='old13' if stage==88 else 'generated6_seed20261016',method=s['method'],date=s['week'],repeat=s['repeat'],controller_s=s['controller_s'],steps=len(steps),search_steps=len(b),all_p50=float(np.percentile(a,50)),all_p95=float(np.percentile(a,95)),all_p99=float(np.percentile(a,99)),search_p50=float(np.percentile(b,50)) if b else None,search_p95=float(np.percentile(b,95)) if b else None,search_p99=float(np.percentile(b,99)) if b else None))
 dump('existing_all_and_search_latency.csv',lat)
 # Public operating-load bins and candidate counts use saved observations, not hidden suffix outcomes.
 import sys
 sys.path.insert(0,str(ROOT/'work/grid08'));from common import make_env
 env=make_env();obs0=env.reset();before=int(env.nb_highres_called);states=csvread(ROOT/'outputs/grid85_mechanism/offline_full_calls.csv');full={ (r['cohort'],r['date'],r['call']):r for r in states if r['method']=='FULL'}
 bins=[]
 try:
  for r in pairs:
   src=ROOT/full[(r['cohort'],r['date'],r['call'])]['source'];call=read(src)
   with np.load(src.with_name(src.stem+'_state.npz')) as z:obs=obs0.copy();obs.from_vect(z['observation'])
   rho=float(obs.rho.max());M=len(call['order']);loadbin='<1.0' if rho<1 else '1.0-1.2' if rho<1.2 else '>=1.2';sizebin='<850' if M<850 else '>=850'
   bins.append(dict(cohort=r['cohort'],date=r['date'],call=r['call'],current_max_rho=rho,candidates=M,load_bin=loadbin,candidate_bin=sizebin,local_minus_global_gain=r['signed_local_minus_global_gain']))
  assert int(env.nb_highres_called)==before
 finally:env.close()
 dump('equal_budget_public_state_bins.csv',bins)
 grouped=[]
 for key in ['load_bin','candidate_bin']:
  gg=defaultdict(list)
  for r in bins:
   if r['local_minus_global_gain'] is not None:gg[(r['cohort'],r[key])].append(r)
  for (cohort,label),rs in gg.items():
   bydate=defaultdict(list)
   for r in rs:bydate[r['date']].append(r['local_minus_global_gain'])
   xs=[statistics.mean(v) for v in bydate.values()]
   grouped.append(dict(cohort=cohort,grouping=key,bin=label,states=len(rs),dates=len(xs),local_wins=sum(r['local_minus_global_gain']>1e-6 for r in rs),global_wins=sum(r['local_minus_global_gain']< -1e-6 for r in rs),date_macro_gain=statistics.mean(xs)))
 dump('equal_budget_grouped.csv',grouped)
 train=read(ROOT/'outputs/grid77/records.json');families={s:sorted({r['family'] for r in train if r['split']==s}) for s in ['train','calibration']}
 assert not(set(families['train'])&set(families['calibration']))
 exposed=read(ROOT/'outputs/grid88/design.json')['weeks'];evaluate_families={x.rsplit('_',1)[1] for x in exposed}
 # Existing source defines family labels; preserve exact labels rather than infer unseen-date independence from a filename.
 meta=dict(states=25,features=read(ROOT/'outputs/grid77/design.json')['features'],actions=['RETURN if incumbent exists','QUERY next64','DELEGATE complete FRAME at initial prefix only'],utility='q=clip(best strict-admissible reward / 2,0,1); U=q-expm1(log1p(cost_level)*queries/M); transition reward=U_next-U_current; STOP reward0',network='25-64-32-3 ReLU DoubleDQN',training_states=sum(r['split']=='train' for r in train),calibration_states=sum(r['split']=='calibration' for r in train),family_sets=families,calibration=read(ROOT/'outputs/grid77/selected.json'),new_positive_cost=.0025,no_test_selection=True,warning='Adapted action space includes FRAME; original tasks A*/RRT* not fully reproduced. Original calibration quality match failed. Zero cost was selected after three predeclared alternatives, not assigned arbitrarily.')
 (OUT/'meta_interface_and_split.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'方法性质与停止规则说明.md').write_text('''# 方法性质与停止规则

设当前候选代表集合有限，查询批次大小为正；可用性按正文严格筛选规则确定，排序键为float32预测奖励与动作编号破并列。

1. **不重复查询。** 全局阶段只访问序列中尚未访问的后续元素，补查集合显式排除已查询集合。因此一次搜索中每个代表最多查询一次，总查询数至多为候选数。
2. **最佳已观测选择。** 每次查询后，仅当新候选可用且排序键更好时更新当前最佳。由查询次数归纳，返回候选是全部已查询可用候选中排序键最大的；集合为空则返回空动作。
3. **已观测奖励单调性。** 新增候选不会删除旧候选，当前最佳只向更优排序键更新。因此其float32奖励不下降。同奖励时使用固定编号规则。
4. **有限终止与边界。** 每个非空批次推进序列或消耗有限补查集合，故算法终止；最坏情况下查询全部候选。上述性质针对单次搜索的已观测集合，不提供全局最优、固定节省比例或长期闭环成本保证。

负载率0.9沿用基础控制器的分支阈值。初始前缀尚未达到这一阈值时，继续扩展优先寻找能进一步降低负载率的动作；找到达标候选后可结束全局扩展，再按归属执行必要补查。这是面向计算开销的满意解停止规则，不代表该候选的预测奖励已全局最大。初始已达阈值的分支继续核查奖励改善，最低核查量和连续停滞批数控制搜索深度。前缀、批次、最低核查量和停滞数为经验设置，本次单因素检查不用于挑选新默认值。

完整周成本容差0.5%是本研究预先采用的投资筛查尺度，不是电网安全标准或等价性界限。除达标计数外，继续报告逐日期成本连续变化、运行时长和完成数。FULL是共同参考搜索流程，不作为长期最优控制器。
''',encoding='utf-8')
 (OUT/'既有证据补充说明.md').write_text('''# 既有证据补充

原始ID顺序实际承接作者动作库的教师频次降序，运行时筛选和等价合并保留该相对次序；频次是外部先验，不能解读为当前状态下的收益预测。新顺序检查保留动作ID及平局规则，只改变访问次序。

GRID85已经计算同站与等预算全局扩展的奖励遗憾及胜负幅度，新增按日期汇总与公开状态分组；置信范围按日期簇重采样。新六日期存在幅度更大的反向案例，不把较高动作一致率转述为更小平均奖励遗憾。

65.10%查询减少与14.98%完整控制时间减少来自GRID81六日期FRAME对FULL；0.42%的全部步P95减少来自GRID88原十三日期FRAME对固定512。两组不得混用。当前另列搜索触发步P95及P99，零搜索日期保持缺失而非记为零。
''',encoding='utf-8')
 dump('技术对照与任务边界.csv',[
  dict(对象='固定512 本实验',停止='前min(512,M)个代表查询结束',后续对象='相同全局顺序',搜索控制训练='无',适用范围='相同电网接口',来源='work/grid86/policies.py'),
  dict(对象='FRAME 本实验',停止='分支阈值和连续无改进反馈',后续对象='全局续查及当前最佳动作所属变电站',搜索控制训练='无额外训练',适用范围='相同电网接口',来源='work/grid70/policies.py'),
  dict(对象='META 电网适配',停止='学习策略返回动作',后续对象='全局续查64或初始委托完整FRAME',搜索控制训练='DoubleDQN',适用范围='本实验电网适配',来源='work/grid77/metacontrol.py'),
  dict(对象='Bhatia等 ICAPS2022原工作',停止='元控制策略选择停止',后续对象='底层规划器搜索及元策略调整内部参数',搜索控制训练='深度强化学习',适用范围='原文AWA*和RRT*任务',来源='outputs/grid83_alignment/ICAPS2022对标与贡献重写方案.md')])
 print(json.dumps(dict(passed=True,teacher_frequency_sorted=True,existing_latency_runs=len(lat),state_bins=len(bins),new_simulations=0),ensure_ascii=False))
if __name__=='__main__':main()
