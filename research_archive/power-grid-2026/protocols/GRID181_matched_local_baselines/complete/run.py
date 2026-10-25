"""Current frozen consensus against the two historical nonlearning local baselines."""
from pathlib import Path
import collections,hashlib,importlib.util,json,math,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid181_matched_local_baselines';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID181_matched_local_baselines'
sp=importlib.util.spec_from_file_location('g181_current',R/'work/grid179_cheap_consensus/run.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C);read=C.read;sha=C.sha;H=C.H
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    assert not O.exists();O.mkdir(parents=True);C.verify();assert read(C.O/'complete.json')['passed_audit'];assert read(R/'outputs/grid180_consensus_runtime/complete.json')['passed_audit']
    save('design.json',dict(id='GRID181_MATCHED_LOCAL_NONLEARNING',authorization='User continue; compare thecurrentfixedconsensus to existing nonlearning references, no newrule/model.',
        references=['BATCH1:one min64knownbatchthenstop','FULL_LOCAL:allremaininglockedstationbatches'],
        population='Sameall2378originaltrainingstarts/16dates/4OOFfolds;3frozenlearnedseeds. Baselines deterministic2378,not3independent repetitions.',
        task='Oldfee/gain_scale andobservedbestactionunchanged. This is saved FULL-teacherlocal-search replay,not original FRAMEphysicaltrajectory or wholeweek quality.',
        metrics='Netutility,queries,fraction,gaincapture,physicalgroupmean;allperseed/date andfirstdifferencecost/gaincontributions. ReproduceGRID127original BATCH1/FULLvalues.',
        prohibitions=['newfits/forwards','baselinebudgetorfeechange','newdata/physical/final8','paper','adoption from proxyonly']))
    historical=R/'outputs/grid127_mean_gain_objective/nonlearning_comparison/comparison.json';paths=[historical,C.O/'oof_trajectories.json',C.O/'family_summary.json',C.O/'complete.json',R/'outputs/grid102_search_rl/cases.pkl',C.D.O/'design.json',R/'outputs/grid180_consensus_runtime/family_summary.json',R/'outputs/grid180_consensus_runtime/key_changes.json',Path(__file__),O/'design.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
    cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'];task=read(C.D.O/'design.json');fee=task['cost_per_query'];scale=task['gain_scale'];control=[];checks=collections.Counter()
    for c in cases:
        for name,p in [('BATCH1',1),('FULL_LOCAL',len(c['states']))]:
            go=[j<p for j in range(len(c['states']))];r=H.trajectory(c,go,name,fee,scale);assert r['batches']==p;assert r['queries']==sum(int(v) for v in c['queries'][:p]);assert abs(r['gain']-math.fsum(float(v) for v in c['gains'][:p]))<1e-12;assert r['chosen_pool_id']==c['chosen'][p];control.append(r);checks['baseline_searches']+=1
    ss={name:H.summarize([r for r in control if r['method']==name]) for name in ['BATCH1','FULL_LOCAL']};old=read(historical)['oof']
    for name in ss:
        for key in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture']:assert abs(ss[name]['macro'][key]-old[name]['macro'][key])<1e-12,(name,key)
        checks['historical_baseline_macros']+=5
    learned=read(C.O/'oof_trajectories.json');family=read(C.O/'family_summary.json');family.update({name:r['macro'] for name,r in ss.items()});bymethod={(r['method'],r['sample_id']):r for r in learned+control};datecount=collections.Counter(c['date'] for c in cases);events=[];pairs=[]
    for seed in [0,1,2]:
        rr=[r for r in learned if r['method']==f'CONSENSUS_seed{seed}'];s=H.summarize(rr);base=ss['BATCH1'];pairs.append(dict(seed=seed,relative_utility_vs_batch1=s['macro']['mean_net_utility']/base['macro']['mean_net_utility']-1,delta=s['macro']['mean_net_utility']-base['macro']['mean_net_utility'],per_date=[dict(date=a['date'],delta=a['mean_net_utility']-b['mean_net_utility']) for a,b in zip(s['per_date'],base['per_date'])]))
        for c in cases:
            a=bymethod[f'CONSENSUS_seed{seed}',c['sample_id']];b=bymethod['BATCH1',c['sample_id']];dg=a['gain']-b['gain'];qs=b['queries']-a['queries'];nd=a['net_utility']-b['net_utility'];assert abs(nd-(dg+fee*qs)/scale)<1e-12
            events.append(dict(seed=seed,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],kind='SKIP_FIRST_BATCH' if a['batches']==0 else ('EXTRA_BATCHES' if a['batches']>1 else 'SAME_ONE_BATCH'),consensus_batches=a['batches'],gain_delta=dg,saved_queries=qs,net_delta=nd,same_choice=a['chosen_pool_id']==b['chosen_pool_id'],weight=1/(3*16*datecount[c['date']])))
    directions=[]
    for kind in sorted({r['kind'] for r in events}):
        rr=[r for r in events if r['kind']==kind];directions.append(dict(kind=kind,pairs=len(rr),physical_groups=len({r['physical_state_group'] for r in rr}),positive_pairs=sum(r['net_delta']>0 for r in rr),negative_pairs=sum(r['net_delta']<0 for r in rr),net_contribution=math.fsum(r['weight']*r['net_delta'] for r in rr),gain_contribution=math.fsum(r['weight']*r['gain_delta']/scale for r in rr),query_saving_contribution=math.fsum(r['weight']*fee*r['saved_queries']/scale for r in rr)))
    con=family['CONSENSUS'];base=family['BATCH1'];full=family['FULL_LOCAL'];nd=con['mean_net_utility']-base['mean_net_utility'];assert abs(nd-sum(r['net_contribution'] for r in directions))<1e-12
    dates=[dict(date=x['date'],delta=float(np.mean([next(v['delta'] for v in p['per_date'] if v['date']==x['date']) for p in pairs]))) for x in pairs[0]['per_date']]
    key=dict(relative_utility_vs_batch1=nd/base['mean_net_utility'],query_reduction_vs_batch1=1-con['mean_queries']/base['mean_queries'],gain_capture_percentage_point_delta_vs_batch1=100*(con['gain_capture']-base['gain_capture']),relative_utility_vs_full=con['mean_net_utility']/full['mean_net_utility']-1,query_reduction_vs_full=1-con['mean_queries']/full['mean_queries'],positive_seeds_vs_batch1=sum(p['delta']>0 for p in pairs),positive_dates_vs_batch1=sum(d['delta']>0 for d in dates),not_same_local_gain_quality=True)
    save('control_trajectories.json',control);save('control_summary.json',ss);save('family_summary.json',family);save('all7134_batch1_pairs.json',events);save('direction_decomposition.json',directions);save('paired_summary.json',dict(pairs=pairs,per_date=dates));save('key_changes.json',key)
    lines=['# GRID181：与不学习方法的同起点局部比较','',f"固定共识策略相对固定一批BATCH1，局部净效用提高{key['relative_utility_vs_batch1']:.2%}，候选查询减少{key['query_reduction_vs_batch1']:.2%}，但收益保留率降低{abs(key['gain_capture_percentage_point_delta_vs_batch1']):.2f}个百分点。这是给定原查询费用下的收益—计算取舍，不能写成同质量优势。",'',
        '本次没有训练、预测或新策略。BATCH1与FULL_LOCAL都是已有对照，在全部2378个相同起点重新记账，原GRID127的效用、收益、查询及比例数值精确重现。BATCH1固定查下一批（最多64），FULL_LOCAL固定查完原锁定站的所有剩余候选。二者确定性结果只算一份，不把与三个模型种子的配对当成独立三次实验。','',
        '| 方案 | 局部净效用 | 平均查询 | 完整局部查询比例 | 收益保留率 | 调用均衡效用 |','|---|---:|---:|---:|---:|---:|']
    for name in ['FULL_LOCAL','BATCH1','FACTOR','STATE4_HELPER','POWER8_HELPER','CONSENSUS']:
        v=family[name];lines.append(f"| {name} | {v['mean_net_utility']:.8f} | {v['mean_queries']:.3f} | {v['query_fraction']:.2%} | {v['gain_capture']:.2%} | {v['physical_group_mean_utility']:.8f} |")
    lines+=['','## 相对固定一批，省掉了什么、补回了什么','','| 首次路径类型 | 配对数 | 净正/净负 | 收益贡献 | 费用节省贡献 | 总效用贡献 |','|---|---:|---:|---:|---:|---:|']
    for r in directions:lines.append(f"| {r['kind']} | {r['pairs']} | {r['positive_pairs']}/{r['negative_pairs']} | {r['gain_contribution']:+.8f} | {r['query_saving_contribution']:+.8f} | {r['net_contribution']:+.8f} |")
    lines+=['','| 种子 | 相对BATCH1效用变化 |','|---|---:|']
    for r in pairs:lines.append(f"| {r['seed']} | {r['relative_utility_vs_batch1']:+.2%} |")
    lines+=['',f"三个种子中{key['positive_seeds_vs_batch1']}个提高，16日期中{key['positive_dates_vs_batch1']}个平均提高。全量正反变化均保留，不能只用跳过零收益的成功例子解释全部成本。",'',
        '## 此时最需要验证什么','',
        '现在已经有一个可保留的局部学习信号：电气小模型的判断能通过固定共识传递到组合中。但相对不学习的固定一批，减少查询同时减少了局部增益。减少的奖励是否影响整周存活和运行成本，需要真实闭环核查；局部奖励保留率既不能直接当成整周质量，也不能被忽略。',
        '', 'GRID180只核对指定的输入/模型流水线，它不包含候选仿真与共同控制器。相对F增加查询时，不能把模型调用更少直接写成控制更快。旧FULL计时未混入本轮。',
        '', '下一步应准备可复现的冻结部署候选和小闭环比较：保留全部三个种子，与原未学习FRAME、固定一批及原FACTOR分别比较；先完成训练/接口资格，再给出具体回放范围供用户决定。这里没有启动新的物理回放或扩大数据，最终8和论文仍未动。']
    (O/'GRID181_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
    C.verify();save('review.json',dict(passed=True,checks=dict(checks),all7134_pairs_accounted=True,all16dates3seeds_retained=True,new_fits=0,new_model_forwards=0,new_physics=0,final8_absent=True));save('decision.json',dict(local_positive_signal_retained=True,local_same_quality_not_established=True,ready_for_deployment_preparation=True,ready_to_scale=False,physical_runs_started=False,next='Prepare frozen fulltrainweights andonlineinterfacequalification without newphysical runs;propose concrete loop scope foruserdecision.'));save('complete.json',dict(passed_audit=True,new_fits=0,new_physics=0,final8_absent=True))
    dest=A/'complete';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');collab=R/'论文协作资料/FRAME_当前学习与固定补查比较_GRID181';collab.mkdir(exist_ok=False)
    for n in ['GRID181_report.md','key_changes.json','family_summary.json','paired_summary.json','direction_decomposition.json','review.json','decision.json','complete.json']:shutil.copy2(O/n,collab/n)
    msg=f"GRID181当前共识与历史BATCH1/FULL_LOCAL同2378起点比较并精确重现基线：对BATCH1局部效用+{key['relative_utility_vs_batch1']:.2%}、查询少{key['query_reduction_vs_batch1']:.2%}、收益保留率{key['gain_capture_percentage_point_delta_vs_batch1']:+.2f}pp，{key['positive_seeds_vs_batch1']}/3种子、{key['positive_dates_vs_batch1']}/16日期正。保留学习信号，明确不是同局部质量优势。准备冻结可部署权重与接口资格，再提具体小闭环范围等待决定；当前零新物理/拟合/最终8。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;oldtxt=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID181与不学习方法比较\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=oldtxt.split('\n',1);oldtxt=a+'\n'+entry+'\n'+b
        else:oldtxt+=entry
        p.write_text(oldtxt,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 准备冻结部署与小闭环范围\n\n1. 保留179共识正向，180指定输入/模型计算节省与额外查询分列；181固定一批代价明确，不当作同质量优势。\n2. 下一项只准备完整16训练日期的小模型权重与已有F权重核对、无物理的在线接口资格；新拟合前通知，不扫组合/阈值。\n3. 资格后提供明确候选、未学习FRAME/BATCH1/F对照、全部3种子、已有4开发日期的回放范围，让用户决定新物理运行；当前不启动回放、不扩大数据/最终8，不改论文。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID181当前学习与固定补查\r\n\r\n[报告](protocols/GRID181_matched_local_baselines/complete/GRID181_report.md)；收益与查询取舍明确，准备部署资格，未启动新物理。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID181_COMPLETE',json.dumps(key))
if __name__=='__main__':main()
