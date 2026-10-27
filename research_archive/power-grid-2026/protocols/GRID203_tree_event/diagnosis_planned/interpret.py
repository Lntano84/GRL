"""Read-only decomposition of the frozen GRID203 versus GRID201 choices."""
from pathlib import Path
import collections,json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read

def prepare():
    assert read(O/'complete.json')['passed'] and not (O/'interpretation_plan.json').exists()
    plan=dict(scope='All4681 eligible saved choices GRID203 vs201;zero fits/forwards/physics.',groups='NEW_ONLY OLD_ONLY BOTH NEITHER;actual macro contributions conserve new-minus-old reward queries utility. Perseed/date/call/prefix all retained.',query_bins='1-4,5-16,17-32,33-64 descriptive;not fitted or deployed trigger.',limits='Actual strategy decomposition is distinct from matched-query comparison. Shared teacher starts and seeds not independent. No threshold/fee/label changes or new training.')
    C.save('interpretation_plan.json',plan);files=[Path(__file__),O/'complete.json',O/'gate_predictions.json',O/'family_summary.json',O/'oof_trajectories.json',C.OLD/'gate_predictions.json',C.OLD/'family_summary.json',O/'interpretation_plan.json'];C.save('interpretation_freeze.json',{p.relative_to(R).as_posix():C.sha(p) for p in files});C.archive('diagnosis_planned');print('GRID203_INTERPRETATION_PLANNED')

def evaluate():
    assert not (O/'interpretation.json').exists()
    for p,h in read(O/'interpretation_freeze.json').items():assert C.sha(R/p)==h
    C.trained();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];_,_,cases=C.B.load();nc=collections.Counter(c['date'] for c in cases);old={(r['seed'],r['sample_id']):r for r in read(C.OLD/'gate_predictions.json')};rows=[]
    for p in read(O/'gate_predictions.json'):
        prev=old[p['seed'],p['sample_id']];a,b=p['gate_continue'],prev['gate_continue'];assert p['actual_next_gain']==prev['actual_next_gain'] and p['actual_next_queries']==prev['actual_next_queries'];group='BOTH' if a and b else 'NEW_ONLY' if a else 'OLD_ONLY' if b else 'NEITHER';sign=int(a)-int(b);mass=1/(3*16*nc[p['date']]);q=p['actual_next_queries'];g=p['actual_next_gain'];rows.append(dict(seed=p['seed'],sample_id=p['sample_id'],date=p['date'],call=p['physical_state_group'],group=group,queries=q,gain=g,profitable=p['actual_next_NET']>0,p_old=prev['event_probability'],p_new=p['event_probability'],mu=p['conditional_normalized_gain'],known_charge=p['known_charge'],macro_delta_gain=sign*mass*g,macro_delta_queries=sign*mass*q,macro_delta_utility=sign*mass*(g-fee*q)/scale))
    def summarize(rr):
        return dict(pairs=len(rr),calls=len({r['call'] for r in rr}),dates=len({r['date'] for r in rr}),unique_prefixes=len({r['sample_id'] for r in rr}),positive_gain=sum(r['gain']>0 for r in rr),profitable=sum(r['profitable'] for r in rr),raw_queries=sum(r['queries'] for r in rr),mean_batch_queries=float(np.mean([r['queries'] for r in rr])) if rr else 0.,macro_delta_gain=sum(r['macro_delta_gain'] for r in rr),macro_delta_queries=sum(r['macro_delta_queries'] for r in rr),macro_delta_utility=sum(r['macro_delta_utility'] for r in rr))
    groups={k:summarize([r for r in rows if r['group']==k]) for k in ['NEW_ONLY','OLD_ONLY','BOTH','NEITHER']};new=read(O/'family_summary.json')['GATE'];prev=read(C.OLD/'family_summary.json')['GATE']
    for k,metric in [('macro_delta_gain','mean_gain'),('macro_delta_queries','mean_queries'),('macro_delta_utility','mean_net_utility')]:assert abs(sum(r[k] for r in rows)-(new[metric]-prev[metric]))<1e-10
    bins=[]
    for k in ['NEW_ONLY','OLD_ONLY']:
        for lo,hi in [(1,4),(5,16),(17,32),(33,64)]:bins.append(dict(group=k,query_range=[lo,hi],**summarize([r for r in rows if r['group']==k and lo<=r['queries']<=hi])))
    callrows=[]
    for call in sorted({r['call'] for r in rows}):
        rr=[r for r in rows if r['call']==call];seed=[dict(seed=s,delta_utility=sum(r['macro_delta_utility'] for r in rr if r['seed']==s)*3) for s in range(3)];callrows.append(dict(call=call,**summarize(rr),seeds=seed,positive_seeds=sum(r['delta_utility']>0 for r in seed),negative_seeds=sum(r['delta_utility']<0 for r in seed)))
    seeds=[]
    for s in range(3):
        rr=[r for r in rows if r['seed']==s];z=summarize(rr)
        for k in ['macro_delta_gain','macro_delta_queries','macro_delta_utility']:z[k]*=3
        seeds.append(dict(seed=s,**z))
    dates=[dict(date=d,**summarize([r for r in rows if r['date']==d])) for d in sorted(nc)]
    C.save('interpretation_rows.json',rows);result=dict(groups=groups,query_bins=bins,per_seed=seeds,per_date_contributions=dates,per_call_contributions=callrows,all4681_decomposed=True,contribution_conservation=True,new_fits=0,new_forwards=0,new_physics=0);C.save('interpretation.json',result)
    lines=['# GRID203：改善来自哪里','', '本补充只读已固定的新旧决策，未训练、改门槛或重做仿真。NEW_ONLY表示新版追加而201不追加；OLD_ONLY相反。两组带来的奖励、查询与代理净效用变化按原日期宏平均加权，和整体差值逐项守恒。','', '|选择组|种子—起点配对|调用|日期|正收益/回本配对|平均一批查询|奖励变化|查询变化|净效用变化|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,z in groups.items():lines.append(f"|{k}|{z['pairs']}|{z['calls']}|{z['dates']}|{z['positive_gain']}/{z['profitable']}|{z['mean_batch_queries']:.3f}|{z['macro_delta_gain']:+.9f}|{z['macro_delta_queries']:+.6f}|{z['macro_delta_utility']:+.9f}|")
    lines+=['', '旧版取消的追加即使有正奖励，也可能不足以抵消查询费用；新版新增的追加大多零奖励，仍需看少数收益能否覆盖全部查询支出。下面只按预定区间描述成本分布，不将区间作为在线触发规则。','', '|选择组|每批查询范围|配对|正收益/回本|净效用变化|','|---|---|---:|---:|---:|']
    for z in bins:lines.append(f"|{z['group']}|{z['query_range'][0]}–{z['query_range'][1]}|{z['pairs']}|{z['positive_gain']}/{z['profitable']}|{z['macro_delta_utility']:+.9f}|")
    lines+=['','## 实际策略的种子结果','', '|种子|奖励变化|查询变化|净效用变化|','|---|---:|---:|---:|']
    for z in seeds:lines.append(f"|{z['seed']}|{z['macro_delta_gain']:+.9f}|{z['macro_delta_queries']:+.6f}|{z['macro_delta_utility']:+.9f}|")
    lines+=['', '这里是实际固定策略差值。与报告中的同预期查询结果不同：逐日期匹配会压低两种策略不同位置的查询支出，因此两种口径的种子方向可能不同，不选择其中更好看的口径。', '', '## 主要正向与反向调用','']
    for label,rr in [('正向',sorted(callrows,key=lambda r:-r['macro_delta_utility'])[:5]),('反向',sorted(callrows,key=lambda r:r['macro_delta_utility'])[:5])]:
        lines.append(label+'：')
        for z in rr:lines.append(f"- {z['call']}：整体净效用贡献{z['macro_delta_utility']:+.9f}，种子正/负为{z['positive_seeds']}/{z['negative_seeds']}；{z['pairs']}个相关配对。")
        lines.append('')
    lines+=['全部调用、日期和4681行见interpretation.json与interpretation_rows.json。调用内多起点及跨种子共享教师反馈，不能当作多个独立成功场景。','', '当前证据支持：事件预测模块更换后，查询分配确实改变，并带来局部净效用改善。仍有相同预期查询下的种子1退化和大量零收益查询，不能称补查时机已经学准。下一步应核查错误集中在何种已有输入和费用关系，先保留现有改善；不自动提高门槛、扩网络或重新训练。']
    (O/'改善来源与反例.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.archive('diagnosis_complete');dest=R/'论文协作资料/FRAME_浅树概率_GRID203'
    for n in ['改善来源与反例.md','interpretation.json','interpretation_plan.json']:shutil.copy2(O/n,dest/n)
    text='GRID203改善来源：新增追加680配对、取消114配对，分开核算收益/查询并逐项守恒；两组成本分布和全部调用反例保留。实际策略差与逐日期同预期查询差不混用，不据新增次数或正例率单项定有效。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID203选择变化解释\n\n'+text+'\n',encoding='utf8',newline='\n')
    print('GRID203_INTERPRETED',json.dumps(dict(groups=groups,seeds=seeds,bins=bins),ensure_ascii=False))

if __name__=='__main__':globals()[sys.argv[1]]()
