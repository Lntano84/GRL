"""Read-only audit of gate label stakes, hierarchical sampling and date support."""
from pathlib import Path
import collections,hashlib,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid195_gate_label_audit';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID195_gate_label_audit'
OLD=R/'outputs/grid190_extension_gate';MIX=R/'outputs/grid192_gate_support';PRE=R/'outputs/grid189_baseline_extension'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
def prepare():
    assert not O.exists();O.mkdir(parents=True)
    save('design.json',dict(id='GRID195_READONLY_GATE_LABEL_AUDIT',authorization='User继续; inspect existing gate labels before proposing further training.',
        scope='All12 fit pools from existing4folds x3 frozenNETseeds, STOP and actualpreSTOPCONT supports. Existing held STOP outcomes only for diagnostic comparison after thresholds fixed on that pool fitSTOP.',
        fixed_metrics='Reconstruct every gain/change/nextNET label; independent date/call/start/position sampling masses; concentration in dates/calls, effective count1/sumsquaredshares; positive reward weighted95percentile from fitSTOP, min/max positive; ratio ofreward share tobinarypositive label mass. All rows/seeds/dates retained.',
        weights='Compare current hierarchical masses with raw uniform row masses descriptively. For potential utilityweightedBCE diagnose mass proportional originalmass*abs(nextNET), not fitting or choosing it. Effective counts describe this finite data only.',
        high_stake_rule='Threshold weighted95percentile ofpositive gain in currentfitSTOP only, no held quantile. Ties retained as >=threshold; no threshold sweep or decision policy. Compute STOP/CONT coverage and existing190/193 held selections.',
        old_negative_control='GRID120 utilityweightedclassification improved overunweightedSIGN but failed tobeat originalHuber. Newgate is boundedafterNETSTOP rather than replacing wholepolicy; no inference itwillwork here.',
        limits='Teacherpath localreward/fee distinct from closedloopweek quality. Prefixes are differentstates, not tobe blindlydeduplicated. OOFstate inspection exposed,not blindtest. No fitting,NETforward,physics,newdata,final8 or paper.'))
    files=[W/'run.py',O/'design.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid113_relational_learning/train_metadata.json',PRE/'fit_stop_rows.json',PRE/'fit_support.json',OLD/'gate_predictions.json',OLD/'complete.json',R/'outputs/grid193_gate_precondition/gate_predictions.json',R/'outputs/grid193_gate_precondition/complete.json',R/'outputs/grid194_gate_case_attribution/complete.json']
    for d in [OLD,MIX]:files+=list((d/'fit_features').glob('*.npz'))
    p=R/'outputs/grid120_decision_objective'
    files += [f for f in p.iterdir() if f.is_file() and f.suffix in ['.md','.json']]
    save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID195_PLANNED')
def hierarchy(rows):
    dates={r['date'] for r in rows};gd=collections.defaultdict(set);gs=collections.defaultdict(set);positions=collections.Counter(r['sample_id'] for r in rows)
    for r in rows:gd[r['date']].add(r['group']);gs[r['group']].add(r['sample_id'])
    w=np.array([1/(len(dates)*len(gd[r['date']])*len(gs[r['group']])*positions[r['sample_id']]) for r in rows]);assert abs(w.sum()-1)<1e-10
    return w
def weighted_quantile(x,w,q):
    order=np.argsort(x,kind='stable');xx=x[order];ww=w[order];return float(xx[np.searchsorted(np.cumsum(ww),q*ww.sum(),side='left')])
def concentration(rows,weights):
    w=np.array(weights,float);total=float(w.sum())
    if total==0:return dict(total=0.,dates=0,calls=0,max_date_share=0.,max_call_share=0.,effective_dates=0.,effective_calls=0.)
    ds=collections.defaultdict(float);gs=collections.defaultdict(float)
    for r,v in zip(rows,w):ds[r['date']]+=float(v/total);gs[r['group']]+=float(v/total)
    ds={k:v for k,v in ds.items() if v>0};gs={k:v for k,v in gs.items() if v>0}
    return dict(total=total,dates=len(ds),calls=len(gs),max_date_share=max(ds.values()),max_call_share=max(gs.values()),effective_dates=1/sum(v*v for v in ds.values()),effective_calls=1/sum(v*v for v in gs.values()),per_date=ds,per_call=gs)
def summarize(rows,w):
    gains=np.array([r['gain'] for r in rows]);u=np.array([r['NET'] for r in rows]);y=gains>0
    assert y.any();positive_mass=float(w@y);q95=weighted_quantile(gains[y],w[y],.95);top=gains>=q95;reward=w*gains
    return dict(rows=len(rows),starts=len({r['sample_id'] for r in rows}),calls=len({r['group'] for r in rows}),dates=len({r['date'] for r in rows}),positive_rows=int(y.sum()),positive_NET_rows=int((u>0).sum()),positive_mass=positive_mass,positive_NET_mass=float(w@(u>0)),positive_gain_min=float(gains[y].min()),positive_gain_median=weighted_quantile(gains[y],w[y],.5),positive_gain_p95=q95,positive_gain_max=float(gains.max()),positive_gain_span=float(gains.max()/gains[y].min()),
        fit_p95_tail_positive_label_share=float(w[top].sum()/positive_mass),fit_p95_tail_reward_share=float(reward[top].sum()/reward.sum()),
        current_mass=concentration(rows,w),uniform_row_mass=concentration(rows,np.full(len(rows),1/len(rows))),binary_positive_mass=concentration(rows,w*y),reward_mass=concentration(rows,reward),abs_NET_weighted_mass=concentration(rows,w*abs(u)))
def evaluate():
    verify();assert not (O/'complete.json').exists()
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'];cm={c['sample_id']:c for c in cases};lookup={(v['sample_id'],v['position']):i for i,v in enumerate(meta)}
    task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale']
    oldpred=read(OLD/'gate_predictions.json');newpred=read(R/'outputs/grid193_gate_precondition/gate_predictions.json');newidx={(r['seed'],r['sample_id']):r for r in newpred};original=read(PRE/'fit_stop_rows.json');pools=[];labeled=[];held=[];mass_checks=0
    def label(idx):
        m=meta[int(idx)];c=cm[m['sample_id']];p=m['position'];gain=float(c['gains'][p]);queries=int(c['queries'][p]);change=bool(c['chosen'][p]!=c['chosen'][p+1]);assert gain>=0 and queries==m['next_queries'];assert change==(gain>0)
        return dict(metadata_index=int(idx),sample_id=c['sample_id'],date=c['date'],group=c['physical_state_group'],position=p,gain=gain,queries=queries,NET=(gain-fee*queries)/scale,action_change=change)
    for f in range(4):
        for s in range(3):
            z=np.load(OLD/'fit_features'/f'f{f}_s{s}.npz');mz=np.load(MIX/'fit_features'/f'f{f}_s{s}.npz');n=len(z['ids']);assert n==int(mz['original_STOP_count']);rr=[label(i) for i in z['ids']];cr=[label(i) for i in mz['ids'][n:]];w=hierarchy(rr);cw=hierarchy(cr)
            assert np.allclose(w,z['mass'],atol=1e-12,rtol=1e-12);assert np.allclose(.5*np.r_[w,cw],mz['mass'],atol=1e-12,rtol=1e-12);assert np.array_equal(z['y'],[r['action_change'] for r in rr]);assert np.array_equal(mz['y'][n:],[r['action_change'] for r in cr]);mass_checks+=1
            orows=[r for r in original if r['fold']==f and r['seed']==s];assert len(orows)==n
            for r,p in zip(rr,orows):assert r['metadata_index']==p['metadata_index'] and r['gain']==p['private_label']['next_gain'] and abs(r['NET']-p['private_label']['next_NET'])<1e-12
            ss=summarize(rr,w);cs=summarize(cr,cw);tau=ss['positive_gain_p95'];fitdates={r['date'] for r in rr};prs=[p for p in oldpred if p['fold']==f and p['seed']==s];hr=[]
            for p in prs:
                r=label(lookup[p['sample_id'],p['position']]);assert r['date'] not in fitdates and r['gain']==p['actual_next_gain'];q=newidx[s,p['sample_id']];r.update(fold=f,seed=s,fit_p95=tau,fit_max=ss['positive_gain_max'],at_least_fit_p95=r['gain']>=tau,above_fit_max=r['gain']>ss['positive_gain_max'],old_continue=p['gate_continue'],new_continue=q['gate_continue']);hr.append(r);held.append(r)
            hw=hierarchy(hr);hs=summarize(hr,hw);hg=np.array([r['gain'] for r in hr]);above=hg>=tau;fitabove=np.array([r['gain']>=tau for r in rr]);contabove=np.array([r['gain']>=tau for r in cr]);total=float(hw@hg)
            def coverage(rows,mask):return dict(rows=int(mask.sum()),starts=len({r['sample_id'] for r,t in zip(rows,mask) if t}),calls=len({r['group'] for r,t in zip(rows,mask) if t}),dates=len({r['date'] for r,t in zip(rows,mask) if t}))
            hm={key:dict(selected_rows=sum(r[key] and r['at_least_fit_p95'] for r in hr),weighted_reward_recall=float(sum(ww*r['gain'] for r,ww in zip(hr,hw) if r[key] and r['at_least_fit_p95'])/sum(ww*r['gain'] for r,ww in zip(hr,hw) if r['at_least_fit_p95'])) if any(r['at_least_fit_p95'] for r in hr) else None) for key in ['old_continue','new_continue']}
            pool=dict(fold=f,seed=s,STOP=ss,CONT=cs,HELD_STOP=hs,fit_only_threshold=tau,fit_high=coverage(rr,fitabove),cont_at_same_fit_threshold=coverage(cr,contabove),held_at_fit_threshold=coverage(hr,above),held_above_fit_max=coverage(hr,hg>ss['positive_gain_max']),held_high_reward_share=float((hw*hg)[above].sum()/total),held_high_selection=hm)
            pools.append(pool)
            for domain,aa,ww in [('STOP',rr,w),('CONT',cr,cw)]:
                for r,weight in zip(aa,ww):labeled.append(dict(r,fold=f,seed=s,domain=domain,mass=float(weight),fit_p95=tau))
    # Range/mean over12 pools is descriptive; overlapping fits are not12 independent datasets.
    def ranges(domain,key):
        arr=np.array([p[domain][key] for p in pools]);return dict(min=float(arr.min()),mean=float(arr.mean()),max=float(arr.max()))
    summary=dict(pools=12,fit_row_records=len(labeled),held_seed_prefix_records=len(held),domains={d:{k:ranges(d,k) for k in ['rows','positive_rows','positive_NET_rows','positive_mass','positive_gain_span','fit_p95_tail_positive_label_share','fit_p95_tail_reward_share']} for d in ['STOP','CONT','HELD_STOP']},
        fit_high_call_range=[min(p['fit_high']['calls'] for p in pools),max(p['fit_high']['calls'] for p in pools)],fit_high_date_range=[min(p['fit_high']['dates'] for p in pools),max(p['fit_high']['dates'] for p in pools)],cont_high_call_range=[min(p['cont_at_same_fit_threshold']['calls'] for p in pools),max(p['cont_at_same_fit_threshold']['calls'] for p in pools)],
        reward_effective_fit_calls=[min(p['STOP']['reward_mass']['effective_calls'] for p in pools),max(p['STOP']['reward_mass']['effective_calls'] for p in pools)],utility_weight_max_fit_call=[min(p['STOP']['abs_NET_weighted_mass']['max_call_share'] for p in pools),max(p['STOP']['abs_NET_weighted_mass']['max_call_share'] for p in pools)])
    save('fit_label_rows.json',labeled);save('held_diagnostic_rows.json',held);save('pool_statistics.json',pools);save('summary.json',summary)
    save('review.json',dict(passed=True,all12_current_STOP_and_mixture_sampling_masses_independently_reconstructed=True,all_fit_action_gain_NET_labels_reproduced=True,held_thresholds_fit_only=True,all_seed_date_and_prefix_records_retained=True,change_equals_strict_positive_gain_all_checked=True,new_fits=0,new_NET_forwards=0,new_physics=0))
    make_report(pools,summary)
    verify();save('complete.json',dict(passed=True,new_fits=0,new_NET_forwards=0,new_physics=0,all12pools=True,all4681eligibleheld_rows=True,no_new_policy_or_threshold_search=True));archive('complete')
    dest=R/'论文协作资料/FRAME_门控标签幅度_GRID195';dest.mkdir(exist_ok=False)
    for name in ['GRID195_report.md','summary.json','pool_statistics.json','review.json','complete.json']:shutil.copy2(O/name,dest/name);assert sha(O/name)==sha(dest/name)
    update_docs(pools,summary);print('GRID195_COMPLETE',json.dumps(summary,ensure_ascii=False))
def make_report(pools,summary):
    st=summary['domains']['STOP']; lines=['# GRID195：补查标签的收益幅度、采样权重与日期覆盖','',
        '本轮只核查已有标签，没有训练。原门控将下一批“改选”和“不改选”作为二元标签，本轮把局部奖励增量与固定查询费用重新接回这些标签，检查高价值机会在训练侧究竟有多少、是否集中。','',
        '## 前缀多会不会放大一个调用','',
        '原STOP门控已按日期、物理调用组、该组内起点分层均衡；CONT补充域还对同一起点的继续位置均分。12组权重全部由独立代码重建，并逐项匹配保存的训练权重。因此，同一调用起点多，并不直接增加该调用的总训练权重。','',
        '这里的多个前缀是不同可见搜索状态，不能因为来自同一次调用就直接删成一个样本。现有分层权重避免了数量主导，但不能解决有利模式来自太少日期的问题。','',
        '## 二元标签掩盖了多大的差别','',
        '|折/种子|STOP正例数|净收益正例数|最小正奖励|最大正奖励|幅度倍数|正例最高约5%权重对应的奖励占比|','|---|---:|---:|---:|---:|---:|---:|']
    for p in pools:
        z=p['STOP'];lines.append(f"|{p['fold']}/{p['seed']}|{z['positive_rows']}|{z['positive_NET_rows']}|{z['positive_gain_min']:.9g}|{z['positive_gain_max']:.9g}|{z['positive_gain_span']:.1f}|{100*z['fit_p95_tail_reward_share']:.2f}%|")
    lines+=['','尾部由各拟合池正例奖励的加权95分位数确定；包含阈值处的全部并列，因此尾部正例权重不必恰好5%。一个极小改选和一个大收益改选都标为1，二元监督本身不要求模型分清它们的价值。这个事实不等于已经证明加权训练有效。','',
        '## 大收益机会的覆盖与集中','',
        '|折/种子|STOP高收益调用组/日期|CONT达到同一阈值的调用组/日期|奖励权重有效STOP调用数|若按绝对NET加权，最大单调用权重|','|---|---:|---:|---:|---:|']
    for p in pools:
        z=p['STOP'];a=p['fit_high'];b=p['cont_at_same_fit_threshold'];lines.append(f"|{p['fold']}/{p['seed']}|{a['calls']}/{a['dates']}|{b['calls']}/{b['dates']}|{z['reward_mass']['effective_calls']:.2f}|{100*z['abs_NET_weighted_mass']['max_call_share']:.2f}%|")
    lines+=['','“有效调用数”定义为各调用归一化贡献份额平方和的倒数，用来描述集中程度，不是独立样本量或统计置信度。绝对NET加权列只计算潜在权重分布，没有执行加权拟合；它会重新分配原先均衡的物理调用权重。','',
        '## 留出停止点面对什么','',
        '|折/种子|达到拟合95分位的留出配对/调用组|高收益部分占留出可得奖励|GRID190抓住该部分奖励|GRID193抓住该部分奖励|超过拟合最大收益的留出配对|','|---|---:|---:|---:|---:|---:|']
    for p in pools:
        h=p['held_at_fit_threshold'];a=p['held_high_selection']['old_continue']['weighted_reward_recall'];b=p['held_high_selection']['new_continue']['weighted_reward_recall'];aa='—' if a is None else f'{100*a:.2f}%';bb='—' if b is None else f'{100*b:.2f}%'
        lines.append(f"|{p['fold']}/{p['seed']}|{h['rows']}/{h['calls']}|{100*p['held_high_reward_share']:.2f}%|{aa}|{bb}|{p['held_above_fit_max']['rows']}|")
    lines+=['','阈值只由相应折的拟合STOP标签确定，未用留出数据选择。留出统计是已暴露历史日期上的描述性核查，不能作为新的独立测试。12个池相互重叠，三个种子也共用物理记录，不能把这些表格行当作12次独立确认。','',
        '## 既有加权分类结果不能跳过','',
        'GRID120已经用绝对净收益加权BCE，与未加权正负分类及原Huber回归比较。旧加权模型比只判断正负好，但未超过原回归；它改变的是整段搜索策略，不能直接等同当前原NET停止后最多追加一批的门控。当前受保护结构不同，既不能用旧结果否定一切，也不能把同一加权公式当作未经尝试的新办法。','',
        '## 接下来先解决什么','',
        '当前数据不能只靠增加正例次数来改善。首先应把“识别改选机会”与“估计收益幅度”分开核验，并检查大收益信号能否在不同日期和调用间迁移。下一项训练若使用幅度信息，需与当前GRID190保留同一特征、原NET和追加边界；同时明确是否因加权改变了调用分布，保留全部对照与反例。','',
        '本轮没有选新阈值、组合模型、扫描损失或启动训练。局部即时奖励、带代理查询费用的效用、整周运行成本是三种口径，不能由本轮标签统计推出整周质量或实际加速。']
    (O/'GRID195_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
def update_docs(pools,summary):
    st=summary['domains']['STOP'];t=st['fit_p95_tail_reward_share'];msg=f'GRID195只读标签核查完成，12池STOP/CONT层次采样权重独立重现，非前缀多就放大调用总权重。STOP正例{int(st["positive_rows"]["min"])}–{int(st["positive_rows"]["max"])}，高收益尾部贡献奖励{100*t["min"]:.1f}%–{100*t["max"]:.1f}%；高收益STOP仅{summary["fit_high_call_range"]}调用组/{summary["fit_high_date_range"]}日期。保留GRID190和旧GRID120加权分类反例；不由标签差异直接归因所有失败，不自动训练/加权/组合/扩物理或最终8。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;txt=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID195门控标签幅度与采样审查\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':h,rest=txt.split('\n',1);txt=h+'\n'+entry+'\n'+rest
        else:txt+=entry
        p.write_text(txt,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID195后\n\n1. 保留GRID190与互补好例，既有分层采样已消除简单的前缀数放大，不盲目去重或扩大网络。\n2. 设计下一项幅度识别核验前，对照GRID120/123等旧结果；区分目标改变、调用权重改变、训练支持与跨日期迁移，不重复旧加权实验后包装为新机制。\n3. 新训练先通知固定范围；同一NET/特征/最多追加一批边界，全部种子、日期和反例保留；不自动新物理、最终8或论文。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID195门控标签幅度审查\n\n[报告](protocols/GRID195_gate_label_audit/complete/GRID195_report.md)；分层采样、收益集中及拟合侧阈值覆盖已核查，零拟合，保留GRID190。\n\n'.encode('utf8')+p.read_bytes())
if __name__=='__main__':globals()[sys.argv[1]]()
