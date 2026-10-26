"""Read-only separation of event and conditional amount bottlenecks."""
from pathlib import Path
import collections,hashlib,importlib.util,json,pickle,shutil,sys
import numpy as np
from sklearn.metrics import roc_auc_score
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid202_factor_bottleneck';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID202_factor_bottleneck';SRC=R/'outputs/grid201_tree_amount';FIT=R/'outputs/grid192_gate_support';LAB=R/'outputs/grid196_gain_readout/fit_labels';CACHE=R/'outputs/grid190_extension_gate'
sp=importlib.util.spec_from_file_location('grid202_fixed_helpers',R/'work/grid201_tree_amount/run.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
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
    M.B.H.sealed()
def prepare():
    assert not O.exists();O.mkdir(parents=True)
    save('design.json',dict(id='GRID202_FACTOR_BOTTLENECK',authorization='User继续: identifymissedvaluableopportunities beforetargetedinternalchange.',scope='All4681eligibleSTOPs/all12fixed201pairs plusfitSTOP/fitCONT cachedrows;no newfits or thresholds.',
        diagnostics='Eventprevalence/pmean/Brier/BCE/AUC allstates;conditionalmean/MSE only actualpositives;predictedmeanreturn overall. Allfixedfit/helddomains,date/call/start balanced.',
        missed_profit_partition='OnactualnextNET>0 butpolicySTOP only: EITHER if p=1holdingmu andmu=realizedgainholdingp eachcouldcrossfixedcost; EVENT_ONLY;AMOUNT_ONLY;JOINT otherwise. Realizedgain hindsight isnot trueconditionalmean;not causal/learnability proof or evaluated oraclepolicy.',
        boundaries='Break-evenp=cost/mu andmu=cost/p only algebraicfixedfee descriptors,no thresholdsearch. Zeroeventdoesnot implyconditionalmu targetzero;no zero labels added toamounthead.',
        limits='No NNforwards/physics/data/final8/paper/newFRAMEcomparison. Sharedteacherprefixes/seedsnotindependent. Allcounterexamples kept.'))
    files=[W/'run.py',R/'work/grid201_tree_amount/run.py',O/'design.json',SRC/'complete.json',SRC/'gate_predictions.json',SRC/'family_summary.json',R/'outputs/grid102_search_rl/design.json']+list((SRC/'models').iterdir())+list((FIT/'fit_features').glob('*.npz'))+list(LAB.glob('*.npz'))+list(CACHE.glob('held_f*.npz'))+[R/p for p in read(CACHE/'code_freeze.json')];save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID202_PLANNED')
def profile(p,logit,mu,y,w):
    event=y>0;posw=w[event]/w[event].sum();gain=w@y;return dict(rows=len(y),positive_mass=float(w@event),predicted_event_mean=float(w@p),event_Brier=float(w@((p-event)**2)),event_BCE=float(w@(np.logaddexp(0.,logit)-event*logit)),event_AUC=float(roc_auc_score(event,p,sample_weight=w)),actual_gain_mean=float(gain),predicted_gain_mean=float(w@(p*mu)),actual_positive_gain_mean=float(posw@y[event]),predicted_mu_on_positive=float(posw@mu[event]),conditional_positive_MSE=float(posw@((mu[event]-y[event])**2)),gain_weighted_event_probability=float(w@(y*p)/gain),predicted_gain_from_realized_zeros=float(w@((~event)*p*mu)))
def hierarchy(rows):
    ds={r['date'] for r in rows};gd=collections.defaultdict(set);gs=collections.defaultdict(set)
    for r in rows:gd[r['date']].add(r['physical_state_group']);gs[r['physical_state_group']].add(r['sample_id'])
    w=np.array([1/(len(ds)*len(gd[r['date']])*len(gs[r['physical_state_group']])) for r in rows]);assert abs(w.sum()-1)<1e-10;return w
def evaluate():
    verify();assert not (O/'complete.json').exists();pred=read(SRC/'gate_predictions.json');_,_,cases=M.B.load();nc=collections.Counter(c['date'] for c in cases);task=read(R/'outputs/grid102_search_rl/design.json');scale=task['gain_scale'];rows=[];pools=[]
    for p in pred:
        prob=p['event_probability'];mu=p['conditional_normalized_gain'];cost=p['known_charge'];g=p['actual_next_gain']/scale;profit=p['actual_next_NET']>0;event_repair=mu>cost;amount_repair=prob*g>cost
        if p['gate_continue']:category='APPEND_PROFITABLE' if profit else 'APPEND_UNPROFITABLE'
        elif profit:category='MISS_EITHER' if event_repair and amount_repair else 'MISS_EVENT_ONLY' if event_repair else 'MISS_AMOUNT_ONLY' if amount_repair else 'MISS_JOINT'
        else:category='STOP_POSITIVE_BUT_UNPROFITABLE' if g>0 else 'STOP_ZERO'
        mass=1/(3*16*nc[p['date']]);rows.append(dict(seed=p['seed'],sample_id=p['sample_id'],date=p['date'],group=p['physical_state_group'],category=category,actual_gain=p['actual_next_gain'],actual_NET=p['actual_next_NET'],p=prob,mu=mu,cost=cost,p_break_even=cost/mu,mu_break_even=cost/prob if prob>0 else None,macro_reward=mass*p['actual_next_gain'],macro_queries=mass*p['actual_next_queries'],append=p['gate_continue']))
    for f in range(4):
        for s in range(3):
            z=np.load(FIT/'fit_features'/f'f{f}_s{s}.npz');lab=np.load(LAB/f'f{f}_s{s}.npz');cp=np.load(SRC/'models'/f'f{f}_s{s}.npz');forest=pickle.loads((SRC/'models'/f'f{f}_s{s}.pkl').read_bytes());x=z['standard'];logit=x@cp['event_coef']+cp['event_intercept'][0];p=np.exp(-np.logaddexp(0.,-logit));mu=forest.predict(x);n=int(z['original_STOP_count']);stats={}
            for domain,idx in [('fit_STOP',np.arange(n)),('fit_CONT',np.arange(n,len(p)))]:w=z['mass'][idx]/z['mass'][idx].sum();stats[domain]=profile(p[idx],logit[idx],mu[idx],lab['target'][idx],w)
            hz=np.load(CACHE/f'held_f{f}_s{s}.npz');hx=(hz['features']-cp['mean'])/cp['scale'];hl=hx@cp['event_coef']+cp['event_intercept'][0];hp=np.exp(-np.logaddexp(0.,-hl));hm=forest.predict(hx);pp=[r for r in pred if r['fold']==f and r['seed']==s];assert np.allclose(hp,[r['event_probability'] for r in pp],atol=1e-12,rtol=1e-10) and np.allclose(hm,[r['conditional_normalized_gain'] for r in pp],atol=1e-10,rtol=1e-10);stats['held_STOP']=profile(hp,hl,hm,np.array([r['actual_next_gain']/scale for r in pp]),hierarchy(pp));pools.append(dict(fold=f,seed=s,domains=stats))
    categories={}
    for k in sorted({r['category'] for r in rows}):
        rr=[r for r in rows if r['category']==k];categories[k]=dict(pairs=len(rr),calls=len({r['group'] for r in rr}),dates=len({r['date'] for r in rr}),macro_reward=sum(r['macro_reward'] for r in rr),macro_queries=sum(r['macro_queries'] for r in rr))
    means={d:{k:float(np.mean([v['domains'][d][k] for v in pools])) for k in pools[0]['domains'][d] if k!='rows'} for d in ['fit_STOP','fit_CONT','held_STOP']};family=read(SRC/'family_summary.json');assert len(rows)==4681;assert abs(sum(r['macro_reward'] for r in rows)-family['ADD1']['mean_gain']+family['NET']['mean_gain'])<1e-12;assert abs(sum(r['macro_reward'] for r in rows if r['append'])-family['GATE']['mean_gain']+family['NET']['mean_gain'])<1e-12
    missed=sum(v['macro_reward'] for k,v in categories.items() if k.startswith('MISS_'));shares={k:v['macro_reward']/missed for k,v in categories.items() if k.startswith('MISS_')};summary=dict(categories=categories,mean_profiles=means,missed_profitable_reward_shares=shares);save('case_rows.json',rows);save('factor_profiles.json',pools);save('summary.json',summary)
    lines=['# GRID202：遗漏高收益机会时，两部分估计分别卡在哪里','','本轮读取固定模型，未训练或改决策。真实局部净收益为正而当前停止的样本，按固定费用下的单因素事后可修复性分组。它不是因果误差归因：实际单次奖励不等于未知的条件平均奖励，p=1也不是可部署预测。','','|类别|配对数|调用组|日期|可得奖励的宏平均贡献|','|---|---:|---:|---:|---:|']
    for k,v in categories.items():lines.append(f"|{k}|{v['pairs']}|{v['calls']}|{v['dates']}|{v['macro_reward']:.9f}|")
    lines+=['','MISS_EVENT_ONLY：仅将事件概率设为1可跨费用；MISS_AMOUNT_ONLY：仅用实际奖励替代幅度可跨费用；MISS_EITHER：任一替代均可；MISS_JOINT：两种单独替代都不足。仅在漏掉的实际有利补查中计算，不评估这些事后替代策略的总性能。','','|数据域|实际事件比例|概率均值|概率AUC|概率Brier|实际正例收益均值|正例幅度预测均值|总体实际收益|总体预测收益|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d,v in means.items():lines.append(f"|{d}|{v['positive_mass']:.4f}|{v['predicted_event_mean']:.4f}|{v['event_AUC']:.4f}|{v['event_Brier']:.4f}|{v['actual_positive_gain_mean']:.6f}|{v['predicted_mu_on_positive']:.6f}|{v['actual_gain_mean']:.6f}|{v['predicted_gain_mean']:.6f}|")
    lines+=['','均为各池日期/调用/起点均衡后12池描述均值；正例部分只在真实正收益样本上重新归一。不能把零收益样本的条件幅度目标填为0，事件误差与条件幅度误差须分别评价。','','概率头若拟合较好、留出明显下降，应先检验概率读出的迁移与校准；若幅度明显不足，也要保留该证据，不能将所有遗漏归于一个头。下一项内部学习改动只固定修改一个模块，不按这些事后类别直接拟触发规则。','','无新优化、神经网络前向、物理仿真、日期、最终测试或原FRAME比较。全部案例与反例保留。']
    (O/'GRID202_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');verify();save('review.json',dict(passed=True,all4681_fixed_factor_predictions_reproduced=True,selected_and_available_rewards_conserve=True,no_oracle_policy_evaluated=True,all12profiles_kept=True,new_fits=0,new_NET_forwards=0,new_physics=0));save('complete.json',dict(passed=True,new_fits=0,new_NET_forwards=0,new_physics=0));archive('complete');dest=R/'论文协作资料/FRAME_概率幅度瓶颈_GRID202';dest.mkdir(exist_ok=False)
    for n in ['GRID202_report.md','summary.json','factor_profiles.json','review.json','complete.json']:shutil.copy2(O/n,dest/n)
    v=means['held_STOP'];msg=f'GRID202固定两头瓶颈诊断：held事件AUC{v["event_AUC"]:.4f}/概率均值{v["predicted_event_mean"]:.4f}/实际事件{v["positive_mass"]:.4f}；事后单因素可修复分组全保存，非因果/可部署保证。保持零标签不进入条件幅度目标，下一改动先依据两头证据定单模块，不新FRAME闭环。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID202两头瓶颈诊断\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,r=t.split('\n',1);t=h+'\n'+entry+'\n'+r
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID202后\n\n1. 继续内部学习改进，不新FRAME闭环。\n2. 结合概率迁移与条件幅度不足，固定下一项单模块对照；不得把事后修复分类直接用作在线规则。\n3. 新训练先通知，全部种子/日期/反例保留，不扫描阈值、费用，不扩最终8或论文。\n',encoding='utf8',newline='\n');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID202两头瓶颈诊断\n\n[报告](protocols/GRID202_factor_bottleneck/complete/GRID202_report.md)；零拟合，事后因子诊断不作为部署策略。\n\n'.encode('utf8')+p.read_bytes());print('GRID202_COMPLETE',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
