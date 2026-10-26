"""Fixed-output audit: conditional gain filtering, negative predictions and extrapolation."""
from pathlib import Path
import collections,hashlib,importlib.util,json,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid200_amount_role';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID200_amount_role';LEARN=R/'outputs/grid198_conditional_gain';CONST=R/'outputs/grid199_constant_amount';FIT=R/'outputs/grid192_gate_support';CACHE=R/'outputs/grid190_extension_gate'
sp=importlib.util.spec_from_file_location('grid200_fixed_readouts',R/'work/grid198_conditional_gain/run.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
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
    save('design.json',dict(id='GRID200_AMOUNT_ROLE',authorization='User prioritizes internal learning improvement before newFRAMEcomparison.',scope='All4681eligibleSTOPs,all12fixed198headsand199constants;zero fits/rules/thresholdsearch.',metrics='Allfourdecisioncategories,positive/zero outcomes,gain/query contributionconservation,fitpositiveMSEvsconstant,negativeandabovefitpositivemax predictions. Fitonlybounds descriptive,not deployedclipping or OODproof.',limits='No causalclaim,NNforward,physics,data/final8/paper. Original sources andfailures kept.'))
    files=[W/'run.py',R/'work/grid198_conditional_gain/run.py',O/'design.json',LEARN/'gate_predictions.json',CONST/'gate_predictions.json',LEARN/'family_summary.json',CONST/'family_summary.json',LEARN/'complete.json',CONST/'complete.json',R/'outputs/grid102_search_rl/design.json']+list((LEARN/'models').glob('*.npz'))+list((LEARN/'fit_labels').glob('*.npz'))+list((CONST/'models').glob('*.npz'))+list((FIT/'fit_features').glob('*.npz'))+list(CACHE.glob('held_f*.npz'))+[R/p for p in read(CACHE/'code_freeze.json')];save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID200_PLANNED')
def evaluate():
    verify();assert not (O/'complete.json').exists();learn=read(LEARN/'gate_predictions.json');ci={(r['seed'],r['sample_id']):r for r in read(CONST/'gate_predictions.json')};_,lookup,cases=M.B.load();nc=collections.Counter(c['date'] for c in cases);rows=[];pools=[]
    for f in range(4):
        for s in range(3):
            z=np.load(FIT/'fit_features'/f'f{f}_s{s}.npz');lab=np.load(LEARN/'fit_labels'/f'f{f}_s{s}.npz');cp=np.load(LEARN/'models'/f'f{f}_s{s}.npz');constant=float(np.load(CONST/'models'/f'f{f}_s{s}.npz')['conditional_constant'][0]);x=z['standard'][lab['indices']];y=lab['target'];w=lab['sample_weight']/lab['sample_weight'].sum();value=x@cp['coef']+cp['intercept'][0];fitmse=float(w@((value-y)**2));base=float(w@((constant-y)**2));assert fitmse<=base+1e-8
            hz=np.load(CACHE/f'held_f{f}_s{s}.npz');hx=(hz['features']-cp['mean'])/cp['scale'];raw=hx@cp['coef']+cp['intercept'][0];byidx={int(j):i for i,j in enumerate(hz['ids'])};pp=[r for r in learn if r['fold']==f and r['seed']==s];low,high=float(y.min()),float(y.max());rr=[]
            for p in pp:
                j=byidx[lookup[p['sample_id'],p['position']]];q=ci[s,p['sample_id']];assert abs(p['event_probability']-q['event_probability'])<1e-12;assert np.isclose(raw[j],p['unclipped_conditional_gain'],atol=1e-10,rtol=1e-10);assert q['conditional_normalized_gain']==constant
                key='BOTH_APPEND' if p['gate_continue'] and q['gate_continue'] else 'BOTH_SKIP' if not p['gate_continue'] and not q['gate_continue'] else 'LEARNED_ONLY' if p['gate_continue'] else 'CONSTANT_ONLY';mass=1/(3*16*nc[p['date']]);change=int(p['gate_continue'])-int(q['gate_continue']);gain=p['actual_next_gain'];queries=p['actual_next_queries'];r=dict(fold=f,seed=s,sample_id=p['sample_id'],group=p['physical_state_group'],date=p['date'],category=key,actual_gain=gain,actual_queries=queries,actual_NET=p['actual_next_NET'],event_probability=p['event_probability'],learned_raw_mu=float(raw[j]),learned_mu=p['conditional_normalized_gain'],constant_mu=constant,fit_positive_min=low,fit_positive_max=high,raw_negative=bool(raw[j]<0),above_positive_label_max=bool(raw[j]>high),macro_gain_delta=mass*gain*change,macro_query_delta=mass*queries*change,macro_zero_query_delta=mass*queries*change*(gain==0));rows.append(r);rr.append(r)
            pos=[r for r in rr if r['actual_gain']>0];pools.append(dict(fold=f,seed=s,positive_fit_rows=len(y),fit_positive_linear_MSE=fitmse,fit_positive_constant_MSE=base,held_positive_rows=len(pos),held_positive_negative_predictions=sum(r['raw_negative'] for r in pos),held_positive_above_fit_max=sum(r['above_positive_label_max'] for r in pos),held_all_negative_predictions=sum(r['raw_negative'] for r in rr),held_all_above_fit_max=sum(r['above_positive_label_max'] for r in rr)))
    assert len(rows)==4681;cats={}
    for k in sorted({r['category'] for r in rows}):
        rr=[r for r in rows if r['category']==k];cats[k]=dict(pairs=len(rr),calls=len({r['group'] for r in rr}),zero_gain_pairs=sum(r['actual_gain']==0 for r in rr),positive_gain_pairs=sum(r['actual_gain']>0 for r in rr),positive_NET_pairs=sum(r['actual_NET']>0 for r in rr),negative_mu_pairs=sum(r['raw_negative'] for r in rr),above_fit_max_pairs=sum(r['above_positive_label_max'] for r in rr),macro_gain_delta=float(sum(r['macro_gain_delta'] for r in rr)),macro_query_delta=float(sum(r['macro_query_delta'] for r in rr)),macro_zero_query_delta=float(sum(r['macro_zero_query_delta'] for r in rr)))
    nf=read(LEARN/'family_summary.json')['GATE'];of=read(CONST/'family_summary.json')['GATE'];assert abs(sum(r['macro_gain_delta'] for r in rows)-nf['mean_gain']+of['mean_gain'])<1e-12;assert abs(sum(r['macro_query_delta'] for r in rows)-nf['mean_queries']+of['mean_queries'])<1e-10
    summary=dict(categories=cats,held_positive_rows=sum(p['held_positive_rows'] for p in pools),held_positive_negative_mu=sum(p['held_positive_negative_predictions'] for p in pools),held_positive_above_fit_max=sum(p['held_positive_above_fit_max'] for p in pools),held_all_negative_mu=sum(p['held_all_negative_predictions'] for p in pools),held_all_above_fit_max=sum(p['held_all_above_fit_max'] for p in pools));save('case_rows.json',rows);save('pool_diagnostics.json',pools);save('summary.json',summary)
    lines=['# GRID200：条件收益头的作用与外推核查','','本轮只核查固定学习条件头与正例均值对照，事件概率完全相同，没有新拟合或新门控策略。','','|决定差异|配对数|零奖励|正奖励|净收益正|原始条件预测负|超拟合最大收益|奖励变化贡献|查询变化贡献|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in cats.items():lines.append(f"|{k}|{v['pairs']}|{v['zero_gain_pairs']}|{v['positive_gain_pairs']}|{v['positive_NET_pairs']}|{v['negative_mu_pairs']}|{v['above_fit_max_pairs']}|{v['macro_gain_delta']:+.9f}|{v['macro_query_delta']:+.6f}|")
    lines+=['','贡献为学习条件头减均值对照，复现已发布的奖励与查询差异。未来奖励只用于事后分组，未转成输入或新规则。','','|折/种子|拟合正例数|拟合线性MSE|拟合均值MSE|留出正例数|正例预测负|正例超拟合最大收益|','|---|---:|---:|---:|---:|---:|---:|']
    for p in pools:lines.append(f"|{p['fold']}/{p['seed']}|{p['positive_fit_rows']}|{p['fit_positive_linear_MSE']:.6f}|{p['fit_positive_constant_MSE']:.6f}|{p['held_positive_rows']}|{p['held_positive_negative_predictions']}|{p['held_positive_above_fit_max']}|")
    lines+=['',f"4681个停止点中，原条件预测为负{summary['held_all_negative_mu']}个，超过拟合正例最大收益{summary['held_all_above_fit_max']}个；612个实际正收益停止点中分别为{summary['held_positive_negative_mu']}和{summary['held_positive_above_fit_max']}个。当前负值截为零，未增加上界裁剪。",'','这些结果区分正例幅度错误与零收益筛选，但不证明外推是全部失败原因。拟合正例范围只作描述，不是部署阈值。','',
        '下一项内部改进可检验预测为正例叶子均值的浅树回归：固定概率、标签、权重、特征及费用。它改变容量和归纳偏置，不能把效果仅归于不外推；应保留线性头、均值对照及全部种子。暂不启动新的原FRAME闭环、日期、最终测试或论文。']
    (O/'GRID200_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');verify();save('review.json',dict(passed=True,all4681_cached_scores_reproduced=True,all12_fit_MSE_no_worse_than_constant=True,all_gain_query_contributions_conserve=True,new_fits=0,new_NET_forwards=0,new_physics=0));save('complete.json',dict(passed=True,new_fits=0,new_NET_forwards=0,new_physics=0));archive('complete');dest=R/'论文协作资料/FRAME_条件收益作用_GRID200';dest.mkdir(exist_ok=False)
    for n in ['GRID200_report.md','summary.json','pool_diagnostics.json','review.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg=f'GRID200条件头只读核查：612正收益停止点中{summary["held_positive_negative_mu"]}个条件原预测负、{summary["held_positive_above_fit_max"]}个超拟合最大值。4681点重现、12池拟合MSE优于均值，留出不足；保留筛选小改善，下一候选固定浅树条件均值头，不自动FRAME闭环。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID200条件收益作用核查\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,r=t.split('\n',1);t=h+'\n'+entry+'\n'+r
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID200后\n\n1. 用户优先改进学习内部，暂不启动新原FRAME闭环比较。\n2. 针对条件线性头的拟合/留出差与负值，资格化固定浅树条件均值头；披露容量变化，不扫描参数。\n3. 新训练先通知范围，保留198/199及全部种子和反例，不扩日期/最终8/论文。\n',encoding='utf8',newline='\n');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID200条件收益作用核查\n\n[报告](protocols/GRID200_amount_role/complete/GRID200_report.md)；零拟合，固定读出与决定差异。\n\n'.encode('utf8')+p.read_bytes());print('GRID200_COMPLETE',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
