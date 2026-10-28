"""Decompose fixed220 changes; qualify a fit-only protected correction step."""
from pathlib import Path
import collections,json,hashlib,shutil
import numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid221_step_plan';W=Path(__file__).parent;S=R/'outputs/grid220_full_prior';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID221_step_plan'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def choose_alpha(b,end,positive,fallback=False):
    if fallback:return 0.,1.
    delta=end-b;mask=positive&(delta<0);bound=float(np.min(b[mask]/(-delta[mask]))) if mask.any() else 1.;a=min(1.,bound)
    if a<1:a=float(np.nextafter(a,0.))
    for _ in range(8):
        if np.all(b[positive]+a*delta[positive]>=0):break
        a=float(np.nextafter(a,0.))
    else:raise ArithmeticError('Preserve fit-positive feasibility; no held-informed adjustment.')
    return a,bound

def main():
    assert not O.exists() and read(S/'independent_audit.json')['passed'];O.mkdir();save('design.json',dict(id='GRID221',scope='Saved220decomposition+syntheticqualificationonly. No actualdataalpha fit,newpolicy scoresorNNfits.',next_rule='For eachownouter/seed/arm let b=originalmargin, s=frozen220logit, d=s-b. Use only fitlabelsNET>0: a*=min(1,min_positive_d<0 b/(-d));rounddownonlyforfloating feasibility. New score b+a*d; originalCONT mask retained.',changed_objective='Adds an explicit empirical no-cancel-positive constraint to a line step toward already-fitted endpoint. Not same unconstrained logisticoptimization,notnewpriceorlabel. Includes every positive regardlessvalue,not selectivegaincutoff.',scope_limits='Samecorrelatedfitrows/outerdateancestors;preservesfittedpositives only,not held/fullweekquality. f0_s1no-positive KEEPconstant usesalpha0. No heldlabelcalibration,thresholdgrid/penaltygrid/newfits of main model.',future='22closedform scalar calibrations+2KEEP over bothfrozen220arms,allpublishedbeforeheldscore;all24scorefilespublishedbeforeoutcomejoin. Primary own220alpha1 and208alpha0 references;all3seeds/16dates retained.'))
    metrics=read(S/'search_metrics.json');counts={d['date']:d['cases'] for d in metrics['GATE_seed0']['per_date']};pairs=read(S/'paired_summary.json');changes=read(S/'changed_between_arms.json');decomp=[]
    for seed in range(3):
        rr=[r for r in changes if r['seed']==seed];positive=sum(max(r['NET_delta'],0)/(16*counts[r['date']]) for r in rr);negative=sum(min(r['NET_delta'],0)/(16*counts[r['date']]) for r in rr);ref=next(r for r in pairs if r['base']=='DIRECT' and r['seed']==seed)['macro_delta']['mean_net_utility'];assert abs(positive+negative-ref)<1e-12;decomp.append(dict(seed=seed,positive_contribution=positive,negative_contribution=negative,total=positive+negative,changed_rows=len(rr)))
    lost=collections.defaultdict(float);saved=0.
    for r in read(S/'removed_batches.json'):
        if r['arm']!='ANCHORED':continue
        weight=1/(3*16*counts[r['date']]);v=r['actual_next_NET']
        if v>0:lost[r['physical_state_group']]+=weight*v
        else:saved+=weight*(-v)
    fam=read(S/'family_summary.json');assert abs(saved-sum(lost.values())-(fam['ANCHORED']['mean_net_utility']-fam['GATE']['mean_net_utility']))<1e-12
    # Test formula on fixed synthetic problems only; actual fit alpha will be stage222.
    rng=np.random.default_rng(0);proof=[]
    for k in range(50):
        b=rng.uniform(.001,1,30);end=rng.normal(0,1,30);y=rng.random(30)<.2;y[0]=True;a,bound=choose_alpha(b,end,y);assert 0<=a<=1 and np.all((b+a*(end-b))[y]>=0)
        if bound<1:assert np.any((b+min(1.,bound+1e-7)*(end-b))[y]<0)
        proof.append(dict(case=k,alpha=a,analytical_bound=bound))
    save('decomposition.json',dict(ANCHOR_minus_DIRECT=decomp,ANCHOR_minus208_saved_unprofitable=saved,ANCHOR_minus208_lost_profitable=float(sum(lost.values())),loss_by_call=dict(sorted(lost.items(),key=lambda x:-x[1]))));save('synthetic_qualification.json',dict(passed=True,cases=proof,actual_data_alpha_fits=0,closed_form_maximality_and_rounding_checked=True));save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [W/'run.py',O/'design.json',S/'complete.json',S/'independent_audit.json',S/'search_metrics.json',S/'paired_summary.json',S/'changed_between_arms.json',S/'removed_batches.json']})
    lines=['# GRID221：局部恢复与训练内保护步长','','GRID220的偏置比直接读出多保留10条回本、52条不回本，正负贡献按原日期宏平均精确分开；所有变化来自原固定一步追加，没有执行新策略。','', '|种子|恢复回本的净效用贡献|额外无利成本贡献|合计|','|---|---:|---:|---:|']
    for r in decomp:lines.append(f"|{r['seed']}|{r['positive_contribution']:.9f}|{r['negative_contribution']:.9f}|{r['total']:.9f}|")
    lines+=['', f"相对原208，偏置取消不回本追加的宏平均收益为{saved:.9f}，误删回本追加损失为{sum(lost.values()):.9f}。因此当前主要问题仍是保护已有回本机会。",'', '下一轮不手选缩放：沿原判断b到固定学习终点s的方向，仅用拟合正例求最大可行步长a。约束是拟合集中每条z>0记录的新分数不低于0；当方向向下时，a不超过b/(b-s)，所有上界取最小并截到1。仅为浮点误差向下舍入。','', '它会改变学习后的决策约束，必须明确作为经验保护校准，不冒充原无约束分类损失的等价优化。所有拟合正例都保留，不按回本幅度挑样本；费用、收益标签、查询预算不改。只有拟合保护的保证，不能保证外层或整周质量。','', '50个固定合成问题的可行性和最大步长性质已检查；本轮没有用真实数据拟合a或评估步长策略。下一轮分别在原两臂11个有正例池计算22个标量，另两常量KEEP用0；全部标量发布后才评分，预测发布后连接结果。无新主模型训练、不调阈值或扫惩罚。']
    (O/'GRID221_plan.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');save('complete.json',dict(passed=True,actual_fits=0,new_policy_scores=0,new_physics=0,synthetic_cases=50));A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID221_QUALIFIED',json.dumps(dict(decomposition=decomp,positive_loss=sum(lost.values()),saved=saved),ensure_ascii=False))
if __name__=='__main__':main()
