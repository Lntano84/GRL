"""Training-only qualification of a parent-matched cancellation-cost constraint."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import hashlib,json,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid229_training_budget_plan/revision02';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID229_training_budget_plan/revision02'
P=R/'outputs/grid220_full_prior';T=R/'outputs/grid222_protected_step';S=R/'outputs/grid227_context_newton';Q=R/'outputs/grid224_context_support_plan/revision02';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def saved_mass(b,d,a,mass):return float(mass[b+a*d<0].sum())
def match_alpha(b,d,cap,mass,budget):
    """Largest closed crossing-bound step under a fixed training cancellation mass budget.

    Ties KEEP. ULP-down only ensures the exact numerical score convention;
    the analytic crossing, not an outcome-selected multiplier, sets the step.
    """
    assert (b>0).all() and (mass>0).all() and 0<=cap<=1 and budget>=0
    initial=saved_mass(b,d,cap,mass)
    if initial<=budget:return cap,dict(active=False,cap_mass=initial,budget=budget,analytic_bound=cap,ulps_down=0)
    ids=np.flatnonzero((d<0)&(b+cap*d<0));ratios=b[ids]/(-d[ids]);order=np.argsort(ratios,kind='stable');ids,ratios=ids[order],ratios[order];seen=0.;bound=None;cut=[]
    for ratio in np.unique(ratios):
        group=ids[ratios==ratio];group_mass=float(mass[group].sum())
        if seen+group_mass>budget:bound=float(ratio);cut=group.tolist();break
        seen+=group_mass
    assert bound is not None and bound<=cap
    a=bound
    for ulps in range(65):
        if saved_mass(b,d,a,mass)<=budget and np.all(b[cut]+a*d[cut]>=0):break
        a=float(np.nextafter(a,0.))
    else:raise ArithmeticError('Numerical crossing feasibility failed; no alternate budget/step.')
    assert 0<=a<=cap and saved_mass(b,d,a,mass)<=budget
    return a,dict(active=True,cap_mass=initial,budget=budget,analytic_bound=bound,ulps_down=ulps,limiting_indices=cut,selected_mass=saved_mass(b,d,a,mass))
def main():
    assert not O.exists() and read(S/'complete.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID229',scope='Training-only source diagnostics and synthetic rule qualification;actualnewmodelorscalarfits0. No heldinputs/predictions/outcomes read.',
        question='Does the same positive-protection rule permit more training cancellation query mass after auxiliary support changes? Qualify one fixed constraint matched to the own STOP_ONLY parent.',
        rule='Perownfold/seed compute parent222 cancellation querymass budget on original220STOPfitrows:sum(structural_mass*known_next_queries*I(parentprotectedscore<0)). For fixed227direction choose maximal alpha<=227protectedalpha whose SAME training querymass does not exceed that budget. TiesKEEP,onlyULPdown for feasibility. If current mass alreadywithinbudget keep227alpha;f0_s1constantKEEP. No labels magnitudes or held choice determine budget.',
        purpose='Match training cancellation aggressiveness while preserving all90fitSTOP positives and neveraddingbeyond227. Explicit new empirical query-mass constraint,not equivalent original protectedoptimization. Heldquerybudget/quality isnotguaranteed.',
        prospective='GRID230 exactly11 closedform scalar calibrations+1KEEP,zeroendpoint/NNfits,600s one totalwindow;227/222/208 controls reused.All12effectiveweight+stepfilespublishedbeforescoring,predictionspublishedbeforeoutcomejoin. Only one rule,no alternate capsorposthocheldparameterchoice.',
        unchanged='All fees/labels/scalers/data/date/seeds/controllers/order/aliases/station/128/64/stagnation2/.9. Mixedauxiliarydomainrisk remains explicit0.5+0.5;querymass uses original structuralmass not riskweights. Exposeddevelopmentonly;no final8/physics/paper/newFRAME.'))
    paths=[W/'run.py',O/'design.json',META,P/'qualification.json',S/'complete.json',S/'trained_freeze.json',T/'complete.json']
    for root,folder in [(P,'fit_inputs'),(P,'models'),(T,'models'),(S,'models'),(Q,'qualified_inputs')]:paths+=list((root/folder).glob('*.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
    meta=read(META);records=[];knownqueries={}
    for f in range(4):
        for seed in range(3):
            pool=f'f{f}_s{seed}';z=np.load(P/'fit_inputs'/f'{pool}.npz');q=np.load(Q/'qualified_inputs'/f'{pool}.npz');assert np.array_equal(z['X'],q['calibration_X']) and np.array_equal(z['ids'],q['calibration_ids'])
            old=np.load(P/'models'/f'{pool}_ANCHORED.npz')['theta'];a0=float(np.load(T/'models'/f'{pool}_ANCHORED.npz')['alpha'][0]);new=np.load(S/'models'/f'{pool}.npz');a1=float(new['alpha'][0]);d0=z['X']@old[:-1]+old[-1];d1=z['X']@new['theta'][:-1]+new['theta'][-1];b=z['b'];queries=np.array([meta[int(i)]['next_queries'] for i in z['ids']]);mass=z['structural_mass']*queries;assert (mass>0).all()
            co=b+a0*d0<0;cn=b+a1*d1<0;pos=z['y']>0;assert not np.any(co&pos) and not np.any(cn&pos)
            ratios0=np.divide(b,-d0,out=np.full(len(b),np.inf),where=d0<0);ratios1=np.divide(b,-d1,out=np.full(len(b),np.inf),where=d1<0)
            records.append(dict(pool=pool,STOP_rows=len(b),positive=int(pos.sum()),old_alpha=a0,new_alpha=a1,alpha_expanded=a1>a0,old_cancelled_rows=int(co.sum()),new_cancelled_rows=int(cn.sum()),old_saved_mass=float(mass[co].sum()),new_saved_mass=float(mass[cn].sum()),total_known_query_mass=float(mass.sum()),old_saved_fraction=float(mass[co].sum()/mass.sum()),new_saved_fraction=float(mass[cn].sum()/mass.sum()),new_saved_mass_exceeds_parent=bool(mass[cn].sum()>mass[co].sum()),positive_endpoint_improved=int(((d1>d0)&pos).sum()),positive_endpoint_worsened=int(((d1<d0)&pos).sum()),positive_bound_before=float(np.min(ratios0[pos])) if pos.any() else None,positive_bound_after=float(np.min(ratios1[pos])) if pos.any() else None))
            knownqueries[pool]=dict(ids=z['ids'].tolist(),known_next_queries=queries.tolist(),query_mass=mass.tolist())
    rng=np.random.default_rng(229);tests=[]
    for trial in range(100):
        b=rng.uniform(.01,1,30);d=rng.normal(-1,1,30);mass=rng.uniform(.1,4,30);cap=float(rng.uniform(0,1));budget=float(rng.uniform(0,1)*mass.sum());a,rec=match_alpha(b,d,cap,mass,budget)
        assert saved_mass(b,d,a,mass)<=budget and a<=cap
        if rec['active']:
            above=float(.5*(rec['analytic_bound']+cap));assert rec['analytic_bound']<above<=cap and saved_mass(b,d,above,mass)>budget
        tests.append(dict(trial=trial,passed=True,active=rec['active']))
    b=np.array([.5,.5,.25]);d=np.array([-1.,-1.,-1.]);mass=np.array([2.,3.,1.]);a,tie=match_alpha(b,d,1.,mass,1.);assert abs(a-.5)<1e-12 and saved_mass(b,d,a,mass)==1
    summary=dict(pools=12,active_pools=11,fit_positive_rows=sum(r['positive'] for r in records),alpha_expanded_pools=sum(r['alpha_expanded'] for r in records),querymass_increased_pools=[r['pool'] for r in records if r['new_saved_mass_exceeds_parent']],synthetic_checks=101,new_model_fits=0,new_scalar_fits=0,held_files_read=0,not_equal_held_query_budget=True)
    save('training_diagnostics.json',records);save('known_query_mass.json',knownqueries);save('synthetic_qualification.json',dict(passed=True,random_cases=tests,tied_crossing=tie));save('summary.json',summary)
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    save('qualification.json',dict(passed=True,only_training_inputs_read=True,querymass_positive_and_public=True,all90_originalSTOPpositives_preserved_by_bothparents=True,synthetic_feasibility_maximality_ties_checked=True,new_fits=0,prospective_rule_fixed=True));save('complete.json',dict(passed=True,scope='training_only_plan_qualification',new_model_fits=0,new_scalar_fits=0,new_physics=0,final8_absent=True))
    lines=['# GRID229：先把修正力度的比较条件对齐','','本轮只看各模型自己的训练记录，不读取外层预测或结果，也没有重新拟合。','',f"11个有效学习池中，保护步长有{summary['alpha_expanded_pools']}个增大；但按同一批原STOP记录的查询代价计算，取消的查询总量只有{len(summary['querymass_increased_pools'])}个池增加：{', '.join(summary['querymass_increased_pools'])}。因此，步长数字变大不一定代表策略整体更激进，不能统一把所有步长压回旧值。",'',
    '下一轮固定一个对照：保留混合经验学出的方向，用它自己训练集上的取消查询量作为约束。允许取消的查询代价，不超过旧222在相同训练记录上取消的代价；同时不超过当前正例保护步长。查询代价按原结构权重乘已知候选数计算，不按结果大小重新加权。若当前已经满足约束，保持当前步长。','',
    '这个对照的目的是分开检验学习方向与取消强度。它增加了一项训练内经验约束，不是原问题的等价求解，也不保证外层查询预算相同或质量更好。没有按某个外层失败例子选择阈值。','',
    '100个固定合成问题和一个相同交点的例子通过可行性、最大步长及平局继续检查。真实数据的11个新标量留到下一阶段计算；不重训原227权重，保留全部种子与旧208/222/227。模型与步长发布后评分，预测发布后再接结果。最终8、物理仿真、论文与FRAME比较均不动。']
    (O/'GRID229_plan.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');print('GRID229_QUALIFIED',json.dumps(summary))
if __name__=='__main__':main()
