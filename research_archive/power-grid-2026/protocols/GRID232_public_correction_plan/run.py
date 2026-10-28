"""Training-only qualification of one restricted residual and joint empirical step."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,shutil
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid232_public_correction_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID232_public_correction_plan'
Q=R/'outputs/grid224_context_support_plan/revision02';M=R/'outputs/grid229_training_budget_plan/revision02/known_query_mass.json';B=R/'outputs/grid230_training_budget/calibration_receipts.json'
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
LOSS=module('grid232_loss',R/'work/grid219_prior_objective/run.py');STEP=module('grid232_step',R/'work/grid221_step_plan/run.py');BUDGET=module('grid232_budget',R/'work/grid229_training_budget_plan/run.py')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def joint_step(b,d,positive,mass,budget,fallback=False):
    if fallback:return 0.,dict(constantKEEP=True)
    cap,positive_bound=STEP.choose_alpha(b,b+d,positive)
    a,record=BUDGET.match_alpha(b,d,cap,mass,budget)
    assert np.all((b+a*d)[positive]>=0) and BUDGET.saved_mass(b,d,a,mass)<=budget
    return a,dict(positive_cap=cap,positive_bound=positive_bound,budget_constraint=record)
def main():
    assert not O.exists() and read(Q/'qualification.json')['passed'];O.mkdir();(O/'qualified_inputs').mkdir()
    save('design.json',dict(id='GRID232',scope='Qualification only,zeroactualmodel/scalarfits,noheldinputs/predictions/labels read.',
        prospective_arm='PUBLIC42:fixedoriginal columns83:125 (public41+priorb),43parameters inclintercept. Offsetb retains originalstrong208learnedprior. Force first83 residualcoefficients zero;notremove priorinformation.',
        objective='Same224revision02 separatelynormalized0.5STOP/0.5auxCONT weightedmeanlogistic risk +0.5*L2all43 includingintercept. Intentional constrainedcapacitychange;not samecapacityablation. No classbalance/newscaler/newfee.',
        scalar='One joint scalarargmax alpha in[0,1]:allownSTOPfitpositives b+alpha*d>=0 AND originalSTOPweightedcancelledqueries<=parent222budget from230receipts. One composite constrained calibration perpool,nottwoindependentlyfittedhyperparameters. TiesKEEP;ULPonlyfornumeric feasibility.',
        fit='FutureGRID233 exactly11linear fits+11jointscalars=22;f0_s1KEEPconstant. One600s totalwindow,noextension/retry. L-BFGS-Bmaxiter2000,gtol1e-6,ftol1e-12,maxls50,zero43init. Allsuccessandrawactive43gradinf<.001 beforeheldscore;unqualified stopandpreserve.',
        controls='230BUDGET(primary),222STOP_ONLY,208GATE reuse. No new126dim refit. All3seeds16exposed dates retained. No heldselectedthreshold,featureselection,penalty,seed orparametergrid.',
        barriers='Sourceinputsfirstpublished;all12effectivepublicmodelsandjointstepspublishedbeforeheldscores;all12predictionspublishedbeforeoutcomejoin. Newconstraintandinputchange disclosed,notheldequalbudgetclaim.',
        unchanged='Originalfee/label/controller/global128/64/stagnation2/.9/order/aliases/station. No final8/physics/paper/FRAME/dependencyinstall. Localproxyonly.'))
    known=read(M);budgets={r['pool']:r['parent_STOP_budget'] for r in read(B)};records=[];errors=[]
    for source in sorted((Q/'qualified_inputs').glob('*.npz')):
        pool=source.stem;q=np.load(source);x=q['X'][:,83:125];calx=q['calibration_X'][:,83:125];mass=np.array(known[pool]['query_mass']);assert x.shape[1]==42 and calx.shape[1]==42 and known[pool]['ids']==q['calibration_ids'].tolist();assert np.array_equal(x[:len(calx)],calx)
        theta=np.linspace(-.01,.01,43);v,g=LOSS.objective(theta,x,q['y'],q['weight'],q['b'])
        for k in [0,10,20,30,40,41,42]:
            e=np.zeros(43);e[k]=1e-6;fd=(LOSS.objective(theta+e,x,q['y'],q['weight'],q['b'])[0]-LOSS.objective(theta-e,x,q['y'],q['weight'],q['b'])[0])/2e-6;err=abs(fd-g[k])/max(1,abs(g[k]));assert err<1e-5;errors.append(err)
        oldtheta=np.r_[np.zeros(83),theta[:-1],theta[-1]];assert np.allclose(q['X']@oldtheta[:-1]+oldtheta[-1],x@theta[:-1]+theta[-1],atol=1e-12,rtol=1e-12)
        np.savez_compressed(O/'qualified_inputs'/source.name,X=x,b=q['b'],y=q['y'],weight=q['weight'],ids=q['ids'],calibration_X=calx,calibration_b=q['calibration_b'],calibration_y=q['calibration_y'],calibration_ids=q['calibration_ids'],query_mass=mass,parent_budget=np.array([budgets[pool]]),fallback_KEEP=q['fallback_KEEP'])
        records.append(dict(pool=pool,rows=len(x),positive=int(q['y'].sum()),STOP_rows=len(calx),STOP_positive=int(q['calibration_y'].sum()),parameters=43,max_abs_input=float(abs(x).max()),fallback_KEEP=bool(q['fallback_KEEP'][0])))
    rng=np.random.default_rng(232);tests=[]
    for i in range(100):
        b=rng.uniform(.01,1,30);d=rng.normal(-1,1,30);positive=rng.random(30)<.2;mass=rng.uniform(.1,5,30);budget=float(rng.uniform(0,1)*mass.sum());a,rec=joint_step(b,d,positive,mass,budget);assert np.all((b+a*d)[positive]>=0) and BUDGET.saved_mass(b,d,a,mass)<=budget
        # Check the governing analytical upper crossing, not the ULP-rounded step.
        bound=rec['budget_constraint']['analytic_bound'] if rec['budget_constraint']['active'] else rec['positive_bound'];bound=min(1.,bound)
        if bound<1:
            witness=.5*(bound+1.);assert np.any((b+witness*d)[positive]<0) or BUDGET.saved_mass(b,d,witness,mass)>budget
        tests.append(dict(test=i,passed=True))
    assert sum(not r['fallback_KEEP'] for r in records)==11 and sum(r['STOP_positive'] for r in records)==90
    save('qualification.json',dict(passed=True,new_model_fits=0,new_scalar_fits=0,all12restricted_inputs_exported=True,finite_difference_max_relative_error=max(errors),zero83_embedding_equivalence_checked=True,synthetic_jointconstraint_cases=tests,pools=records,held_files_read=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'qualified_inputs').iterdir()),O/'design.json',O/'qualification.json']})
    paths=[W/'run.py',M,B,Q/'qualified_freeze.json',R/'work/grid219_prior_objective/run.py',R/'work/grid221_step_plan/run.py',R/'work/grid229_training_budget_plan/run.py']+list((Q/'qualified_inputs').glob('*.npz'));save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('complete.json',dict(passed=True,new_model_fits=0,new_scalar_fits=0,held_files_read=0,new_physics=0,final8_absent=True))
    text='''# GRID232：公共摘要修正器的训练内资格

下一轮保留原强模型判断，只把额外修正器限制为已有41维公共摘要和先验裕度b，共42输入、43参数。此前的缓存表示仍参与原强模型先验；本方案只减少修正器的自由度。

训练继续使用原停止/辅助经验各半的风险、原单位L2、原费用与标签。每个池只求一个联合步长，同时满足训练STOP正例不被取消、加权取消查询代价不超过自身旧222预算。这个约束与230一致；不保证外层等预算或质量。

12池输入导出、原126维空间零系数嵌入等价性、固定梯度差分及100个合成联合约束检查通过。本轮没有训练真实模型或标量，也没有读取新的外层预测或结果。

后续233只进行一次11模型加11联合标量，共22拟合/校准，另1常量KEEP。600秒总窗口，固定原L-BFGS-B设置，全部数值资格通过后才评分。原230、222、208复用，全部种子和不利结果保留。限制输入是一项容量变更，不作同容量或必然改善的声明。
'''
    (O/'GRID232_plan.md').write_text(text,encoding='utf8');A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    shutil.copytree(O/'qualified_inputs',A/'qualified_inputs');(A/'archive_manifest.json').write_text(json.dumps({p.relative_to(A).as_posix():sha(p) for p in A.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8');print('GRID232_QUALIFIED',json.dumps(dict(pools=12,real_fits=0,parameters=43,max_fd_error=max(errors))))
if __name__=='__main__':main()
