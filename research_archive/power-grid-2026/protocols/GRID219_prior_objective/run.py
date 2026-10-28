"""Qualify weighted-mean risk plus full correction norm; no optimizer calls."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid219_prior_objective';W=Path(__file__).parent;OLD=R/'outputs/grid218_anchored_logit';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID219_prior_objective'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def objective(theta,x,y,w,offset):
    z=x@theta[:-1]+theta[-1]+offset;p=np.exp(-np.logaddexp(0,-z));e=w*(p-y)/w.sum();return float(w@(np.logaddexp(0,z)-y*z)/w.sum()+.5*(theta@theta)),np.r_[x.T@e,e.sum()]+theta

def main():
    assert not O.exists();assert read(OLD/'complete.json')['passed'];O.mkdir();save('design.json',dict(stage='GRID219',actual_fits=0,next_FIXED_fits=22,constants=2,objective='sum(w*logisticNLL)/sum(w)+.5*sum(theta^2);all126parametersincludingintercept penalized. Same216target andweights,public125features andb. lambda1 fixed unitprior,notderivedoptimal.',change_vs218='Risknormalizedbytotalweight AND interceptregularized;matched DIRECTvsANCHORED within new objective isolatespriorcenter. Cross218difference is notsinglefactor.',reason='Original selectedcount67..492 doesnotmeasureindependent samplecount. Meanobjectiveinvariantto globalweightscaleor duplicateddata,fullpriorcoversglobalbias. No claim meanriskorfullnorm guaranteeslearnedquality.',next_solver='SameL-BFGS-B maxiter2000/gtol1e-6/ftol1e-12/maxls50/zero126init,600s22serialfits,11ownpools2arms+f0s1KEEP. no search/retry/extension/newNN/physics/test.'))
    records=[];theta=np.linspace(-.02,.02,126)
    for file in sorted((OLD/'fit_inputs').glob('*.npz')):
        q=np.load(file);x,y,w,b=q['X'],q['y'],q['weight'],q['b']
        for arm in ['DIRECT','ANCHORED']:
            offset=b if arm=='ANCHORED' else np.zeros(len(b));loss,grad=objective(theta,x,y,w,offset);scaled,sg=objective(theta,x,y,w*10,offset);doubled,dg=objective(theta,np.vstack([x,x]),np.r_[y,y],np.r_[w,w],np.r_[offset,offset]);assert abs(loss-scaled)<1e-12 and abs(loss-doubled)<1e-12 and np.allclose(grad,sg,atol=1e-12,rtol=1e-12) and np.allclose(grad,dg,atol=1e-12,rtol=1e-12)
            z=x@theta[:-1]+theta[-1]+offset;p=np.exp(-np.logaddexp(0,-z));aug=np.c_[x,np.ones(len(x))];hessian=np.eye(126)+aug.T@((w*p*(1-p)/w.sum())[:,None]*aug);mineig=float(np.linalg.eigvalsh(hessian).min());assert mineig>=1-1e-10
            maxerr=0.
            for k in [0,41,82,83,123,124,125]:
                eps=np.zeros(126);eps[k]=1e-6;est=(objective(theta+eps,x,y,w,offset)[0]-objective(theta-eps,x,y,w,offset)[0])/2e-6;maxerr=max(maxerr,abs(est-grad[k]));assert abs(est-grad[k])<1e-6
            records.append(dict(pool=file.stem,arm=arm,rows=len(x),weight_sum=float(w.sum()),min_Hessian_eigenvalue=mineig,max_finite_difference_error=maxerr,global_weight_and_replication_invariance=True))
    save('qualification.json',dict(passed=True,records=records,all24_weight_replication_invariance_and_fullgradient_checks=True,actual_fits=0,held_outcomes_used_for_this_qualification=False));save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [W/'run.py',O/'design.json',OLD/'complete.json',R/'work/grid218_anchored_logit/run.py',*list((OLD/'fit_inputs').glob('*.npz'))]});(O/'GRID219_plan.md').write_text('# GRID219：修正目标资格\n\n下一轮仍为11个有正例模型池、两个同容量线性读出，另一个无正例池两臂KEEP。使用原125维公共输入、126参数、同一费用和代价标签。\n\n目标改为加权平均逻辑损失加单位完整参数L2（含截距）。已验证整体权重缩放及复制相同记录都不改变目标/梯度，24组完整梯度有限差分及Hessian下界检查通过；没有调用优化器或拟合模型。\n\n单位惩罚为本次固定先验强度，不是理论最优值。相对GRID218同时改变损失归一化和截距惩罚，不能称单因素消融；主比较是新目标下直接读出与原判断偏置。所有模型在评分前发布、全部预测连接标签前发布，不改费用/阈值，不择种子或扫惩罚。\n',encoding='utf8',newline='\n');save('complete.json',dict(passed=True,qualification_only=True,actual_fits=0,source_pools=12,future_fits=22));A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID219_QUALIFIED_NO_FITS',len(records))
if __name__=='__main__':main()
