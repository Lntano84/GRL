"""Freeze proper-mean loss comparison,no added model parameters or data."""
from common127 import *

def main():
    assert not O.exists();O.mkdir(parents=True);base.verify();src.sealed();src.configure();assert read(BASE/'review.json')['passed']
    for part in ['train','validation']:shutil.copy2(BASE/f'{part}_targets.npz',O/f'{part}_targets.npz')
    r=torch.tensor([-10.,-1.,-.1,0.,.1,2.],dtype=torch.float64,requires_grad=True);c=torch.tensor([.001,.02,.03,.04,.08,.1],dtype=torch.float64);G=torch.tensor([1.,2.,8.,.1,0.,3.],dtype=torch.float64);q=torch.stack([torch.zeros_like(r),r],1);g=torch_gain(r,c,'POS_KL');h=torch.hypot(r,c)
    for arm in ARMS:
        main,_=losses(q,G,c,arm);der=torch.autograd.grad(main,r,retain_graph=True)[0];expected=(g-G)*(1+r/h)/len(r) if arm=='POS_MSE' else .5*(g-G)/h/len(r);assert torch.allclose(der,expected,rtol=1e-9,atol=1e-11) and torch.isfinite(der).all()
    assert torch.allclose(torch.log(g),torch.log(c)+torch.asinh(r/c),atol=1e-10)
    # Exact finite-support mean identity,not a trained constant model or Poisson-data assumption.
    support=np.array([0.,.1,5.]);p=np.array([.8,.15,.05]);mean=float(p@support);assert abs(float(p@(1-support/mean)))<1e-12
    init=read(BASE/'preflight.json')['original_initial_hashes']
    for s in range(3):assert digest(build(s))==init[str(s)]
    D=read(BASE/'design.json');plan=dict(id='GRID127_MEAN_GAIN_OBJECTIVE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User authorizes autonomous learning exploration with immediate updates. Announced fixed-link square-loss/generalized-KL comparison after auditedGRID126 stopcollapse.',
      question='Under the identical positive gain/known-cost parameterization,does a conditional-mean-consistent loss recover rare valuable CONTINUE decisions,versus the frozenHuber models and originalnetregression?',
      inherited='Original41/6978network/init/7250x32/sampler/Adam/clip10/fee/data/datefolds;positive gain g=r+hypot(r,c),netg-c,continueiffr>0;no auxiliary classifier.',
      arms=dict(NET_ZERO='Reuse15originalcachednetmodels.',POS_HUBER='Reuse15GRID126COST_POSITIVE models;no repeatfits.',POS_MSE='0.5mean((g-G)^2)+0.5Huber(stophead,0).',POS_KL='0.5mean(g-G+G(logG-logg))+0.5Huber(stophead,0);0log0=0,logg=logc+asinh(r/c). G may be any nonnegative continuous label,no Poisson count assumption.'),
      derivation='For fixedpubliccontextandg>0,expectedKLderivativewithrespecttog=1-E[G|context]/g,minimumatconditionalmean;MSEhasthesamemeanminimum. Withchosenlink,dg/dr=g/hypot(r,c),soKLmainrgradient=.5(g-G)/hypot(r,c),avoidsmultiplyingrarelargeunderestimatessolelybyvanishingsmallgain. Populationtargetnotfinitefit/generalizationguarantee.',
      matching='Bothnewarmsidenticallink,initialfunctions,randomweights,main0.5nominalcoefficient,stopHuber,samplesandupdates. Lossandgradientscalesaredifferent,not gradient/compute-time matching;recordclippingandgradientprofiles,noperformance-selectednormalizer.',
      training={**D['training'],'fit_order':[(split,s,arm) for split in ['full','f0','f1','f2','f3'] for s in range(3) for arm in ARMS]},folds=D['folds'],cost_per_query=D['cost_per_query'],gain_scale=D['gain_scale'],
      evaluation='All30newendpointsbeforevalidation/heldinference;original4validationdatesandconditional16dateOOF,complete-searchutility/query/gain,allseedsanddates. No newtest/physical/dataexpansion.',
      criteria='NewarmvsNET_ZEROandvsPOS_HUBERmusthavepositivefamilymean/all3pairedseeds,>=3/4valdatesand>=12/16OOFdatesinbothdomains. KL-specificclaimrequiresanalogousKL>MSEsupport. CompareoldHuber/DQN/BATCH1anchors;no seedselection.',
      limitations='Repeatedexposedteacherpathdevelopments;historicaltaskfee/scaleusedall16dates,notpristinenestedtest. No claimPoissondistribution,RL,wholeweekquality,latencyorstatisticalsignificance. PreserveGRID108MSEandGRID126negativeboundresults.',
      prohibited=['morethan30newfits','link/lr/lossweighttuning','newcost/date/seed/data','fitcontinuation','physicalreplay','final8test','paper/figures'])
    save(O/'design.json',plan);save(O/'preflight.json',dict(passed=True,original_initial_hashes=init,new_gain_inputs_or_targets=False,same_positive_link_all_newarms=True,synthetic_loss_gradient_formula=True,log_link_identity=True,conditional_mean_loss_identity=True,optimizer_instances=0,new_fits=0))
    files=[O/'design.json',O/'preflight.json',*O.glob('*_targets.npz'),*[BASE/n for n in ['design.json','preflight.json','review.json','fits.json','predictions.npz','decision.json','code_freeze.json','data_freeze.json','full_train_trajectories.json','full_validation_trajectories.json','oof_trajectories.json']],*[S/n for n in ['design.json','predictions.npz','preflight.json','train_data.npz','validation_data.npz','train_metadata.json','validation_metadata.json']],R/'outputs/grid102_search_rl/cases.pkl',src.S/'preflight.json',CV/'preflight.json']
    files+=list(CV.glob('fold*_normalization.npz'))+list(CV.glob('fold*_expected_exposure.npz'))+[OLD/f'training_exposure_seed{s}.npz' for s in range(3)]
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),base.W/'common126.py',src.W/'common123.py']});save(O/'state.json',dict(stage='qualified_before_push',new_fits=0));archive('before')
    docs('## 2026-10-23 — GRID127 fixed-positive-link mean-objective learning frozen\n\nAfterauditedGRID126positiveHubercontinuecollapse,autonomoususerauthorizationcoversannounced30fixedPOS_MSE/POS_KLfits,same6978/init/link/7250x32/Adam/clip/sampler/task;15POS_HUBER+15NETcachedcontrolsreused. GeneralizedKLcontinuousnonnegativegains,noPoissondataassumption;gradient.5(g-G)/hypot(r,c),properconditionalmeanpopulationtargetnotfinitefitguarantee. No lossnormalizerselection/extraepochs/linkrescue. Allendpointsbeforeheld/validation,1800scap,no physical/dataexpansion/test.',
      '# Next steps — GRID127均值收益目标对照已封存\n\n1. 在相同非负链接下完成MSE/KL共30次固定训练，不重复已完成Huber。\n2. 评价全部种子和日期的补查效用，区分训练梯度改善与实际决策改善。\n3. 保留失败结果，不调参数或扩数据救结果，不进入物理回放/最终测试。\n')
    print('GRID127_QUALIFIED',flush=True)

if __name__=='__main__':main()
