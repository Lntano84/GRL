"""Freeze paired loss/parameterization controls before all research fits."""
from common126 import *

def main():
    assert not O.exists();O.mkdir(parents=True);src.verify();src.sealed();src.configure();assert read(R/'outputs/grid125_value_underestimate/review.json')['passed']
    cases=src.pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};D=read(S/'design.json');checks={}
    for part in ['train','validation']:
        meta=read(S/f'{part}_metadata.json');g=np.array([cm[m['sample_id']]['gains'][m['position']]/D['gain_scale'] for m in meta]);c=np.array([D['cost_per_query']*m['next_queries']/D['gain_scale'] for m in meta]);y=np.load(S/f'{part}_data.npz')['net'];assert (g>=0).all() and (c>0).all() and np.array_equal((g-c).astype('f'),y)
        assert all(m['next_queries']==cm[m['sample_id']]['queries'][m['position']] for m in meta)
        np.savez_compressed(O/f'{part}_targets.npz',gain=g,charge=c);checks[part]=dict(states=len(y),positive_gain=int((g>0).sum()),positive_net=int((y>0).sum()),charge_min=float(c.min()),charge_max=float(c.max()))
    r=torch.tensor([-3.,-.1,0.,.1,3.],dtype=torch.float64,requires_grad=True);c=torch.tensor([.001,.02,.03,.06,.1],dtype=torch.float64);g=torch_gain(r,c,'COST_POSITIVE');der=torch.autograd.grad(g.sum(),r)[0];expected=1+r.detach()/torch.hypot(r.detach(),c)
    assert torch.allclose(der,expected,atol=1e-12) and (g>0).all() and torch.equal(g>c,r>0) and float(g[2])==float(c[2]) and float(der[2])==1.
    # Known-cost algebra alone is a null change,not a source of efficacy.
    yy=torch.tensor([-.08,.02,.7],dtype=torch.float64);cc=torch.tensor([.1,.03,.02],dtype=torch.float64);q=torch.tensor([-.2,.1,.6],dtype=torch.float64,requires_grad=True)
    a=torch.nn.functional.smooth_l1_loss(q,yy);b=torch.nn.functional.smooth_l1_loss(q+cc,yy+cc);ga=torch.autograd.grad(a,q)[0];gb=torch.autograd.grad(b,q)[0];assert torch.allclose(a,b,atol=1e-15) and torch.allclose(ga,gb,atol=1e-15)
    sourceq=np.load(S/'predictions.npz');unchanged=0
    for key in sourceq.files:
        if '__NET_ZERO_' not in key:continue
        part='validation' if key.endswith('__validation') else 'train';cost=np.load(O/f'{part}_targets.npz')['charge'];raw=sourceq[key][:,1].astype(float);gp=numpy_gain(raw,cost,'COST_POSITIVE');assert np.array_equal(gp>cost,raw>0);unchanged+=len(raw)
    init=read(S/'preflight.json')['original_initial_hashes']
    for seed in range(3):assert digest(build(seed))==init[str(seed)] and sum(p.numel() for p in build(seed).parameters())==6978
    plan=dict(id='GRID126_KNOWN_COST_GAIN',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User: 继续，你有想法可以不用停下来，即时给我汇报就行. Announced matched known-cost/nonnegative-gain learning before fitting.',
      purpose='Does predicting uncertain reward improvement and subtracting exact public query charge improve valuable stopping decisions;does a parameter-free positive gain link help beyond direct gain regression?',
      arms=dict(NET_ZERO='Reuse old15original6978parameterHubermodels,mainq1net>0;no refits.',GAIN_DIRECT='Same random tensors/network,head1regresses raw normalized gain,knowncharge deducted exactly. Continue iffrawhead1>charge.',COST_POSITIVE='Same network,r=head1,gain=r+sqrt(r^2+charge^2),stable equivalentnegativebranch;net=gain-charge,continueiffr>0. Link0->charge,net0andlocalderivative1;gain>=0. No fitted/prior scale or new parameter.'),
      matching='Original41inputs/6978params/randomtensors/stopheadHubertarget0/main0.5Huberbeta1/sampler/Adam7250x32/gainunits/price/dates. POS vsNET has identical latent action boundary andlocalzero slope;GAIN_DIRECT initial action functions differ and are disclosed. Changing link changes gradients/optimization geometry,not claimed to isolate all capacity/statistical effects.',
      objective='0.5SmoothL1(predicted normalized gain,original gain,beta1)+0.5SmoothL1(head0,0,beta1). No auxiliary classification,positive oversampling,lossweightsweep,costchange or threshold tuning. Allzero-gain states retained.',
      null_control='Addingknowncharge to a netprediction andtarget is an exactHuberloss/gradientidentity;applyingpositive link tooldweights leavesallNET_ZEROactions unchanged. New scores require actual training toclaimbenefit.',
      training=dict(seeds=[0,1,2],updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip=10.,wall_cap_s=1800,new_fits=30,fit_order=[(split,seed,arm) for split in ['full','f0','f1','f2','f3'] for seed in range(3) for arm in ARMS]),
      data='Original3582train/856validationstates,32/8scenario replicas and16/4dates;fourfixed12fit/4helddatefolds,foldfit-onlynormalization. Knownchargefrompublicnextqueries,gainlabelprivateuntilqueried. No futuregaininthemodelinput.',
      folds=D['folds'],cost_per_query=D['cost_per_query'],gain_scale=D['gain_scale'],
      diagnostics='Initial/final fitting-side mainHuber,signregret,continueprobability,known-bound violations,linkJacobian includingpositive-targetstates;full7250loss/normcurves. Saturation/gradient changes retained,not rescued by newstepbudget.',
      evaluation='All30fixedendpointsbeforeanynewmodelvalidation/heldforward;wholelocal-searchnetutility,queryfraction,gaincapture,perdate/perseed,pairing,profitablefalseSTOP;notwholeweekphysicalqualityorspeed.',
      criteria='EacharmmustbeatNET_ZEROmeaninbothdomains,all3pairedseedspositive,>=3/4validationdatesand>=12/16OOFdatespositive. POS-specificattributionrequiresanalogousPOS>DIRECTsupport. Also preservelegacyHuber/DQN/BATCH1validationanchors. No new data or physical scaling even if passes.',
      limits='Alreadyexposedteacherpathdevelopmentdates/correlatedprefixes;historicalfee/gainscaleusedall16originaltrainingdates,soOOFconditionalonfixedtask,notnewindependenttest. ExistingMSE/weightedclassification/auxiliarynegativeevidencekept. Noautomaticstepsize/link/seed/budgetrescue.',
      prohibited=['new scenario/data','physical environment/forecast/replay','final8test','installation','paper/figures','seedselection','fitcontinuation','additionalmodelsoutside30'])
    save(O/'design.json',plan);save(O/'preflight.json',dict(passed=True,targets=checks,original_initial_hashes=init,old_projection_unchanged_predictions=unchanged,synthetic_link_gradient_checked=True,null_translation_loss_and_gradient_identity=True,optimizer_instances=0,new_fits=0,new_physical_calls=0))
    files=[O/'design.json',O/'preflight.json',*O.glob('*_targets.npz'),R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid125_value_underestimate/review.json',R/'outputs/grid125_value_underestimate/GRID125_report.md']
    files +=[S/n for n in ['design.json','preflight.json','review.json','predictions.npz','train_data.npz','validation_data.npz','train_metadata.json','validation_metadata.json','decision.json','full_train_trajectories.json','full_validation_trajectories.json','oof_trajectories.json']]
    files +=[src.S/'preflight.json',CV/'preflight.json',CV/'design.json',CV/'train_arrays.npz',CV/'metadata.json']+list(CV.glob('fold*_normalization.npz'))+list(CV.glob('fold*_expected_exposure.npz'))+[OLD/f'training_exposure_seed{s}.npz' for s in range(3)]
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),src.W/'common123.py',R/'work/grid115_matched_module_plan/matched_model.py']});save(O/'state.json',dict(stage='qualified_before_push',new_fits=0));archive('before')
    docs('## 2026-10-23 — GRID126 known-cost/gain paired learning frozen\n\nUserauthorizesautonomousprogresswithupdates.30newfixedfits,original6978/41/7250x32/init/Adam/sampler/fee/full+fourfolds,15originalcachedcontrols. DIRECTrawgainregressionminusknownfee;POSgain=r+hypot(r,fee),stablebranch,nonnegative,netzeroatlatentzero/localderivative1/actionboundaryr>0matchedold. Puretranslationnullgradientidentity/oldweightlinkactionschecked. MainhalfHuber+stophalfHuber0,noauxlossorweightsearch. All30endpointsbeforeheld/validationevaluation;1800s,no physical/newdata/test.',
      '# Next steps — GRID126已知费用与非负收益对照已封存\n\n1. 完成30次固定训练，检查抽样、更新预算和正收益状态的梯度敏感性。\n2. 全部终点完成后，评价原验证日期和四折留出日期的完整补查效用。\n3. 核验数值/轨迹与全种子结论，若失败则仅用现有记录解释，不自动加训或扩数据。\n')
    print('GRID126_QUALIFIED',checks,unchanged,flush=True)

if __name__=='__main__':main()
