"""Freeze full-training deployment weights, not a new held-out result."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid182_deployment_bundle';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID182_deployment_bundle'
sp=importlib.util.spec_from_file_location('g182_helper',R/'work/grid174_power_helper/run.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D);read=D.read;sha=D.sha;digest=D.digest;ARMS=['STATE4','POWER8']
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    if (O/'models').exists():shutil.copytree(O/'models',dest/'models')
    (dest/'archive_manifest.json').write_text(json.dumps({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    D.verify_trained()
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
def prepare():
    assert not O.exists();O.mkdir(parents=True);D.verify_trained();assert read(R/'outputs/grid181_matched_local_baselines/complete.json')['passed_audit']
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');raw=np.load(R/'outputs/grid121_date_transfer/train_arrays.npz')['raw'];physical=np.load(R/'outputs/grid173_public_state/features.npz')['physical8'];ids=np.arange(3582)
    mean,scale=D.S.normalize(raw,meta,ids);m8,s8=D.S.normalize(physical,meta,ids);old=np.load(R/'outputs/grid102_search_rl/normalization.npz');assert np.allclose(mean,old['mean'],atol=1e-12,rtol=1e-12) and np.allclose(scale,old['scale'],atol=1e-12,rtol=1e-12)
    g=np.clip((raw-mean)/scale,-20,20).astype('f');p8=np.clip((physical-m8)/s8,-20,20).astype('f');public=D.G.inputs('train');cost=np.stack([r['known_query_cost'] for r in public]);assert np.array_equal(g,np.load(R/'outputs/grid107_direct_value/train.npz')['X'])
    x8=np.concatenate([g,cost,p8],1).astype('f');x4=x8.copy();x4[:,46:]=0;np.savez_compressed(O/'normalization.npz',mean41=mean,scale41=scale,mean8=m8,scale8=s8);np.savez_compressed(O/'training_inputs.npz',STATE4=x4,POWER8=x8)
    save('design.json',dict(id='GRID182_FROZEN_FULL_TRAIN_BUNDLE',authorization='User continuousprogress; pretrainingnotice6fulltrainhelperfits,oldFweightsreused,no physical.',
        purpose='Prepare deployable frozenconsensus weights. Not newvalidation orstatisticalconfirmation.',
        population='Alloriginal16trainingdates/32scenarios/2378starts/3582states;original4devnotfeature-extracted/scored;final8sealed.',
        new_fits=6,fit_order=[dict(seed=s,arm=a) for s in [0,1,2] for a in ARMS],model='Same7554randominitialized50-64-64-2;STATE4last4zeros,POWER8all8;noarchitecture/feature/targetselection.',
        training=dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip=10.,loss='oldmeanHuberNET[0,y],beta1',wall_cap_s=300),
        normalization='Equaldate/state-weight full16normalization,original41reproduced;currentphysical8fit16only. Historicfee/gainscale unchanged.',
        sampling='Exactoriginal16date/prefix/position probability and232000indexdraws perseed;all6finalweights before anyengineeringforward. No current scientific performance comparison.',
        factor='Copy3GRID165ZERO_MAINfull16endpoints withoutfitting;verifycheckpointhashes/normalization andfrozenoriginaltrain predictions.',
        deployment_rule='STATE4/POWER8agreement selects;disagreement delegatesFACTOR. Strictq1>q0;no newthreshold. Candidatecontrols FRAME_GUARD,BATCH1,FACTORall3seeds.',
        permission_boundary='Bundle/interfacepreparationonly. No newenvironmentconstruction/reset/step/simulate orphysicalruns authorized by this file;concretephysicalscope must be presented later.',
        prohibited=['extraepoch/checkpoint/seed/feature/fee/ruleselection','devscores','newdata/physics/final8','paper']))
    z=np.load(R/'outputs/grid107_direct_value/train.npz');checks=[]
    for seed in [0,1,2]:
        rng=np.random.default_rng(20261022+seed);h=hashlib.sha256();counts=np.zeros(3582,np.int64)
        for _ in range(7250):
            ix=rng.choice(3582,32,replace=True,p=z['sampling_probability']);h.update(ix.astype('<i8').tobytes());np.add.at(counts,ix,1)
        assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws']);expected=next(v for v in read(R/'outputs/grid165_factor_cost_routing/preflight.json')['paired'] if v['seed']==seed);assert h.hexdigest()==expected['sampling_hash'];checks.append(dict(seed=seed,sampling_hash=h.hexdigest(),original_factor_checkpoint=next(v['checkpoint_sha256'] for v in read(R/'outputs/grid165_factor_cost_routing/fits.json')['records'] if (v['seed'],v['arm'])==(seed,'ZERO_MAIN'))))
    save('preflight.json',dict(passed=True,train_states=3582,dates=16,old41normalization_reproduced=True,paired=checks,new_fits=0))
    files=[R/'outputs/grid173_public_state'/n for n in ['features.npz','metadata.json','review.json','complete.json']]+[R/'outputs/grid107_direct_value'/n for n in ['train.npz',*[f'training_exposure_seed{s}.npz' for s in [0,1,2]]]]+[R/'outputs/grid113_relational_learning'/n for n in ['train_public.pkl','train_metadata.json']]+[R/'outputs/grid165_factor_cost_routing'/n for n in ['fits.json','preflight.json','trained_freeze.json','state_scores.npz',*[f'models/ZERO_MAIN_seed{s}.pt' for s in [0,1,2]]]]+[O/'design.json',O/'preflight.json',O/'normalization.npz',O/'training_inputs.npz',D.O/'design.json',R/'outputs/grid181_matched_local_baselines/complete.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in files});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),R/'work/grid174_power_helper/run.py']});verify();archive('frozen');print('GRID182_FROZEN')
def train():
    verify();assert not (O/'training_started.json').exists();D.G.configure();import torch
    save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fits=6));(O/'models').mkdir();x=np.load(O/'training_inputs.npz');z=np.load(R/'outputs/grid107_direct_value/train.npz');xt={a:torch.tensor(x[a]) for a in ARMS};y=z['target'];yt=torch.tensor(np.column_stack([np.zeros_like(y),y]));start=time.perf_counter();records=[]
    for seed in [0,1,2]:
        for arm in ARMS:
            net,_=D.build(seed);initial=digest(net);opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);counts=np.zeros(3582,np.int64);stream=hashlib.sha256();losses=[];fs=time.perf_counter();net.train()
            for update in range(1,7251):
                if time.perf_counter()-start>300:raise TimeoutError('Keep6fulltrainpartial,no extension/repeat')
                ix=rng.choice(3582,32,replace=True,p=z['sampling_probability']);stream.update(ix.astype('<i8').tobytes());np.add.at(counts,ix,1);loss=torch.nn.functional.smooth_l1_loss(net(xt[arm][ix]),yt[ix],beta=1.);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
            assert stream.hexdigest()==next(v['sampling_hash'] for v in read(O/'preflight.json')['paired'] if v['seed']==seed);assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws']);p=O/'models'/f'{arm}_seed{seed}.pt';torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),seed=seed,arm=arm,updates=7250),p);np.savez_compressed(O/f'exposure_{arm}_seed{seed}.npz',state_draws=counts,losses=np.asarray(losses));records.append(dict(seed=seed,arm=arm,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(p),sampling_hash=stream.hexdigest(),fit_wall_s=time.perf_counter()-fs));save('fit_receipts.json',records);print('FULL_TRAIN_FIT',seed,arm,len(records),flush=True)
    factor=[]
    for seed in [0,1,2]:
        old=R/f'outputs/grid165_factor_cost_routing/models/ZERO_MAIN_seed{seed}.pt';dest=O/'models'/f'FACTOR_seed{seed}.pt';shutil.copy2(old,dest);expected=next(v['original_factor_checkpoint'] for v in read(O/'preflight.json')['paired'] if v['seed']==seed);assert sha(dest)==expected;factor.append(dict(seed=seed,source=old.relative_to(R).as_posix(),copied_checkpoint_sha256=expected,new_fit=False))
    save('fits.json',dict(passed=True,new_fits=6,records=records,reused_factor=factor,total_wall_s=time.perf_counter()-start));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*O.glob('exposure_*.npz'),O/'fits.json',O/'fit_receipts.json',O/'normalization.npz',O/'training_inputs.npz']});verify();archive('trained');print('GRID182_WEIGHTS_TRAINED')
def verify_trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def publication():
    verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote;rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json');save('prequalification_publication.json',dict(passed=True,local_sha=local,remote_sha=remote));print('GRID182_MODELS_PUBLISHED')
def qualify():
    verify_trained();assert read(O/'prequalification_publication.json')['passed'];assert not (O/'review.json').exists();D.G.configure();import torch
    old=D.G.old_api();public=D.G.inputs('train');x=np.load(O/'training_inputs.npz');scores={};maxerr=0.;checked=0
    for seed in [0,1,2]:
        for arm in ARMS:
            cp=torch.load(O/'models'/f'{arm}_seed{seed}.pt',weights_only=False,map_location='cpu');net,_=D.build(seed);net.load_state_dict(cp['state_dict']);net.eval();assert all(int(v['step'])==7250 for v in cp['optimizer']['state'].values())
            with torch.no_grad():q=net(torch.tensor(x[arm])).numpy()
            w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};nq=D.S.numpy_mlp(w,x[arm],'');assert np.allclose(q,nq,atol=3e-5,rtol=3e-5) and np.array_equal(q[:,1]>q[:,0],nq[:,1]>nq[:,0]);maxerr=max(maxerr,float(abs(q-nq).max()));checked+=3582;scores[f'{arm}_seed{seed}']=q
        net,_=D.G.build(seed,'ZERO_MAIN');net.load_state_dict(torch.load(O/'models'/f'FACTOR_seed{seed}.pt',weights_only=False,map_location='cpu')['state_dict']);q=old.predict(net,public,batch=32);prior=np.load(R/'outputs/grid165_factor_cost_routing/state_scores.npz')[f'ZERO_MAIN_seed{seed}__train'];assert np.allclose(q,prior,atol=3e-5,rtol=3e-5) and np.array_equal(q[:,1]>q[:,0],prior[:,1]>prior[:,0]);scores[f'FACTOR_seed{seed}']=q
    np.savez_compressed(O/'engineering_train_scores.npz',**scores);save('deployment_manifest.json',dict(id='GRID182_FROZEN_CONSENSUS_BUNDLE',new_fits=6,reused_factor=3,seeds=[0,1,2],rule='STATE4andPOWER8agreement,elsedelegateFACTOR;strictq1>q0',models={p.name:sha(p) for p in (O/'models').iterdir()},normalization_sha256=sha(O/'normalization.npz'),source_population='16trainingdatesonly',held_date_benchmark_claim=False,online_interface_qualified=False,physical_runs_started=False))
    verify_trained();save('review.json',dict(passed=True,independent_numpy_rows=checked,numpy_maxerror=maxerr,old_factor_decisions_reproduced=10746,all_weights_published_beforeengineeringforward=True,new_physics=0,dev_predictions=0,final8_absent=True));save('complete.json',dict(passed_audit=True,new_fits=6,reused_models=3,new_physics=0,dev_predictions=0,interface_qualification_pending=True));archive('complete')
    lines=['# GRID182：冻结可部署训练权重','', '两个小模型在全部16个原训练日期上各完成3个种子，共6次新拟合；原FACTOR复用GRID165的3份完整训练权重，逐文件与训练决定核对一致。没有重新拟合FACTOR。', '', '特征、NET标签、损失、更新数、原抽样和共识规则均未选择或调整。41维归一化完整复现旧值，8项电气描述只用原16训练日期计算归一化。全部9份权重在工程前向核验之前发布。','',f'6次拟合耗时{read(O/"fits.json")["total_wall_s"]:.1f}秒；21492行小模型独立NumPy、10746行原F决定复现、优化器步数、原始采样流和封存哈希通过。这里的训练集前向只核验实现，不提供新的泛化成绩。', '', '原4个开发日期没有新增特征或评分，最终8未生成，没有创建电网环境或发出物理预测。在线接口仍待无物理的保存状态核验；任何闭环回放需在具体范围明确后由用户决定。']
    (O/'GRID182_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    # Report written after core completion, separately addressed rather than altering the frozen snapshot.
    dest=A/'report';dest.mkdir(exist_ok=False);shutil.copy2(O/'GRID182_report.md',dest/'GRID182_report.md');(dest/'archive_manifest.json').write_text(json.dumps({'GRID182_report.md':sha(dest/'GRID182_report.md')},indent=2)+'\n',encoding='utf8',newline='\n');collab=R/'论文协作资料/FRAME_冻结部署权重_GRID182';collab.mkdir(exist_ok=False)
    for n in ['GRID182_report.md','deployment_manifest.json','review.json','complete.json']:shutil.copy2(O/n,collab/n)
    msg='GRID182部署权重6次全16训练小模型拟合+复用3旧F，全部权重先推送；小模型21492独立前向/原F10746决定重现/原采样和归一化通过。仅训练集工程核验，非新泛化结果；未原4dev新评分/最终8/物理。在线接口资格待做，物理范围先明确再由用户决定。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;oldtxt=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID182冻结部署权重\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=oldtxt.split('\n',1);oldtxt=a+'\n'+entry+'\n'+b
        else:oldtxt+=entry
        p.write_text(oldtxt,encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID182冻结部署权重\r\n\r\n[报告](protocols/GRID182_deployment_bundle/report/GRID182_report.md)；6新拟合、3旧F复用，无物理接口待核验。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID182_BUNDLE_COMPLETE')
if __name__=='__main__':globals()[sys.argv[1]]()
