"""One origin-defined input ablation: keep GRID98 state4, remove new power/plan4."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid178_power_block_ablation';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID178_power_block_ablation'
sp=importlib.util.spec_from_file_location('g178_power',R/'work/grid174_power_helper/run.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D);read=D.read;sha=D.sha;digest=D.digest
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def data(f):
    x,z=D.xdata(f,'POWER8');x=x.copy();x[:,46:50]=0;assert x.shape==(3582,50);return x,z
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
    assert not O.exists();O.mkdir(parents=True);D.verify_trained();assert read(D.O/'complete.json')['passed_audit']
    save('design.json',dict(id='GRID178_STATE4_INPUT_ABLATION',authorization='User sustainedmechanismresearch; pretrainingnotice12fits,one origin-definedablation.',
        question='Is POWER8 improvement mainly restored publicloadstate4 or incrementally the fournewfrozenpower/plan descriptors?',
        intervention='Same50input7554MLP,keep normalizedcolumns42:46state4;setcolumns46:50power-plan4tozero. Original42unchanged. ReuseallZERO8andPOWER8originalsampler endpoints,trainonly12STATE4controls.',
        fixed='Original fullsampler,notGRID177mixture; sameall4folds3seeds/randominit/7250x32/NETHuber/Adam/normalization/price,F/R,andgate. No fieldwise search.',
        interpretation='Jointblockablation,not eachfieldcausalproof. STATE4 includes duplicatedmaxrho. All8totalparameterssame,activecolumnsnot same. Exposeddate development,not a newtest.',
        training=dict(new_fits=12,updates=7250,batch=32,lr=.0003,clip=10.,wall_cap_s=300),
        evaluation='Publishall12weightsbeforeheldpredictions. Helper andgate bothreported,allseeds/all16dates,POWER8vsSTATE4 incremental difference andSTATE4vsZERO8. Do not choose a winner then callit independentconfirmation.',
        prohibited=['newfields/budget/fees/thresholds/seeds/folds','mixtureorsinglefieldsearch','newdata/physics/final8','original4devscoring','paper']))
    paths=[D.O/n for n in ['complete.json','review.json','design.json','preflight.json','trained_freeze.json','oof_trajectories.json','summary.json',*[f'fold{f}_power.npz' for f in range(4)]]]+[p/n for p in D.sources() for n in ['fold_arrays.npz','metadata.json','state_scores.npz','preflight.json','design.json']]+[R/'outputs/grid102_search_rl/cases.pkl',O/'design.json']
    paths +=[R/f'outputs/grid121_date_transfer/fold{f}_seed{s}_expected_exposure.npz' for f in range(4) for s in [0,1,2]]
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),R/'work/grid174_power_helper/run.py']});verify();archive('frozen');print('GRID178_FROZEN')
def train():
    verify();assert not (O/'training_started.json').exists();D.G.configure();import torch
    save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fits=12));(O/'models').mkdir();start=time.perf_counter();records=[]
    for fold,path in enumerate(D.sources()):
        x,z=data(fold);fi=z['fit_ids'];hi=z['held_ids'];prob=z['fit_probability'];xt=torch.tensor(x[fi]);y=z['target'][fi];yt=torch.tensor(np.column_stack([np.zeros_like(y),y]))
        for seed in [0,1,2]:
            net,_=D.build(seed);initial=digest(net);assert initial==next(v['hash'] for v in read(D.O/'preflight.json')['initial_hashes'] if v['seed']==seed);expected=next(v for v in read(path/'preflight.json')['paired'] if v['seed']==seed)
            opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);stream=hashlib.sha256();counts=np.zeros(3582,np.int64);losses=[];fs=time.perf_counter();net.train()
            for update in range(1,7251):
                if time.perf_counter()-start>300:raise TimeoutError('Keep12fixedablationpartialoutputs,noextension/repeat')
                ix=rng.choice(len(fi),32,replace=True,p=prob);globalix=fi[ix];stream.update(globalix.astype('<i8').tobytes());np.add.at(counts,globalix,1);loss=torch.nn.functional.smooth_l1_loss(net(xt[ix]),yt[ix],beta=1.);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
            assert stream.hexdigest()==expected['sampling_hash'] and counts[hi].sum()==0;assert np.array_equal(counts,np.load(R/f'outputs/grid121_date_transfer/fold{fold}_seed{seed}_expected_exposure.npz')['state_draws']);assert not np.any(net.q_net[0].weight.detach().numpy()[:,46:50])
            name=f'fold{fold}_seed{seed}';p=O/'models'/f'{name}.pt';torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),fold=fold,seed=seed,updates=7250),p);np.savez_compressed(O/f'exposure_{name}.npz',state_draws=counts,losses=np.asarray(losses));records.append(dict(fold=fold,seed=seed,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(p),sampling_hash=stream.hexdigest(),held_draws=0,fit_wall_s=time.perf_counter()-fs));save('fit_receipts.json',records);save('fit_progress.json',dict(completed=len(records),fold=fold,seed=seed,total_wall_s=time.perf_counter()-start));print(json.dumps(read(O/'fit_progress.json')),flush=True)
    save('fits.json',dict(passed=True,new_fits=12,records=records,total_wall_s=time.perf_counter()-start));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*O.glob('exposure_*.npz'),O/'fits.json',O/'fit_receipts.json']});archive('trained');print('GRID178_12_FITS_COMPLETE')
def verify_trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def publication():
    verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote;rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json');save('preheld_publication.json',dict(passed=True,local_sha=local,remote_sha=remote));print('GRID178_MODELS_PUBLISHED')
def evaluate():
    verify_trained();assert read(O/'preheld_publication.json')['passed'];assert not (O/'evaluation_started.json').exists();D.G.configure();import torch
    save('evaluation_started.json',dict(all12published=True));cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'];tr=[];scores={};maxerr=0.;checks=0
    for fold,path in enumerate(D.sources()):
        x,z=data(fold);meta=read(path/'metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};old=np.load(path/'state_scores.npz');dates=set(read(path/'design.json')['fold']['held_dates'])
        for seed in [0,1,2]:
            p=O/'models'/f'fold{fold}_seed{seed}.pt';rec=next(v for v in read(O/'fits.json')['records'] if (v['fold'],v['seed'])==(fold,seed));assert sha(p)==rec['checkpoint_sha256'];cp=torch.load(p,weights_only=False,map_location='cpu');assert all(int(v['step'])==7250 for v in cp['optimizer']['state'].values());net,_=D.build(seed);net.load_state_dict(cp['state_dict']);net.eval();assert digest(net)==rec['final_hash']
            with torch.no_grad():q=net(torch.tensor(x)).numpy()
            w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};nq=D.S.numpy_mlp(w,x,'');assert np.allclose(nq,q,atol=3e-5,rtol=3e-5) and np.array_equal(nq[:,1]>nq[:,0],q[:,1]>q[:,0]);maxerr=max(maxerr,float(abs(nq-q).max()));checks+=3582;scores[f'fold{fold}_seed{seed}']=q
            F=old[f'FACTOR_seed{seed}'];U=old[f'ROUTED_seed{seed}'];go=q[:,1]>q[:,0];gate=(F[:,1]>F[:,0])&((U[:,1]>U[:,0])|go);assert np.all(gate<=(F[:,1]>F[:,0]))
            for c in cases:
                if c['date'] not in dates:continue
                ids=[lookup[c['sample_id'],j] for j in range(len(c['states']))]
                for arm,dec in [('STATE4_HELPER',go),('STATE4_GATE',gate)]:tr.append(D.H.trajectory(c,dec[ids],f'{arm}_seed{seed}',read(D.O/'design.json')['cost_per_query'],read(D.O/'design.json')['gain_scale']))
    tr+=read(D.O/'oof_trajectories.json');summary={name:D.H.summarize([v for v in tr if v['method']==name]) for name in {v['method'] for v in tr}};assert all(v['cases']==2378 and len(v['per_date'])==16 for v in summary.values());arms=['ZERO8_HELPER','STATE4_HELPER','POWER8_HELPER','ZERO8_GATE','STATE4_GATE','POWER8_GATE','FACTOR'];family={a:{k:float(np.mean([summary[f'{a}_seed{s}']['macro'][k] for s in [0,1,2]])) for k in ['mean_net_utility','mean_queries','gain_capture','query_fraction','physical_group_mean_utility']} for a in arms}
    pairs=[]
    for seed in [0,1,2]:
        a=summary[f'POWER8_HELPER_seed{seed}'];b=summary[f'STATE4_HELPER_seed{seed}'];pairs.append(dict(seed=seed,all8_minus_state4=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],per_date=[dict(date=x['date'],delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
    dates=[dict(date=d['date'],delta=float(np.mean([next(v['delta'] for v in p['per_date'] if v['date']==d['date']) for p in pairs]))) for d in pairs[0]['per_date']];key=dict(state4_vs_zero8=family['STATE4_HELPER']['mean_net_utility']/family['ZERO8_HELPER']['mean_net_utility']-1,all8_vs_state4=family['POWER8_HELPER']['mean_net_utility']/family['STATE4_HELPER']['mean_net_utility']-1,all8_vs_zero8=family['POWER8_HELPER']['mean_net_utility']/family['ZERO8_HELPER']['mean_net_utility']-1,all8_increment_positive_seeds=sum(p['all8_minus_state4']>0 for p in pairs),all8_increment_positive_dates=sum(d['delta']>0 for d in dates))
    np.savez_compressed(O/'state_scores.npz',**scores);save('oof_trajectories.json',tr);save('summary.json',summary);save('family_summary.json',family);save('paired_summary.json',dict(pairs=pairs,per_date=dates));save('key_changes.json',key);verify_trained();save('review.json',dict(passed=True,numpy_rows_checked=checks,numpy_maxerror=maxerr,all12weights_beforeheld=True,all_zero_removedcolumns=True,same_original_sampling_streams=True,all_dates_seeds_kept=True,new_physics=0,new_data=0,final8_absent=True));print('GRID178_AUDITED',json.dumps(key))
def finish():
    verify_trained();assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();k=read(O/'key_changes.json');f=read(O/'family_summary.json');p=read(O/'paired_summary.json')
    lines=['# GRID178：状态信息与功率—分区描述的分块消融','',f"小模型单独决策时，STATE4相对ZERO8净效用变化{k['state4_vs_zero8']:+.2%}；完整POWER8相对STATE4变化{k['all8_vs_state4']:+.2%}，{k['all8_increment_positive_seeds']}/3种子、{k['all8_increment_positive_dates']}/16日期的增量为正。完整POWER8对ZERO8的原{ k['all8_vs_zero8']:+.2%}结果保留。",'',
        'STATE4保留前四项当前负载/站内过载描述，移除后四项功率规模与候选分区结合的描述。各组网络都是50—64—64—2、7554个总参数；同随机初始化、全部原采样索引、标签、归一化和7250次更新；移除列置0，不改变原42项。新增12份STATE4模型，ZERO8与POWER8直接复用既有端点，全部新权重先推送再评分。','',
        '| 方案 | 局部净效用 | 平均查询 | 收益保留率 | 调用均衡效用 |','|---|---:|---:|---:|---:|']
    for a in ['ZERO8_HELPER','STATE4_HELPER','POWER8_HELPER','ZERO8_GATE','STATE4_GATE','POWER8_GATE','FACTOR']:
        v=f[a];lines.append(f"| {a} | {v['mean_net_utility']:.8f} | {v['mean_queries']:.3f} | {v['gain_capture']:.2%} | {v['physical_group_mean_utility']:.8f} |")
    lines+=['','| 种子 | 完整8项 − STATE4小模型效用 |','|---|---:|']
    for s in p['pairs']:lines.append(f"| {s['seed']} | {s['all8_minus_state4']:+.8f} |")
    lines+=['','这是按特征来源划分的一个预定消融，不是逐项选择、混合比例或阈值搜索。输入块互相作用，不能把两个百分比当成独立可加贡献；也不证明某一个字段单独必要。STATE4中的最大负载率重复原输入，活跃列数不同的事实保留。','',
        f"新增12次拟合耗时{read(O/'fits.json')['total_wall_s']:.1f}秒，独立NumPy与原采样/移除列梯度核验通过。三个种子和16个日期全部保留。结果仍是历史同网局部NET任务，不是独立新场景测试或完整周质量/控制提速。",'',
        '下一步保留有正向作用的输入方向，结合完整搜索里恢复有利机会与节省无收益查询的记录解释，不据此临时选新字段或宣布最终模型。旧三模型组合和独立小模型结果继续分列。']
    (O/'GRID178_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');save('decision.json',dict(all8_increment_positive_mean=k['all8_vs_state4']>0,not_all_win_gate=True,adopted=False,ready_to_scale=False,next='Interpret complete fixedblockcomparison;no fieldorpolicysearch.'));save('complete.json',dict(passed_audit=True,new_fits=12,new_physics=0,new_data=0,final8_absent=True));archive('complete');collab=R/'论文协作资料/FRAME_电气输入分块消融_GRID178';collab.mkdir(exist_ok=False)
    for n in ['GRID178_report.md','key_changes.json','family_summary.json','paired_summary.json','review.json','decision.json','complete.json']:shutil.copy2(O/n,collab/n)
    msg=f"GRID178来源分块消融12STATE4拟合，原ZERO8/POWER8复用，同架构/初始/采样/目标/预算全配对。STATE4对ZERO{k['state4_vs_zero8']:+.2%}，完整8对STATE4{k['all8_vs_state4']:+.2%}，增量{k['all8_increment_positive_seeds']}/3种子、{k['all8_increment_positive_dates']}/16日期正。保留完整8原+4.36，不声称字段独立可加或最终采用；无新数据/物理/最终8/论文改动。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID178电气输入分块\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(f'# 最新：GRID178电气输入分块消融\r\n\r\n[报告](protocols/GRID178_power_block_ablation/complete/GRID178_report.md)；完整8项对STATE4增量{k["all8_vs_state4"]:+.2%}，全部配对结果保留。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID178_COMPLETE',msg)
if __name__=='__main__':globals()[sys.argv[1]]()
