"""One fixed mixture sampler, same POWER8 helper and fixed outer gate."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,hashlib,importlib.util,json,math,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid177_support_mixture';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID177_support_mixture'
sp=importlib.util.spec_from_file_location('g177_power',R/'work/grid174_power_helper/run.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D);read=D.read;sha=D.sha;digest=D.digest;P=R/'outputs/grid176_gate_support'
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
    assert not O.exists();O.mkdir(parents=True);D.verify_trained();assert read(P/'complete.json')['passed_audit'];assert read(D.O/'complete.json')['passed_audit']
    save('design.json',dict(id='GRID177_SUPPORT_MIXTURE',authorization='User continue; pretraining notice12fixedfits, sampleronly.',
        intervention='Onlysampler0.5original+0.5conditionalFCONT/RSTOPFITsupport, GRID176probability/streams alreadyfrozen. All12foldseedconfigurations;no mixweightorlabelselection.',
        fixed='SamePOWER8normalized50inputs/7554MLP/randominit/7250x32/NETHuber/Adam/fee/outerFAND(RORC). Reuseall12oldPOWER8models ascontrols;do not refitcontrols.',
        fit_order=[dict(fold=f,seed=s) for f in range(4) for s in [0,1,2]],new_fits=12,training=dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip=10.,wall_cap_s=300),
        evaluation='Publishall12weightsbeforeheldpredictions. All16OOFdates3seeds,completehelper andgate searches,utility/gain/query,actualoldlossrecovery. Smallpositivesretained,allnegativesreported.',
        limitations='Samplingexposure intentionallydiffersfromoriginal;not an exactsame-indexcomparison. FitF/Rsupport mayunderrepresentheldsupport;notnested/pristinetest. No newphysics or controltime claim.',
        prohibited=['extraiterations/seeds/folds/mixtures/losses/thresholds','heldlabels in sampler','newdata/physics/final8','original4devscores','paper','automatic scale']))
    paths=[P/n for n in ['design.json','review.json','complete.json','support_records.json',*[f'fold{f}_seed{s}_sampling.npz' for f in range(4) for s in [0,1,2]]]]
    paths +=[D.O/n for n in ['design.json','preflight.json','complete.json','review.json','trained_freeze.json','family_summary.json','oof_trajectories.json','all7134_explained_events.json',*[f'fold{f}_power.npz' for f in range(4)]]]
    paths +=[p/n for p in D.sources() for n in ['fold_arrays.npz','metadata.json','state_scores.npz','design.json']]+[R/'outputs/grid102_search_rl/cases.pkl',O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),R/'work/grid174_power_helper/run.py',R/'work/grid176_gate_support/run.py']});verify();archive('frozen');print('GRID177_FROZEN')
def train():
    verify();assert not (O/'training_started.json').exists();D.G.configure();import torch
    save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fits=12));(O/'models').mkdir();start=time.perf_counter();receipts=[]
    for fold,path in enumerate(D.sources()):
        xx,z=D.xdata(fold,'POWER8');fi=z['fit_ids'];hi=z['held_ids'];x=torch.tensor(xx[fi]);y=z['target'][fi];yt=torch.tensor(np.column_stack([np.zeros_like(y),y]))
        for seed in [0,1,2]:
            plan=np.load(P/f'fold{fold}_seed{seed}_sampling.npz');assert np.array_equal(fi,plan['fit_ids']);prob=plan['probability'];expected=next(v for v in read(P/'support_records.json') if (v['fold'],v['seed'])==(fold,seed));net,_=D.build(seed);initial=digest(net);assert initial==next(v['hash'] for v in read(D.O/'preflight.json')['initial_hashes'] if v['seed']==seed)
            opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);stream=hashlib.sha256();counts=np.zeros(3582,np.int64);losses=[];fs=time.perf_counter();net.train()
            for update in range(1,7251):
                if time.perf_counter()-start>300:raise TimeoutError('Preservefixed12fits,noextension/repeat')
                ix=rng.choice(len(fi),32,replace=True,p=prob);gix=fi[ix];stream.update(gix.astype('<i8').tobytes());np.add.at(counts,gix,1);loss=torch.nn.functional.smooth_l1_loss(net(x[ix]),yt[ix],beta=1.);assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
            assert stream.hexdigest()==expected['sampling_hash'] and np.array_equal(counts,plan['expected_state_draws']) and counts[hi].sum()==0
            name=f'fold{fold}_seed{seed}';pathcp=O/'models'/f'{name}.pt';torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),fold=fold,seed=seed,updates=7250),pathcp);np.savez_compressed(O/f'exposure_{name}.npz',state_draws=counts,losses=np.asarray(losses));rec=dict(fold=fold,seed=seed,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(pathcp),sampling_hash=stream.hexdigest(),held_draws=0,fit_wall_s=time.perf_counter()-fs);receipts.append(rec);save('fit_receipts.json',receipts);save('fit_progress.json',dict(completed=len(receipts),fold=fold,seed=seed,total_wall_s=time.perf_counter()-start));print(json.dumps(read(O/'fit_progress.json')),flush=True)
    save('fits.json',dict(passed=True,new_fits=12,records=receipts,total_wall_s=time.perf_counter()-start));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*O.glob('exposure_*.npz'),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID177_12_FITS_COMPLETE')
def verify_trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def publication():
    verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote
    rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json');save('preheld_publication.json',dict(passed=True,local_sha=local,remote_sha=remote));print('GRID177_MODELS_PUBLISHED')
def evaluate():
    verify_trained();assert read(O/'preheld_publication.json')['passed'];assert not (O/'evaluation_started.json').exists();D.G.configure();import torch
    save('evaluation_started.json',dict(all12published=True));cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'};tr=[];scores={};checks=collections.Counter();error=0.
    for fold,path in enumerate(D.sources()):
        x,z=D.xdata(fold,'POWER8');meta=read(path/'metadata.json');lookup={(v['sample_id'],v['position']):i for i,v in enumerate(meta)};dates=set(read(path/'design.json')['fold']['held_dates']);old=np.load(path/'state_scores.npz')
        for seed in [0,1,2]:
            name=f'fold{fold}_seed{seed}';rec=next(v for v in read(O/'fits.json')['records'] if (v['fold'],v['seed'])==(fold,seed));p=O/'models'/f'{name}.pt';assert sha(p)==rec['checkpoint_sha256'];cp=torch.load(p,weights_only=False,map_location='cpu');assert all(int(v['step'])==7250 for v in cp['optimizer']['state'].values());net,_=D.build(seed);net.load_state_dict(cp['state_dict']);net.eval();assert digest(net)==rec['final_hash']
            with torch.no_grad():q=net(torch.tensor(x)).numpy()
            w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};nq=D.S.numpy_mlp(w,x,'');assert np.allclose(q,nq,atol=3e-5,rtol=3e-5) and np.array_equal(q[:,1]>q[:,0],nq[:,1]>nq[:,0]);error=max(error,float(abs(q-nq).max()));checks['numpy_rows']+=3582;scores[name]=q
            F=old[f'FACTOR_seed{seed}'];U=old[f'ROUTED_seed{seed}'];go=q[:,1]>q[:,0];gate=(F[:,1]>F[:,0])&((U[:,1]>U[:,0])|go);assert np.all(gate<=(F[:,1]>F[:,0]))
            for c in cases.values():
                if c['date'] not in dates:continue
                ids=[lookup[c['sample_id'],j] for j in range(len(c['states']))]
                for arm,actions in [('MIX_HELPER',go),('MIX_GATE',gate)]:tr.append(D.H.trajectory(c,actions[ids],f'{arm}_seed{seed}',read(D.O/'design.json')['cost_per_query'],read(D.O/'design.json')['gain_scale']))
    prior=read(D.O/'oof_trajectories.json');keep=['POWER8_HELPER','POWER8_GATE','ZERO8_GATE','FACTOR'];tr +=[v for v in prior if any(v['method']==f'{a}_seed{s}' for a in keep for s in [0,1,2])];summary={name:D.H.summarize([v for v in tr if v['method']==name]) for name in {v['method'] for v in tr}};assert all(v['cases']==2378 and len(v['per_date'])==16 for v in summary.values());family={arm:{key:float(np.mean([summary[f'{arm}_seed{s}']['macro'][key] for s in [0,1,2]])) for key in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']} for arm in ['MIX_HELPER','MIX_GATE',*keep]}
    by={(v['method'],v['sample_id']):v for v in tr};rows=[];datecounts=collections.Counter(c['date'] for c in cases.values())
    for seed in [0,1,2]:
        for c in cases.values():
            a=by[f'MIX_GATE_seed{seed}',c['sample_id']];b=by[f'POWER8_GATE_seed{seed}',c['sample_id']];f=by[f'FACTOR_seed{seed}',c['sample_id']];assert a['queries']<=f['queries'];checks['query_bound_searches']+=1
            rows.append(dict(seed=seed,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],net_delta=a['net_utility']-b['net_utility'],gain_delta=a['gain']-b['gain'],saved_queries=b['queries']-a['queries'],mix_batches=a['batches'],power_batches=b['batches'],weight=1/(3*16*datecounts[c['date']])))
    pairs=[]
    for seed in [0,1,2]:
        a=summary[f'MIX_GATE_seed{seed}'];b=summary[f'POWER8_GATE_seed{seed}'];pairs.append(dict(seed=seed,delta=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],per_date=[dict(date=x['date'],delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
    dates=[dict(date=date,delta=float(np.mean([next(v['delta'] for v in p['per_date'] if v['date']==date) for p in pairs]))) for date in sorted(datecounts)];nd=family['MIX_GATE']['mean_net_utility']-family['POWER8_GATE']['mean_net_utility'];assert abs(nd-math.fsum(r['weight']*r['net_delta'] for r in rows))<1e-12
    key=dict(gate_relative_vs_power=nd/family['POWER8_GATE']['mean_net_utility'],gate_relative_vs_oldgate=family['MIX_GATE']['mean_net_utility']/family['ZERO8_GATE']['mean_net_utility']-1,gate_relative_vs_factor=family['MIX_GATE']['mean_net_utility']/family['FACTOR']['mean_net_utility']-1,query_reduction_vs_factor=1-family['MIX_GATE']['mean_queries']/family['FACTOR']['mean_queries'],helper_relative_vs_power=family['MIX_HELPER']['mean_net_utility']/family['POWER8_HELPER']['mean_net_utility']-1,positive_seeds=sum(p['delta']>0 for p in pairs),positive_dates=sum(v['delta']>0 for v in dates))
    np.savez_compressed(O/'state_scores.npz',**scores);save('oof_trajectories.json',tr);save('summary.json',summary);save('family_summary.json',family);save('paired_summary.json',dict(pairs=pairs,per_date=dates));save('all7134_paired_events.json',rows);save('key_changes.json',key);verify_trained();save('review.json',dict(passed=True,checks=dict(checks),numpy_maxerror=error,all12published_beforeheld=True,all_dates_seeds_kept=True,sampler_label_independent_fitonly=True,new_physics=0,new_data=0,final8_absent=True));print('GRID177_AUDITED',json.dumps(key))
def finish():
    verify_trained();assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();k=read(O/'key_changes.json');f=read(O/'family_summary.json');pairs=read(O/'paired_summary.json');events=read(O/'all7134_paired_events.json')
    lines=['# GRID177：让训练更关注决定能被采用的位置','',f"固定半混合采样后，组合相对原POWER8组合的留出净效用变化{k['gate_relative_vs_power']:+.2%}；{k['positive_seeds']}/3种子、{k['positive_dates']}/16日期正向。相对原FACTOR为{k['gate_relative_vs_factor']:+.2%}，查询减少{k['query_reduction_vs_factor']:.2%}。",'',
        '模型、8项电气输入、仅拟合日期归一化、NET标签、费用、7250次更新、原F/R和组合规则全部不变。只把抽样改为原分布与F继续/R停止区域的原条件分布各半。区域由拟合日期原模型预测定义，不按正负标签挑样本；保留全局一半，避免区域没有正例时丢掉所有有利样本。不是调混合比例或新增风险门槛。','',
        '| 方案 | 局部净效用 | 平均查询 | 收益保留率 | 调用均衡效用 |','|---|---:|---:|---:|---:|']
    for arm in ['FACTOR','ZERO8_GATE','POWER8_GATE','MIX_GATE','POWER8_HELPER','MIX_HELPER']:
        v=f[arm];lines.append(f"| {arm} | {v['mean_net_utility']:.8f} | {v['mean_queries']:.3f} | {v['gain_capture']:.2%} | {v['physical_group_mean_utility']:.8f} |")
    lines+=['','| 种子 | 组合相对POWER8效用变化 |','|---|---:|']
    for p in pairs['pairs']:lines.append(f"| {p['seed']} | {p['delta']:+.8f} |")
    lines+=['','| 日期 | 三种子平均效用变化 |','|---|---:|']
    for p in pairs['per_date']:lines.append(f"| {p['date']} | {p['delta']:+.8f} |")
    lines+=['',f"7134配对中{sum(e['net_delta']>0 for e in events)}条净正、{sum(e['net_delta']<0 for e in events)}条净负，其余不变；全部保存，不选最好种子。12份新模型拟合耗时{read(O/'fits.json')['total_wall_s']:.1f}秒，旧POWER8全部端点直接复用。",'',
        '这项对照检验训练分布与固定组合决策区域的对应，不证明所有旧漏停都来自样本不足。拟合F/R的区域可能仍比留出区域容易；当前没有新增独立场景或真正嵌套的门控数据。','',
        '保留GRID174/175小模型三种子改善，不因本次组合结果混杂就抹去输入进步。下一步依据实际首次分歧解释正向与损失，再决定是否研究更直接的模型角色；不追加混合比例、标签、阈值或网络扫描。','',
        '独立NumPy、采样流、优化器步数和查询上界均核验。局部NET、收益、查询与完整周质量/控制时间分别看；原4开发新评分、最终8、物理扩大和论文均未动。']
    (O/'GRID177_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');save('decision.json',dict(positive_mean=k['gate_relative_vs_power']>0,not_all_win_gate=True,adopted=False,ready_to_scale=False,next='Read-only first differences,retain POWER8standalone improvement;no furthermixture/rule scan.'));save('complete.json',dict(passed_audit=True,new_fits=12,new_physics=0,new_data=0,final8_absent=True));archive('complete')
    collab=R/'论文协作资料/FRAME_决策区域混合采样_GRID177';collab.mkdir(exist_ok=False)
    for n in ['GRID177_report.md','key_changes.json','family_summary.json','paired_summary.json','review.json','decision.json','complete.json']:shutil.copy2(O/n,collab/n);assert sha(collab/n)==sha(O/n)
    msg=f"GRID177唯一半混合采样12拟合，POWER8/NET/F/R/规则全不变，原全局分布保留一半不按标签选样。组合对POWER8{k['gate_relative_vs_power']:+.2%}，{k['positive_seeds']}/3种子、{k['positive_dates']}/16日期正；对F{k['gate_relative_vs_factor']:+.2%}、查询少{k['query_reduction_vs_factor']:.2%}。小模型对POWER8{k['helper_relative_vs_power']:+.2%}，角色成绩分列。保留原输入三种子改善，不扫混合比例/门槛救结果。无新数据/物理/最终8/论文修改。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID177决策区域混合采样\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 保留输入改善，解释组合训练结果\n\n1. POWER8独立小模型+4.36%与组合/混合采样分开，全部日期和种子保留；不扫混合比例、阈值、网络或标签。\n2. 只读解释固定混合采样的实际首次分歧，区分少查无损、恢复有利机会、无利多查和漏收益；新训练另给针对性方案及通知。\n3. 不新物理/数据/最终8，不改论文，不混局部NET与整周质量/完整控制时间。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(f'# 最新：GRID177固定半混合采样\r\n\r\n[报告](protocols/GRID177_support_mixture/complete/GRID177_report.md)；组合相对POWER8{k["gate_relative_vs_power"]:+.2%}，全日期种子保留。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID177_COMPLETE',msg)
if __name__=='__main__':globals()[sys.argv[1]]()
