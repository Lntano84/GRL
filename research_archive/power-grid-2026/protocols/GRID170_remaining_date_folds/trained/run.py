"""Complete all remaining pre-existing date folds under the unchanged GRID167 rule."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,hashlib,importlib.util,json,math,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid170_remaining_date_folds'
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID170_remaining_date_folds'
spec=importlib.util.spec_from_file_location('g170_fixed_fold_api',R/'work/grid168_one_sided_transfer/run.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
SOURCE0=S.O;G=S.G;H=S.H;read=S.read;sha=S.sha;digest=S.digest
ARMS=S.ARMS;SEEDS=S.SEEDS;FOLDS=[1,2,3]

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')

def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in W.glob('*.py'):shutil.copy2(p,dest/p.name)
    for p in O.rglob('*'):
        if p.is_file():q=dest/p.relative_to(O);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})

def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    H.sealed()

def prepare():
    assert not O.exists();O.mkdir(parents=True);S.verify_trained()
    assert read(SOURCE0/'complete.json')['passed_audit'] and read(R/'outputs/grid169_held_stop_mechanism/complete.json')['passed_audit']
    old=read(SOURCE0/'design.json');oldfolds=read(R/'outputs/grid121_date_transfer/design.json')['folds']
    meta=read(SOURCE0/'metadata.json');z=np.load(R/'outputs/grid121_date_transfer/train_arrays.npz');cost=np.load(SOURCE0/'fold_arrays.npz')['cost']
    cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train']
    design=dict(id='GRID170_REMAINING_DATE_FOLDS',authorization='User sustained work; pretraining notice will precede27fixedfits.',
        question='Does the fixed GRID167one-sided small improvement recur beyond the first date fold? Complete ALLremaining3 existingfolds, never choose one by performance.',
        folds=oldfolds[1:],existing_fold0='GRID168,kept unchanged and included in final16dateOOFsummary.',arms=ARMS,seeds=SEEDS,new_fits=27,
        rule=old['rule'],model=old['model'],training=dict(old['training'],wall_cap_s=5400),normalization=old['normalization'],sampling=old['sampling'],
        cost_per_query=old['cost_per_query'],gain_scale=old['gain_scale'],historical_dependency=old['historical_dependency'],
        training_order=[dict(fold=f,seed=s,arm=a) for f in FOLDS for s in SEEDS for a in ARMS],
        evaluation='All27 finalweights published before any remaining-fold held prediction. Complete allfolds/allseeds/allhelddates even if unfavorable. Combine oldfold0 plusnewfold1/2/3, each16originaltraindate predictedonlybyitsheldfold.',
        inference_scope='SameoldlocalNETtask; unchanged candidates/contexts/prefixes. No newphysical runs,not wholeweekquality or actualtime. Original4dev notscored;final8sealed.',
        research_interpretation='No all-seed/date gate to retain small recurring positive mechanisms; report positive/negative groups and feesaving/gainloss separately.',
        forbidden=['morefits/rule/weight/threshold/seed/fold/fee selection','newdata/physics/final8','original4dev predictions','paper','automatic scale'])
    save(O/'design.json',design);checks={}
    for f in FOLDS:
        folder=O/f'fold{f}';folder.mkdir();fold=oldfolds[f];fit=np.array([m['date'] in fold['fit_dates'] for m in meta]);fi=np.flatnonzero(fit);hi=np.flatnonzero(~fit)
        assert not {meta[i]['physical_state_group'] for i in fi}&{meta[i]['physical_state_group'] for i in hi}
        mean,scale=S.normalize(z['raw'],meta,fi);prior=np.load(R/f'outputs/grid121_date_transfer/fold{f}_normalization.npz')
        assert np.allclose(mean,prior['mean'],atol=1e-12,rtol=1e-12) and np.allclose(scale,prior['scale'],atol=1e-12,rtol=1e-12)
        normalized=np.clip((z['raw']-mean)/scale,-20,20).astype('f');prob=z['original_probability'][fi];prob=prob/prob.sum()
        np.savez_compressed(folder/'fold_arrays.npz',normalized=normalized,mean=mean,scale=scale,fit_ids=fi,held_ids=hi,fit_probability=prob,target=z['target'],cost=cost)
        save(folder/'metadata.json',meta);population=dict(fit_states=len(fi),held_states=len(hi),fit_dates=12,held_dates=4,fit_starts=sum(c['date'] in fold['fit_dates'] for c in cases),held_starts=sum(c['date'] in fold['held_dates'] for c in cases))
        save(folder/'design.json',dict(old,id=f'GRID170_FOLD{f}',fold=fold,population=population,selection='Prespecifiedremainingfold,all3jointlyfrozen;no subset selection.',fits_permitted=9))
        paired=[]
        for seed in SEEDS:
            rng=np.random.default_rng(20261022+seed);stream=hashlib.sha256();count=np.zeros(len(meta),np.int64)
            for _ in range(7250):
                ix=fi[rng.choice(len(fi),32,replace=True,p=prob)];stream.update(ix.astype('<i8').tobytes());np.add.at(count,ix,1)
            assert stream.hexdigest()==read(R/'outputs/grid121_date_transfer/preflight.json')['sampling_streams'][f'fold{f}_seed{seed}']
            assert np.array_equal(count,np.load(R/f'outputs/grid121_date_transfer/fold{f}_seed{seed}_expected_exposure.npz')['state_draws']) and count[hi].sum()==0
            oldpair=next(v for v in read(SOURCE0/'preflight.json')['paired'] if v['seed']==seed)
            paired.append(dict(seed=seed,initial_hashes=oldpair['initial_hashes'],sampling_hash=stream.hexdigest()))
        save(folder/'preflight.json',dict(passed=True,fold=f,paired=paired,fit_only_normalization_reproduced=True,held_optimizer_draws=0,population=population,new_fits=0))
        checks[str(f)]=population
    save(O/'preflight.json',dict(passed=True,folds=checks,new_fits=0,all_remaining_folds_fixed=True))
    code=[*W.glob('*.py'),R/'work/grid168_one_sided_transfer/run.py']+[R/p for p in read(SOURCE0/'code_freeze.json') if not p.startswith('work/grid168_one_sided_transfer/')]
    freeze={p.relative_to(R).as_posix():sha(p) for p in code};save(O/'code_freeze.json',freeze)
    files=[SOURCE0/n for n in ['complete.json','design.json','preflight.json','code_freeze.json','source_hashes.json','trained_freeze.json','held_trajectories.json','held_paired_events.json','paired_summary.json','family_summary.json','review.json','metadata.json','fold_arrays.npz']]
    files +=[R/'outputs/grid121_date_transfer'/n for n in ['design.json','preflight.json','train_arrays.npz',*[f'fold{f}_normalization.npz' for f in FOLDS],*[f'fold{f}_seed{s}_expected_exposure.npz' for f in FOLDS for s in SEEDS]]]
    files +=[R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid113_relational_learning/train_public.pkl',R/'outputs/grid169_held_stop_mechanism/complete.json',O/'design.json',O/'preflight.json']
    files +=[p for f in FOLDS for p in (O/f'fold{f}').iterdir() if p.is_file()]
    source={p.relative_to(R).as_posix():sha(p) for p in files};save(O/'source_hashes.json',source)
    for f in FOLDS:save(O/f'fold{f}/source_hashes.json',source);save(O/f'fold{f}/code_freeze.json',freeze)
    verify();save(O/'state.json',dict(stage='frozen',new_fits=0));archive('frozen');print('GRID170_FROZEN',json.dumps(checks))

def train():
    verify();assert not (O/'training_started.json').exists();G.configure();import torch
    start=time.perf_counter();save(O/'training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fits=27,wall_cap_s=5400));receipts=[]
    for f in FOLDS:
        folder=O/f'fold{f}';z=np.load(folder/'fold_arrays.npz');meta=read(folder/'metadata.json');fi=z['fit_ids'];hi=z['held_ids'];prob=z['fit_probability']
        public=G.inputs('train');fitrows=[dict(public[i],global41=z['normalized'][i]) for i in fi]
        take=G.old_api().tensor_cache(fitrows);x=torch.tensor(np.concatenate([z['normalized'][fi],z['cost'][fi]],1));target=z['target'][fi];yt=torch.tensor(np.column_stack([np.zeros_like(target),target]));(folder/'models').mkdir();foldrecords=[];foldstart=time.perf_counter()
        for seed in SEEDS:
            expected=next(v for v in read(folder/'preflight.json')['paired'] if v['seed']==seed)
            for arm in ARMS:
                net=S.build(seed,arm);initial=digest(net);assert initial==expected['initial_hashes'][arm]
                opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed)
                count=np.zeros(len(meta),np.int64);stream=hashlib.sha256();losses=[];fs=time.perf_counter();net.train()
                for update in range(1,7251):
                    if time.perf_counter()-start>5400:raise TimeoutError('Keep partialfixed27-fittask;no automaticextension/retry')
                    ix=rng.choice(len(fi),32,replace=True,p=prob);globalix=fi[ix];np.add.at(count,globalix,1);stream.update(globalix.astype('<i8').tobytes())
                    q=net(x[ix]) if arm=='COST42' else net(**take(ix));loss=torch.nn.functional.smooth_l1_loss(q,yt[ix],beta=1.);assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
                    if update%1000==0 or update==7250:
                        p=dict(fold=f,seed=seed,arm=arm,update=update,completed=len(receipts),total_wall_s=time.perf_counter()-start,mean_last_loss=float(np.mean(losses[-500:])))
                        save(O/'fit_progress.json',p);print(json.dumps(p),flush=True)
                assert stream.hexdigest()==expected['sampling_hash'] and count[hi].sum()==0
                assert np.array_equal(count,np.load(R/f'outputs/grid121_date_transfer/fold{f}_seed{seed}_expected_exposure.npz')['state_draws'])
                name=f'{arm}_seed{seed}';path=folder/'models'/f'{name}.pt';torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),seed=seed,arm=arm,updates=7250),path)
                np.savez_compressed(folder/f'exposure_{name}.npz',state_draws=count,losses=np.asarray(losses))
                rec=dict(fold=f,seed=seed,arm=arm,updates=7250,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(path),sampling_hash=stream.hexdigest(),parameters=sum(p.numel() for p in net.parameters()),fit_wall_s=time.perf_counter()-fs,held_draws=0)
                receipts.append(rec);foldrecords.append(rec);save(O/'fit_receipts.json',receipts);save(folder/'fit_receipts.json',foldrecords)
        save(folder/'fits.json',dict(passed=True,new_fits=9,records=foldrecords,total_wall_s=time.perf_counter()-foldstart,held_used_for_fit=False))
        save(folder/'trained_freeze.json',{p.relative_to(folder).as_posix():sha(p) for p in [*list((folder/'models').iterdir()),*folder.glob('exposure_*.npz'),folder/'fits.json',folder/'fit_receipts.json']})
    assert len(receipts)==27
    save(O/'fits.json',dict(passed=True,new_fits=27,records=receipts,total_wall_s=time.perf_counter()-start,no_held_used_for_fit=True))
    save(O/'trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for f in FOLDS for p in (O/f'fold{f}').rglob('*') if p.is_file()})
    verify();save(O/'state.json',dict(stage='trained_before_all_remaining_held_predictions',new_fits=27));archive('trained');print('GRID170_ALL27_FITS_COMPLETE')

def verify_trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p

def publication():
    verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote
    rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json')
    pub=dict(passed=True,local_sha=local,remote_sha=remote,all27_weights_before_any_remaining_held_forward=True)
    save(O/'preheld_publication.json',pub)
    for f in FOLDS:save(O/f'fold{f}/preheld_publication.json',pub)
    print('GRID170_ALL27_WEIGHTS_PUBLISHED')

def evaluate_fold():
    f=int(sys.argv[2]);assert f in FOLDS;verify_trained();assert read(O/'preheld_publication.json')['passed']
    G.old_api()  # Establish existing common/model import paths before independent audit import.
    S.O=O/f'fold{f}';S.evaluate()

def finish():
    verify_trained();assert not (O/'complete.json').exists();paths=[SOURCE0,*[O/f'fold{f}' for f in FOLDS]]
    records=[];events=[];perfold=[]
    for k,path in enumerate(paths):
        assert read(path/'review.json')['passed'];tr=read(path/'held_trajectories.json');ev=read(path/'held_paired_events.json');records+=tr
        for e in ev:events.append(dict(e,fold=k,weight=e['weight']/4))
        per=read(path/'paired_summary.json')['held'];perfold.append(dict(fold=k,**per))
    methods=[f'{a}_seed{s}' for a in ARMS+['ONE_SIDED'] for s in SEEDS]
    summaries={name:H.summarize([r for r in records if r['method']==name]) for name in methods}
    assert all(v['cases']==2378 and len(v['per_date'])==16 for v in summaries.values())
    family={a:{key:float(np.mean([summaries[f'{a}_seed{s}']['macro'][key] for s in SEEDS])) for key in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']} for a in ARMS+['ONE_SIDED']}
    pairs=[]
    for seed in SEEDS:
        a=summaries[f'ONE_SIDED_seed{seed}'];b=summaries[f'FACTOR_seed{seed}'];pairs.append(dict(seed=seed,one_sided=a['macro']['mean_net_utility'],factor=b['macro']['mean_net_utility'],net_delta=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],per_date=[dict(date=x['date'],net_delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
    dates=[dict(date=x['date'],mean_delta=float(np.mean([next(v['net_delta'] for v in p['per_date'] if v['date']==x['date']) for p in pairs]))) for x in pairs[0]['per_date']]
    nd=family['ONE_SIDED']['mean_net_utility']-family['FACTOR']['mean_net_utility'];rel=nd/family['FACTOR']['mean_net_utility'];qr=1-family['ONE_SIDED']['mean_queries']/family['FACTOR']['mean_queries']
    assert abs(nd-math.fsum(e['weight']*e['net_delta'] for e in events))<1e-12
    event_summary=[]
    for cat in sorted({e['category'] for e in events}):
        rr=[e for e in events if e['category']==cat]
        event_summary.append(dict(category=cat,pairs=len(rr),starts=len({e['sample_id'] for e in rr}),physical_groups=len({e['physical_state_group'] for e in rr}),dates=sorted({e['date'] for e in rr}),net_contribution=math.fsum(e['weight']*e['net_delta'] for e in rr),same_choice_pairs=sum(e['same_choice'] for e in rr)))
    positive_seeds=sum(p['net_delta']>0 for p in pairs);positive_dates=sum(p['mean_delta']>0 for p in dates);positive_seed_dates=sum(v['net_delta']>0 for p in pairs for v in p['per_date']);same_seed_dates=sum(v['net_delta']==0 for p in pairs for v in p['per_date'])
    key=dict(net_delta=nd,relative_change=rel,mean_query_reduction=qr,gain_capture_percentage_points=100*(family['ONE_SIDED']['gain_capture']-family['FACTOR']['gain_capture']),positive_seeds=positive_seeds,positive_dates=positive_dates,positive_seed_dates=positive_seed_dates,unchanged_seed_dates=same_seed_dates,positive_folds=sum(p['mean_net_delta']>0 for p in perfold),new_fits=27,total_folds=4,total_models_including_grid168=36)
    save(O/'oof_summary.json',summaries);save(O/'family_summary.json',family);save(O/'paired_summary.json',dict(pairs=pairs,per_date=dates,per_fold=perfold));save(O/'key_changes.json',key);save(O/'all_oof_trajectories.json',records);save(O/'all7134_paired_events.json',events);save(O/'paired_event_summary.json',event_summary)
    checks=collections.Counter()
    for path in paths:checks.update(read(path/'review.json')['checks'])
    save(O/'review.json',dict(passed=True,all4prespecified_folds=True,all36models_retained=True,all16dates_predicted_only_when_held=True,all3seeds_retained=True,checks=dict(checks),all27newweights_published_before_remaining_held=True,no_new_data=True,no_new_physics=True,original4dev_not_scored=True,final8_absent=True))
    save(O/'decision.json',dict(small_positive_mean_retained=nd>0,positive_folds=key['positive_folds'],positive_seeds=positive_seeds,positive_dates=positive_dates,not_all_win_gate=True,adopted=False,ready_to_scale=False,next='Read-only recurring positive/negative first-difference explanation and assess three-model overhead before proposing a concrete physical validation.'))
    lines=['# GRID170：固定单向停止规则的四折日期迁移','',
        f"补齐另外3折后，16个日期的留出宏平均净效用相对同折FACTOR变化{rel:+.2%}，平均查询减少{qr:.2%}；{positive_seeds}/3个种子、{positive_dates}/16个日期平均正向，{key['positive_folds']}/4个日期折正向。48个种子×日期组合中{positive_seed_dates}个提高、{same_seed_dates}个不变。",'',
        '这里复用了GRID168第0折，并一次性补齐旧GRID121规定的其余3折，没有挑选表现较好的折。每折12日期从头拟合、4日期留出，每个日期的比较模型都未在该日期拟合。三个组成模型与单向规则保持不变。','',
        '## 留出日期汇总','', '| 策略 | 净效用 | 平均查询 | 完整补查查询比例 | 收益保留率 | 调用均衡效用 |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS+['ONE_SIDED']:
        v=family[arm];lines.append(f"| {arm} | {v['mean_net_utility']:.8f} | {v['mean_queries']:.3f} | {v['query_fraction']:.2%} | {v['gain_capture']:.2%} | {v['physical_group_mean_utility']:.8f} |")
    lines+=['','收益保留率变化'+f"{key['gain_capture_percentage_points']:+.4f}"+'个百分点。费用节省与收益损失分别保留，不能把净效用改善叫作质量完全无损。','',
        '| 日期折 | 效用相对变化 | 均查询减少 | 正向种子 |','|---|---:|---:|---:|']
    for p in perfold:lines.append(f"| {p['fold']} | {p['relative_change']:+.2%} | {p['query_reduction']:.2%} | {p['positive_seeds']}/3 |")
    lines+=['','| 日期 | 三种子平均效用变化 |','|---|---:|']
    for p in dates:lines.append(f"| {p['date']} | {p['mean_delta']:+.8f} |")
    lines+=['','## 正反变化全部保留','','| 类型 | 种子×起点对 | 不同起点 | 物理状态组 | 宏平均效用贡献 |','|---|---:|---:|---:|---:|']
    for e in event_summary:lines.append(f"| {e['category']} | {e['pairs']} | {e['starts']} | {e['physical_groups']} | {e['net_contribution']:+.8f} |")
    lines+=['','## 训练与核验','',
        f"本轮27次拟合总耗时{read(O/'fits.json')['total_wall_s']:.1f}秒；连同前一轮共36份模型，全部种子均保留。各折使用拟合日期归一化，抽样只含拟合行，同种子的三个模型复用相同索引序列。各模型均7250次更新，预算、优化器、输入与NET目标不变。全部27份权重在剩余折留出预测前统一推送。",'',
        '完整检查了独立NumPy分数、逐状态停止约束、逐搜索查询上界、原始目标和查询成本、抽样流、归一化及封存哈希。此次没有新场景生成或物理回放。','',
        '## 如何使用这组证据','',
        '本轮是在已暴露的同网历史日期上验证迁移，小而反复出现的查询节省可以继续研究，不因少数日期或种子吃亏就自动否定。它仍不是独立盲测；费用与收益尺度是历史任务单位，各日期折和前缀之间也有关联。',
        '', '单向规则只保证同起点、同顺序下一次局部补查的查询不增加，不保证动作、局部收益或完整周成本无损。三个模型的实际推理开销尚未计入旧查询费用，不能据此主张控制时间优势。',
        '', '下一步先对全部留出正反变化做机制解释，并核对三模型推理开销是否会吃掉候选查询节省，再决定一组具体的真实控制验证范围。不自动扩物理数据、不生成最终8、不修改论文。']
    (O/'GRID170_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    save(O/'complete.json',dict(passed_audit=True,new_fits=27,total_models_with_grid168=36,new_data=0,new_physics=0,all4folds_retained=True,final8_absent=True));save(O/'state.json',dict(stage='complete',new_fits=27));archive('complete')
    collab=R/'论文协作资料/FRAME_四折单向停止迁移_GRID170';collab.mkdir(exist_ok=False)
    for name in ['GRID170_report.md','key_changes.json','family_summary.json','paired_summary.json','paired_event_summary.json','review.json','decision.json','complete.json']:shutil.copy2(O/name,collab/name);assert sha(collab/name)==sha(O/name)
    msg=f"GRID170补齐剩余3固定折27全新拟合，含GRID168共4折36模型/16留出日期，单向停止相对同折FACTOR净效用{rel:+.2%}、均查询少{qr:.2%}，{positive_seeds}/3种子、{positive_dates}/16日期、{key['positive_folds']}/4折平均正。全折/反例保留，不选规则/日期救结果。独立前向/归一化/抽样/查询界通过，旧局部NET非整周质量或实际提速；继续追正向复现和推理开销，不自动扩物理/新数据/最终8。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        path=REPO/'docs'/name;old=path.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID170四折单向停止迁移\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        path.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID170迁移后机制与开销\n\n1. 全部四折已固定完成，不再挑折/改停止规则。解释重复省查询与损失的物理组、立即/延迟收益来源。\n2. 核对三模型特征/推理开销与候选仿真节省的量级；不混旧计时作控制提速。\n3. 新物理或数据扩大先给具体范围，最终8不动，论文不改。保留小而反复的正向线索，不要求全种子全日期赢。\n',encoding='utf8',newline='\n')
    index=REPO/'research_archive/power-grid-2026/README.md';prefix=f'# 最新：GRID170四折单向停止迁移\r\n\r\n[报告](protocols/GRID170_remaining_date_folds/complete/GRID170_report.md)；16留出日期效用{rel:+.2%}、均查询减少{qr:.2%}，全部日期折保留。\r\n\r\n';index.write_bytes(prefix.encode('utf8')+index.read_bytes())
    for phase in A.iterdir():
        for p,h in read(phase/'archive_manifest.json').items():assert sha(phase/p)==h
    print('GRID170_COMPLETE',json.dumps(key,ensure_ascii=False))

if __name__=='__main__':globals()[sys.argv[1]]()
