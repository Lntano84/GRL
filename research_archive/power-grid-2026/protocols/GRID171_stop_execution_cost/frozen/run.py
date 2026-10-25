"""Exact short-circuit execution of the fixed stop rule; local model-head timing only."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,hashlib,importlib.util,json,math,pickle,platform,shutil,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid171_stop_execution_cost';REPO=R/'work/GRL-results-integration'
A=REPO/'research_archive/power-grid-2026/protocols/GRID171_stop_execution_cost'
sp=importlib.util.spec_from_file_location('g171_transfer',R/'work/grid170_remaining_date_folds/run.py');T=importlib.util.module_from_spec(sp);sp.loader.exec_module(T)
S=T.S;G=S.G;H=S.H;read=S.read;sha=S.sha;MODES=['FACTOR','EAGER_ONE_SIDED','SHORT_ONE_SIDED']
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    T.verify_trained()
    for p,h in read(T.SOURCE0/'trained_freeze.json').items():assert sha(T.SOURCE0/p)==h,p
    for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
    for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
    H.sealed()
def prepare():
    assert not O.exists();O.mkdir(parents=True);T.verify_trained();assert read(T.O/'complete.json')['passed_audit']
    save('design.json',dict(id='GRID171_STOP_EXECUTION_COST',authorization='User continue explaining and developing small gains; equivalent execution only,no fit or new physical data.',
        purpose='Measure exact one-sided model-head overhead and savings from short-circuiting logically unnecessary helper calls.',
        modes=MODES,rule='F AND(R OR C),unchanged. SHORT:ifFSTOP stop;else evaluatecheapC;ifCCONT continue;else evaluateR. Order fixed by already-known simpleMLP versus structured complexity before timing;no order search. EAGER evaluatesall3 each reachedstate.',
        population='All4fixedOOFfolds,16dates,2378savedstarts x3seeds. Same held-foldweights asGRID170. Original4dev/final8excluded.',
        timing='CPU1thread,no_grad,batch1. All3models loadedandwarmed outside timer for allmodes. Timer includes each reached-state public tensor packing,forward,conditional selection andsearch loop. Two reverse mode orders per foldseed,allcases sameorder. No other fits/benchmarks concurrently.',
        repeats=[MODES,list(reversed(MODES))],
        excluded='Environment simulation,construction of raw public action/feedback features,fold normalization,weight load/warmup,labels/query accounting/disk writes. Therefore NOT fullcontroltime;this is local modelhead execution cost on existing inputs.',
        checks='All decisions must match saved batch32strictq1>q0 decisions;short/eager trajectories/queries/choice exactlyidentical. Allmodes/seed/date/timing repeats retained.',
        analysis='Date macro thenseed/repeat mean. Short versus eager is equivalent-policy execution optimization. Extraheadms divided bysavedqueries is a conditional break-even perquery cost,not observed simulation/controlspeedup.',
        limits='2timingrepeats,no hardwaregenerality. No rule/threshold tuning,models/labels unchanged;source teacherlocal task not fullweekquality.',
        prohibited=['fitting/policychange','newphysics/data/final8','comparehistoricalFULLtimesascurrent','paper']))
    paths=[T.O/n for n in ['complete.json','review.json','design.json','trained_freeze.json','all_oof_trajectories.json','key_changes.json']]+[T.SOURCE0/n for n in ['held_trajectories.json','state_scores.npz','metadata.json','fold_arrays.npz','trained_freeze.json']]
    paths +=[T.O/f'fold{f}'/n for f in [1,2,3] for n in ['state_scores.npz','metadata.json','fold_arrays.npz','held_trajectories.json','trained_freeze.json']]
    paths +=[R/'outputs/grid113_relational_learning/train_public.pkl',R/'outputs/grid102_search_rl/cases.pkl',O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),R/'work/grid170_remaining_date_folds/run.py',R/'work/grid168_one_sided_transfer/run.py']})
    table=[]
    for f in [0,1]:
        for r in [0,1]:
            for c in [0,1]:
                eager=bool(f and (r or c));short=False if not f else (True if r else bool(c));assert eager==short;table.append(dict(f=f,r=r,c=c,output=eager))
    save('preflight.json',dict(passed=True,all8logical_cases=table,new_fits=0,new_model_forwards=0));verify();archive('frozen');print('GRID171_FROZEN')

def benchmark():
    verify();assert not (O/'timing_started.json').exists();G.configure();import torch
    save('timing_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),platform=platform.platform(),processor=platform.processor(),torch=torch.__version__,threads=torch.get_num_threads(),no_physics=True))
    Old=G.old_api();public=G.inputs('train');cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'}
    out=[];checks=collections.Counter();wall=time.perf_counter()
    for fold,path in enumerate([T.SOURCE0,*[T.O/f'fold{f}' for f in [1,2,3]]]):
        meta=read(path/'metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};z=np.load(path/'fold_arrays.npz');rows=[dict(r,global41=z['normalized'][i]) for i,r in enumerate(public)]
        scores=np.load(path/'state_scores.npz');reference={(r['method'],r['sample_id']):r for r in read(path/'held_trajectories.json')}
        held_dates=read(path/'design.json')['fold']['held_dates'];cc=[c for c in cases.values() if c['date'] in held_dates]
        for seed in S.SEEDS:
            nets={}
            for arm in S.ARMS:
                net=S.build(seed,arm);cp=torch.load(path/'models'/f'{arm}_seed{seed}.pt',weights_only=False,map_location='cpu');net.load_state_dict(cp['state_dict']);net.eval();nets[arm]=net
            firstrow=rows[lookup[cc[0]['sample_id'],0]];warm=Old.collate([firstrow])
            with torch.no_grad():
                for _ in range(3):
                    nets['FACTOR'](**warm);nets['ROUTED'](**warm);nets['COST42'](torch.cat([warm['global41'],warm['known_query_cost']],-1))
            for repeat,order in enumerate([MODES,list(reversed(MODES))]):
                for mode in order:
                    for case in cc:
                        ids=[lookup[case['sample_id'],j] for j in range(len(case['states']))];p=0;actions=[];counts=collections.Counter();observed=[];started=time.perf_counter()
                        with torch.no_grad():
                            while p<len(ids):
                                row=rows[ids[p]];inp=Old.collate([row]);counts['pack']+=1
                                a=nets['FACTOR'](**inp)[0];counts['factor']+=1;af=bool(a[1]>a[0]);observed.append(('FACTOR',ids[p],a.numpy().copy()))
                                if mode=='FACTOR':go=af
                                elif mode=='EAGER_ONE_SIDED':
                                    b=nets['ROUTED'](**inp)[0];c=nets['COST42'](torch.cat([inp['global41'],inp['known_query_cost']],-1))[0];counts['routed']+=1;counts['small']+=1
                                    observed.extend([('ROUTED',ids[p],b.numpy().copy()),('COST42',ids[p],c.numpy().copy())]);go=af and (bool(b[1]>b[0]) or bool(c[1]>c[0]))
                                elif not af:go=False
                                else:
                                    c=nets['COST42'](torch.cat([inp['global41'],inp['known_query_cost']],-1))[0];counts['small']+=1;observed.append(('COST42',ids[p],c.numpy().copy()))
                                    if bool(c[1]>c[0]):go=True
                                    else:
                                        b=nets['ROUTED'](**inp)[0];counts['routed']+=1;observed.append(('ROUTED',ids[p],b.numpy().copy()));go=bool(b[1]>b[0])
                                actions.append(int(go))
                                if not go:break
                                p+=1
                        elapsed=time.perf_counter()-started
                        for arm,i,q in observed:
                            expected=scores[f'{arm}_seed{seed}'][i];assert np.allclose(q,expected,atol=3e-5,rtol=3e-5)
                            assert bool(q[1]>q[0])==bool(expected[1]>expected[0]);checks['saved_scores_checked']+=1
                        refname=('FACTOR' if mode=='FACTOR' else 'ONE_SIDED')+f'_seed{seed}';ref=reference[refname,case['sample_id']]
                        assert p==ref['batches'] and actions==ref['actions'];checks['saved_searches_checked']+=1
                        out.append(dict(fold=fold,seed=seed,repeat=repeat,mode=mode,date=case['date'],sample_id=case['sample_id'],physical_state_group=case['physical_state_group'],head_ms=1000*elapsed,calls=dict(counts),batches=p,queries=ref['queries'],gain=ref['gain'],chosen_pool_id=ref['chosen_pool_id']))
                    save('timing_progress.json',dict(fold=fold,seed=seed,repeat=repeat,mode=mode,rows=len(out),wall_s=time.perf_counter()-wall));print(json.dumps(read(O/'timing_progress.json')),flush=True)
    save('all_timing_records.json',out);summary={};perdate=[]
    for mode in MODES:
        rr=[r for r in out if r['mode']==mode];dates=sorted({r['date'] for r in rr});dm=[]
        for date in dates:
            records=[r for r in rr if r['date']==date];row=dict(mode=mode,date=date,mean_head_ms=float(np.mean([r['head_ms'] for r in records])),mean_queries=float(np.mean([r['queries'] for r in records])),factor_calls=float(np.mean([r['calls'].get('factor',0) for r in records])),routed_calls=float(np.mean([r['calls'].get('routed',0) for r in records])),small_calls=float(np.mean([r['calls'].get('small',0) for r in records])))
            dm.append(row);perdate.append(row)
        summary[mode]={key:float(np.mean([r[key] for r in dm])) for key in ['mean_head_ms','mean_queries','factor_calls','routed_calls','small_calls']}
    short=summary['SHORT_ONE_SIDED'];eager=summary['EAGER_ONE_SIDED'];base=summary['FACTOR'];assert short['mean_queries']==eager['mean_queries']
    qsave=base['mean_queries']-short['mean_queries'];extra=short['mean_head_ms']-base['mean_head_ms']
    key=dict(short_vs_eager_head_reduction=1-short['mean_head_ms']/eager['mean_head_ms'],extra_head_ms_vs_factor=extra,mean_queries_saved_vs_factor=qsave,conditional_break_even_ms_per_query=extra/qsave if qsave>0 else None,total_wall_s=time.perf_counter()-wall,not_full_control_time=True)
    save('family_summary.json',summary);save('per_date_summary.json',perdate);save('key_changes.json',key);verify();save('review.json',dict(passed=True,checks=dict(checks),all4folds_all16dates_all3seeds_all2repeats=True,no_new_physics=True,no_new_fits=True,short_eager_all_saved_decisions_identical=True));print('GRID171_TIMING_AUDITED',json.dumps(key))

def finish():
    verify();assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();f=read(O/'family_summary.json');k=read(O/'key_changes.json')
    lines=['# GRID171：同一停止规则的执行开销','',f"只改变等价执行顺序，单向规则的模型头平均耗时相对三模型每次全部运行减少{k['short_vs_eager_head_reduction']:.2%}；所有停止位置、候选查询和最终选择均保持一致。",'',
        '原FACTOR停止时，直接停止；原FACTOR继续时，先运行计算便宜的小费用MLP。如果小模型也继续，就直接继续；只有小模型建议停时，才需要路由结构模型。这是布尔规则的短路执行，顺序在计时前按模型结构固定，没有更换策略或引入新阈值。','',
        '| 执行方式 | 每起点模型头毫秒 | 每起点查询 | FACTOR调用 | 路由调用 | 小模型调用 |','|---|---:|---:|---:|---:|---:|']
    for a in MODES:
        r=f[a];lines.append(f"| {a} | {r['mean_head_ms']:.4f} | {r['mean_queries']:.3f} | {r['factor_calls']:.3f} | {r['routed_calls']:.3f} | {r['small_calls']:.3f} |")
    lines+=['',f"相对原FACTOR，短路版本每个起点的模型头额外耗时{k['extra_head_ms_vs_factor']:+.4f}毫秒，平均少查{k['mean_queries_saved_vs_factor']:.4f}个候选。",'',
        f"只考虑本轮已计入的模型头开销，抵消它所需的每候选查询耗时为{k['conditional_break_even_ms_per_query']:.4f}毫秒。这是开销与节省查询的比值，不是本轮测得的仿真耗时，更不是完整控制提速。" if k['conditional_break_even_ms_per_query'] is not None else '本轮没有正的平均查询节省，无法定义按查询抵消的门槛。', '',
        '计时使用相同CPU、单线程、batch1，16日期×3种子、两次反向执行顺序，权重加载与预热在计时外。计入每步公共张量打包、模型前向、逻辑选择和搜索循环；公共动作/反馈原始特征构造、归一化和物理仿真不在范围内。计时内为核验保存分数的轻量复制也包含在内。',
        '', '全部模型分数的停止符号与原batch32保存结果核对，短路与全部运行的搜索轨迹逐条相同。没有训练、新物理数据或新的策略成绩。完整控制是否划算，需要后续真正计入特征构造、物理预测和共同控制器；质量仍沿用GRID170的正反结果。']
    (O/'GRID171_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');save('complete.json',dict(passed_audit=True,new_fits=0,new_physics=0,execution_only=True));archive('complete')
    collab=R/'论文协作资料/FRAME_单向停止执行开销_GRID171';collab.mkdir(exist_ok=False)
    for n in ['GRID171_report.md','key_changes.json','family_summary.json','review.json','complete.json']:shutil.copy2(O/n,collab/n);assert sha(collab/n)==sha(O/n)
    msg=f"GRID171固定F AND(R OR C)短路执行，四折16日期三种子双反向重复，轨迹不变，学习头相对全三模型运行减少{k['short_vs_eager_head_reduction']:.2%}；相对FACTOR额外{k['extra_head_ms_vs_factor']:+.4f}ms/起点、少查{k['mean_queries_saved_vs_factor']:.3f}。只模型头，不含原特征构造/物理，不能声称控制提速。无新训练/数据/最终8。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        path=REPO/'docs'/name;old=path.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID171等价停止执行\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        path.write_text(old,encoding='utf8',newline='\n')
    index=REPO/'research_archive/power-grid-2026/README.md';index.write_bytes('# 最新：GRID171等价停止执行开销\r\n\r\n[报告](protocols/GRID171_stop_execution_cost/complete/GRID171_report.md)；固定策略短路执行，单独核对模型头开销。\r\n\r\n'.encode('utf8')+index.read_bytes())
    print('GRID171_COMPLETE',json.dumps(k,ensure_ascii=False))

if __name__=='__main__':globals()[sys.argv[1]]()
