"""Compute current NN/feature costs, exact saved decisions, without powerflow calls."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,gzip,hashlib,importlib.util,json,pickle,platform,shutil,sys,time
import numpy as np
from runtime_features import state4,power8
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid180_consensus_runtime';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID180_consensus_runtime'
sp=importlib.util.spec_from_file_location('g180_consensus',R/'work/grid179_cheap_consensus/run.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C);S=C.S;D=C.D;read=C.read;sha=C.sha;MODES=['FACTOR','STATE4','POWER8','EAGER3','CONSENSUS']
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    C.verify()
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    for path in D.sources():
        for rec in read(path/'fits.json')['records']:
            if rec['arm']=='FACTOR':assert sha(path/'models'/f"FACTOR_seed{rec['seed']}.pt")==rec['checkpoint_sha256']
def prepare():
    assert not O.exists();O.mkdir(parents=True);C.verify();assert read(C.O/'complete.json')['passed_audit']
    save('design.json',dict(id='GRID180_CURRENT_CONSENSUS_RUNTIME',authorization='User continue; actual current invocation/input cost check, no newtraining/physical.',
        modes=MODES,population='All4oldOOFfolds/16dates/3seeds/2378starts;tworeverseorders,allcases,weights unchanged.',
        included='Everyreachedstate:original41normalization,neededcurrentphysical4or8calculation+normalization,inputconcatenation,tensorpacking,NNforward,decision/searchloopandlightscoretracecopies. EAGER3 all3,CONSENSUScheappairthenFACTORonlyondisagreement.',
        excluded='Raw observation decode/knowncandidateplanconstruction andcanonical68/paidfeedback8 encoding,original41raw projection,weightload/warmup,physicalsimulations/control/queries/diskwrites. Not fullcontroltime; source cachedpublicinputs shared.',
        implementation='Vectorize the same GRID173fourpowerdescriptors;all3582rawpackets must reproduceoriginalqualified8values beforetiming. No newfield,labelsorpolicy.',
        matching='CPU1thread/no_grad/batch1. Three models loaded/warmed outside timers for allmodes. Allnative decisions match savedbatch32 scores and eachcomplete search.',
        limitations='Two timingrepeats only;not hardwaregeneral. Candidatequery increase remains separate;old171times not reused. Prediction/head runtime cannot establish wholeweekadvantage.',
        prohibited=['newfits/features/weights/thresholds/rules','concurrenttraining/heavyjobs','newphysical/data/final8','paper']))
    meta=read(D.sources()[0]/'metadata.json');m173=read(R/'outputs/grid173_public_state/metadata.json');assert [(m['sample_id'],m['position']) for m in meta]==[(m['sample_id'],m['position']) for m in m173]
    ctx={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['partition']=='train':ctx[row['sample_id'],row['position']]=row['public_input']
    schema=read(R/'outputs/grid173_public_state/vector_schema.json');static=read(R/'outputs/grid03/scoring_smoke/agent_episode/dict_action_space.json');fields={}
    for source in read(R/'outputs/grid173_public_state/state_sources.json'):
        path=R/source['source_call'];z=np.load(path.with_name(path.stem+'_state.npz'))['observation'];parts={r['name']:z[r['offset']:r['offset']+r['size']] for r in schema}
        p=np.zeros(540,float)
        for pos,name,sign in [('gen_pos_topo_vect','gen_p',1),('load_pos_topo_vect','load_p',-1),('line_or_pos_topo_vect','p_or',-1),('line_ex_pos_topo_vect','p_ex',-1),('storage_pos_topo_vect','storage_power',-1)]:p[np.asarray(static[pos],int)]=sign*parts[name].astype(float)
        p[parts['topo_vect']<=0]=0;fields[source['source_call']]=(p,parts['rho'],parts['line_status'].astype(bool))
    bounds=np.r_[0,np.cumsum(static['sub_info'])];packets=[];reference=np.load(R/'outputs/grid173_public_state/features.npz')['physical8'];maxerr=0.
    spec=importlib.util.spec_from_file_location('g180_reference_feature',R/'work/grid173_public_state/feature_core.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
    for i,m in enumerate(m173):
        c=ctx[m['sample_id'],m['position']];p,rho,live=fields[m['source_call']];station=m['station'];a,b=bounds[station:station+2]
        public=dict(terminal_power=p[a:b],rho=rho,line_status=live,incident_lines=(np.asarray(static['line_or_to_subid'])==station)|(np.asarray(static['line_ex_to_subid'])==station),anchor_plan=c['incumbent_plan'],next_plans=c['next_plans'])
        x=power8(public);old=F.compute(public);maxerr=max(maxerr,float(abs(x-old).max()));assert np.allclose(x,old,atol=1e-7,rtol=1e-7) and np.array_equal(old,reference[i]);assert np.array_equal(state4(public),old[:4]);packets.append(public)
    with (O/'runtime_contexts.pkl').open('xb') as f:pickle.dump(packets,f,protocol=5)
    save('preflight.json',dict(passed=True,public_packets=3582,vectorized_vs_qualified_maxerror=maxerr,all4statecolumns_exact=True,new_forwards=0,new_fits=0,new_physics=0))
    paths=[C.O/n for n in ['complete.json','review.json','oof_trajectories.json','fixed_decisions.npz','family_summary.json']]+[R/'outputs/grid173_public_state'/n for n in ['metadata.json','state_sources.json','source_hashes.json','features.npz','vector_schema.json']]+[R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz',R/'outputs/grid113_relational_learning/train_public.pkl',R/'outputs/grid121_date_transfer/train_arrays.npz',R/'outputs/grid03/scoring_smoke/agent_episode/dict_action_space.json',O/'design.json',O/'preflight.json',O/'runtime_contexts.pkl']
    paths +=[path/n for path in D.sources() for n in ['metadata.json','fold_arrays.npz','state_scores.npz','trained_freeze.json','fits.json']]+[D.O/'state_scores.npz',S.O/'state_scores.npz']
    inherited=read(R/'outputs/grid173_public_state/source_hashes.json');inherited.update({p.relative_to(R).as_posix():sha(p) for p in paths});save('source_hashes.json',inherited);save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),R/'work/grid173_public_state/feature_core.py']});verify();archive('frozen');print('GRID180_FROZEN',maxerr)
def benchmark():
    verify();assert not (O/'timing_started.json').exists();D.G.configure();import torch
    save('timing_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),platform=platform.platform(),torch=torch.__version__,threads=torch.get_num_threads()));Old=D.G.old_api();pub=pickle.loads((R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes());raw=np.load(R/'outputs/grid121_date_transfer/train_arrays.npz')['raw'];packets=pickle.loads((O/'runtime_contexts.pkl').read_bytes());refs=read(C.O/'oof_trajectories.json');out=[];checks=collections.Counter();wall=time.perf_counter()
    for fold,path in enumerate(D.sources()):
        z=np.load(path/'fold_arrays.npz');powernorm=np.load(D.O/f'fold{fold}_power.npz');meta=read(path/'metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};old=np.load(path/'state_scores.npz');q4=np.load(S.O/'state_scores.npz');q8=np.load(D.O/'state_scores.npz')
        for seed in [0,1,2]:
            nets={};fnet,_=D.G.build(seed,'ZERO_MAIN');fnet.load_state_dict(torch.load(path/'models'/f'FACTOR_seed{seed}.pt',weights_only=False,map_location='cpu')['state_dict']);fnet.eval();nets['FACTOR']=fnet
            for arm,src,name in [('STATE4',S.O,f'fold{fold}_seed{seed}'),('POWER8',D.O,f'fold{fold}_POWER8_seed{seed}')]:
                net,_=D.build(seed);net.load_state_dict(torch.load(src/'models'/f'{name}.pt',weights_only=False,map_location='cpu')['state_dict']);net.eval();nets[arm]=net
            expected={'FACTOR':old[f'FACTOR_seed{seed}'],'STATE4':q4[f'fold{fold}_seed{seed}'],'POWER8':q8[f'fold{fold}_POWER8_seed{seed}']}
            seedrefs={mode:[r for r in refs if r['method']==f"{('CONSENSUS' if mode in ['CONSENSUS','EAGER3'] else mode+'_HELPER' if mode!='FACTOR' else mode)}_seed{seed}" and r['date'] in read(path/'design.json')['fold']['held_dates']] for mode in MODES}
            assert all(len(v)>0 for v in seedrefs.values());first=seedrefs['FACTOR'][0];i=lookup[first['sample_id'],0];warm=dict(pub[i],global41=z['normalized'][i]);inputf=Old.collate([warm]);vec=np.r_[z['normalized'][i],pub[i]['known_query_cost'],powernorm['normalized'][i]].astype('f');vec4=vec.copy();vec4[46:]=0
            with torch.no_grad():
                for _ in range(3):fnet(**inputf);nets['STATE4'](torch.tensor(vec4[None]));nets['POWER8'](torch.tensor(vec[None]))
            for repeat,order in enumerate([MODES,list(reversed(MODES))]):
                for mode in order:
                    for reference in seedrefs[mode]:
                        sid=reference['sample_id'];ids=[lookup[sid,j] for j in range(sum(m['sample_id']==sid for m in meta))];p=0;actions=[];observed=[];counts=collections.Counter();parts=collections.Counter();started=time.perf_counter()
                        def infer(name,arg,index):
                            t=time.perf_counter();q=(nets[name](**arg) if name=='FACTOR' else nets[name](arg))[0];parts[name+'_forward_s']+=time.perf_counter()-t;counts[name]+=1;observed.append((name,index,q.numpy().copy()));return bool(q[1]>q[0])
                        def inputf_at(index,norm):
                            t=time.perf_counter();value=Old.collate([dict(pub[index],global41=norm)]);parts['packing_s']+=time.perf_counter()-t;counts['FACTOR_pack']+=1;return value
                        def tensor(x):
                            t=time.perf_counter();value=torch.from_numpy(x[None]);parts['packing_s']+=time.perf_counter()-t;return value
                        with torch.no_grad():
                            while p<len(ids):
                                if time.perf_counter()-wall>900:raise TimeoutError('Preservepartialtiming,noextension')
                                i=ids[p];t=time.perf_counter();norm=np.clip((raw[i]-z['mean'])/z['scale'],-20,20).astype('f');parts['features_s']+=time.perf_counter()-t
                                if mode=='FACTOR':go=infer('FACTOR',inputf_at(i,norm),i)
                                else:
                                    t=time.perf_counter()
                                    if mode=='STATE4':
                                        physical=np.r_[np.clip((state4(packets[i])-powernorm['mean'][:4])/powernorm['scale'][:4],-20,20),np.zeros(4)].astype('f');counts['state4_features']+=1
                                    else:physical=np.clip((power8(packets[i])-powernorm['mean'])/powernorm['scale'],-20,20).astype('f');counts['power8_features']+=1
                                    x8=np.r_[norm,pub[i]['known_query_cost'],physical].astype('f');x4=x8.copy();x4[46:]=0;parts['features_s']+=time.perf_counter()-t
                                    if mode=='STATE4':go=infer('STATE4',tensor(x4),i)
                                    elif mode=='POWER8':go=infer('POWER8',tensor(x8),i)
                                    else:
                                        a=infer('STATE4',tensor(x4),i);b=infer('POWER8',tensor(x8),i)
                                        if mode=='EAGER3':f=infer('FACTOR',inputf_at(i,norm),i);go=(int(a)+int(b)+int(f))>=2
                                        elif a==b:go=a
                                        else:go=infer('FACTOR',inputf_at(i,norm),i)
                                actions.append(int(go))
                                if not go:break
                                p+=1
                        elapsed=time.perf_counter()-started;exclusive=sum(parts.values());assert exclusive<=elapsed+1e-9
                        for name,index,value in observed:
                            q=expected[name][index];assert np.allclose(value,q,atol=3e-5,rtol=3e-5) and bool(value[1]>value[0])==bool(q[1]>q[0]);checks['saved_scores_checked']+=1
                        assert p==reference['batches'] and actions==reference['actions'];checks['complete_searches_checked']+=1
                        out.append(dict(fold=fold,seed=seed,repeat=repeat,mode=mode,sample_id=sid,date=reference['date'],physical_state_group=reference['physical_state_group'],elapsed_ms=1000*elapsed,features_ms=1000*parts['features_s'],packing_ms=1000*parts['packing_s'],forward_ms=1000*sum(v for k,v in parts.items() if k.endswith('_forward_s')),other_ms=1000*(elapsed-exclusive),calls=dict(counts),queries=reference['queries'],gain=reference['gain'],chosen_pool_id=reference['chosen_pool_id']))
                    save('timing_progress.json',dict(fold=fold,seed=seed,repeat=repeat,mode=mode,rows=len(out),wall_s=time.perf_counter()-wall));print(json.dumps(read(O/'timing_progress.json')),flush=True)
    save('all_timing_records.json',out);families={};perdate=[];perrepeat=[]
    for mode in MODES:
        rr=[r for r in out if r['mode']==mode];dd=[]
        for date in sorted({r['date'] for r in rr}):
            rows=[r for r in rr if r['date']==date];d=dict(mode=mode,date=date,**{k:float(np.mean([r[k] for r in rows])) for k in ['elapsed_ms','features_ms','packing_ms','forward_ms','other_ms','queries']},factor_calls=float(np.mean([r['calls'].get('FACTOR',0) for r in rows])));dd.append(d);perdate.append(d)
        families[mode]={k:float(np.mean([r[k] for r in dd])) for k in ['elapsed_ms','features_ms','packing_ms','forward_ms','other_ms','queries','factor_calls']}
        for repeat in [0,1]:
            rows=[r for r in rr if r['repeat']==repeat];perrepeat.append(dict(mode=mode,repeat=repeat,elapsed_ms=float(np.mean([np.mean([r['elapsed_ms'] for r in rows if r['date']==date]) for date in {r['date'] for r in rows}]))))
    con=families['CONSENSUS'];base=families['FACTOR'];eager=families['EAGER3'];assert con['queries']==eager['queries'];qextra=con['queries']-base['queries'];ms_saved=base['elapsed_ms']-con['elapsed_ms'];key=dict(current_pipeline_reduction_vs_factor=1-con['elapsed_ms']/base['elapsed_ms'],equivalent_execution_reduction_vs_eager=1-con['elapsed_ms']/eager['elapsed_ms'],pipeline_ms_saved_vs_factor=ms_saved,additional_queries_vs_factor=qextra,conditional_break_even_ms_per_extra_query=ms_saved/qextra if qextra>0 else None,total_wall_s=time.perf_counter()-wall,not_full_control_time=True)
    save('family_summary.json',families);save('per_date_summary.json',perdate);save('per_repeat_summary.json',perrepeat);save('key_changes.json',key);verify();save('review.json',dict(passed=True,checks=dict(checks),all4folds16dates3seeds2repeats=True,electrical_computation_included=True,all_decisions_unchanged=True,exclusive_time_buckets=True,new_fits=0,new_physics=0,final8_absent=True));print('GRID180_TIMING_COMPLETE',json.dumps(key))
def finish():
    verify();assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();k=read(O/'key_changes.json');f=read(O/'family_summary.json');r=read(O/'per_repeat_summary.json')
    lines=['# GRID180：共识与分歧委托的当前输入和模型开销','',f"共识后委托FACTOR的当前局部输入/模型流水线，相对原FACTOR耗时变化{-k['current_pipeline_reduction_vs_factor']:+.2%}；相对等价的三模型全部运行减少{k['equivalent_execution_reduction_vs_eager']:.2%}。每次停止决定和完整补查轨迹都与GRID179一致。",'',
        '本次把原41项归一化、新增电气描述计算及归一化、张量打包、NN前向、决策循环和轻量核验记录计入。功率—分区4项采用数组运算，3582个公开输入全部复现原GRID173描述；没有更改模型、特征定义或规则。','',
        '未计入原41项的原始投影、公开候选计划构造与68维动作/8维反馈编码、物理仿真、权重加载/预热或共同控制器。原FACTOR使用已保存的动作/反馈编码，新增电气描述则现场计算。这里是明确划定范围的局部流水线计时，不是完整控制时间。','',
        '| 方法 | 总毫秒/起点 | 特征 | 打包 | NN前向 | 其余 | F调用/起点 | 查询/起点 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,v in f.items():lines.append(f"| {method} | {v['elapsed_ms']:.4f} | {v['features_ms']:.4f} | {v['packing_ms']:.4f} | {v['forward_ms']:.4f} | {v['other_ms']:.4f} | {v['factor_calls']:.3f} | {v['queries']:.3f} |")
    lines+=['','| 方法 | 计时重复 | 宏平均毫秒/起点 |','|---|---:|---:|']
    for v in r:lines.append(f"| {v['mode']} | {v['repeat']} | {v['elapsed_ms']:.4f} |")
    lines+=['',f"共识策略相对原F平均增加{k['additional_queries_vs_factor']:.4f}个候选查询，当前已计入流水线节省{k['pipeline_ms_saved_vs_factor']:+.4f}毫秒。若平均新增查询耗时超过{k['conditional_break_even_ms_per_extra_query']:.4f}毫秒，这部分候选成本就会超过上述流水线节省；这是条件比值，不是本轮测得的仿真时间。" if k['conditional_break_even_ms_per_extra_query'] is not None else '没有正的新增查询数，不定义这个抵消比值。','',
        '本轮CPU单线程、batch1、四折16日期三种子、两次反向执行顺序。所有方法加载同一组模型并预热，计时期间没有训练或其它实验；全部记录和不利计时结果保留。查询/收益仍沿用GRID179的取舍，不把更多收益说成更少查询，也不套用旧GRID171时间。',
        '', '当前可以说明哪些模型调用/输入计算被省掉，尚不能据此证明整周更快或质量达标。任何物理闭环验证都应先冻结候选、对照和运行范围。无新预测、回放、训练、最终8或论文改动。']
    (O/'GRID180_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');save('complete.json',dict(passed_audit=True,new_fits=0,new_physics=0,final8_absent=True));archive('complete');collab=R/'论文协作资料/FRAME_共识委托实际开销_GRID180';collab.mkdir(exist_ok=False)
    for n in ['GRID180_report.md','key_changes.json','family_summary.json','per_repeat_summary.json','review.json','complete.json']:shutil.copy2(O/n,collab/n)
    msg=f"GRID180当前电气描述/归一化/打包/NN/循环全部指定计入，四折16日期3种子双反向，决定及轨迹不变。共识对F流水线减少{k['current_pipeline_reduction_vs_factor']:.2%}，对等价全三模型减少{k['equivalent_execution_reduction_vs_eager']:.2%}；候选查询仍增加{k['additional_queries_vs_factor']:.3f}/起点，必须单列。不含原投影/动作编码/物理共同控制，非完整控制提速；无新物理/训练/最终8。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID180共识当前开销\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID180当前共识流水线开销\r\n\r\n[报告](protocols/GRID180_consensus_runtime/complete/GRID180_report.md)；新增电气描述计入、查询代价分列，不当作整周提速。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID180_COMPLETE',json.dumps(k))
if __name__=='__main__':globals()[sys.argv[1]]()
