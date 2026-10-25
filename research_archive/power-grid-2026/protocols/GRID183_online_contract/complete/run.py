"""Saved-state engineering qualification; no new physical predictions or fitting."""
from pathlib import Path
import collections,copy,gzip,hashlib,json,pickle,shutil,sys,time
import numpy as np
import runtime as N
R=N.R;W=Path(__file__).parent;O=R/'outputs/grid183_online_contract';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID183_online_contract'
read=N.D.read;sha=N.D.sha
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for name in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/name).items():assert sha(R/p)==h,p
    N.D.H.sealed()
def prepare():
    assert not O.exists();O.mkdir(parents=True);assert read(R/'outputs/grid182_deployment_bundle/complete.json')['passed_audit']
    save('design.json',dict(id='GRID183_NATIVE_PUBLIC_INPUT_AND_COMPLETION_CONTRACT',authorization='User continue, deployment engineering only; no new physical approval implied.',
        population='All3582originaltrainprefixstates/2378starts/222savedcalls/16train dates. Original4devnotdecoded,final8sealed.',
        question='Can online currentobs+preparedcommands+paidfeedback reproduce qualified traininginputs andcomplete savedsearch choices?',
        implementation='No truth/target argument in PublicState or model. Exact old41projection,canonical68/feedback8,currentphysical8;frozen182normalization/models;strictq1>q0 andfixedconsensus.',
        global_search='Delegate original91 NO_LOCAL,thenlockedinitialstation,batches64,observedfloat32reward/lowIDbest;controls originalFRAME andBATCH1.',
        fallback='If knownintendedreconnectionbus is unknown or schema unsupported,finish all remaining lockedlocalcandidates as originalFRAME,logengineeringreason. Never invent bus/input. No broadexception catch.',
        qualification='All3582input/score rows,all2378localpaths x3seeds x2learnedarms plusBATCH1;222wholeglobalsavedcalls samearms,FRAMEsamequerysequence. Candidatealias/masking/savedcommandhash invariants;no doublequeries/returnedbest. Probeunrevealedtruthisnotpassed;unsupportedinputexplicitfallback.',
        restrictions=['no fits/optimizer','no environment make/init/reset/step/simulate','no dev/testfeatures','no physicaltime/qualityclaim','no rule/feature/seed search','no manuscript'],
        next='Ifpassed,prepare concrete bounded physicalprotocol andreviewable runner;userdecides actual replay. Offlinepassisnotfullclosedloopquality.'))
    paths=[R/'outputs/grid182_deployment_bundle'/n for n in ['deployment_manifest.json','normalization.npz','trained_freeze.json','engineering_train_scores.npz','review.json','complete.json']]
    paths += list((R/'outputs/grid182_deployment_bundle/models').glob('*.pt'))
    paths += [R/'outputs/grid173_public_state'/n for n in ['features.npz','metadata.json','source_hashes.json']]
    paths += [R/'outputs/grid113_relational_learning'/n for n in ['train_public.pkl','train_metadata.json']]
    paths += [R/'outputs/grid121_date_transfer/train_arrays.npz',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json',R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz',O/'design.json']
    paths += [R/p for p in ['work/grid174_power_helper/run.py','work/grid91_mechanism/policies.py','work/grid101_gain_learning/features.py','work/grid102_search_rl/observations.py','work/grid112_relational_plan/schema.py','work/grid180_consensus_runtime/runtime_features.py','work/grid111_action_feedback/public.py','work/grid111_action_feedback/common111.py']]
    inherited=read(R/'outputs/grid173_public_state/source_hashes.json');inherited.update({p.relative_to(R).as_posix():sha(p) for p in paths});save('source_hashes.json',inherited)
    save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();archive('frozen');print('GRID183_FROZEN')

def qualify():
    verify();assert not (O/'qualification_started.json').exists();save('qualification_started.json',dict(pid=__import__('os').getpid(),new_fits=0,new_physics=0));started=time.perf_counter();N.D.G.configure()
    # Decoder additionally disables actual simulator entrypoints globally.
    sys.path.insert(0,str(R/'work/grid111_action_feedback'));from public import Decoder,av
    decoder=Decoder();models=[N.FrozenModels(seed) for seed in [0,1,2]]
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)}
    raw_expected=np.load(R/'outputs/grid121_date_transfer/train_arrays.npz')['raw'];power_expected=np.load(R/'outputs/grid173_public_state/features.npz')['physical8']
    inputs_expected=np.load(R/'outputs/grid182_deployment_bundle/training_inputs.npz');public_expected=pickle.loads((R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes());score_expected=np.load(R/'outputs/grid182_deployment_bundle/engineering_train_scores.npz')
    contexts={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['partition']=='train':contexts[row['sample_id'],row['position']]=row['public_input']
    samples={v['sample_id']:v for v in read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples'] if v['partition']=='train'}
    cases=[v for v in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if v['partition']=='train']
    decoded={};providers={};checks=collections.Counter();max_score_error=0.;traces=[]
    def source(sm):
        rel=sm['source_call']
        if rel not in decoded:
            c,ob,plans,unknown,aliases=decoder.decode(rel);decoded[rel]=(c,ob,plans,unknown,aliases)
            path=R/rel
            with np.load(path.with_name(path.stem+'_state.npz')) as z:base=decoder.ac.from_vect(z['base_action'])
            op=decoder.ac.line_or_pos_topo_vect;ep=decoder.ac.line_ex_pos_topo_vect
            intent=(base.line_set_status>0)|base.line_change_status|(base.set_bus[op]>0)|(base.set_bus[ep]>0)
            disconnected=(~ob.line_status)&~intent;maskpos=np.r_[op[disconnected],ep[disconnected]];actions={}
            for i in c['representatives']:
                action=decoder.library[i].copy();bus=action.set_bus.copy();bus[maskpos]=0;action.set_bus=bus;actions[i]=action
            providers[rel]=N.PublicState(ob,base,actions,decoder.ac)
            # Native object contains no outcomes, reward oracle or source call.
            assert set(vars(providers[rel]))=={'ob','base','actions','static','bounds','plans','unknown','power'}
        return decoded[rel],providers[rel]
    for ci,case in enumerate(cases):
        sid=case['sample_id'];sm=samples[sid];x=read(R/sm['dataset_path']/sm['input_file']);aliases={int(k):v for k,v in x['public_aliases'].items()};subids=x['public_subids'];order=x['public_order'];station=int(subids[x['incumbent_id']]);targets={int(v) for k,v in aliases.items() if subids[k]==station}
        data,provider=source(sm);call,ob,plans,unknown,_=data;truth={r['pool_id']:r for r in call['outcomes']};visible=copy.deepcopy(x['visible']);pending=x['remaining_same_station_ids'].copy();budget=len(pending)
        for j in range(len(case['states'])):
            i=lookup[sid,j];ids=pending[:64];raw=N.observe(visible,order,aliases,subids,x['rho_limit'],targets,pending,budget,j);assert np.array_equal(raw,raw_expected[i]) and np.array_equal(raw,case['states'][j]);checks['raw41_rows']+=1
            c=provider.context(visible,ids,station,targets);assert c==contexts[sid,j];checks['canonical_contexts']+=1
            for action_id in c['seen_ids']+ids:
                assert np.array_equal(provider.plan(action_id),plans[action_id]) and provider.unknown[action_id]==unknown[action_id]
                checks['native_plan_checks']+=1
            power=provider.electrical(c,station);assert np.array_equal(power,power_expected[i]);checks['physical8_rows']+=1
            for seed,model in enumerate(models):
                g,x4,x8=model.inputs(raw,power,len(ids));assert np.array_equal(x4,inputs_expected['STATE4'][i]) and np.array_equal(x8,inputs_expected['POWER8'][i]);checks['normalized_rows']+=1
                pub=N.S.pack(c,g,model.fee,model.scale)
                for k,v in pub.items():assert np.array_equal(v,public_expected[i][k]),(sid,j,k)
                for arm,argument in [('STATE4',x4),('POWER8',x8),('FACTOR',pub)]:
                    q=model.factor(argument) if arm=='FACTOR' else model.small(arm,argument);expected=score_expected[f'{arm}_seed{seed}'][i];max_score_error=max(max_score_error,float(abs(q-expected).max()));assert np.allclose(q,expected,atol=3e-5,rtol=3e-5) and bool(q[1]>q[0])==bool(expected[1]>expected[0]);checks['native_model_rows']+=1
            # Only the external saved oracle advances paid feedback, after the input check.
            for action_id in ids:visible.append({k:truth[action_id][k] for k in ['pool_id',*N.FEEDBACK]})
            pending=pending[len(ids):]
        assert not pending
        for seed,model in enumerate(models):
            ix=[lookup[sid,j] for j in range(len(case['states']))]
            for mode in ['FACTOR','CONSENSUS']:
                f=score_expected[f'FACTOR_seed{seed}'][ix];go=f[:,1]>f[:,0]
                if mode=='CONSENSUS':
                    a=score_expected[f'STATE4_seed{seed}'][ix];b=score_expected[f'POWER8_seed{seed}'][ix];a=a[:,1]>a[:,0];b=b[:,1]>b[:,0];go=np.where(a==b,a,go)
                expected=N.D.H.trajectory(case,go,mode,model.fee,model.scale)
                received=[];initial=copy.deepcopy(x['visible']);paid={r['pool_id'] for r in initial}
                def reveal(action_id,phase):
                    assert action_id not in paid;paid.add(action_id);received.append(action_id);return {k:truth[action_id][k] for k in ['pool_id',*N.FEEDBACK]}
                result=dict(outcomes=initial,events=[],completion_candidate_ids=x['remaining_same_station_ids'].copy(),local_substation=station)
                result=N.CompletionPolicy(mode,model).complete(result,order,reveal,subids,aliases,x['rho_limit'],provider)
                assert result['engineering_fallback'] is None and result['chosen_pool_id']==expected['chosen_pool_id'] and len(received)==expected['queries']
                assert abs(result['local_reward_gain']-expected['gain'])<1e-12
                assert [r['action'] for r in result['learned_decisions']]==expected['actions'];checks['complete_local_searches']+=1
                assert result['chosen_pool_id']==N.best(result['outcomes'])['pool_id'];traces.append(dict(sample_id=sid,seed=seed,mode=mode,actions=expected['actions'],queries=len(received),chosen_pool_id=result['chosen_pool_id']))
        # Deterministic fixed one-batch control only once, not three independent repeats.
        got=[]
        def batch_reveal(i,phase):got.append(i);return {k:truth[i][k] for k in ['pool_id',*N.FEEDBACK]}
        stub=dict(outcomes=copy.deepcopy(x['visible']),events=[],completion_candidate_ids=x['remaining_same_station_ids'].copy(),local_substation=station)
        batch=N.CompletionPolicy('LOCAL_BATCH1').complete(stub,order,batch_reveal,subids,aliases,x['rho_limit'],None)
        assert len(got)==min(64,budget) and batch['chosen_pool_id']==case['chosen'][1];checks['batch1_searches']+=1
        if ci%300==0:print('SAVED_STARTS',ci,'INPUTS',checks['raw41_rows'],flush=True)
    # Actual global-prefix + learned-completion wiring, using the saved external oracle only.
    for rel,(call,ob,plans,unknown,aliases) in decoded.items():
        provider=providers[rel];truth={r['pool_id']:r for r in call['outcomes']};order=sorted(call['representatives']);subs=decoder.subs
        def run(policy):
            ids=[]
            def reveal(i,phase):
                assert i not in ids;ids.append(i);return {k:truth[i][k] for k in ['pool_id',*N.FEEDBACK]}
            result=policy.run(order,reveal,subs,aliases,call['rho_limit'],provider=provider)
            inc=N.best(result['outcomes']);assert result['chosen_pool_id']==(None if inc is None else inc['pool_id']);assert len(ids)==result['public_queries'];return ids,result
        first,frame=run(N.CompletionPolicy('FRAME_GUARD'));again=N.G.run(order,lambda i,phase:{k:truth[i][k] for k in ['pool_id',*N.FEEDBACK]},'FRAME_GUARD',subs,aliases,limit=call['rho_limit'])
        assert first==[r['pool_id'] for r in again['outcomes']] and frame['chosen_pool_id']==again['chosen_pool_id'];checks['whole_frame_calls']+=1
        for model in models:
            for mode in ['FACTOR','CONSENSUS']:
                ids,result=run(N.CompletionPolicy(mode,model));assert result['engineering_fallback'] is None;checks['whole_learned_calls']+=1
        ids,result=run(N.CompletionPolicy('LOCAL_BATCH1'));checks['whole_batch1_calls']+=1
    # Unsupported public command plans fall back to full original local completion.
    case=cases[0];x=read(R/samples[case['sample_id']]['dataset_path']/samples[case['sample_id']]['input_file']);rel=samples[case['sample_id']]['source_call'];truth={r['pool_id']:r for r in decoded[rel][0]['outcomes']}
    class Unsupported:
        def context(self,*args):raise N.InputUnavailable('qualification_unknown_reconnection')
    received=[]
    def reveal(i,phase):received.append(i);return {k:truth[i][k] for k in ['pool_id',*N.FEEDBACK]}
    stub=dict(outcomes=copy.deepcopy(x['visible']),events=[],completion_candidate_ids=x['remaining_same_station_ids'].copy(),local_substation=x['public_subids'][x['incumbent_id']])
    fallback=N.CompletionPolicy('CONSENSUS',models[0]).complete(stub,x['public_order'],reveal,x['public_subids'],{int(k):v for k,v in x['public_aliases'].items()},x['rho_limit'],Unsupported())
    assert received==x['remaining_same_station_ids'] and fallback['engineering_fallback']=='qualification_unknown_reconnection';checks['explicit_fallback']=1
    verify();save('qualification_counts.json',dict(checks));save('local_search_traces.json',traces)
    save('review.json',dict(passed=True,states=3582,starts=2378,train_dates=16,saved_calls=len(decoded),checks=dict(checks),max_score_error=max_score_error,decoder_checks=dict(decoder.checks),new_fits=0,new_physics=0,dev_feature_rows=0,final8_absent=True,wall_s=time.perf_counter()-started,
        remaining='Live Search adapter/common controller and fresh physical trajectory not yet exercised. Offlinequalification is not closedloopbenchmark.'))
    save('complete.json',dict(passed_audit=True,native_input_and_completion_qualified=True,physical_runs_started=False,live_adapter_qualification_pending=True))
    lines=['# GRID183：在线输入与补查核心资格','',f'使用原16个训练日期的3582个保存状态，核对当前观测、公共动作和已付费反馈到模型输入的完整转换。{checks["native_model_rows"]}行模型输出与已封存部署权重的参考输出一致，最大数值差{max_score_error:.3g}，所有停止决定一致。','',f'{checks["complete_local_searches"]}条完整局部搜索、{checks["whole_learned_calls"]}条含原全局搜索的保存状态调用，以及FRAME和固定一批对照通过查询序列、最终最佳已观察动作与不重复查询核验。它们是保存反馈下的工程重放，不是新的物理轨迹或独立验证成绩。','', '特征构造器只接收当前观测、已准备动作、静态设备位置和已经付费的反馈；未来结果只存在外部保存反馈回调中。原41维、拓扑与反馈编码、8项电气描述和归一化逐行复现，没有重训、创建电网环境或调用物理预测。','', '未知重连母线等不支持输入会显式记录原因，并完成原同站补查；没有随意填特征。该回退也已核验。原4个开发日期未读取新特征，最终8未生成。','', '下一步准备与共同控制器连接的Search适配及明确小闭环方案，完成工程检查后再由用户决定启动物理回放。当前结果不用于声称整周质量或控制时间改善。']
    (O/'GRID183_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');archive('complete')
    dest=R/'论文协作资料/FRAME_在线输入资格_GRID183';dest.mkdir(exist_ok=False)
    for n in ['GRID183_report.md','review.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg='GRID183原16训练日期3582状态在线输入复现、32238模型行、14268局部搜索及222保存全局调用各臂核验；零新物理/拟合/开发集特征，原FRAME全局不改。只取得输入和补查核心资格，实际Search适配和完整物理回放尚待。未知计划显式完整同站回退；不当新科学成绩。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID183在线输入与补查核心\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID183核心资格后\n\n1. 182部署权重、183原输入与保存搜索已核对；不把训练集工程核验当新收益。\n2. 准备实际Search接口和完整计时/质量审计协议，确定原4开发日期、全3种子、FRAME/BATCH1/F对照的范围。\n3. 新物理回放范围具体可检查后由用户决定；当前不启动，不新数据/最终8/论文。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID183在线输入资格\r\n\r\n[报告](protocols/GRID183_online_contract/complete/GRID183_report.md)；核心接口保存状态核验，尚未新物理。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID183_CORE_QUALIFIED',json.dumps(read(O/'review.json'),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
