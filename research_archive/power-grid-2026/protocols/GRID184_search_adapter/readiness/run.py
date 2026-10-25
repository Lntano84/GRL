"""Reviewable physical plan and no-environment Search adapter qualification."""
from pathlib import Path
import ast,collections,copy,gzip,hashlib,importlib.util,json,pickle,shutil,sys,time,types
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid184_search_adapter';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID184_search_adapter'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for n in ['code_freeze.json','data_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    old=read(R/'outputs/grid100_data_foundation/design.json')
    for item in old['scenario_plan']:
        if item['partition']=='test':assert not (R/old['environment']/'chronics'/item['scenario']).exists()
def extract(path,name,cls=None,namespace=None):
    source=path.read_text(encoding='utf8');tree=ast.parse(source)
    nodes=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={} if namespace is None else namespace;exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);return ns[name]
def prepare():
    assert not O.exists();O.mkdir(parents=True);assert read(R/'outputs/grid183_online_contract/complete.json')['passed_audit']
    old=read(R/'outputs/grid104_learning_closedloop/design.json');methods=[dict(method=m,completion_mode=m,model_seed=None) for m in ['FRAME_GUARD','LOCAL_BATCH1']]
    methods += [dict(method=f'{m}_seed{s}',completion_mode=m,model_seed=s) for m in ['FACTOR','CONSENSUS'] for s in [0,1,2]]
    for m in methods:m.update(policy=m['completion_mode'],rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True,order_mode='STATIC_ID')
    execution=[]
    for repeat in [0,1]:
        dates=old['selected'] if repeat==0 else list(reversed(old['selected']))
        for j,date in enumerate(dates):
            shift=j%len(methods);order=methods[shift:]+methods[:shift]
            if repeat:order=list(reversed(order))
            for method in order:execution.append(dict(method,repeat=repeat,week=date['scenario'],date=date['date'],stage_phase='all',partition='validation_development'))
    assert len(execution)==64
    save('design.json',dict(id='GRID184_CONSENSUS_CLOSED_LOOP_PROPOSAL',status='PREPARED_NOT_AUTHORIZED_TO_RUN_PHYSICS',
        preparation_authorization='User continue; weights/nativecore and Search integration preparation. Actualphysical replay awaits explicit scope decision.',
        question='Does the fixed cheapconsensus preserve originalwholeweekquality and retain small recurring efficiency gains after actual feature/model and physicalquery cost?',
        environment=old['environment'],environment_seed=0,selected=old['selected'],full_references=old['full_references'],methods=methods,execution_order=execution,repeats=2,
        weights='All9GRID182frozenfull16trainweights;zero fitting,weights chosenbeforedevelopment replay. All3seeds retained,not bestseed.',
        matching='OriginalFRAME global128/64/patience2/.9,initial lockedstation,candidatealiases/idorder/controller/reward/DC/safeQP/priorcache. Same savedgraphcontext for allarms.',
        seed_pairing='FACTOR_seedS versus CONSENSUS_seedS. All8 arms all4dates,firstexisting replicate100. Timingrepeat not newscenario. No newdata/test8.',
        timing='Fresh serial worker. Full ctl.propose includes candidate preparation,graphcontext,raw41/knownplans/physicaldescriptors,normalization/packing/neededNNs/control and actualforecasts. Loading/dummywarmup outside timer but retained setup. Allsteps+searchstepsP95/P99;exclusive optimizer/forecast/searchoverhead/other.',
        caps=dict(episode_s=1200,wall_s=14400,physical_steps=130000,public_forecasts=6000000),
        quality_threshold=.005,quality='FULLcomplete=>require completion and cost<=100.5%FULL. FULLincomplete=>REFERENCE_LIMITED,report survival/horizon and commonprefix costs,not wholeweekqualitypass. Earlyfailure never as speedup. FRAME/BATCH1 samebatch costs also reported.',
        readout='No mandatory5%gain orallseedwins to keep exploratory signal. Report eachseed/date/repeat,pairedfamilymacro,quality/survival/cost/query/actualtime separately. Positive small recurringeffect remains researchworthy;not automatic adoption.',
        stop='Stop on engineering failure or original4h benchmark window,keep partial outputs/no autoextension. Never stop selectively for interim method outcome,train/rerank/tune or droplosingseeds.',
        audit='Only after all64workersfinish:serializedvector/costledger/feedback/context/input/model/global+local reexecution;no newphysicalforecast/independentACsolve. Exact repeattrajectories. Report engineeringfallbacks,allqueries and modelcost.',
        live_qualification='First plannedFRAMErun uses commonworker;any interfacefault stops and retains failure. No extra uncounted rollout. Offline Search test does not qualify solver/live forecast.',
        boundaries=['notsecondgrid/blindtest','notmanuscript update','no final8generation','no additionalfits','no newbaseline/thresholdsearch','noautomatic expansion']))
    data=read(R/'outputs/grid104_learning_closedloop/data_freeze.json');data.update(read(R/'outputs/grid183_online_contract/source_hashes.json'))
    for p in [R/'outputs/grid182_deployment_bundle'/n for n in ['normalization.npz','deployment_manifest.json','trained_freeze.json','engineering_train_scores.npz']]+list((R/'outputs/grid182_deployment_bundle/models').glob('*.pt'))+[O/'design.json']:
        data[p.relative_to(R).as_posix()]=sha(p)
    code=read(R/'outputs/grid104_learning_closedloop/code_freeze.json');code.update(read(R/'outputs/grid183_online_contract/code_freeze.json'))
    for p in [*W.glob('*.py'),R/'work/grid113_relational_learning/model.py',R/'work/grid26/engine.py',R/'work/grid08/common.py']:code[p.relative_to(R).as_posix()]=sha(p)
    save('data_freeze.json',data);save('code_freeze.json',code);verify();archive('frozen');print('GRID184_PREPARED_NO_PHYSICS')

def qualify():
    verify();assert not (O/'offline_review.json').exists();started=time.perf_counter()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sentinel=types.ModuleType('common');sentinel.live_controller_sentinel=True;assert 'common' not in sys.modules;sys.modules['common']=sentinel
    sp=importlib.util.spec_from_file_location('g184_adapter',W/'adapter.py');B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B)
    models=[B.M.FrozenModels(s) for s in [0,1,2]];assert sys.modules['common'] is sentinel
    sys.path.insert(0,str(R/'work/grid111_action_feedback'));from public import Decoder,av
    decoder=Decoder();prepare_fn=extract(R/'work/grid26/engine.py','prepare','Search',dict(np=np,action_vector=av));graph=extract(R/'work/grid08/common.py','graph_features',namespace=dict(np=np))
    checks=collections.Counter();maxerr=0.;scores=np.load(R/'outputs/grid182_deployment_bundle/engineering_train_scores.npz');x=np.load(R/'outputs/grid182_deployment_bundle/training_inputs.npz');pub=pickle.loads((R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes())
    for seed,model in enumerate(models):
        for arm in ['STATE4','POWER8']:
            with torch.inference_mode():q=model.nets[arm](torch.from_numpy(x[arm])).numpy()
            old=scores[f'{arm}_seed{seed}'];assert np.allclose(q,old,atol=3e-5,rtol=3e-5) and np.array_equal(q[:,1]>q[:,0],old[:,1]>old[:,0]);checks['isolated_model_rows']+=len(q);maxerr=max(maxerr,float(abs(q-old).max()))
        for i,row in enumerate(pub):
            q=model.factor(row);old=scores[f'FACTOR_seed{seed}'][i];assert np.allclose(q,old,atol=3e-5,rtol=3e-5) and bool(q[1]>q[0])==bool(old[1]>old[0]);checks['isolated_model_rows']+=1;maxerr=max(maxerr,float(abs(q-old).max()))
    for j,source in enumerate(read(R/'outputs/grid173_public_state/state_sources.json')):
        rel=source['source_call'];call,ob,plans,unknown,aliases=decoder.decode(rel)
        with np.load((R/rel).with_name((R/rel).stem+'_state.npz')) as z:base=decoder.ac.from_vect(z['base_action'])
        truth={r['pool_id']:r for r in call['outcomes']}
        class Search:
            prepare=prepare_fn
        search=Search();search.env=decoder.ac;search.subids=decoder.subs;search.library=decoder.library;search.rule='STATIC_ID';search.policy='unused'
        eligible,reps,aa,actions,changed=search.prepare(ob,base);assert reps==call['representatives'] and aa==aliases and changed==call['masked_ids'];checks['real_prepare_functions']+=1
        key_to_id={av(a+base).tobytes():i for i,a in actions.items()};query_ids=[]
        class SavedForecast:
            def __getattr__(self,k):return getattr(ob,k)
            def simulate(self,action):
                i=key_to_id[av(action).tobytes()];assert i not in query_ids;query_ids.append(i);row=truth[i]
                return types.SimpleNamespace(rho=np.array([row['rho']],dtype=np.float32)),row['reward'],row['done'],{k:row[k] for k in ['illegal','ambiguous','exceptions']}
        obs=SavedForecast()
        for method in read(O/'design.json')['methods']:
            seed=method['model_seed'];policy=B.install(Search,method,control_api=(lambda info:info,graph),models=None if seed is None else models[seed]);query_ids.clear()
            selected,log,context,prior=search.choose(obs,base,0.,rho_threshold=call['rho_limit'])
            ids=[]
            def reveal(i,phase):ids.append(i);return {k:truth[i][k] for k in ['pool_id',*B.N.FEEDBACK]}
            provider=B.N.PublicState(ob,base,actions,decoder.ac)
            expected=policy.run(sorted(reps),reveal,decoder.subs,aliases,limit=call['rho_limit'],provider=provider)
            assert query_ids==ids and log['chosen_pool_id']==expected['chosen_pool_id'] and log['engineering_fallback'] is None
            assert selected is None if log['chosen_pool_id'] is None else np.array_equal(av(selected),av(actions[log['chosen_pool_id']]))
            for field in ['raw_features','physical8','action','next_ids','remaining','paid_queries','delegated','scores']:
                assert [v.get(field) for v in log['learned_decisions']]==[v.get(field) for v in expected['learned_decisions']],(j,method,field)
            exclusive=log['prepare_s']+log['forecast_s']+log['feature_s']+log['local_inference_s'];assert exclusive<=log['source_s']+1e-7
            assert set(context)=={'x','edges','glob'} and prior is None;checks['search_choose_calls']+=1
        if j%60==0:print('ADAPTER_SAVED_CALLS',j,flush=True)
    assert sys.modules['common'] is sentinel;verify()
    save('offline_review.json',dict(passed=True,checks=dict(checks),isolated_max_score_error=maxerr,common_name_collision_avoided=True,actual_prepare_and_graph_function_bodies=True,saved_oracle_wrapper=True,real_environment_constructions=0,new_physical_forecasts=0,new_fits=0,dev_feature_rows=0,final8_absent=True,wall_s=time.perf_counter()-started))
    save('preparation_complete.json',dict(passed=True,offline_adapter_qualified=True,physical_authorization_present=(O/'physical_authorization.json').exists(),physical_runs_started=False))
    (O/'GRID184_preparation_report.md').write_text('# GRID184：小闭环准备完成，尚未启动\n\n独立部署模型定义解决训练与控制器同名模块风险；32238行输出/停止决定复现原权重。使用真实候选准备与图状态构造函数体、保存观测和外部保存反馈回调，222个调用×8臂的Search.choose通过输入/查询/选动作与互斥计时检查，没有创建物理环境。\n\n准备的范围为原4个已暴露开发日期，各取已有scenario100；原FRAME、固定一批、原FACTOR三种子、共识三种子，共8臂，两次交错反向计时，总64条。仍是同一电网的开发验证，不是新增场景或最终测试。训练、控制器、排序和费用均冻结。\n\n每条最多1200秒，回放总窗口4小时，不自动延长；所有种子与早停、不利结果保留。重点核对整周完成/成本与FULL的100.5%标准、实际完整控制时间、候选与总查询、全步骤/搜索步骤P95/P99、特征与模型开销。FULL未完周单列，不把早停当提速。不要求每个日期都赢或超过5%才保留探索价值。\n\n当前只取得无物理的接口资格。真实求解器/在线forecast尚未测试；首条计划内FRAME承担运行接口检查，发生工程错误即停并保留。实际物理运行须用户确认上述64条范围。\n',encoding='utf8',newline='\n')
    archive('qualified');dest=R/'论文协作资料/FRAME_小闭环准备_GRID184';dest.mkdir(exist_ok=False)
    for n in ['GRID184_preparation_report.md','design.json','offline_review.json','preparation_complete.json']:shutil.copy2(O/n,dest/n)
    msg='GRID184部署定义隔离common同名风险，32238行权重复现，222保存调用×8臂真实prepare/graph函数体及Search.choose核验通过。准备64条原4开发日期同批闭环，原FRAME/BATCH1/F三种子/共识三种子双反向，尚未授权或启动新物理。小改善保留，整周质量标准不改，原FULL时间不混用。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID184小闭环准备\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID184等待闭环范围决定\n\n1. 182权重、183补查核心、184Search接口已通过无物理核对，资料已归档。\n2. 提请用户决定64条已有开发场景同批闭环范围；没有physical_authorization.json不得运行worker/pipeline。\n3. 获同意后原权重零拟合执行全部种子/日期/双重复，完成后零预测审计，报告小改善及反例，不自动扩大/最终8/论文。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID184小闭环准备\r\n\r\n[报告](protocols/GRID184_search_adapter/qualified/GRID184_preparation_report.md)；64条范围等待决定，当前零新物理。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID184_OFFLINE_ADAPTER_PASSED',json.dumps(read(O/'offline_review.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
