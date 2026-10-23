"""Postfit OOF evaluation, independent audit, selective publication."""
import argparse,shutil
from common121 import *

def sync_docs(note,next_text):
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+note+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(next_text,encoding='utf8')

def archive(phase):
    dest=REPO/f'research_archive/power-grid-2026/protocols/GRID121_date_transfer/{phase}';dest.mkdir(parents=True,exist_ok=False)
    for p in W.glob('*.py'):shutil.copy2(p,dest/p.name)
    for p in O.iterdir():
        if not p.is_file():continue
        if phase=='before' and p.name not in ['design.json','preflight.json','code_freeze.json','data_freeze.json','state.json'] and 'normalization' not in p.name:continue
        shutil.copy2(p,dest/p.name)
    if phase=='complete':shutil.copytree(O/'models',dest/'models')
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
    return dest

def before():
    verify();assert read(O/'preflight.json')['passed'];archive('before')
    sync_docs('## 2026-10-23 — GRID121 grouped-date transfer diagnostic frozen\n\nAfter GRID120 training improvement/validation decline,user broad learning authorization covers announced24fit diagnostic:4fixedroundrobin folds of16originaltraindates,eachfit12dates,HUBER/UTILITY_BCE seeds0/1/2,original6978network/init/7250x32/Adam/price. Eachdate prefixes/scenarioreplicas stay together; per-fold normalization andweightnormalizer fit12datesonly. Originalhistoricalprice/gainscale usedall16dates,so diagnosticconditionalonfixedtask,not fullynested/pristineindependenttest. Nooriginal4val predictions orfinal8test/physical. All24finalendpointsbeforeheldinference,1800s cap,no tuning.',
        '# Next steps — GRID121按日期留出诊断已封存\n\n1. 完成24次配对训练，全部结束后汇总16日期折外净效用及逐折/种子结果。\n2. 核对日期分组、训练侧标准化/权重、留出日期抽样计数为0和独立前向结果。\n3. 与GRID120的训练/验证分离一起解释，不自动扩大、改费或启动新的RL/电网回放。\n')
    print('GRID121_BEFORE_ARCHIVED',flush=True)

def evaluate():
    assert not (O/'evaluation_receipt.json').exists();verify();configure();F=read(O/'fits.json');assert F['passed'] and len(F['records'])==24
    D=read(O/'design.json');meta=read(O/'metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};cases=pickle.loads((O/'train_cases.pkl').read_bytes());Q={};tr=[];fitresults={}
    for rec in F['records']:
        fold,seed,arm=rec['fold'],rec['seed'],rec['arm'];name=f'fold{fold}__{arm}_seed{seed}';x,y,ids,held,prob,Z=fold_data(fold);net=build(seed);cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu');net.load_state_dict(cp['state_dict']);assert digest(net)==rec['final_hash']
        q=forward(net,x);Q[name]=q;held_dates=set(D['folds'][fold]['held_dates']);parts={'fit':[],'held':[]}
        for c in cases:
            ii=[lookup[c['sample_id'],j] for j in range(len(c['states']))];part='held' if c['date'] in held_dates else 'fit'
            t=trajectory(c,f'{arm}_seed{seed}',q[ii,1]>q[ii,0],D['cost_per_query'],D['gain_scale']);t['fold']=fold;t['model_name']=name;parts[part].append(t)
            if part=='held':tr.append(t)
        fitresults[name]={part:summarize(rows) for part,rows in parts.items()}
    results=summaries(tr);u=lambda name:results[name]['all']['macro']['mean_net_utility'];families={a:float(np.mean([u(f'{a}_seed{s}') for s in range(3)])) for a in D['arms']}
    pairs=[dict(seed=s,huber=u(f'HUBER_seed{s}'),utility=u(f'UTILITY_BCE_seed{s}'),difference=u(f'UTILITY_BCE_seed{s}')-u(f'HUBER_seed{s}')) for s in range(3)]
    dates=sorted({m['date'] for m in meta});delta={d:float(np.mean([results[f'UTILITY_BCE_seed{s}']['all']['per_date'][d]['mean_net_utility']-results[f'HUBER_seed{s}']['all']['per_date'][d]['mean_net_utility'] for s in range(3)])) for d in dates}
    support=families['UTILITY_BCE']>families['HUBER'] and all(p['difference']>0 for p in pairs) and sum(x>0 for x in delta.values())>=12
    decision=dict(family_oof_utility=families,paired_seeds=pairs,date_utility_minus_huber=delta,positive_dates=sum(x>0 for x in delta.values()),objective_transfer_support=support,ready_to_scale=False,
        scope='OOF development diagnosis on16previously exposed train dates with fixedhistoricaltaskprice; no new independent confirmation.')
    np.savez_compressed(O/'predictions.npz',**Q);save(O/'oof_trajectories.json',tr);save(O/'oof_results.json',results);save(O/'fit_results.json',fitresults);save(O/'decision.json',decision)
    save(O/'evaluation_receipt.json',dict(passed=True,all24_endpoints_before_held_predictions=True,oof_trajectories=len(tr),original_validation_evaluations=0,new_physical_calls=0,final_test_materialized=False));save(O/'state.json',dict(stage='evaluated_waiting_audit',fits=24));print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)

def audit():
    assert not (O/'review.json').exists();verify();configure();F=read(O/'fits.json');D=read(O/'design.json');P=read(O/'preflight.json');Q=np.load(O/'predictions.npz');raw,y,pp=arrays();meta=read(O/'metadata.json');err=0.;n=0
    cases=pickle.loads((O/'train_cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)}
    for i,m in enumerate(meta):
        c=cm[m['sample_id']];j=m['position'];assert np.array_equal(raw[i],c['states'][j]);assert np.float32((c['gains'][j]-D['cost_per_query']*c['queries'][j])/D['gain_scale'])==y[i]
    for fold in range(4):
        held_dates=set(D['folds'][fold]['held_dates']);held=np.array([m['date'] in held_dates for m in meta]);fit=~held;dates=sorted({m['date'] for m in meta if m['date'] not in held_dates})
        mean=np.mean([raw[np.array([m['date']==d for m in meta])].astype(float).mean(0) for d in dates],axis=0)
        scale=np.sqrt(np.mean([((raw[np.array([m['date']==d for m in meta])].astype(float)-mean)**2).mean(0) for d in dates],axis=0));scale[scale<1e-8]=1.
        norm=np.load(O/f'fold{fold}_normalization.npz');assert np.allclose(mean,norm['mean'],atol=1e-10,rtol=1e-10) and np.allclose(scale,norm['scale'],atol=1e-10,rtol=1e-10)
        X=np.clip((raw-norm['mean'])/norm['scale'],-20,20).astype('f');prob=pp[fit]/pp[fit].sum();Z=float(prob@abs(y[fit]).astype(float));assert Z==P['fold_checks'][str(fold)]['weight_normalizer']
        for rec in [r for r in F['records'] if r['fold']==fold]:
            name=f"fold{fold}__{rec['arm']}_seed{rec['seed']}";path=O/'models'/f'{name}.pt';assert sha(path)==rec['checkpoint_sha256'];cp=torch.load(path,weights_only=False,map_location='cpu')
            net=build(rec['seed']);assert digest(net)==cp['initial_hash']==P['original_initial_hashes'][str(rec['seed'])];net.load_state_dict(cp['state_dict']);assert digest(net)==cp['final_hash']==rec['final_hash']
            assert cp['updates']==7250 and len(cp['optimizer']['state'])==len(cp['state_dict']) and all(int(s['step'])==7250 for s in cp['optimizer']['state'].values())
            ref=X.astype(float)
            for layer in [0,2,4]:
                ref=ref@cp['state_dict'][f'q_net.{layer}.weight'].numpy().astype(float).T+cp['state_dict'][f'q_net.{layer}.bias'].numpy().astype(float)
                if layer!=4:ref=np.maximum(ref,0)
            actual=Q[name];err=max(err,float(abs(ref-actual).max()));assert np.allclose(ref,actual,atol=3e-5,rtol=3e-5) and np.array_equal(ref[:,1]>ref[:,0],actual[:,1]>actual[:,0]);n+=len(y)
            ex=np.load(O/f'exposure_{name}.npz');expected=np.load(O/f"fold{fold}_seed{rec['seed']}_expected_exposure.npz")['state_draws'];assert np.array_equal(ex['state_draws'],expected) and ex['state_draws'][held].sum()==0
            assert len(ex['minibatch_losses'])==7250 and np.isfinite(ex['minibatch_losses']).all()
            assert rec['index_sha256']==P['sampling_streams'][f"fold{fold}_seed{rec['seed']}"]
    tr=read(O/'oof_trajectories.json');counts=collections.Counter((r['method'],r['sample_id']) for r in tr);assert len(tr)==6*2378 and all(v==1 for v in counts.values())
    for r in tr:
        c=cm[r['sample_id']];assert c['date'] in D['folds'][r['fold']]['held_dates'];q=Q[r['model_name']];p=0;seq=[]
        while p<len(c['states']):
            score=q[lookup[c['sample_id'],p]];a=int(score[1]>score[0]);seq.append(a)
            if not a:break
            p+=1
        assert seq==r['actions'] and p==r['batches'];assert float(c['gains'][:p].sum())==r['gain'] and int(c['queries'][:p].sum())==r['queries'] and c['chosen'][p]==r['chosen_pool_id']
        assert abs((r['gain']-D['cost_per_query']*r['queries'])/D['gain_scale']-r['net_utility'])<1e-12
    results=read(O/'oof_results.json');independent={}
    for name,res in results.items():
        rr=[r for r in tr if r['method']==name];ds=sorted({r['date'] for r in rr});assert len(ds)==16
        u=float(np.mean([np.mean([r['net_utility'] for r in rr if r['date']==d]) for d in ds]));assert abs(u-res['all']['macro']['mean_net_utility'])<1e-12;independent[name]=u
    dec=read(O/'decision.json');diff=[]
    for seed in range(3):
        a=independent[f'UTILITY_BCE_seed{seed}']-independent[f'HUBER_seed{seed}'];assert a==dec['paired_seeds'][seed]['difference'];diff.append(a)
    dates=sorted({m['date'] for m in meta});dd={d:float(np.mean([np.mean([r['net_utility'] for r in tr if r['date']==d and r['method']==f'UTILITY_BCE_seed{s}'])-np.mean([r['net_utility'] for r in tr if r['date']==d and r['method']==f'HUBER_seed{s}']) for s in range(3)])) for d in dates};assert dd==dec['date_utility_minus_huber']
    assert (np.mean(diff)>0 and all(x>0 for x in diff) and sum(x>0 for x in dd.values())>=12)==dec['objective_transfer_support']
    assert F['wall_s']<1801;verify();sealed();save(O/'review.json',dict(passed=True,all_state_numpy_forwards=n,max_abs_error=err,oof_trajectories_checked=len(tr),all24_step7250=True,held_date_draws_zero=True,normalization_independently_recomputed=True,train_only_fold_weight_normalizers=True,all_date_groups_kept_together=True,all_frozen_hashes_unchanged=True,new_physical_calls=0,original_validation_predictions=0,final_test_materialized=False));print('GRID121_AUDIT_PASSED',n,len(tr),err,flush=True)

def complete():
    assert not (O/'complete.json').exists();verify();assert read(O/'review.json')['passed'];D=read(O/'design.json');dec=read(O/'decision.json');F=read(O/'fits.json');res=read(O/'oof_results.json');fr=read(O/'fit_results.json')
    fam={a:{k:float(np.mean([res[f'{a}_seed{s}']['all']['macro'][k] for s in range(3)])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_call_balanced_utility']} for a in D['arms']}
    decomp=dict(gain=(fam['UTILITY_BCE']['mean_gain']-fam['HUBER']['mean_gain'])/D['gain_scale'],query_charge=-D['cost_per_query']*(fam['UTILITY_BCE']['mean_queries']-fam['HUBER']['mean_queries'])/D['gain_scale'],net=fam['UTILITY_BCE']['mean_net_utility']-fam['HUBER']['mean_net_utility']);assert abs(decomp['gain']+decomp['query_charge']-decomp['net'])<1e-12
    save(O/'summary_analysis.json',dict(family_oof_metrics=fam,decomposition=decomp))
    lines=['# GRID121：按日期留出的学习迁移诊断','', '**24次训练及保存结果核验完成。本轮只诊断同一电网的日期迁移，不是新增独立测试。**','',
        '## 为什么做','',
        'GRID120的收益代价加权分类在训练日期上比原回归目标更好，在原四个验证日期上却更差。本轮使用16个原训练日期重新分组，比较两种目标在各自没有参与拟合的日期上表现如何。这样可以检查差异是否只集中在原四个验证日期，不能把它当成完全独立的新数据证据。','',
        '## 分组与训练','',
        '16日期排序后，按下标模4分成四折。每折留出4日期，使用其余12日期训练；一个日期的全部前缀、物理调用和场景副本始终在同一侧。每个日期只由将其留出的模型评价。',
        '每折比较HUBER和UTILITY_BCE两个目标、种子0/1/2，共24次；每次均完成7250次更新、每批32状态。网络仍为原41→64→64→2、6978参数，随机初始化逐种子一致，不在折间续训。Adam、梯度裁剪和两个目标的定义沿用前轮，未选择更好端点。',
        '每折的输入均值/标准差只用12个拟合日期计算，收益权重归一化常数也只用这12日期。原层次抽样概率在12日期内重新归一化，两个目标的具体抽样索引相同。留出日期的优化抽样计数为0；没有用原四个验证日期评价或调整本轮。',
        '既有查询单价和收益尺度保持固定。必须说明：它们历史上由全部16个训练日期定义。本轮是以已固定任务为条件的折外开发诊断，并非对原始费用制定流程也进行完全嵌套的无信息泄露评估。这些日期此前也已参与其他开发实验，不能称为新独立测试。',
        f'24次训练总耗时{F["wall_s"]:.2f}秒，1800秒窗口未延长；全部训练结束后才计算留出日期预测。没有新电网回放、场景、安装或最终测试。','',
        '## 16日期折外结果','', '以下先在16日期上作宏平均，再平均三个种子。']
    for a,m in fam.items():lines.append(f'- {a}：净效用{m["mean_net_utility"]:.8f}，查询比例{100*m["query_fraction"]:.3f}%，收益保留{100*m["gain_capture"]:.3f}%，物理调用均衡净效用{m["physical_call_balanced_utility"]:.8f}。')
    lines+=['','配对种子（加权分类减Huber）：']
    for p in dec['paired_seeds']:lines.append(f'- 种子{p["seed"]}：Huber {p["huber"]:.8f}，加权分类{p["utility"]:.8f}，差值{p["difference"]:+.8f}。')
    lines += ['',f'逐日期正向为{dec["positive_dates"]}/16：']
    for date,delta in dec['date_utility_minus_huber'].items():lines.append(f'- {date}：{delta:+.8f}。')
    lines+=['','## 各折表现','']
    for fold in range(4):
        for a in D['arms']:
            fit=float(np.mean([fr[f'fold{fold}__{a}_seed{s}']['fit']['macro']['mean_net_utility'] for s in range(3)]));held=float(np.mean([fr[f'fold{fold}__{a}_seed{s}']['held']['macro']['mean_net_utility'] for s in range(3)]))
            lines.append(f'- 折{fold}，{a}：拟合侧净效用{fit:.8f}，留出侧{held:.8f}；留出日期{D["folds"][fold]["held_dates"]}。')
    lines+=['',f'总体差异分解：收益项{decomp["gain"]:+.8f}，查询费用项{decomp["query_charge"]:+.8f}，净差{decomp["net"]:+.8f}。','',
        '## 判读与边界','',f'预定描述性迁移条件为：加权分类均值高于Huber，三个配对种子均为正，且至少12/16日期同向。本轮结果={dec["objective_transfer_support"]}。',
        '不同目标都只使用12日期训练，需在本轮折内公平比较，不能把它们的绝对数值直接与前轮16日期训练、四验证日期的数值拼接。拟合侧与留出侧日期的基础难度也不同，不能仅凭两侧绝对值之差断言过拟合；应结合配对目标、逐折和逐日期比较。',
        '这项结果将与GRID120一起用于选择下一条学习问题：目标取舍、可见反馈预测能力，或真实路径与教师路径差异。没有改门槛、费用、网络或挑最好种子救结果，不自动扩规模或新增强化学习。',
        '即时补查净收益不是整周运行收益，相关前缀不是独立样本；这里没有物理安全、实际时延、显著性或跨电网结论。最终8个测试场景仍未生成。','',
        '## 核验','',
        '核验覆盖24个模型共85968个独立NumPy前向、3582个原标签、全部折的训练侧标准化、权重归一化、原初始化、24次优化器7250步、留出日期零抽样，以及14268条折外完整补查序列。所有源码、数据及历史任务常数的冻结哈希保持一致。']
    (O/'GRID121_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');collab=R/'论文协作资料/FRAME_日期迁移诊断_GRID121';collab.mkdir(exist_ok=False);dest=REPO/'research_archive/power-grid-2026/protocols/GRID121_date_transfer/complete'
    save(O/'complete.json',dict(passed=True,completed_fits=24,review_passed=True,report=str(O/'GRID121_report.md'),archive=str(dest),collaboration=str(collab),ready_to_scale=False,new_physical_calls=0,final_test_materialized=False));archive('complete')
    for n in ['GRID121_report.md','decision.json','summary_analysis.json','review.json','fits.json','complete.json']:shutil.copy2(O/n,collab/n)
    sync_docs(f"## 2026-10-23 — GRID121 grouped-date transfer diagnostic complete\n\nAll24fits,4fold12fit/4helddates,original6978/7250x32/3seeds,pairedHuber/utilityBCE,train-onlyfoldnormalization/weights,zeroholdoutdraws. Wall{F['wall_s']:.2f}s. OOFfamily{dec['family_oof_utility']};paired{[p['difference'] for p in dec['paired_seeds']]};positivedates{dec['positive_dates']}/16;support={dec['objective_transfer_support']}. Independent85968NumPyforwards/14268OOFtrajectories/normalization/optimizer/samplingauditpassed. Historicallyfixedprice/scalehadall16datedependency,anddatespreviouslyexposed;conditionalOOFdiagnosticnotpristinenestedindependenttest. No original4validationevaluation/physical/finaltest/autoscaling.",
        '# Next steps — GRID120/121两项学习实验完成\n\n1. 联合汇报决策目标对照与按日期留出的实际结果，区分目标收益、日期迁移和训练内表现。\n2. 保留所有负结果，不自动扫参或增加更多网络；下一学习路线围绕仍有证据缺口的具体问题。\n3. 新RL需先明确长期目标与公平对照；不自动扩电网、生成最终测试或改论文。\n\n报告：research_archive/power-grid-2026/protocols/GRID120_decision_objective/complete/GRID120_report.md 与 research_archive/power-grid-2026/protocols/GRID121_date_transfer/complete/GRID121_report.md\n')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID121按日期留出的学习迁移诊断](protocols/GRID121_date_transfer/complete/GRID121_report.md)：24个固定训练，日期分组、折内标准化和完整折外搜索核验；开发诊断，非独立最终测试。\n')
    print('GRID121_COMPLETE_ARCHIVED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','evaluate','audit','complete'],required=True);a=p.parse_args();globals()[a.phase]()
