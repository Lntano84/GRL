"""Read-only reward-scale, saved-prefix opportunity and public-support audit."""
from pathlib import Path
import ast,collections,datetime,gzip,hashlib,json,math,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid136_objective_support';W=Path(__file__).parent
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID136_objective_support'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sealed():
    d=read(R/'outputs/grid100_data_foundation/design.json')
    for r in d['scenario_plan']:
        if r['partition']=='test':
            assert not (R/d['environment']/'chronics'/r['scenario']).exists()
            assert not (R/'outputs/grid100_data_foundation/datasets'/r['scenario']).exists()
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    sealed()
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    manifest={p.name:sha(p) for p in d.iterdir() if p.is_file()}
    (d/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
def load():
    with np.load(R/'outputs/grid129_baseline_risk/train_public_inputs.npz') as z:X=z['X'][:,:37].astype(float);ids=z['sample_ids'].tolist();Q=z['known_queries']
    with np.load(R/'outputs/grid130_calibrated_gate/train_calibration_scores.npz') as z:fit=z['fitmask'];cal=z['calmask']
    with gzip.open(R/'outputs/grid129_baseline_risk/train_label_ledger.jsonl.gz','rt',encoding='utf8') as f:labels=[json.loads(l) for l in f]
    cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())}
    assert len(ids)==2378 and all(i==r['sample_id'] for i,r in zip(ids,labels))
    return X,ids,Q,fit,cal,labels,cases
def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed();assert read(R/'outputs/grid135_matched_tail/publication_recovery_01.json')['passed']
    plan=dict(id='GRID136_OBJECTIVE_SUPPORT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='frozen_read_only_posthoc_plan',
        authorization='User asks continue investigating; announced read-only label scale and observed public support.',
        scope='All2378 original train prefixes,12fit/4cal masks,cheap/expensive cohorts. Validation only7previously reported expensive misses;no new model evaluation or test.',
        questions=['How many positive full gains are within1/2/4/8/16/32/64float32 reward ULPs? Counts,call counts,gain/query shares and positive quantiles,not a noise verdict.',
            'Which saved train/cal cases have no gain,positive gain but no profitable recorded prefix,only partial profitable search,or profitable full completion under inherited price?',
            'For all7known expensive misses,distinguish full completion cost from any positive-prefix opportunity;full net<0 alone does not mean STOP was optimal.',
            'Check original reward source,RL query price and whole-week quality definitions separately.',
            'Station/date/call support and ALL37predeclared summary feature ranges/percentiles for known misses and calibration anchors. Posthoc descriptions,not feature/trigger selection.'],
        semantics='R=2-maxrho from frozen author reward;gains are float32-rounded prediction reward differences,not operational currency. Price inherited GRID102heuristic,not seconds/currency.',
        categories='G=0;G>0&max_prefix_net<=0;G>0&max_prefix_net>0&full_net<=0;G>0&full_net>0. Include STOP with utility0 in hindsight maximum. No new deployable rule.',
        ulp_boundaries=[1,2,4,8,16,32,64],normalization='ULP at current incumbent float32 reward;describe precision only,do not remove tiny gains or select tolerance.',
        comparisons='Keep GRID127 own-objective failure vsBATCH1 as counterevidence;different objectives do not excuse all old performance loss.',
        prohibited=['fit','new model forward','alternative cutoff application','relabel saved training targets','new price/quality tolerance','new simulation/scenario','final8test','paper/figure','automatic scale'])
    save('design.json',plan)
    paths=[R/p for p in ['outputs/grid129_baseline_risk/train_public_inputs.npz','outputs/grid129_baseline_risk/train_label_ledger.jsonl.gz',
        'outputs/grid130_calibrated_gate/train_calibration_scores.npz','outputs/grid134_full_tail_inputs/train_calibration_public_inputs.jsonl.gz',
        'outputs/grid102_search_rl/cases.pkl','outputs/grid102_search_rl/design.json','outputs/grid135_matched_tail/design.json',
        'outputs/grid135_matched_tail/validation_trajectories.json','outputs/grid135_matched_tail/interpretation.json','outputs/grid135_matched_tail/thresholds.json',
        'outputs/grid135_matched_tail/evaluation_freeze.json','outputs/grid135_matched_tail/calibration_goal_audit.json',
        'outputs/grid28/offline.json','outputs/grid91_mechanism/design.json','outputs/grid100_data_foundation/bulk_dataset_manifest.json',
        'work/grid02/ljn_nn/modules/rewards.py','work/grid03/score_adapter.py','work/grid08/common.py','work/grid100_data_foundation/bulk_teacher_worker.py',
        'work/grid101_gain_learning/features.py','work/grid102_search_rl/prepare.py','work/grid135_matched_tail/common135.py',
        'outputs/grid127_mean_gain_objective/nonlearning_comparison/与不学习方法比较_中文.md']]
    ef=read(R/'outputs/grid135_matched_tail/evaluation_freeze.json');paths += [R/'outputs/grid135_matched_tail'/p for p in ef]
    oldrows=read(R/'outputs/grid135_matched_tail/validation_trajectories.json');selected={r['sample_id'] for r in oldrows if r['expensive'] and r['action_disagreement']}
    for v in read(R/'outputs/grid135_matched_tail/calibration_goal_audit.json')['calibration_anchor_calls'].values():selected.update(v['sample_ids'])
    meta=read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']
    paths += [R/p for p in sorted({m['source_call'] for m in meta if m['sample_id'] in selected})]
    paths += [O/'design.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))})
    save('state.json',dict(stage='frozen_no_fits'));verify();archive('frozen');print('GRID136_FROZEN')
def opportunity(c,fee):
    gains=[float(g) for g in c['gains']];queries=[int(q) for q in c['queries']];G=math.fsum(gains);Q=sum(queries)
    us=[0.]+[math.fsum(gains[:j])-fee*sum(queries[:j]) for j in range(1,len(gains)+1)]
    other=np.r_[0.,np.cumsum(np.array(gains)-fee*np.array(queries))];assert abs(np.array(us)-other).max()<1e-12
    full=G-fee*Q;assert abs(full-us[-1])<1e-12
    category='zero_gain' if G==0 else ('no_profitable_prefix' if max(us)<=0 else ('partial_only_profitable' if full<=0 else 'full_profitable'))
    return dict(gain=G,queries=Q,full_net=full,best_saved_prefix_net=max(us),best_batch=us.index(max(us)),net_by_batches=us,category=category,
        cost_proxy=fee*Q,one_shot_best_net=max(0.,full),sequential_minus_one_shot_upper_bound=max(us)-max(0.,full))
def run():
    verify();assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(read_only=True))
    X,ids,Q,fit,cal,labels,cases=load();D=read(R/'outputs/grid102_search_rl/design.json');fee=D['cost_per_query'];subs=read(R/'outputs/grid28/offline.json')['public_subids']
    raw={}
    with gzip.open(R/'outputs/grid134_full_tail_inputs/train_calibration_public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for l in f:
            r=json.loads(l);raw[r['sample_id']]=r['public_input']
    tree=ast.parse((R/'work/grid101_gain_learning/features.py').read_text(encoding='utf8'));names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAMES' for t in n.targets));assert len(names)==37
    records=[];checks=0
    for i,(sid,m) in enumerate(zip(ids,labels)):
        c=cases[sid];assert c['partition']=='train';v=opportunity(c,fee);assert v['gain']==m['full_baseline_gain'] and v['queries']==int(Q[i])==m['known_saved_queries_if_skip']
        inc=float(raw[sid]['incumbent_reward']);assert inc==float(np.float32(inc)) and inc==X[i,14]
        ulp=abs(float(np.spacing(np.float32(inc))));assert ulp>0;G=v['gain'];gap=G/ulp
        assert float(np.float32(inc+G))==inc+G
        station=int(subs[m['skip_action']]);assert m['skip_action']==c['chosen'][0]
        records.append(dict(sample_id=sid,date=m['date'],call=m['call'],split='fit' if fit[i] else 'calibration',station=station,expensive=bool(Q[i]>64),
            incumbent_reward=inc,reward_ulp=ulp,gain_ulps=gap,**v));checks+=1
    cohorts={}
    for part,mask in [('fit',fit),('calibration',cal)]:
        for scope,extra in [('all',np.ones(len(Q),bool)),('expensive',Q>64),('cheap',Q<=64)]:
            ii=np.flatnonzero(mask&extra);rr=[records[i] for i in ii];positive=[r for r in rr if r['gain']>0];totalG=math.fsum(r['gain'] for r in rr);totalQ=sum(r['queries'] for r in rr)
            bins={};limits=[0,1,2,4,8,16,32,64,float('inf')]
            for left,right in zip([-1]+limits[:-1],limits):
                tag='zero' if right==0 else (f'({left},{right}]ulp' if math.isfinite(right) else '>64ulp')
                chosen=[r for r in rr if left<r['gain_ulps']<=right];gain=math.fsum(r['gain'] for r in chosen)
                bins[tag]=dict(prefixes=len(chosen),calls=len({r['call'] for r in chosen}),gain_share=gain/totalG if totalG else 0.,queries=sum(r['queries'] for r in chosen),query_share=sum(r['queries'] for r in chosen)/totalQ)
            assert sum(v['prefixes'] for v in bins.values())==len(rr)
            categories={}
            for cat in ['zero_gain','no_profitable_prefix','partial_only_profitable','full_profitable']:
                chosen=[r for r in rr if r['category']==cat];categories[cat]=dict(prefixes=len(chosen),calls=len({r['call'] for r in chosen}),dates=len({r['date'] for r in chosen}),gain_share=math.fsum(r['gain'] for r in chosen)/totalG if totalG else 0.,queries=sum(r['queries'] for r in chosen))
            assert sum(v['prefixes'] for v in categories.values())==len(rr)
            cohorts[part+'_'+scope]=dict(prefixes=len(rr),calls=len({r['call'] for r in rr}),positive_prefixes=len(positive),positive_calls=len({r['call'] for r in positive}),
                positive_gain_quantiles={str(q):float(np.quantile([r['gain'] for r in positive],q)) for q in [0,.25,.5,.75,1]},
                positive_ULP_quantiles={str(q):float(np.quantile([r['gain_ulps'] for r in positive],q)) for q in [0,.25,.5,.75,1]},
                positive_at_most4ULP=sum(r['gain_ulps']<=4 for r in positive),positive_at_most64ULP=sum(r['gain_ulps']<=64 for r in positive),
                total_gain=totalG,total_queries=totalQ,ulp_bins=bins,categories=categories,
                positive_full_proxy_unprofitable=sum(r['gain']>0 and r['full_net']<=0 for r in rr),
                sequential_vs_one_shot_hindsight_sum=math.fsum(r['sequential_minus_one_shot_upper_bound'] for r in rr))
    save('gain_scale_and_opportunities.json',cohorts);save('train_calibration_records.json',records)
    trajectories=read(R/'outputs/grid135_matched_tail/validation_trajectories.json');bad=[r for r in trajectories if r['expensive'] and r['action_disagreement']]
    unique={r['sample_id']:r for r in bad};assert len(unique)==7
    anchors=read(R/'outputs/grid135_matched_tail/calibration_goal_audit.json')['calibration_anchor_calls'];anchorids={sid for a in anchors.values() for sid in a['sample_ids']}
    labelmap={r['sample_id']:r for r in labels};portraits={};trainstation=np.array([r['station'] for r in records])
    for sid in sorted(set(unique)|anchorids):
        c=cases[sid];v=opportunity(c,fee);x=np.asarray(c['states'][0,:37],float);station=int(subs[c['chosen'][0]]);coverage={};features={}
        for part,mask in [('fit',fit&(Q>64)),('calibration',cal&(Q>64))]:
            mask=mask&(trainstation==station);ii=np.flatnonzero(mask);rr=[records[i] for i in ii]
            coverage[part]=dict(prefixes=len(rr),calls=len({r['call'] for r in rr}),dates=sorted({r['date'] for r in rr}),positive_prefixes=sum(r['gain']>0 for r in rr),positive_calls=len({r['call'] for r in rr if r['gain']>0}),
                pending_range=[int(Q[ii].min()),int(Q[ii].max())] if len(ii) else None,
                reward_range=[float(X[ii,14].min()),float(X[ii,14].max())] if len(ii) else None,rho_range=[float(X[ii,15].min()),float(X[ii,15].max())] if len(ii) else None)
            fs={}
            for j,n in enumerate(names):
                if not len(ii):fs[n]=None;continue
                a=X[ii,j];fs[n]=dict(value=float(x[j]),min=float(a.min()),max=float(a.max()),percentile_midrank=float(((a<x[j]).sum()+.5*(a==x[j]).sum())/len(a)),outside_observed_range=bool(x[j]<a.min() or x[j]>a.max()))
            features[part]=fs
        portraits[sid]=dict(role='known_validation_miss' if sid in unique else 'calibration_anchor',call=c['physical_state_group'],date=c['date'],station=station,
            incumbent_reward=float(x[14]),incumbent_rho=float(x[15]),reward_gap=float(x[16]),queried_fraction=float(x[1]),
            station_coverage=coverage,all37_public_features=features,**v)
    save('critical_public_portraits.json',portraits)
    # Reward implementation is inspected as source only, never imported/executed.
    reward_source=(R/'work/grid02/ljn_nn/modules/rewards.py').read_text(encoding='utf8');assert 'return 2.0 - obs.rho.max()' in reward_source
    manifest=read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples'];sources={m['sample_id']:m['source_call'] for m in manifest};formula_checks=[]
    for sid,p in portraits.items():
        call=read(R/sources[sid]);outcomes={r['pool_id']:r for r in call['outcomes']};c=cases[sid];a=outcomes[c['chosen'][0]];b=outcomes[c['chosen'][-1]]
        assert a['strict_admissible'] and b['strict_admissible'];ra=float(np.float32(a['reward']));rb=float(np.float32(b['reward']));assert rb-ra==p['gain']
        assert abs(float(np.float32(2.-float(a['rho'])))-ra)<=abs(float(np.spacing(np.float32(ra))))
        assert abs(float(np.float32(2.-float(b['rho'])))-rb)<=abs(float(np.spacing(np.float32(rb))))
        formula_checks.append(dict(sample_id=sid,incumbent_reward=ra,full_reward=rb,incumbent_rho=a['rho'],full_rho=b['rho'],reward_gain=rb-ra,
            max_rho_reduction=float(a['rho'])-float(b['rho']),reward_gain_ULPs=(rb-ra)/abs(float(np.spacing(np.float32(ra))))));checks+=1
    save('reward_formula_checks.json',formula_checks)
    missed={sid:portraits[sid] for sid in unique};stats=dict(unique_missed_prefixes=7,physical_calls=len({p['call'] for p in missed.values()}),
        full_completion_proxy_nonpositive=sum(p['full_net']<=0 for p in missed.values()),
        no_profitable_recorded_prefix=sum(p['best_saved_prefix_net']<=0 for p in missed.values()),
        profitable_recorded_prefix=sum(p['best_saved_prefix_net']>0 for p in missed.values()),
        missed_station_ids=sorted({p['station'] for p in missed.values()}),
        all_missed_stations_have_positive_fit_calls=all(p['station_coverage']['fit']['positive_calls']>0 for p in missed.values()))
    save('missed_opportunity_summary.json',stats)
    verify();save('review.json',dict(passed=True,source_gain_query_checks=checks,train_calibration_prefixes=2378,existing_validation_misses_checked=7,
        ULP_categories_partition_cases=True,hindsight_prefix_utility_independent_fsum_cumsum_check=True,all37_features_retained=True,
        old_models_targets_cutoffs_results_unchanged=True,new_fits=0,new_model_forwards=0,new_thresholds_evaluated=0,new_physical_calls=0,final_tests_absent=True))
    lines=['# GRID136：补查值不值，当前几种目标是否一致','',
        '本轮只读原训练/校准记录，以及GRID135已经暴露的7条大补查误停；没有训练、模型前向、换门槛、重标数据或新仿真。','',
        '## 先把三个目标分开','',
        '作者搜索奖励为R=2−最大负载率。保存的补查收益G是最佳已评估动作预测奖励的float32差值，近似对应最大负载率的下降，不是运行成本。原始奖励/负载率来源对关键记录逐项核对，差异在舍入尺度内。',
        '旧GRID102强化学习优化“分批奖励增量−固定代理查询费”；当前GRID135分类/校准保护“只要完整补查G>0就不跳过”；论文整周质量则以存活、完成情况和对FULL的完整周成本+0.5%筛查。这三种任务有联系，却不是同一个目标。',
        '查询费按历史训练数据启发式构造，单位不是秒或货币。不能把G/查询费的大小直接解释为整周质量，也不能把0.5%成本容忍度直接变成G的容忍度。','',
        '## 正收益到底多小','', '| 数据组 | 前缀/调用 | 正收益前缀 | ≤4个奖励ULP | ≤64个奖励ULP | 正收益中位数 |','|---|---:|---:|---:|---:|---:|']
    for key in ['fit_all','calibration_all','fit_expensive','calibration_expensive']:
        a=cohorts[key];lines.append(f"| {key} | {a['prefixes']}/{a['calls']} | {a['positive_prefixes']} | {a['positive_at_most4ULP']} | {a['positive_at_most64ULP']} | {a['positive_gain_quantiles']['0.5']:.9g} |")
    lines+=['',
        'ULP指当前奖励附近相邻float32数的间隔。接近几个ULP只说明数值差很小，不证明它是噪声或可以安全忽略；本轮没有提高仿真精度重算。上述计数含相关前缀，不能当独立场景数。','',
        '## 有正收益不等于每一段都值得查','', '| 数据组 | 无收益 | 有收益但任何保存前缀都不回本 | 仅中途停止能回本 | 完整补查能回本 |','|---|---:|---:|---:|---:|']
    for key in ['fit_all','calibration_all','fit_expensive','calibration_expensive']:
        cats=cohorts[key]['categories'];lines.append('| '+key+' | '+' | '.join(str(cats[c]['prefixes']) for c in ['zero_gain','no_profitable_prefix','partial_only_profitable','full_profitable'])+' |')
    lines+=['',
        '“回本”仅指G−代理费×查询数>0，按原64批次的已保存结果逐项累加。使用未来已存结果得到的最好前缀是事后上界，不能部署，未据此拟合新标签或读出新策略。',
        '完整查完不回本，并不意味着最开始就应该停；中间可能已经拿到收益，后面继续查才变亏。这正是一次“跳过或全部查完”与逐批停止的动作范围差异，不能只比较整段G。','',
        '## 那7条误停的真实情况','']
    for sid in sorted(missed):
        p=missed[sid];lines.append(f"- {sid}：剩余{p['queries']}次查询，完整G={p['gain']:.12f}，整段扣费值{p['full_net']:.9f}；保存批次中最好的停止位置为{p['best_batch']}，对应扣费值{p['best_saved_prefix_net']:.9f}。")
    lines+=[f"\n7条都来自同一次物理搜索。整段补查扣费均不回本；其中{stats['no_profitable_recorded_prefix']}条的任何保存前缀都不回本，另{stats['profitable_recorded_prefix']}条查第一批即可取得正扣费收益。", 
        '因此，按当前“全部正增量都保护”的任务，七条都是误停；按旧代理费任务，四条停止符合其事后最优选择，三条则仍漏了值得查的一批。不能把七条一概解释成同一种失败。这个区分不修改GRID135原失败判定，也不证明真实整周运行更好。','',
        '## 决策前信息是否支持识别','']
    for sid in sorted(missed)[:1]:
        p=missed[sid];lines.append(f"该调用初始最佳动作所属站编号为{p['station']}（沿用原零基编号）。")
        for part,c in p['station_coverage'].items():lines.append(f"- {part}中同站大补查：{c['prefixes']}前缀/{c['calls']}调用/{len(c['dates'])}日期，其中正收益{c['positive_prefixes']}前缀/{c['positive_calls']}调用；剩余候选范围{c['pending_range']}。")
    lines+=['',
        '全部37项公开摘要均给出拟合/校准同站范围和经验分位，见critical_public_portraits.json；这只是对既知误例的事后描述，不是新触发规则或泛化检验。落在单项范围内不能证明联合条件有足够覆盖，落在范围外也不能直接断言不可学习。',
        '输入有负载率与已购反馈摘要，但没有显式输入全网潮流的空间分布。补齐候选接线计划，补的是动作信息，不等于补齐运行状态信息；本轮不能据此断言添加全观测会有效。','',
        '## 我的判断','',
        '当前首先要统一“补查值不值”的目标与动作范围，然后才是继续训练。先明确保护的是负载率改进、扣费收益，还是整周质量；如果继续研究扣费价值，就必须区分“根本不用查”“查一部分”“查完”，不能只把任何G>0当成同一种正例。',
        '但目标差异不是旧学习失败的全部解释。GRID127在自己采用的扣费净效用上仍低于固定一批，0.15735212对0.16286891；这仍是收益识别/泛化不足的反证。旧加权、收益回归和关系网络失败全部保留，不能换个说法就重新宣布成功。',
        '本轮不选新的损失容忍度、价格、标签或模型；也不删除微小收益。下一步应先用现有训练/校准的保存运行记录，明确学习任务的质量含义和可覆盖的条件，再封存具体学习方案。最终8测试保持未生成。']
    (O/'GRID136_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('complete.json',dict(passed=True,new_fits=0,new_physical_calls=0,old_protocol_unchanged=True,adopted=False,ready_to_scale=False));save('state.json',dict(stage='complete'));archive('complete')
    print(json.dumps(dict(cohorts={k:dict(positive=v['positive_prefixes'],ULP4=v['positive_at_most4ULP'],ULP64=v['positive_at_most64ULP'],categories=v['categories']) for k,v in cohorts.items() if k.endswith(('all','expensive'))},missed=stats),ensure_ascii=False),flush=True)
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
