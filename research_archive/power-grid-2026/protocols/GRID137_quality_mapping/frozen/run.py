"""Read-only inherited FULL quality mapping for56 existing single omissions."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,math,shutil,sys
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid137_quality_mapping';W=Path(__file__).parent
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID137_quality_mapping'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sealed():
    d=read(R/'outputs/grid100_data_foundation/design.json')
    for m in d['scenario_plan']:
        if m['partition']=='test':
            assert not (R/d['environment']/'chronics'/m['scenario']).exists()
            assert not (R/'outputs/grid100_data_foundation/datasets'/m['scenario']).exists()
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    sealed()
def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    manifest={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
    (dest/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
def mapping():
    d91=read(R/'outputs/grid91_mechanism/design.json');d96=read(R/'outputs/grid96_value_learning/design.json');labels={l['case_id']:l for l in read(R/'outputs/grid96_value_learning/labels.json')}
    refs={(m['cohort'],m['week'],m['environment']):m for m in d91['references']};out=[]
    for c in d96['cases']:
        ref=refs[(c['cohort'],c['week'],c['environment'])];assert ref['frame']==c['reference']
        base=f"outputs/grid96_value_learning/runs/r0__{c['week']}__FRAME_VALUE_QUAL"
        skip=c['reused_branch'] if c['pilot_known'] else f"outputs/grid96_value_learning/runs/r0__{c['week']}__SKIP_{c['id']}"
        out.append(dict(case=c,label=labels[c['id']],full=ref['full'],historical_frame=ref['frame'],factual=base,skip=skip))
    assert len(out)==56;return d91,d96,out
def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed();d91,d96,items=mapping();assert d91['quality_threshold']==.005
    assert read(R/'outputs/grid96_value_learning/review.json')['passed_saved_data']
    D=dict(id='GRID137_INHERITED_QUALITY_MAPPING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User asks continue;announced read-only inherited whole-week quality mapping,no new training/replay.',
        threshold=d91['quality_threshold'],nominal_steps=d96['nominal_episode_steps'],cases=56,
        mapping='Exact pre-existing GRID91 cohort/week/environment FULL and FRAME reference mapping;not a new reference choice.',
        historical_quality='If FULL completes,branch must complete;otherwise no earlier termination thanFULL. Compare totalcost/FULLtotalcost-1<=0.005 only when both complete.',
        evidence_labels=['PASS_COMPLETE','FAIL_COST','FAIL_SURVIVAL','REFERENCE_LIMITED','UNRESOLVED_EXTERNAL','BASELINE_NOT_QUALIFIED'],
        incomplete='A historical survival-relative pass with incompleteFULL is REFERENCE_LIMITED for complete-week cost supervision;do not call it a complete-week pass.',
        cost_basis='Full episode raw costs,not suffix percentage. Sharedprefix+suffix identity independently checked. Branches only cancelone completion then originalFRAME suffix.',
        checks=['whole-run cost and query sums','historicalFRAME factual physical trajectory equality','shared preintervention prefix and starting state hash','old suffix costs and query differences','external-event differences equal old audited list','division-vs-multiplication quality threshold check','legacyFRAME quality matchesGRID91'],
        old_labels='Keep all oldSKIP/MIXED/KEEP labels unchanged;derive new evaluation-only fields and retain risk vectors. No training data files rewritten.',
        optional_saved_readout_audit='Recount ALL original GRID96saved model/baseline decisions at original0.5/0.95 readouts,calendarLOO and old-to-generated,excludingpilot headline. No refit/forward/newcutoff or claim probabilities calibrated to new target.',
        prohibited=['fit','new model forward','risk/cost threshold search','new physical/scenario/test','paper edit','deploy or scale','pool with2378teacher prefixes','copy single-omission label to eachbatch or many-cancel policy'])
    save('design.json',D);save('case_mapping.json',[dict(case_id=m['case']['id'],week=m['case']['week'],cohort=m['case']['cohort'],environment=m['case']['environment'],full=m['full'],historical_frame=m['historical_frame'],factual=m['factual'],skip=m['skip']) for m in items])
    paths=[R/p for p in ['outputs/grid91_mechanism/design.json','outputs/grid91_mechanism/review.json','work/grid91_mechanism/audit.py',
        'outputs/grid96_value_learning/design.json','outputs/grid96_value_learning/labels.json','outputs/grid96_value_learning/review.json','outputs/grid96_value_learning/qualification.json',
        'outputs/grid96_value_learning/predictions.json','outputs/grid96_value_learning/learning_review.json','outputs/grid96_value_learning/learning_results.json','work/grid96_value_learning/learn.py',
        'outputs/grid136_objective_support/publication_receipt.json']]
    for folder in sorted({m[k] for m in items for k in ['full','historical_frame','factual','skip']}):
        paths += [R/folder/'summary.json',R/folder/'steps.jsonl.gz']
    paths += [O/'design.json',O/'case_mapping.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))})
    save('state.json',dict(stage='frozen_read_only'));verify();archive('frozen');print('GRID137_FROZEN')
def inspect(folder):
    s=read(R/folder/'summary.json');cost=[];queries=[];before=[];opponents=[];prefix=[];ph=hashlib.sha256();physical=hashlib.sha256();initial=ph.hexdigest()
    with gzip.open(R/folder/'steps.jsonl.gz','rt',encoding='utf8') as f:
        for j,line in enumerate(f):
            row=json.loads(line);assert row['step']==j+1;cost.append(float(row['raw_cost']));queries.append(int(row['simulations']));before.append(row['before_hash'])
            core={k:row[k] for k in ['step','before_hash','action_hash','after_hash','raw_cost','done','complete']}
            physical.update(json.dumps(core,sort_keys=True,separators=(',',':')).encode())
            core['simulations']=row['simulations'];ph.update(json.dumps(core,sort_keys=True,separators=(',',':')).encode());prefix.append(ph.hexdigest())
            opponents.append(json.dumps(row.get('label_audit',{}).get('opponent_realization'),sort_keys=True,separators=(',',':')))
    assert len(cost)==s['steps'] and abs(math.fsum(cost)-s['cost'])<1e-6 and sum(queries)==s['public_forecasts']
    return dict(summary=s,cost=cost,queries=queries,before=before,opponents=opponents,prefix=prefix,empty_prefix=initial,physical_hash=physical.hexdigest())
def screen(s,full,tau):
    surv=bool(s['complete']) if full['complete'] else s['steps']>=full['steps'];gap=s['cost']/full['cost']-1 if s['complete'] and full['complete'] else None
    passed=surv and (gap is None or gap<=tau)
    if gap is not None:assert (gap<=tau)==(s['cost']<=(1+tau)*full['cost'])
    return dict(reference_screen_pass=bool(passed),survival_relative_pass=bool(surv),cost_gap_full=gap)
def run():
    verify();assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(new_fits=0,new_physical=0));D=read(O/'design.json');d91,d96,items=mapping()
    old91={(m['cohort'],m['week']):m for m in read(R/'outputs/grid91_mechanism/review.json')['comparisons']};cache={};rows=[];checks=collections.Counter()
    def get(folder):
        if folder not in cache:cache[folder]=inspect(folder);checks['saved_runs']+=1;checks['saved_steps']+=len(cache[folder]['cost'])
        return cache[folder]
    for m in items:
        c,l=m['case'],m['label'];full=get(m['full']);hist=get(m['historical_frame']);a=get(m['factual']);b=get(m['skip']);sf,sa,sb=full['summary'],a['summary'],b['summary'];t=c['step']
        assert all(s['week']==c['week'] for s in [sf,sa,sb]);assert hist['physical_hash']==a['physical_hash'];assert hist['summary']['n1_forecasts']==sa['n1_forecasts']
        assert full['before'][0]==a['before'][0]==b['before'][0]
        assert (a['prefix'][t-2] if t>1 else a['empty_prefix'])==(b['prefix'][t-2] if t>1 else b['empty_prefix'])
        assert a['before'][t-1]==b['before'][t-1]==c['state_hash'];checks['preintervention_prefixes']+=1
        assert (a['physical_hash']==b['physical_hash'])==l['physical_path_identical']
        delta=[j+1 for j in range(t-1,min(len(a['cost']),len(b['cost']))) if a['opponents'][j]!=b['opponents'][j]];assert delta==l['external_differences']
        prefix=math.fsum(a['cost'][:t-1]);asuffix=math.fsum(a['cost'][t-1:]);bsuffix=math.fsum(b['cost'][t-1:])
        assert abs(prefix+asuffix-sa['cost'])<1e-6 and abs(prefix+bsuffix-sb['cost'])<1e-6
        assert abs(asuffix-l['horizons']['primary']['completion']['cost'])<1e-6 and abs(bsuffix-l['horizons']['primary']['skip']['cost'])<1e-6
        assert sum(a['queries'])-sum(b['queries'])==l['queries_saved'];checks['full_cost_suffix_query_identities']+=1
        qa=screen(sa,sf,D['threshold']);qb=screen(sb,sf,D['threshold']);ref=old91[(c['cohort'],c['week'])];assert qa['reference_screen_pass']==ref['frame_quality']
        if qa['cost_gap_full'] is not None:assert abs(qa['cost_gap_full']-ref['frame_cost_delta_full'])<1e-12
        if delta:status='UNRESOLVED_EXTERNAL'
        elif not qa['reference_screen_pass']:status='BASELINE_NOT_QUALIFIED'
        elif not qb['survival_relative_pass']:status='FAIL_SURVIVAL'
        elif not sf['complete']:status='REFERENCE_LIMITED'
        elif not qb['reference_screen_pass']:status='FAIL_COST'
        else:status='PASS_COMPLETE'
        headroom=(1+D['threshold'])*sf['cost']-sa['cost'] if sf['complete'] and sa['complete'] else None
        both=sa['complete'] and sb['complete'];change=sb['cost']-sa['cost'] if both else None
        if change is not None:
            assert abs(change-(bsuffix-asuffix))<1e-6
            if headroom is not None:assert qb['reference_screen_pass']==(change<=headroom)
        risk={k:dict(completion=l['horizons']['primary']['completion'][k],skip=l['horizons']['primary']['skip'][k]) for k in ['overload_steps','overload_excess_sum','max_rho','illegal','exceptions']}
        rows.append(dict(case_id=c['id'],cohort=c['cohort'],week=c['week'],calendar_date=c['calendar_date'],environment=c['environment'],step=t,pilot_known=c['pilot_known'],original_label=l['label'],
            quality_evidence_label=status,full_reference=m['full'],full_complete=bool(sf['complete']),full_steps=sf['steps'],full_cost=sf['cost'],
            completion_complete=bool(sa['complete']),completion_steps=sa['steps'],completion_total_cost=sa['cost'],completion_quality=qa,
            skip_complete=bool(sb['complete']),skip_steps=sb['steps'],skip_total_cost=sb['cost'],skip_quality=qb,
            wholeweek_cost_comparable=bool(sf['complete'] and sb['complete']),nominal_cost_supervision_known=status in ['PASS_COMPLETE','FAIL_COST','FAIL_SURVIVAL'],
            common_prefix_cost=prefix,completion_suffix_cost=asuffix,skip_suffix_cost=bsuffix,skip_cost_minus_completion=change,remaining_FULL_quality_headroom=headroom,
            completion_all_queries=sa['public_forecasts'],skip_all_queries=sb['public_forecasts'],queries_saved=l['queries_saved'],candidate_queries_saved=l['candidate_queries_saved'],
            matched_frame_skip_horizon=sa['steps']==sb['steps'] and sa['complete']==sb['complete'],physical_path_identical=l['physical_path_identical'],external_differences=delta,risk_vector=risk))
    save('quality_evidence_ledger.json',rows);counts={};cross={}
    for scope,rr in [('all56',rows),('nonpilot53',[r for r in rows if not r['pilot_known']]),('action_changed',[r for r in rows if not r['physical_path_identical']])]:
        status=dict(collections.Counter(r['quality_evidence_label'] for r in rr));fail=[r for r in rr if r['quality_evidence_label'].startswith('FAIL_')]
        counts[scope]=dict(cases=len(rr),status=status,historical_reference_pass=sum(r['skip_quality']['reference_screen_pass'] for r in rr),failure_dates=sorted({r['calendar_date'] for r in fail}),failure_cases=[r['case_id'] for r in fail],
            known_quality_dates=sorted({r['calendar_date'] for r in rr if r['nominal_cost_supervision_known']}),all_original_labels_retained=True)
        cross[scope]={lab:dict(collections.Counter(r['quality_evidence_label'] for r in rr if r['original_label']==lab)) for lab in ['SKIP','MIXED','KEEP']}
    save('quality_mapping_summary.json',counts);save('old_new_label_crosswalk.json',cross)
    # Recount original saved readouts only; probabilities still refer to old3class target.
    pr=read(R/'outputs/grid96_value_learning/predictions.json');lr=read(R/'outputs/grid96_value_learning/learning_review.json');skipindex=lr['classes'].index('SKIP');assert lr['thresholds']==[.5,.95]
    byid={r['case_id']:r for r in rows};readouts=[]
    for scheme in sorted({p['scheme'] for p in pr}):
        for model in sorted({p['model'] for p in pr}):
            rr=[p for p in pr if p['scheme']==scheme and p['model']==model and not p['pilot_known']]
            if not rr:continue
            for cut in lr['thresholds']:
                selected=[p for p in rr if p['prediction']=='SKIP' and p['probabilities'][skipindex]>=cut]
                readouts.append(dict(scheme=scheme,model=model,original_cutoff=cut,headline_cases=len(rr),proposed_skips=len(selected),
                    original_labels=dict(collections.Counter(p['label'] for p in selected)),derived_quality_evidence=dict(collections.Counter(byid[p['case_id']]['quality_evidence_label'] for p in selected)),
                    selected_case_ids=[p['case_id'] for p in selected],no_new_target_training=True))
    save('saved_readout_crosscheck.json',readouts)
    # Independent baseline-level threshold check, separate from ratio implementation.
    for r in rows:
        if r['wholeweek_cost_comparable']:assert (r['skip_total_cost']<=1.005*r['full_cost'])==r['skip_quality']['reference_screen_pass'];checks['independent_cost_threshold_checks']+=1
    verify();save('review.json',dict(passed=True,checks=dict(checks),cases=56,mapped_FULL_references=len({r['full_reference'] for r in rows}),
        original_labels_predictions_models_unchanged=True,derived_evaluation_fields_only=True,all_costs_full_episode_not_suffix_percent=True,
        reference_incomplete_not_fullweek_positive=True,all_risk_vectors_retained=True,new_fits=0,new_model_forwards=0,new_physical_replays=0,new_thresholds=0,final_tests_absent=True))
    lines=['# GRID137：56条真实分支按原论文质量标准重新核对','',
        '本轮只添加按旧标准派生的评价字段；GRID96原SKIP/MIXED/KEEP标签、模型、概率和所有原文件不变。没有新训练、模型前向或物理回放。','',
        '## 对齐的标准与成本口径','',
        '严格沿用GRID91的已审核FULL参照映射与0.5%成本阈值。FULL完成时，取消分支必须完成、整周总成本不超过FULL的100.5%；FULL未完成时，只按原规则检查不早于FULL终止，但单独标为REFERENCE_LIMITED，不能作为整周成本达标正例。',
        '全部56条的场景/电网环境/原FRAME参照与GRID91对应。独立核对每条实际FRAME重现、共同干预前缀和起始状态哈希；共同前缀成本加两条后缀分别等于整周总成本，不能把干预后剩余成本百分比当整周百分比。',
        '对已完成分支，取消造成的成本增量应不超过1.005×FULL整周成本−FRAME整周成本。这个剩余空间依赖已有FRAME相对FULL的位置，不能再给FRAME额外叠加一个0.5%。','',
        '## 评价字段的分布','', '| 范围 | 例数 | 按原标准派生的状态 | 失败日期 |','|---|---:|---|---|']
    for name,v in counts.items():lines.append(f"| {name} | {v['cases']} | {v['status']} | {v['failure_dates']} |")
    lines+=['',
        'PASS_COMPLETE：有完整周证据且通过原筛查；FAIL_COST/FAIL_SURVIVAL：对应原成本/存活标准失败；REFERENCE_LIMITED：FULL本身未完成，仅相对存活可比。外部事件不一致或基线未通过的记录保留为单独未知状态。','',
        '## 与原多维标签的关系','']
    for lab,v in cross['all56'].items():lines.append(f'- 原{lab}：{v}。')
    lines+=['',
        '两个标签系统回答不同问题。原MIXED可能在论文完成/成本标准下通过，但过载或查询指标仍有权衡；不能改称所有维度都安全。原路径等价的共同提前结束，也不能因为原标签是SKIP就当完整周正例。风险原值全部保留在逐例账本。','',
        '## 全部失败与证据不足案例','']
    for r in rows:
        if r['quality_evidence_label']!='PASS_COMPLETE':lines.append(f"- {r['case_id']} {r['cohort']}/{r['week']} step{r['step']}：{r['quality_evidence_label']}；原{r['original_label']}；FULL/取消完成={r['full_complete']}/{r['skip_complete']}，步数{r['full_steps']}/{r['skip_steps']}，对FULL成本差{r['skip_quality']['cost_gap_full']}。")
    lines+=['',
        '原0.5%是实验筛查容差，不是物理安全保证。所有状态均为已暴露开发记录，单次干预后继续原FRAME；不能把这些标签复制给每批动作，也不能把多次取消的收益相加。','',
        '## 既有模型建议的回查','',
        'saved_readout_crosscheck.json按原0.5/0.95读数、原日历分组与old-to-generated划分，回查全部原模型和常数参照。不产生新概率、不挑最好读数、不训练新目标；旧概率是Pareto类别概率，不是本轮完整周通过概率。',
        '这只是把已知建议与新评价字段并列，不能当一次预先设计的目标对齐学习实验，更不能据少量已知失败宣布部署可靠。','',
        '## 下一步判断','',
        '现在可明确区分“有完整质量证据”“原筛查失败”“仅相对存活可比”，而不再把即时G>0、MIXED或SKIP简单当训练答案。能否进行有意义的新训练，应看失败日期与非等价轨迹样本的覆盖，而不是56这个总数。',
        '本轮不扩大场景、训练或修改论文；最终8测试仍封存。']
    (O/'GRID137_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('complete.json',dict(passed=True,evaluation_fields_derived=56,original_targets_unchanged=True,new_fits=0,new_replays=0,adopted=False,ready_to_scale=False));save('state.json',dict(stage='complete'));archive('complete')
    print(json.dumps(dict(summary=counts,crosswalk=cross,checks=dict(checks)),ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
