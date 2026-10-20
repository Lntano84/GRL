"""Read-only separation of effect evidence and preference evidence. No fitting."""
from pathlib import Path
from collections import Counter
import csv,datetime,hashlib,json,math
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/frame_label_cleanup_20261020';SRC=ROOT/'outputs/grid96_value_learning'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(n,rr):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
def direction(delta,tol):return '取消更好' if delta < -tol else '保留更好' if delta > tol else '数值相当'
def main():
    assert not OUT.exists(),'Preserve any existing cleanup output'
    OUT.mkdir(parents=True)
    source_names=['labels.json','design.json','review.json','learning_review.json','fits.json','predictions.json','learning_results.json','decision.json']
    frozen={n:sha(SRC/n) for n in source_names}
    assert read(SRC/'review.json')['passed_saved_data'] and read(SRC/'learning_review.json')['passed']
    rows=read(SRC/'labels.json');design=read(SRC/'design.json');names=design['learning']['features'];cases={c['id']:c for c in design['cases']}
    assert len(rows)==len({r['case_id'] for r in rows})==56 and len(names)==15
    outcomes=[];inputs=[];metadata=[];full=[];groups=Counter()
    for r in rows:
        a=r['horizons']['primary']['completion'];b=r['horizons']['primary']['skip'];both=a['reaches_horizon'] and b['reaches_horizon'];same=r['physical_path_identical']
        assert r['features']==cases[r['case_id']]['features'] and all(math.isfinite(r['features'][n]) for n in names)
        if same:group='轨迹相同_均完成' if both else '轨迹相同_均未完成'
        elif r['label']=='SKIP':group='轨迹改变_取消占优'
        elif r['label']=='KEEP':group='轨迹改变_保留存活优势'
        else:assert r['label']=='MIXED';group='轨迹改变_指标权衡'
        groups[group]+=1
        dims={};tol=max(.01,32*2.220446049250313e-16*(abs(a['cost'])+abs(b['cost'])))
        for k,t in [('cost',tol),('all_queries',0),('overload_steps',0),('max_rho',1e-6),('overload_excess_sum',1e-6)]:
            dims[k+'_skip_minus_keep']=b[k]-a[k] if both else None
            dims[k+'_preference']=direction(b[k]-a[k],t) if both else '整周不作数值比较'
        if both and not same:
            dd=[dims[k+'_preference'] for k in ['cost','all_queries','overload_steps','max_rho','overload_excess_sum']]
            independent='SKIP' if '取消更好' in dd and '保留更好' not in dd else 'KEEP' if '保留更好' in dd and '取消更好' not in dd else 'MIXED'
            assert independent==r['label']
        if same:assert r['label']=='SKIP' and r['queries_saved']>0
        if r['label']=='KEEP':assert r['completion_complete'] and not r['skip_complete']
        common=dict(case_id=r['case_id'],cohort=r['cohort'],week=r['week'],step=r['step'],calendar_date=r['calendar_date'],pilot_known=r['pilot_known'])
        result=dict(**common,cleanup_group=group,physical_path_identical=same,original_label=r['label'],original_reason=r['reason'],both_complete=both,
            keep_steps=r['completion_steps'],skip_steps=r['skip_steps'],keep_complete=r['completion_complete'],skip_complete=r['skip_complete'],
            local_queries=r['local_queries'],queries_saved=r['queries_saved'],whole_week_cost_delta=r['cost_skip_minus_completion'] if both else None,**dims)
        outcomes.append(result);inputs.append(dict(case_id=r['case_id'],**r['features']))
        metadata.append(dict(**common,source=cases[r['case_id']]['source'],state_hash=cases[r['case_id']]['state_hash'],
            trajectory_effect_label='SAME_PATH' if same else 'CHANGED_PATH',preference_label=r['label'] if not same else '',original_label=r['label'],
            main_evaluation_eligible=not r['pilot_known'],not_a_new_sample=True))
        full.append(dict(metadata=metadata[-1],pre_completion_features=dict(r['features']),outcome=result,original_horizon_evidence=r['horizons']))
    assert groups==Counter({'轨迹相同_均完成':30,'轨迹相同_均未完成':9,'轨迹改变_取消占优':5,'轨迹改变_保留存活优势':1,'轨迹改变_指标权衡':11})
    csvsave('01_轨迹不变_39例.csv',[r for r in outcomes if r['physical_path_identical']])
    csvsave('02_轨迹改变_17例.csv',[r for r in outcomes if not r['physical_path_identical']])
    csvsave('03_权衡明细_11例.csv',[r for r in outcomes if r['original_label']=='MIXED'])
    csvsave('04_补查前特征_56例.csv',inputs);csvsave('05_标签与分组_56例.csv',metadata);save('clean_records.json',full)
    # Preserve original calendar folds, not a newly optimized split.
    folds=[{k:r[k] for k in ['scheme','fold','train_ids','test_ids']} for r in read(SRC/'fits.json') if r['model']=='LOGISTIC']
    for f in folds:
        assert set(f['train_ids']).isdisjoint(f['test_ids'])
        if f['scheme']=='calendar_loo':
            mm={m['case_id']:m for m in metadata};assert {mm[i]['calendar_date'] for i in f['train_ids']}.isdisjoint(mm[i]['calendar_date'] for i in f['test_ids'])
    save('06_既有日期留组索引.json',dict(scope='Original GRID96 evaluation folds copied for traceability; no evaluation executed',folds=folds,known_pilot_ids=[r['case_id'] for r in metadata if r['pilot_known']]))
    patterns=Counter()
    for r in outcomes:
        if r['original_label']=='MIXED':patterns[tuple(r[k+'_preference'] for k in ['cost','all_queries','overload_steps','max_rho','overload_excess_sum'])]+=1
    save('summary.json',dict(rows=56,scenario_entries=21,calendar_dates=17,features=15,groups=dict(groups),original_labels=dict(Counter(r['label'] for r in rows)),
        new_labels_with_keep=sum(r['original_label']=='KEEP' and r['main_evaluation_eligible'] for r in metadata),
        mixed_patterns=[dict(dimensions=['cost','all_queries','overload_steps','max_rho','overload_excess_sum'],directions=list(k),count=v) for k,v in patterns.items()]))
    text='''# 补查记录清理说明

这次只整理现有56例，没有删除样本、重写原标签、修改模型或再跑实验。

记录分为两层：先看取消是否改变物理运行轨迹；对于改变轨迹的状态，再看两条分支的存活、成本、查询和过载指标。后一层没有统一偏好时仍保留MIXED，不强行改成可取消或必须保留。这是整理数据的方式，尚未构建两阶段学习模型。

## 清理后的五组

- 30例：两条分支均完成整周，轨迹相同，取消只减少查询。
- 9例：两条分支都提前结束，实际轨迹仍相同。它们是路径等价记录，不能当作“完整周运行成功”的证据，也没有删除。
- 5例：取消后轨迹改变，整周成本、查询及所记录过载指标均不差，并有至少一项改善。
- 1例：取消后1906步结束，保留补查则完成2017步。它是旧试点，单独保留；新标签中还没有第二例这类样本。
- 11例：取消与保留各有所得。逐维方向见03_权衡明细_11例.csv，原MIXED标签不改。

## 文件怎么用

01、02分别保留39个轨迹不变和17个轨迹改变的记录；03展开11个权衡样本。数据中的keep表示执行这一次补查，skip表示只取消这一次，之后仍用原FRAME。

04只含case_id与15个补查前可见特征，不含结果、原标签、日期、场景或站点ID。case_id只是对齐键，不是模型输入。05保存标签、来源和日期分组；06照录原来的日期留组索引，没有执行新的评价。clean_records.json保留全部1步、1小时、1天和计划整周证据。

轨迹是否相同、最终成本、最终过载和存活均是事后答案，不能加入在线输入。完全相同的15维特征也不自动视为重复样本：不同状态或未来时序可能有不同后果。本次只检查ID与来源，不凭特征相似删样本。

## 评价口径

原GRID96的44 SKIP、1 KEEP、11 MIXED和全部预测成绩保持不变。新分类只帮助阅读，不能回头用它选择模型、阈值或筛掉难例。三条已知试点仍单列；同日的G1/G2仍放在同一留出组。这里没有发现新的独立场景或新的学习成绩。

目前暂停新增训练和门控部署。神经网络、预训练表格模型均可留作候选；先明确11例权衡对应的任务偏好和重要补查的证据。后续若预测连续代价向量，可沿用这些记录，不必现在把它们强行压成一个好坏分数。
'''
    (OUT/'清理说明_中文.md').write_text(text,encoding='utf-8')
    for n,h in frozen.items():assert sha(SRC/n)==h,'Historical source changed'
    receipt=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=frozen,script_sha256=sha(W/'prepare.py'),rows_preserved=56,rows_deleted=0,original_labels_changed=0,new_replays=0,new_fits=0,new_predictions=0,group_leakage_checks=len(folds),all_source_hashes_unchanged=True,
        output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    save('cleanup_review.json',receipt);print(json.dumps(read(OUT/'summary.json'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
