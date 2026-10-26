"""Finish passed audit delivery, descriptive comparisons, and lossless raw-record archive."""
from pathlib import Path
import collections,copy,datetime,gzip,hashlib,json,math,shutil,sys,zipfile
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid184_search_adapter';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID184_search_adapter'
sys.path.insert(0,str(R/'work/grid184_search_adapter'));import run as original
read=original.read;sha=original.sha
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def stage(name,files):
    dest=A/name;dest.mkdir(exist_ok=False)
    for p in files:
        if p.is_file():shutil.copy2(p,dest/p.name)
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
def steps(row):
    p=O/'runs'/f"r{row['repeat']}__{row['week']}__{row['method']}"/'steps.jsonl.gz'
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(x) for x in f]
def describe():
    original.verify();review=read(O/'review.json');decision=read(O/'decision.json');assert review['passed_saved_data'] and review['runs']==64 and review['checks']['repeat_pairs']==32
    assert read(O/'audit_recovery_progress.json')['completed_runs']==64
    assert not (O/'results_summary.json').exists();rs=review['runrows'];d=read(O/'design.json');key=lambda r:(r['repeat'],r['week'],r['method']);lookup={key(r):r for r in rs};family={}
    for mode in ['FRAME_GUARD','LOCAL_BATCH1','FACTOR','CONSENSUS']:
        rows=[r for r in rs if r['method']==mode or r['method'].startswith(mode+'_seed')];first=[r for r in rows if r['repeat']==0]
        nseeds=3 if mode in ['FACTOR','CONSENSUS'] else 1
        family[mode]=dict(configurations=len(rows),seed_count=nseeds,quality_dates_per_seed={str(s):sum(r['quality']=='PASS_COMPLETE' for r in first if r['method']==(f'{mode}_seed{s}' if nseeds==3 else mode)) for s in range(nseeds)},complete_dates_per_seed={str(s):sum(r['complete'] for r in first if r['method']==(f'{mode}_seed{s}' if nseeds==3 else mode)) for s in range(nseeds)},reference_limited_dates=sorted({r['date'] for r in first if r['quality']=='REFERENCE_LIMITED'}),cost_failure_dates=sorted({r['date'] for r in first if r['quality']=='FAIL_COST'}),
            macro_control_s=float(np.mean([r['controller_s'] for r in rows])),macro_candidate_queries=float(np.mean([r['candidate_queries'] for r in rows])),macro_total_queries=float(np.mean([r['all_queries'] for r in rows])),macro_all_p95_s=float(np.mean([r['all_p95'] for r in rows])),macro_search_p95_s=float(np.mean([r['search_p95'] for r in rows if r['search_p95'] is not None])),macro_feature_s=float(np.mean([r['local_feature_s'] for r in rows])),macro_inference_s=float(np.mean([r['local_inference_s'] for r in rows])))
    comparisons={};allpairs=[]
    for reference in ['FRAME_GUARD','LOCAL_BATCH1','FACTOR']:
        pairs=[]
        for a in rs:
            if not a['method'].startswith('CONSENSUS'):continue
            seed=int(a['method'].split('seed')[1]);name=f'FACTOR_seed{seed}' if reference=='FACTOR' else reference;b=lookup[a['repeat'],a['week'],name]
            assert a['steps']==b['steps'] and a['complete']==b['complete']
            pairs.append(dict(seed=seed,date=a['date'],repeat=a['repeat'],reference=name,control_reduction=1-a['controller_s']/b['controller_s'],candidate_query_reduction=1-a['candidate_queries']/b['candidate_queries'],total_query_reduction=1-a['all_queries']/b['all_queries'],all_p95_reduction=1-a['all_p95']/b['all_p95'],search_p95_reduction=1-a['search_p95']/b['search_p95'] if a['search_p95'] is not None and b['search_p95'] is not None else None,complete=a['complete'],cost_delta=a['cost']/b['cost']-1 if a['complete'] else None,signature_equal=a['signature']==b['signature']))
        full=[v for v in pairs if v['complete']];q={k:float(np.mean([v[k] for v in pairs if v[k] is not None])) for k in ['control_reduction','candidate_query_reduction','total_query_reduction','all_p95_reduction','search_p95_reduction']}
        q.update(complete_dates_control_reduction=float(np.mean([v['control_reduction'] for v in full])),complete_dates_cost_delta=float(np.mean([v['cost_delta'] for v in full])),same_observed_horizon_all_pairs=True,included_reference_limited_date='2037-05-04',per_seed={str(s):dict(control_reduction=float(np.mean([v['control_reduction'] for v in pairs if v['seed']==s])),candidate_query_reduction=float(np.mean([v['candidate_query_reduction'] for v in pairs if v['seed']==s])),complete_cost_delta=float(np.mean([v['cost_delta'] for v in full if v['seed']==s]))) for s in [0,1,2]},per_date={date:dict(control_reduction=float(np.mean([v['control_reduction'] for v in pairs if v['date']==date])),candidate_query_reduction=float(np.mean([v['candidate_query_reduction'] for v in pairs if v['date']==date]))) for date in sorted({v['date'] for v in pairs})},repeat_macros=[float(np.mean([v['control_reduction'] for v in pairs if v['repeat']==rep])) for rep in [0,1]],identical_trajectory_pairs=sum(v['signature_equal'] for v in pairs))
        comparisons[reference]=q;allpairs+=pairs
    divergences=[]
    for seed in [0,1,2]:
        for date in sorted({r['week'] for r in rs}):
            a=lookup[0,date,f'CONSENSUS_seed{seed}'];b=lookup[0,date,f'FACTOR_seed{seed}'];sa=steps(a);sb=steps(b);different=next((i for i,(x,y) in enumerate(zip(sa,sb)) if x['action_hash']!=y['action_hash']),None)
            item=dict(seed=seed,week=date,physical_signature_equal=a['signature']==b['signature'],cost_delta=a['cost']/b['cost']-1 if a['complete'] and b['complete'] else None,first_physical_action_divergence_step=None if different is None else sa[different]['step'])
            if different is not None:
                item['same_before_observation']=sa[different]['before_hash']==sb[different]['before_hash'];assert item['same_before_observation']
                def at(row):
                    folder=O/'runs'/f"r0__{date}__{row['method']}"
                    return [read(p) for p in folder.glob('call[0-9][0-9][0-9].json') if read(p)['step']==sa[different]['step']]
                ca=at(a);cb=at(b);details=[]
                for x,y in zip(ca,cb):
                    assert x['state_hash']==y['state_hash'];prefix=[]
                    for dx,dy in zip(x['learned_decisions'],y['learned_decisions']):
                        if dx['action']!=dy['action']:
                            assert dx['raw_features']==dy['raw_features'] and dx['next_ids']==dy['next_ids'];prefix.append(dict(batch=dx['batch'],paid_queries=dx['paid_queries'],same_raw41=True,consensus_action=dx['action'],factor_action=dy['action'],consensus_scores=dx.get('scores'),factor_scores=dy.get('scores'),consensus_delegated=dx.get('delegated')));break
                    details.append(dict(state_hash=x['state_hash'],consensus_selected=x['chosen_pool_id'],factor_selected=y['chosen_pool_id'],first_local_decision_difference=prefix))
                item['common_state_call_details']=details
            divergences.append(item)
    save(O/'first_divergences.json',dict(entries=divergences,scope='Firstactualphysicalactiondifferenceoncommonobservedprefix;notacompletecausalexplanationofwholeweekcost. No newpredictions.'))
    result=dict(families=family,consensus_comparisons=comparisons,pairs=allpairs,macro_convention='Equal4dates,2timingrepeats,3seeds;oneREF_LIMITEDdateincludedforobserved-horizoncomputeonly. Also report3complete dates separately. Costratiosonlycompleteweeks.',quality_definition='FULLcomplete:completion+cost<=100.5%;referenceincompleteungraded. Allseed/datefailuresretained.',new_fits=0,new_physics=0)
    save(O/'results_summary.json',result)
    # The original audit assertions and outputs passed before a copy-only footer error.
    save(O/'audit_recovery_execution.json',dict(audit_assertions_passed=True,assertions_unchanged=True,checks=review['checks'],audit_process_exit=1,post_assertion_error="NameError: shutil not imported in original copy footer",publication_only_repair=True,no_second_audit=True,post_footer_original_freeze_verify_passed=True,new_physics=0,new_fits=0))
    oldreport=O/'GRID184_report.md';shutil.copy2(oldreport,O/'GRID184_report_audit_initial.md')
    pct=lambda x:f'{100*x:.2f}%'
    lines=['# GRID184：固定共识的真实闭环开发比较','', '64条固定回放和保存数据核验均已完成。三种子、四日期、两次严格反向计时全部保留；本轮没有拟合、生成新场景或打开最终8。审计逐个验证238112个向量、164862条预测反馈、486次搜索与32对重复轨迹，未知计划的工程回退为0。','',
        '原审计因关机中断；恢复版仅把每一步反复解压NPZ改成每个run解压两成员各一次，全部29处assert与核查标准保留。64条核查和review/decision已成功落盘，随后复制协作资料的收尾因原脚本缺少shutil导入退出；此次单独补复制并再次核对冻结，没有重复审计或物理回放。','',
        '## 完成与质量','',
        '所有方法和种子均在3个日期完成整周，在2037-05-04都运行到1390步。该日期的FULL参照自身未完成，统一标为REFERENCE_LIMITED；不能按整周成本判质量，也不能把其提前终止算提速。三个可判整周质量的日期中，所有方法/种子都是2个通过、1个超成本阈值。因此“2/4”包括一个不可判日期，准确说法为可判日期2/3通过。','',
        '2037-03-02是共同的成本失败日期。相对FULL，FRAME成本增加1.0268%，固定一批增加0.5712%；FACTOR三种子分别增加2.1498%、1.0268%、0.5712%，共识分别增加2.8026%、1.1819%、0.5712%。不能把共识称为保持全部整周质量的计算优势。','',
        '## 实际控制时间与查询','',
        '以下先对每个日期、重复和种子算相对变化，再等权宏平均；数值为减少比例，负号表示增加。四日期计算统计含一个共同未完周，只描述相同已运行时域；另给三个完整日期的时间统计。', '']
    for ref,label in [('FRAME_GUARD','原FRAME'),('LOCAL_BATCH1','固定一批'),('FACTOR','同种子原学习模型FACTOR')]:
        c=comparisons[ref];lines.append(f"相对{label}：相同已运行时域控制时间减少{pct(c['control_reduction'])}，候选查询减少{pct(c['candidate_query_reduction'])}，总查询减少{pct(c['total_query_reduction'])}；只看三个完整日期，控制时间减少{pct(c['complete_dates_control_reduction'])}，完整周成本平均变化{pct(c['complete_dates_cost_delta'])}。全步骤P95减少{pct(c['all_p95_reduction'])}，搜索步骤P95减少{pct(c['search_p95_reduction'])}。")
    lines += ['', '各配置原始时间（每日期、每重复的控制秒数均值）与查询：']
    for mode,v in family.items():lines.append(f"- {mode}：控制{v['macro_control_s']:.3f}秒，候选{v['macro_candidate_queries']:.2f}次，总查询{v['macro_total_queries']:.2f}次；全步骤P95 {1000*v['macro_all_p95_s']:.3f}毫秒，搜索步骤P95 {1000*v['macro_search_p95_s']:.3f}毫秒。学习家族再对三种子等权。")
    lines += ['', '## 哪些改善保留，哪些不能合并','',
        '当前组合相对未学习FRAME存在计算节省，但与同种子FACTOR相比，新增便宜共识并未同时带来稳定的整周成本与计算改善。3月种子0/1的整周成本更高，种子2与对应FACTOR的全部四日期物理轨迹一致。离线局部NET改善不能直接当作整周运行成本改善。小的收益仍保留，不以全赢门槛否定；下一步应先解释已有有利/不利分歧，不立即扩大场景或继续拼门槛。','',
        'first_divergences.json只记录共同已观察物理前缀上的首次实际动作差异和当时已付费的模型反馈。不把第一次差异当成整周成本变化的完整因果解释；没有独立AC重求解。','',
        '## 证据与适用范围','',
        '本轮仅为同一公开电网原4已暴露开发日期的第一份已有场景，重复计时不是新场景，3个训练种子不是独立电网。FULL只用于质量/存活/时域参照，没有混用旧FULL时间。原始逐步成本、向量、预测反馈、模型决定与全部失败按原样保留；results_summary.json给出全部配对与宏平均口径，review.json与decision.json保持审计原件不改。']
    oldreport.write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    dest=R/'论文协作资料/FRAME_小闭环结果_GRID184';dest.mkdir(exist_ok=True)
    for n in ['GRID184_report.md','review.json','decision.json','results_summary.json','first_divergences.json','audit_recovery_execution.json']:
        p=dest/n;assert not p.exists();shutil.copy2(O/n,p);assert sha(p)==sha(O/n)
    original.verify();stage('audit_and_results',[*W.glob('*.py'),*W.glob('*.json'),*O.iterdir()]);print('GRID184_RESULTS_SUMMARIZED',json.dumps(comparisons,ensure_ascii=False))

def records():
    assert read(O/'review.json')['passed_saved_data'];dest=A/'raw_records';dest.mkdir(exist_ok=False);objects=dest/'objects';objects.mkdir();entries={};known={};large={};total=0
    with zipfile.ZipFile(dest/'small_files.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted((O/'runs').rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(O).as_posix();h=sha(p);n=p.stat().st_size;total+=n
            if h not in known:
                if n>=1048576:
                    target=objects/(h+'.blob');shutil.copyfile(p,target);assert sha(target)==h;known[h]='objects/'+target.name;large[h]=n
                else:z.write(p,h);known[h]='small_files.zip!'+h
            entries[rel]=dict(sha256=h,bytes=n,location=known[h])
    save(dest/'records_manifest.json',dict(files=entries,logical_files=len(entries),logical_bytes=total,unique_objects=len(known),large_objects=len(large),large_unique_bytes=sum(large.values()),small_zip_sha256=sha(dest/'small_files.zip'),encoding='Content-addressed lossless originals;all repeated paths retained in manifest,only identical SHA256 bytes deduplicated. Large objects useGitLFS.'))
    # Verify every logical member against the original byte hash, including ZIP reads.
    with zipfile.ZipFile(dest/'small_files.zip') as z:
        for h,where in known.items():
            actual=hashlib.sha256(z.read(h)).hexdigest() if where.startswith('small_files') else sha(dest/where);assert actual==h
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
    g=REPO/'.gitattributes';text=g.read_text(encoding='utf8');line='research_archive/power-grid-2026/protocols/GRID184_search_adapter/raw_records/objects/*.blob filter=lfs diff=lfs merge=lfs -text'
    assert line not in text;g.write_text(text.rstrip()+'\n'+line+'\n',encoding='utf8',newline='\n')
    save(O/'raw_archive_review.json',dict(passed=True,logical_files=len(entries),logical_bytes=total,large_unique_objects=len(large),large_unique_bytes=sum(large.values()),small_zip_bytes=(dest/'small_files.zip').stat().st_size,every_unique_object_readback=True,lfs_path_pattern=line))
    print('GRID184_RAW_RECORDS_PACKAGED',json.dumps(read(O/'raw_archive_review.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
