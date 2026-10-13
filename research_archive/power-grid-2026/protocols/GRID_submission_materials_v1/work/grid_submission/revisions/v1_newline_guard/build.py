"""Offline author materials, unified tables and traceable source/evidence receipts."""
import csv
import hashlib
import io
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid_submission_materials_v1'
POLICIES = ['FULL', 'FIRST_SAFE_BEST', 'NO_LOCAL', 'FRAME']
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
avg = lambda v: math.fsum(v) / len(v) if v else None


def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if p.exists():
        assert p.read_bytes() == data, ('different partial artifact', p)
    else:
        p.write_bytes(data)


def csvfile(name, rows):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    data = b'\xef\xbb\xbf' + f.getvalue().encode('utf-8')
    if p.exists():
        assert p.read_bytes() == data, ('different partial CSV', p)
    else:
        p.write_bytes(data)


def stage_tables(stage):
    design = read(ROOT / 'outputs' / stage / 'design.json')
    finished = read(ROOT / 'outputs' / stage / 'finished.json')
    review = read(ROOT / 'outputs' / stage / 'review.json')
    assert finished['passed_engineering'] and review['passed_saved_data']
    ix = {(s['repeat'], s['week'], s['method']): s for s in finished['summaries']}
    families, summary = [], []
    for week in design['weeks']:
        fs = [ix[rep, week, 'FULL'] for rep in (0, 1)]
        for method in POLICIES:
            ss = [ix[rep, week, method] for rep in (0, 1)]
            equal = all(s['steps'] == f['steps'] and s['complete'] == f['complete'] for s, f in zip(ss, fs))
            full_complete, complete = all(s['complete'] for s in fs), all(s['complete'] for s in ss)
            survive = all(s['complete'] if f['complete'] else s['steps'] >= f['steps'] for s, f in zip(ss, fs))
            fc, sc = avg([s['cost'] for s in fs]), avg([s['cost'] for s in ss])
            cost = (sc - fc) / abs(fc) if complete and full_complete else None
            control = avg([s['controller_s'] for s in ss])
            fq, sq = avg([s['n1_forecasts'] for s in fs]), avg([s['n1_forecasts'] for s in ss])
            families.append({'stage': stage.upper(), 'data_role': 'development' if stage == 'grid61' else 'limited_fresh_confirmation',
                'week': week, 'method': method, 'timing_repeats': 2, 'complete': complete, 'steps': ss[0]['steps'],
                'no_earlier_failure': survive, 'equal_horizon_full': equal, 'cost': sc, 'cost_delta_full': cost,
                'quality_pass': bool(survive and (cost is None or cost <= .005)), 'controller_s': control,
                'control_r0_s': ss[0]['controller_s'], 'control_r1_s': ss[1]['controller_s'],
                'control_reduction_full': 1-control/avg([s['controller_s'] for s in fs]) if equal else None,
                'n1_queries': sq, 'query_reduction_full': 1-sq/fq if equal and fq else None,
                'n1_source_s': avg([s['n1_source_s'] for s in ss])})
    for method in POLICIES:
        rows = [x for x in families if x['method'] == method]
        legacy = next(x for x in review['summaries'] if x['method'] == method)
        summary.append({'stage': stage.upper(), 'data_role': rows[0]['data_role'], 'method': method,
            'families': len(rows), 'runs': 2*len(rows), 'completed_families': sum(x['complete'] for x in rows),
            'quality_families': sum(x['quality_pass'] for x in rows),
            'control_comparable_families': sum(x['control_reduction_full'] is not None for x in rows),
            'query_comparable_families': sum(x['query_reduction_full'] is not None for x in rows),
            'macro_control_reduction_full': avg([x['control_reduction_full'] for x in rows if x['control_reduction_full'] is not None]),
            'macro_query_reduction_full': avg([x['query_reduction_full'] for x in rows if x['query_reduction_full'] is not None]),
            'worst_complete_cost_delta_full': max((x['cost_delta_full'] for x in rows if x['cost_delta_full'] is not None), default=None),
            'legacy_report_macro_control': legacy['macro_control_reduction_full'],
            'estimator': 'average repeat time within date, then ratio, then equal-date macro'})
    return families, summary


def main():
    assert not (OUT / 'validation.json').exists(), 'Material version already built.'
    (OUT / 'reproducibility').mkdir(parents=True, exist_ok=True)
    sources = []
    for p in ['outputs/grid61/design.json', 'outputs/grid61/finished.json', 'outputs/grid61/review.json',
              'outputs/grid62/design.json', 'outputs/grid62/finished.json', 'outputs/grid62/review.json',
              'outputs/grid62_timing_review/review.json', 'outputs/grid_paper_v2/source_receipts.json',
              'work/grid60/policies.py', 'work/grid27/policy.py', 'work/grid61/mechanism_adapter.py',
              'work/grid55/worker.py', 'work/grid03/score_adapter.py']:
        sources.append({'path': p, 'sha256': sha(ROOT/p), 'bytes': (ROOT/p).stat().st_size})
    all_families, all_summary = [], []
    for stage in ('grid61', 'grid62'):
        fs, ss = stage_tables(stage); all_families.extend(fs); all_summary.extend(ss)
    csvfile('tables/main_results_unified.csv', all_summary)
    csvfile('tables/all_families.csv', all_families)
    reconciliation = [{'stage': x['stage'], 'method': x['method'], 'legacy_control_reduction': x['legacy_report_macro_control'],
        'unified_control_reduction': x['macro_control_reduction_full'],
        'difference_percentage_points': 100*(x['macro_control_reduction_full']-x['legacy_report_macro_control']),
        'note': 'Offline estimator reconciliation, original report unchanged, no rerun or verdict change.'} for x in all_summary]
    csvfile('tables/estimator_reconciliation.csv', reconciliation)
    for x in all_summary:
        if x['stage'] == 'GRID62':
            assert abs(x['macro_control_reduction_full'] - x['legacy_report_macro_control']) < 1e-12
    lines = ['# 可用于重写的统一主表', '', '先平均同日期两计时，再算对FULL比例；只对可比较视野计算。原报告不覆盖。', '']
    tex = ['% Generated table fragment; requires booktabs. Ratios from summary receipts.', '\\begin{tabular}{llrrrr}', '\\toprule', 'Set & Policy & Quality & Complete & Time saved & Queries saved \\\\', '\\midrule']
    for stage in ('GRID61', 'GRID62'):
        lines += ['## '+stage+('（四旧开发日期）' if stage == 'GRID61' else '（三新封存日期）'), '',
            '|策略|质量日期|完成日期|控制减少|查询减少|最坏完整周成本变化|', '|---|---:|---:|---:|---:|---:|']
        for x in [x for x in all_summary if x['stage'] == stage]:
            n = x['families']; t = 100*x['macro_control_reduction_full']; q = 100*x['macro_query_reduction_full']; c = 100*x['worst_complete_cost_delta_full']
            lines.append(f"|{x['method']}|{x['quality_families']}/{n}|{x['completed_families']}/{n}|{t:.2f}%|{q:.2f}%|{c:+.5f}%|")
            method_tex = x['method'].replace('_', '\\_')
            tex.append(f"{stage} & {method_tex} & {x['quality_families']}/{n} & {x['completed_families']}/{n} & {t:.2f}\\% & {q:.2f}\\% \\\\")
        lines += ['']
    lines += ['GRID62查询平均只有两个FULL非零查询日期进入分母；五月所有方法未调用拓扑搜索。五月仍保留在原三日期控制平均和存活结果中。', '',
        '开发组GRID61的统一FRAME控制节省13.1121%，与原逐次比例汇总13.0588%相差0.0533个百分点；这只是不同聚合顺序，旧原稿13.06%不删除。']
    tex += ['\\bottomrule', '\\end{tabular}', '% Quality includes no earlier failure; does not mean completed episodes.']
    (OUT/'tables/main_tables.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (OUT/'tables/main_table_fragment.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
    claims = [
        {'id':'C01','claim':'FRAME相对同排序FULL有有限质量—计算改善','status':'SUPPORTED_IN_FIXED_MATRICES','source':'tables/main_results_unified.csv','locator':'method=FRAME;stage=GRID61/GRID62','limit':'不是对所有成熟策略全面领先，也不是生产或跨电网保证'},
        {'id':'C02','claim':'开发组FRAME控制减少13.1121%、查询减少69.3671%','status':'OFFLINE_UNIFIED_ESTIMATOR','source':'tables/main_results_unified.csv','locator':'GRID61/FRAME','limit':'四旧日期；原报告13.0588%保留；无新实验'},
        {'id':'C03','claim':'封存FRAME控制减少6.0858%、查询减少83.5429%','status':'FROZEN_SCREEN_PASSED','source':'outputs/grid62/review.json','locator':'summaries[method=FRAME]','limit':'3日期，查询分母仅2日期；质量3/3不等于完成3/3'},
        {'id':'C04','claim':'三月补全把完整周成本差4.81017%降到0.37200%','status':'CONTROLLED_DEVELOPMENT_MECHANISM','source':'outputs/grid61/per_run.csv','locator':'week=2035-03-05_0;FRAME vs NO_LOCAL','limit':'强质量修复集中一个旧日期'},
        {'id':'C05','claim':'1733步额外198次补查找回FULL同选动作309','status':'SINGLE_STATE_WITNESS','source':'outputs/grid60/qualification.json','locator':'known critical state1733;also GRID61 method draft','limit':'不是全池或整周最优证明'},
        {'id':'C06','claim':'补全不降低同状态已观测选择键，新增查询|L|','status':'CONDITIONAL_PROPERTY','source':'work/grid60/policies.py','locator':'run, local completion branch','limit':'只在触发补全时，不推导长期安全/整周成本'},
        {'id':'C07','claim':'新日期FIRST_SAFE_BEST也全部通过质量','status':'ADVERSE_COMPETITIVE_REFERENCE','source':'outputs/grid62/review.json','locator':'summaries[FIRST_SAFE_BEST]','limit':'其五月无搜索计时差不能归因停止政策'},
        {'id':'C08','claim':'两有调用日期FRAME少8.2496%、FIRST少11.9210%控制时间','status':'POSTHOC_DESCRIPTIVE_ONLY','source':'outputs/grid62_timing_review/review.json','locator':'active_only_descriptive','limit':'不换主判据，不删五月，不当新确认'},
        {'id':'C09','claim':'同路径重复仍有接近两倍控制/CPU计时差','status':'TIMING_LIMITATION','source':'outputs/grid62_timing_review/review.json','locator':'repeat_pairs','limit':'未定位负载/频率根因，无待机不排除所有变化'},
        {'id':'C10','claim':'共有缓存属于共享工程优化','status':'ATTRIBUTION_BOUNDARY','source':'outputs/grid62/design.json','locator':'cache_attribution;methods cache flags','limit':'缓存收益不归GNN、不同批比例不相加'},
        {'id':'C11','claim':'没有建立GNN对强频次顺序稳定增量','status':'NOT_ESTABLISHED','source':'outputs/grid57_prior_review/review.json;outputs/grid59/review.json','locator':'historical comparable ranking results','limit':'不否定所有GNN或RL，不用新训练救当前主张'},
        {'id':'C12','claim':'共同失败不计为质量收益','status':'FAILURES_RETAINED','source':'tables/all_families.csv','locator':'June1901 development;May747 fresh','limit':'失败成本不作完整周成本比'},
        {'id':'C13','claim':'FULL穷举当前canonical动作池','status':'REFERENCE_SCOPE','source':'work/grid60/policies.py','locator':'FULL branch','limit':'不是所有电网拓扑/全周最优'},
        {'id':'C14','claim':'benchmark成本含curtailment前后变化','status':'ACCOUNTING_DEFINITION','source':'work/grid03/score_adapter.py','locator':'public_cost_ledger','limit':'不声称现实财务/总弃电收益'}]
    csvfile('claims_evidence.csv', claims); save('claims_evidence.json', claims)
    code_map = []
    tokens = {'work/grid27/policy.py':['def key','def staged','initial =','stale =','len(results) >= 256'],
        'work/grid60/policies.py':['def run','sub=int','targets=','local_completion','FIRST_SAFE_BEST'],
        'work/grid61/mechanism_adapter.py':['def rank','def choose','limit=min','def dispatch','valid='],
        'work/grid55/worker.py':['def worker','decision_s =','raw =','public_cost_ledger','summary ='],
        'work/grid03/score_adapter.py':['def public_cost_ledger','delta_curtailed=','def cost_rounding_tolerance']}
    for source, needles in tokens.items():
        content = (ROOT/source).read_text(encoding='utf-8').splitlines()
        for needle in needles:
            hits = [i+1 for i,line in enumerate(content) if needle in line]
            assert hits, (source, needle)
            code_map.append({'source':source,'line':hits[0],'anchor':needle,'sha256':sha(ROOT/source)})
        dst=OUT/'method/source'/source;dst.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(ROOT/source,dst)
    csvfile('method/code_mapping.csv',code_map)
    for name, path in [('fresh_confirmation','outputs/grid_paper_v2/fresh_confirmation'),('development_mechanism_legacy','outputs/grid61_figure_v1/framework_mechanism')]:
        for suffix in ('png','pdf'):
            src = ROOT/(path+'.'+suffix);dst=OUT/'figures'/(name+'.'+suffix)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    shutil.copy2(ROOT/'outputs/grid_paper_v2/manuscript.md',OUT/'英文V2研究工作稿_供核对非提交.md')
    shutil.copy2(ROOT/'outputs/grid_paper_v2/REPRODUCE.md',OUT/'reproducibility/原复现边界说明.md')
    shutil.copy2(ROOT/'work/grid_paper/reproduce_tables.py',OUT/'reproducibility/reproduce_tables.py')
    shutil.copy2(ROOT/'outputs/grid_paper_v2/software_environment.json',OUT/'reproducibility/software_environment.json')
    for stage in ('grid61','grid62'):
        folder = OUT/'reproducibility/summaries'/stage;folder.mkdir(parents=True, exist_ok=True)
        for name in ('design.json','finished.json','review.json','per_run.csv','code_freeze.json'):
            shutil.copy2(ROOT/'outputs'/stage/name,folder/name)
    payloads = []
    for stage in ('grid61','grid62'):
        for p in sorted((ROOT/'outputs'/stage/'runs').rglob('*')):
            if p.is_file():
                payloads.append({'stage':stage,'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    save('reproducibility/raw_payload_manifest.json',payloads)
    save('reproducibility/hardware_snapshot.json',{'checked_date':'2026-10-13','processor':'AMD Ryzen 9 8940HX with Radeon Graphics',
        'physical_cores':16,'logical_processors':32,'scope':'Current host snapshot. Historical traces record wall and process CPU; instantaneous clock/load were not independently measured.',
        'formal_execution':'One serial worker; numerical and torch threads constrained. No model fits in GRID61/62.'})
    sources.extend({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
                   for p in [ROOT/'outputs/grid60/qualification.json', ROOT/'outputs/grid61_figure_v1/revision.json'])
    save('reproducibility/source_receipts.json',sources)
    save('reproducibility/material_audit.json',{'passed':True,'new_simulator_runs':0,'new_fits':0,
        'raw_files':len(payloads),'raw_bytes':sum(x['bytes'] for x in payloads),'source_receipts':len(sources),
        'table_rows':len(all_summary),'family_rows':len(all_families),'claims':len(claims),'code_anchors':len(code_map),
        'scope':'Offline material preparation and estimator reconciliation. Original scientific outputs and verdicts unchanged. Not independent AC solving or native LaTeX typesetting.'})
    print(json.dumps({'passed':True,'tables':len(all_summary),'families':len(all_families),'claims':len(claims),'raw_bytes':sum(x['bytes'] for x in payloads)}))


if __name__ == '__main__':
    main()
