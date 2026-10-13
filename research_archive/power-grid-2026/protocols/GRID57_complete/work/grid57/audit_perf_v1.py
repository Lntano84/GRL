"""Read each NPZ vector array once; retain structural zero-query cases as missing ratios."""
import hashlib,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid57';p=ROOT/'work/grid57/audit.py'
source=p.read_text(encoding='utf-8')
replacements={
 "with np.load(folder/'vectors.npz') as z:\n            assert len(z['action'])==len(z['observation'])==len(actual)":
 "with np.load(folder/'vectors.npz') as z:\n            actions=z['action'];observations=z['observation']\n            assert len(actions)==len(observations)==len(actual)",
 "digest(z['action'][i])==r['action_hash'] and digest(z['observation'][i])==r['after_hash']":
 "digest(actions[i])==r['action_hash'] and digest(observations[i])==r['after_hash']",
 "cheap_ratio=mean([r['static_queries']/r['gnn_seed_average_queries'] for r in results])":
 "assert all(r['static_queries']>0 or r['gnn_seed_average_queries']==0 for r in results)\n    cheap_ratio=mean([r['static_queries']/r['gnn_seed_average_queries'] for r in results if r['gnn_seed_average_queries']>0])",
 "'mean_static_over_gnn_query_ratio':cheap_ratio,":
 "'mean_static_over_gnn_query_ratio':cheap_ratio,'comparable_query_families':len(gains),'zero_query_families_retained':[r['week'] for r in results if r['static_queries']==r['gnn_seed_average_queries']==0],",
 "        text.append(f\"|{r['week']}|{r['complete']}|{cost}|{gain}|{r['static_queries']}|{r['gnn_seed_average_queries']:.0f}|{100*r['gnn_query_reduction_vs_static']:.3f}%|\")":
 "        qgain='不定义（无查询）' if r['gnn_query_reduction_vs_static'] is None else f\"{100*r['gnn_query_reduction_vs_static']:.3f}%\"\n        text.append(f\"|{r['week']}|{r['complete']}|{cost}|{gain}|{r['static_queries']}|{r['gnn_seed_average_queries']:.0f}|{qgain}|\")"
}
for a,b in replacements.items():
    assert source.count(a)==1,a
    source=source.replace(a,b)
record={'original_audit_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'v0_stopped':'Audit process stopped for repeated NPZ decompression inside per-step indexing. No scientific run was active/repeated; original audit retained.',
    'array_read_change':'Decompress action/observation once per run; identical vector and hash checks.',
    'structural_zero_query_case':'2035-05-28_0 has zero STATIC_ID/FULL/GNN topology queries. All8cost/survival states retained; 0/0 query ratios missing, query statistics use7eligible families. Original>=6positive-family continuation requirement unchanged.',
    'new_runs':0,'new_fits':0,'predeclared_quality_threshold_unchanged':True,'no_early_failure_removed':True}
(OUT/'audit_v0_read_performance_and_zero_query.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
namespace={'__file__':str(p),'__name__':'grid57_audit_perf_v1'}
try:
    exec(compile(source,str(p),'exec'),namespace);namespace['main']()
    r=json.loads((OUT/'review.json').read_text())
    text='\n查询统计只在有拓扑调用的7个日期定义；2035-05-28_0双方0查询，保留其质量/成本结果，不强造0/0比例。报告中正向分子/总体8仍沿原门槛；详见comparable_query_families和工程修正记录。\n'
    with (OUT/'GRID57_report.md').open('a',encoding='utf-8') as f:f.write(text)
except BaseException:
    (OUT/'audit_v1_error.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2),encoding='utf-8')
    raise
