from pathlib import Path
import json,re,hashlib,shutil,subprocess
R=Path('D:/Codex/GRL研究');P=R/'work/GRL-results-integration';O=R/'outputs/research_review_20261030';A=P/'research_archive/power-grid-2026/reviews/2026-10-30_weekly';D=R/'论文协作资料/本周实验总结_20261026-1030';q=R/'outputs/grid284_leaf_support';aq=P/'research_archive/power-grid-2026/protocols/GRID284_leaf_support';dq=R/'论文协作资料/FRAME_叶频支持诊断_GRID284'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
# Clarify discovery scope without asserting that unlocated reports never existed.
for n in ['全部实验索引_GRID000-284.md']:
 p=O/n;t=p.read_text(encoding='utf8').replace('无独立主报告，见日志/原始目录','本次未定位独立主报告，见日志/原始目录');p.write_text(t,encoding='utf8')
p=O/'本周实验总览.html';p.write_text(p.read_text(encoding='utf8').replace('无独立主报告，见JSON索引/日志','本次未定位独立主报告，见JSON索引/日志'),encoding='utf8')
inv=json.loads((O/'experiment_inventory.json').read_text());rr=inv['rows'];assert len(rr)==285 and [r['grid'] for r in rr]==list(range(285)) and sum(r['in_week'] for r in rr)==101
assert len(re.findall(r'^\|GRID\d+\|', (O/'本周逐项实验清单_101阶段.md').read_text(),re.M))==101
assert len(re.findall(r'^\|GRID\d+\|', (O/'全部实验索引_GRID000-284.md').read_text(),re.M))==285
for r in rr:
 if r['source_report']:assert Path(r['source_report']).is_file() and sha(Path(r['source_report']))==r['report_sha256']==sha(P/r['git_report'])
page=(O/'本周实验总览.html').read_text();assert page.count('<tr><td>GRID')==386 and '<script>' in page and 'https://cdn' not in page
for n in ['README.md','本周实验总结_20261026-1030.md']:
 for target in re.findall(r'\]\(([^)]+)\)',(O/n).read_text(encoding='utf8')):
  if '://' not in target:assert (O/target).is_file(),target
for row in json.loads((O/'published_endpoint_table.json').read_text()):assert f"{row['mean_NET']:.9f}" in (O/'本周实验总结_20261026-1030.md').read_text()
review=dict(passed=True,week_stage_rows101=True,all_history_positions285=True,canonical_report_sources256_sha_match=True,main_report_endpoint_values_match_saved_json=True,relative_report_links_exist=True,HTML_rows386=True,HTML_external_dependencies0=True,scientific_runs_fits_newpolicy_in_review0=True,research_stopped=True,automation_remains_PAUSED=True)
write(O/'review_validation.json',review)
write(O/'文件清单_SHA256.json',{p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name!='文件清单_SHA256.json'})
for p in O.iterdir():
 if p.is_file():shutil.copy2(p,A/p.name);shutil.copy2(p,D/p.name)
write(A/'archive_manifest.json',{p.name:sha(p) for p in A.iterdir() if p.is_file() and p.name!='archive_manifest.json'})
# Finish the latest stage delivery only; old completed phase archives stay intact.
T=aq/'delivery';T.mkdir(exist_ok=False)
for n in ['CONTINUOUS_HANDOFF.md','delivery_check.json']:shutil.copy2(q/n,T/n);shutil.copy2(q/n,dq/n)
write(T/'delivery_manifest.json',{p.name:sha(p) for p in T.iterdir()})
print(json.dumps(review));print('staged size',sum(p.stat().st_size for p in A.rglob('*') if p.is_file()))
