"""Keep setup failure and publish the stdlib-name repair before first result."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026';DEST=ARCH/'protocols/GRID29B_V1_before_results'
assert not DEST.exists() and not (ROOT/'outputs/grid29b/profile.json').exists()
files=sorted((ROOT/'work/grid29b').glob('*.py'))+[ROOT/'outputs/grid29b'/n for n in ['design.json','code_freeze.json']]
files+=[p for p in (ROOT/'outputs/grid29b/revisions/v0_stdlib_collision').rglob('*') if p.is_file()]+[Path(__file__).resolve()]
records=[]
for p in files:
    rel=p.relative_to(ROOT);target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    records.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(DEST/'manifest.json').write_text(json.dumps({'files':records,'scope':'No first numeric or control result. Repair script name to avoid Python stdlib profile collision.'},indent=2),encoding='utf-8')
(DEST/'README.md').write_text('''# GRID29B 初始化修订

profile.py与Python标准库重名，在Torch优化器初始化导入cProfile时递归进入试验脚本。控制器未初始化完成，没有交付控制步、DC配对结果或质量裁决。旧源码、失败日志和初始化目录保留。

改名probe_dc.py，其余数值/控制逻辑不变，重新冻结后运行。继续诊断GRID29的10.93%与0.39%差异：真实控制记忆下隔离DC配对，以及完整solver类型/编译/后端字段；不把带额外诊断的控制总时间用于提速比较。
''',encoding='utf-8')
p=REPO/'docs/EXPERIMENT_LOG.md';p.write_text(p.read_text(encoding='utf-8')+'''\n## 2026-10-11 — GRID29B setup-name repair, no numerical outcome

profile.py shadowed stdlib profile during Torch/cProfile initialization and recursively entered the driver. Failure before control initialization, no delivered physical step/DC pair. Preserve original code/log/setup directory. Rename probe_dc.py only, refreeze before running; existing GRID29 result rows unchanged. Human directs continued diagnosis of10.93% versus0.39%, especially what worked in the10% pair.\n''',encoding='utf-8')
print(json.dumps({'archived':len(records),'before_result':True}))
