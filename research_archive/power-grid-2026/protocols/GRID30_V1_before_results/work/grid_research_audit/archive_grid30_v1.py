"""Preserve setter failure and exact-copy revision before first shadow outcome."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
DEST=ARCH/'protocols/GRID30_V1_before_results';assert not DEST.exists() and not (ROOT/'outputs/grid30/qualification.json').exists()
files=list((ROOT/'work/grid30').glob('*.py'))+[Path(__file__).resolve()]
files += [ROOT/'outputs/grid30'/n for n in ['design.json','code_freeze.json','parameter_copy_diagnostic.json']]
files += [p for p in (ROOT/'outputs/grid30/revisions/v0_parameter_setter').rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md','.log']]
records=[]
for p in files:
    rel=p.relative_to(ROOT);target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    records.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(DEST/'manifest.json').write_text(json.dumps({'files':records,'scope':'V0setter failure before numeric/action shadow result; retained335physical-step fragment stays local. V1exact original array update, before outcome.'},indent=2),encoding='utf-8')
(DEST/'README.md').write_text('''# GRID30 V1影子参数复制修订

作者以原位赋值更新gen_per_bus，公开状态335中有两处负值，最小-0.0006708863656967878。影子拷贝器重新调用非负setter时拒绝，未进入有效数值比较。保留V0源码/协议/错误/335步片段。

V1影子数组原样原位复制并逐值核对，不投影/裁剪，不改变实际参数/控制器或系数。仍需通过手工与153次真实参数下的数值/动作资格，尚无完整提速结果。
''',encoding='utf-8')
p=REPO/'docs/EXPERIMENT_LOG.md';p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-11 — GRID30 replica-copy qualification repair\n\nV0manual8pass but public-state335gen_per_bus includes2negative original author entries(min-0.0006708864). Setter replica refused them before original/shadow safe solve. Retain335step fragment and old code. V1copies arrays in-place exactly as original updater, assertion no clipping; new numeric/action gate still required. No deployment/new fit/hidden dates.\n',encoding='utf-8')
p=REPO/'docs/RESEARCH_STATE.md';text=p.read_text(encoding='utf-8');p.write_text(text.split('\n',1)[0]+'\n\n- GRID30 V0影子setter拒绝原作者负值，非方法裁决；335步片段保留。V1原样复制参数已冻结，数值/动作资格尚未得出。\n'+text.split('\n',1)[1],encoding='utf-8')
print(json.dumps({'archived':len(records),'before_first_numeric_result':True}))
