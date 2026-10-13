"""Make low-cost points legible; preserve original figure and numerical tables."""
import csv,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid56_figure_v1';assert not OUT.exists();OUT.mkdir()
source=ROOT/'work/grid56/build.py';text=source.read_text(encoding='utf-8');table=[]
with (ROOT/'outputs/grid56/main_table.csv').open(encoding='utf-8') as f:
    for r in csv.DictReader(f):
        for k in ['macro_control_reduction_full_pct','worst_family_cost_delta_full_pct','query_reduction_full_pct']:r[k]=float(r[k])
        table.append(r)
start=text.index('import matplotlib\n');end=text.index("save('outputs_manifest.json'")
exec(compile(text[start:end],str(source),'exec'),{'OUT':OUT,'table':table})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'table_sha256':sha(ROOT/'outputs/grid56/main_table.csv'),'source_sha256':sha(source),
    'numerical_changes':0,'old_figures_retained':'outputs/grid56/quality_speed.pdf/png',
    'style':'Symmetric-log cost axis with linear range0.2% and external legend; displayed coordinates unchanged.',
    'initial_inline_render_failure':'Source read_text() without UTF8 failed on Windows GBK before render. Figure-only helper uses explicit UTF8; no scientific outputs changed.'}
(OUT/'revision.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
ARCH=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026';dest=ARCH/'protocols/GRID56_figure_v1';assert not dest.exists();dest.mkdir()
entries=[]
for p in [Path(__file__).resolve()]+list(OUT.iterdir()):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries},indent=2),encoding='utf-8')
print(json.dumps({'figure_only':True,'files':len(entries),'numerical_changes':0}))
