"""Shorten crowded plot labels only; no data or experiment change."""
import hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid61_figure_v1';assert not OUT.exists();OUT.mkdir()
source=ROOT/'outputs/grid61/review.json';R=json.loads(source.read_text(encoding='utf-8'));assert R['passed_saved_data']
labels=['FULL','First-safe','No local\ncompletion','Framework'];colors=['#8997A5','#75A6C9','#D5A556','#294D70']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(12.4,4.5))
for ax,field,title in zip(axes,['macro_query_reduction_full','macro_control_reduction_full','worst_complete_cost_delta_full'],['N1 query reduction','Controller time reduction','Worst complete-week cost increase']):
    v=[100*r[field] for r in R['summaries']];bars=ax.bar(labels,v,color=colors,width=.6)
    for b,n in zip(bars,v):ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(v)*.025,f'{n:.2f}%',ha='center',va='bottom',fontsize=9)
    ax.set_title(title,fontsize=11);ax.set_ylabel('Relative to FULL (%)');ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_ylim(0,max(v)*1.22)
    if field=='worst_complete_cost_delta_full':ax.axhline(.5,color='#7E3136',linestyle='--',lw=1);ax.text(.05,.58,'0.5% screen',fontsize=9,color='#7E3136')
fig.suptitle('Common-order framework ablation: four old development weeks, two timing repeats',fontsize=12)
fig.text(.02,.055,'All policies share ranking, base control and DC/safe-QP caches. One common failed week retained; cost uses complete weeks.',fontsize=8)
fig.text(.02,.012,'First-safe reference includes the stronger best-admissible fallback after exhaustion. This is a search-policy adaptation.',fontsize=8)
fig.tight_layout(rect=(0,.11,1,.91));fig.savefig(OUT/'framework_mechanism.png',dpi=170);fig.savefig(OUT/'framework_mechanism.pdf');plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'revision.json').write_text(json.dumps({'numerical_changes':0,'source_sha256':sha(source),'old_figures_preserved':'outputs/grid61_mechanism/framework_mechanism.pdf/png','change':'Shorter tick labels and explicit fallback footnote'},indent=2),encoding='utf-8')
dest=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID61_figure_v1';assert not dest.exists();dest.mkdir();records=[]
for p in [Path(__file__)]+list(OUT.iterdir()):
    target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    records.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
(dest/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps({'figure_saved':True,'numerical_changes':0}))
