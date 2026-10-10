"""Plot audited validation comparisons, keeping survival separate from costs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08'
report=json.loads((OUT/'delivery.json').read_text());records=report['records']
names=[('v0','NN20'),('v0','ALWAYS_RESTORE'),('v0','IMMEDIATE_COST')]+[(v,k) for v in ['v0','v1','v2'] for k in ['gnn','mlp']]
labels=[f'{v} {k}' if k in ['gnn','mlp'] else k for v,k in names]
weeks=json.loads((OUT/'design.json').read_text())['evaluation_weeks']
survival=[];ratio=[]
for v,k in names:
    selected=[r for r in records if r['version']==v and r['policy']==k]
    survival.append(sum(r['complete'] for r in selected))
    # Only all-week methods are assigned an average ratio to the fixed rule.
    vals=[]
    for w in weeks:
        a=next(r for r in selected if r['scenario']==w)
        b=next(r for r in records if r['version']=='v0' and r['policy']=='ALWAYS_RESTORE' and r['scenario']==w)
        if a['complete'] and b['complete']:vals.append(a['raw_cost']/b['raw_cost'])
    ratio.append(float(np.mean(vals)) if len(vals)==len(weeks) else np.nan)
fig,axes=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[1,2]},layout='constrained')
y=np.arange(len(names));colors=['#777777','#247a49','#b7832e']+['#2764a5','#55a6bd']*3
axes[0].barh(y,survival,color=colors);axes[0].set_yticks(y,labels);axes[0].invert_yaxis()
axes[0].set_xlim(0,4.5);axes[0].set_xticks(range(5));axes[0].set_xlabel('Completed validation weeks / 4')
axes[1].barh(y,ratio,color=colors);axes[1].invert_yaxis();axes[1].set_yticks(y,labels)
axes[1].set_xlim(0,float(np.nanmax(ratio))*1.18)
axes[1].axvline(1,color='#333333',ls='--',lw=1);axes[1].set_xlabel('Mean per-week cost / ALWAYS_RESTORE cost (lower is better)')
for i,(s,r) in enumerate(zip(survival,ratio)):
    axes[0].text(s+.05,i,str(s),va='center',fontsize=9)
    if np.isfinite(r):axes[1].text(r+.02,i,f'{r:.3f}',va='center',fontsize=9)
    else:axes[1].text(.05,i,'Incomplete: no all-week cost rank',va='center',fontsize=9)
fig.suptitle('GRID08: development validation, one training seed\nAction-module gains must be separated from learned-policy gains',fontsize=12)
fig.savefig(OUT/'GRID08_comparison.png',dpi=160);plt.close(fig)
print(OUT/'GRID08_comparison.png')
