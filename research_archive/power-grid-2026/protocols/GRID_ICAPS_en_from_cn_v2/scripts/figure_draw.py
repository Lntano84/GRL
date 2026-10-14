from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path('D:/Codex/GRL研究/outputs/grid_icaps_en_from_cn_v2/figures')
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','pdf.fonttype':42,'font.size':9})
INK='#243849'; BLUE='#3F6C8B'; GREEN='#52796F'; AMBER='#AC7C3C'
def save(fig,name):
    for ext in ('svg','pdf','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
def framework():
    fig,ax=plt.subplots(figsize=(7.2,2.53));fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,100),ylim=(0,36));ax.axis('off')
    def box(x,y,w,h,title,subtitle,color,fill):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1,rounding_size=0.7',lw=.9,ec=color,fc=fill))
        ax.text(x+w/2,y+h*.65,title,ha='center',va='center',color=INK,fontsize=9.1,weight='bold')
        ax.text(x+w/2,y+h*.29,subtitle,ha='center',va='center',color=INK,fontsize=8)
    def arrow(a,b,dash=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.9,color=INK,linestyle='--' if dash else '-'))
    box(1,23,19,10,'Current state','Base action + legality',INK,'#F4F5F6')
    box(24,23,21,10,'Representative pool','Retain original aliases',INK,'#F4F5F6')
    box(49,23,23,10,'Prefix evaluation','First 128 candidates',BLUE,'#EDF3F7')
    box(80,23,19,10,'Topology proposal','Base control + recovery',INK,'#F4F5F6')
    box(49,4,23,10,'Progressive evaluation','Batches of 64',BLUE,'#EDF3F7')
    box(79,4,20,10,'Substation completion','Remaining local candidates',AMBER,'#F8F1E7')
    arrow((20,28),(24,28));arrow((45,28),(49,28));arrow((72,28),(80,28))
    ax.text(76,33.2,'Direct return',ha='center',fontsize=7.7,color=INK)
    arrow((60.5,23),(60.5,14));ax.text(61.7,18.1,'Continue',fontsize=8,color=BLUE)
    arrow((72,9),(79,9));ax.text(75.5,15,'Stop',ha='center',fontsize=8,color=INK)
    arrow((89,14),(89,23))
    arrow((49,7),(46,7),True);arrow((46,7),(46,12),True);arrow((46,12),(49,12),True)
    ax.text(26,9,'Query feedback\nUpdate best; next batch',fontsize=8,color=BLUE)
    ax.text(1,1.1,'Global discovery → local comparison; reuse query feedback and retain the best admissible action.',fontsize=8.1,color=INK)
    save(fig,'framework_en')
def results():
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.35));fig.subplots_adjust(left=.08,right=.98,bottom=.24,top=.8,wspace=.47)
    values=[[69.37,83.54],[13.11,6.09],[.372,.014]]
    titles=['(a) Candidate queries ↓','(b) Controller time ↓','(c) Largest cost increase']
    labels=['Development','Confirmation']
    for ax,vs,title in zip(axs,values,titles):
        bars=ax.bar([0,1],vs,color=[BLUE,GREEN],width=.58)
        ax.set_xticks([0,1],['Dev.','Confirm.']);ax.set_title(title,fontsize=9.3,weight='bold',pad=10)
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',color='#E7EBEF',lw=.6)
        ax.set_ylabel('%',labelpad=1);ax.set_ylim(0,max(vs)*1.26)
        for b,v in zip(bars,vs):ax.text(b.get_x()+b.get_width()/2,v+max(vs)*.035,(f'{v:.3f}%' if title.startswith('(c)') else f'{v:.2f}%'),ha='center',fontsize=8.7,color=INK)
    fig.text(.5,.025,'FRAME relative to FULL; confirmation queries use the two search-active dates.',ha='center',fontsize=8.2,color=INK)
    save(fig,'results_en')
def case():
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw={'width_ratios':[1.25,1]});fig.subplots_adjust(left=.06,right=.98,bottom=.19,top=.82,wspace=.38)
    ax=axs[0];ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
    ax.set_title('(a) Local completion recovers an overlooked action',fontsize=9.1,weight='bold',loc='left')
    names=['NO_LOCAL','FRAME','FULL'];qs=[256,454,909];actions=[26,309,309]
    for y,name,q,a in zip([3.15,1.85,.55],names,qs,actions):
        ax.text(0,y,name,va='center',fontsize=9,weight='bold',color=INK)
        col=AMBER if name=='NO_LOCAL' else BLUE if name=='FRAME' else '#97A2AC'
        ax.barh(y,q/100,height=.48,left=3,color=col)
        ax.text(3+q/100+.1,y,str(q),va='center',fontsize=8.4,color=INK)
        ax.text(3,y-.45,f'Selected action {a}',fontsize=8.1,color=INK)
    ax.text(3,3.75,'Predictive simulation queries',fontsize=8.3,color=INK)
    ax=axs[1];ax.set_title('(b) Full-week cost increase',fontsize=9.1,weight='bold')
    ax.bar([0,1],[4.810,.372],color=[AMBER,BLUE],width=.58)
    ax.set_xticks([0,1],['NO_LOCAL','FRAME']);ax.set_ylabel('% relative to FULL');ax.set_ylim(0,6)
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',color='#E7EBEF',lw=.6)
    for x,v in enumerate([4.810,.372]):ax.text(x,v+.2,f'{v:.3f}%',ha='center',fontsize=9,color=INK)
    fig.text(.5,.035,'March development case: step 1733 (left); closed-loop full-week costs (right).',ha='center',fontsize=8.2,color=INK)
    save(fig,'completion_case_en')
if __name__=='__main__':
    import sys
    framework()
    if '--framework-only' not in sys.argv:results();case()
