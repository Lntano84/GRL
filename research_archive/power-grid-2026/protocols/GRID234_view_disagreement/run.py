"""Read-only two-view changes and the support of a potential learned selector."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,hashlib,json,shutil
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid234_view_disagreement';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID234_view_disagreement'
S=R/'outputs/grid233_public_correction';B=R/'outputs/grid230_training_budget';Q=R/'outputs/grid224_context_support_plan/revision02';P=R/'outputs/grid220_full_prior';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
def main():
    assert not O.exists() and read(S/'complete.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID234',scope='Read-only all182sealedchanges,14profitablechanges,andowntrainingtwo-view disagreements. No model/scalar fits,newNNforward/newpolicy.',
        mechanism='Exact symmetric score delta:mean(alpha)*(d_public-d_full)+mean(direction)*(alpha_public-alpha_full). Description only,notnewhybridpolicyperformance. Include8recoveries6losses,Jan5fourrelatedstates,Jan19/Jun1andlostrepeat savings.',
        selector_support='For originalSTOP andauxCONT separately count calibratedandrawendpoint disagreements andpositive labels. Bothparents enforceallfitSTOPpositivesKEEP,so calibratedSTOPdisagreement positive support must be zero. Do not fit a selector or oracle fromheldlabels.',
        next='Qualify data ancestry and one finite residual-layer cross-fit protocol before claiming reliable selector labels. Existinghead-only cross-generation doesnotexcludeallancestors;do notcall fullynested orblind.',
        boundaries='All3seeds16oldexposeddates,allcounterexamples. No heldselectedthreshold/alpha/penalty/seed orcall exception. No final8/physics/paper/FRAME.'))
    paths=[W/'run.py',O/'design.json',META,S/'complete.json',S/'independent_audit.json',S/'changed_choices.json',S/'gate_predictions.json',S/'fit_receipts.json',B/'calibration_receipts.json',S/'paired_summary.json']
    for root,folder in [(Q,'qualified_inputs'),(P,'public_held_inputs'),(B,'models'),(S,'models')]:paths+=list((root/folder).glob('*.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();meta=read(META);lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};changes=read(S/'changed_choices.json');train=[];rows=[]
    for f in range(4):
        for seed in range(3):
            pool=f'f{f}_s{seed}';z=np.load(Q/'qualified_inputs'/f'{pool}.npz');h=np.load(P/'public_held_inputs'/f'{pool}.npz');idx={int(i):k for k,i in enumerate(h['ids'])};cp=np.load(B/'models'/f'{pool}.npz');t0=cp['theta'];a0=float(cp['alpha'][0]);cp=np.load(S/'models'/f'{pool}.npz');t1=cp['theta'];a1=float(cp['alpha'][0]);x=z['X'];b=z['b'];d0=x@t0[:-1]+t0[-1];d1=x[:,83:125]@t1[:-1]+t1[-1];old=b+a0*d0>=0;new=b+a1*d1>=0;raw0=b+d0>=0;raw1=b+d1>=0;stop=z['domain_STOP'];positive=z['y']>0
            assert np.all(old[stop&positive]) and np.all(new[stop&positive]);r=dict(pool=pool,STOP_positive=int((stop&positive).sum()),domains={})
            for name,mask in [('STOP',stop),('CONT',~stop)]:
                diff=(old!=new)&mask;rawdiff=(raw0!=raw1)&mask;r['domains'][name]=dict(rows=int(mask.sum()),disagreements=int(diff.sum()),profitable_disagreements=int((diff&positive).sum()),nonprofitable_disagreements=int((diff&~positive).sum()),risk_mass=float(z['weight'][diff].sum()/z['weight'].sum()),endpoint_disagreements=int(rawdiff.sum()),endpoint_profitable_disagreements=int((rawdiff&positive).sum()))
            train.append(r)
            for c in changes:
                if c['fold']!=f or c['seed']!=seed:continue
                k=idx[lookup[c['sample_id'],c['position']]];hx=h['X'][k];hb=float(h['b'][k]);hd0=float(hx@t0[:-1]+t0[-1]);hd1=float(hx[83:125]@t1[:-1]+t1[-1]);s0=hb+a0*hd0;s1=hb+a1*hd1;direction=.5*(a0+a1)*(hd1-hd0);step=.5*(hd0+hd1)*(a1-a0);assert abs(direction+step-(s1-s0))<1e-9 and (s0>=0)==c['parent_append'] and (s1>=0)==c['public_append']
                rows.append(dict(**c,prior=hb,full_direction=hd0,public_direction=hd1,full_alpha=a0,public_alpha=a1,full_score=s0,public_score=s1,symmetric_direction_contribution=direction,symmetric_step_contribution=step,larger_component='direction' if abs(direction)>abs(step) else 'step',descriptive_only=True))
    assert len(rows)==182;prof=[r for r in rows if r['actual_next_NET']>0];recovered=[r for r in prof if r['public_append']];lost=[r for r in prof if not r['public_append']];assert (len(recovered),len(lost))==(8,6)
    groups={}
    for name,rr in [('all_changes',rows),('profitable_recoveries',recovered),('profitable_losses',lost),('Jan5_losses',[r for r in lost if r['physical_state_group']=='2037-01-05_101|call001'])]:
        groups[name]=dict(rows=len(rr),dominant_components=dict(collections.Counter(r['larger_component'] for r in rr)),direction_improved=sum(r['public_direction']>r['full_direction'] for r in rr),alpha_increased=sum(r['public_alpha']>r['full_alpha'] for r in rr))
    stopdiff=sum(r['domains']['STOP']['disagreements'] for r in train);stoppos=sum(r['domains']['STOP']['profitable_disagreements'] for r in train);contdiff=sum(r['domains']['CONT']['disagreements'] for r in train);contpos=sum(r['domains']['CONT']['profitable_disagreements'] for r in train);assert (stopdiff,stoppos,contdiff,contpos)==(871,0,411,74)
    summary=dict(changes=182,profitable_recoveries=8,profitable_losses=6,groups=groups,STOP_disagreement_rows=stopdiff,STOP_positive_disagreements=stoppos,CONT_disagreement_rows=contdiff,CONT_positive_disagreements=contpos,training_selector_support_is_endogenous=True,no_new_fits=True,no_new_policy_evaluation=True)
    for name,v in [('score_decomposition.json',rows),('profitable_decomposition.json',prof),('training_disagreements.json',train),('summary.json',summary)]:save(name,v)
    verify();save('review.json',dict(passed=True,all182savedchanges_reconstructed=True,all14profitablechanges_retained=True,all12owntrainingpools_checked=True,zeroSTOPpositive_disagreement_follows_both_constraints=True,new_fits=0,new_scalars=0,new_policy_evaluations=0));save('complete.json',dict(passed=True,read_only=True,changes=182,new_fits=0,new_scalars=0,new_physics=0,final8_absent=True))
    lines=['# GRID234：两种修正器的分歧，能否直接交给一个模型学习？','','本轮只读拆解233相对230的全部182处变化及各自训练支持，没有训练新模型或试组合策略。','',
    '## 成功与失败仍涉及修正强度','',
    '1月5日误停的4条相关状态上，简化修正器的原始方向都比完整修正器更温和，但采用的联合步长从约0.192升至0.526，最后分数反而更低。即使训练取消查询代价有上界，它也不能保证每个新状态上的修正幅度受到同样限制。',
    '1月19日种子2的3条相关回本恢复则主要伴随方向改善，两个步长接近；6月1日恢复时，简化修正器的方向转为正向。不能把这两类改善都解释成单纯多查，也不能由此声称模型整体学得更好。全部8恢复和6误停的分数分解已经保留。','',
    '## 直接学“该信哪一个”有训练答案缺口','',
    '两个已校准模型在原STOP训练记录上共有871条分歧，其中回本正例为0。这不是模型已经解决了所有回本判断：两边的经验保护规则本来就要求训练正例继续，所以它们在这些正例上不可能产生分歧。',
    '若直接拿这些记录训练“分歧时信哪边”的选择器，看到的原STOP答案会全部偏向取消补查；它没有学到外层那些“应保留回本机会”的分歧。不能用已经看过的外层失败记录补进去，再称为独立训练验证。',
    '辅助CONT记录中有411条分歧、74条正例，但它们来自停止前的状态，不能自动替代实际STOP分歧的正例。未校准终点在STOP上也存在正例分歧，但仍是参与拟合的记录，不能直接当成可靠的未拟合误差标签。','',
    '## 下一步','',
    '先核查训练、校准和验证的数据祖先，固定一个有限的修正层交叉拟合方案，让学习器能看到未参与该修正器拟合的训练记录上的错误。原先只有收益头的交叉生成，不能说成整条控制链都已隔离。',
    '这一步保留完整修正器的省查询能力与简化修正器在1月19日、6月1日的恢复线索；不直接上二级选择器，也不按外层结果选阈值、种子或组合方式。下一轮先做资格，不扩数据或网络。',
    '当前均为暴露开发数据的机制诊断，分数代数分解不是组合策略的因果效果。局部代理收益不等于整周质量或实际时延，最终8与物理仿真不动。']
    (O/'GRID234_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');dest=R/'论文协作资料/FRAME_双修正器分歧_GRID234';dest.mkdir(exist_ok=False)
    for name in ['GRID234_report.md','summary.json','training_disagreements.json','profitable_decomposition.json','review.json','complete.json']:shutil.copy2(O/name,dest/name)
    msg='GRID234只读182变化/14回本变化完成。Jan5四相关误停方向更温和但alpha .192→.526放大；Jan19三恢复主要方向改善，Jun1方向转正。更关键：两校准器原STOP训练分歧871条/正例0，由双方正例保护约束内生造成；辅助CONT分歧411/正例74不能冒充STOP支持。直接训练选择器会缺少回本分歧答案。下一步235仅数据祖先/有限修正层交叉拟合资格，不用已看外层补标签，不直上二级gate；0拟合/0新策略。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID234双修正器分歧与答案缺口\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID234后\n\n1. 234只读完成，保留230主候选/233局部恢复，不重复旧拟合或评分。\n2. 235仅资格核查数据祖先：原STOP训练两视图分歧871/正例0是保护约束造成，不能直接据此训练选择器；辅助411/74与未校准拟合分歧也不是独立STOP误差标签。\n3. 固定一个有限修正层交叉拟合方案前，追踪214收益头、原208/NET及STOP选择器、标准化、辅助先验、预算父模型、校准记录的标签祖先。明确只能隔离哪一层，不称全链嵌套。优先评估已有固定inner8/4日期方案的支持，不按外层择fold/阈值/组合。235拟合0。\n4. 后续学习阶段最多22拟合/校准、600秒，源码先发、模型先发、预测先发后连标签，Git严格等helper退出后再下一操作。最终8/物理/电网/论文/FRAME不动，原费用/标签/控制器/排序别名/锁站/128/64/停滞2/.9保留。自主推进直到喊停。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID234双修正器分歧\n\n[报告](protocols/'+A.name+'/GRID234_report.md)；发现拟合分歧的正例答案缺口，下一步先做交叉拟合资格。\n\n').encode('utf8')+p.read_bytes());print('GRID234_COMPLETE',json.dumps(summary))
if __name__=='__main__':main()
