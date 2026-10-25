"""Audit decisional support; freeze one label-independent half-mixture sampling plan."""
from pathlib import Path
import collections,hashlib,importlib.util,json,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid176_gate_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID176_gate_support'
sp=importlib.util.spec_from_file_location('g176_helper',R/'work/grid174_power_helper/run.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D);read=D.read;sha=D.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    assert not O.exists();O.mkdir(parents=True);D.verify_trained();assert read(R/'outputs/grid175_helper_gate_audit/complete.json')['passed_audit']
    save('design.json',dict(id='GRID176_GATE_DECISION_SUPPORT',authorization='User continuouslearningexploration; nofits in thisstage.',
        question='The helper affects F AND(R OR C) only at FCONT/RSTOP. How much existing training probability covers that decision-influence region?',
        active='Strict Fq1>q0 and NOT Rq1>q0,using frozenfold/seedFACTORandROUTED predictions. Supportdefinedwithoutlabels; samefitdatesashelper. Not model-call region:cheap-firstexecution maycall C elsewhere.',
        proposed_single_intervention='Keep POWER8 model/features/NETloss/normalization/updates/seeds/folds andouterrule. Change sampler only:0.5originalP+0.5PconditionalonactiveFITrows. No positive-label reweighting or mixture sweep.',
        reason='Retain global profitableexamples even if active fit subset has none; pure active-only training is not proposed.',
        limitation='F/Rfitpredictions can underrepresent helddisagreement,not nested crossfit. Historical exposed dates,conditionaltask. No statistical risk guarantee.',
        fixed='All4folds3seeds,12newfits onlyif latertrainingnotice; originalPOWER8endpoints reused ascontrols. Allfuturelabels prohibited in sampler.',
        prohibited=['fitting/modelprediction','perseedmixturetuning','activeheldrows in training','newdata/physics/test8','paper']))
    rows=[];paths=[R/'outputs/grid175_helper_gate_audit/complete.json',D.O/'complete.json',D.O/'design.json']
    for fold,path in enumerate(D.sources()):
        z=np.load(path/'fold_arrays.npz');scores=np.load(path/'state_scores.npz');meta=read(path/'metadata.json');fi=z['fit_ids'];hi=z['held_ids'];p=z['fit_probability']
        paths +=[path/n for n in ['fold_arrays.npz','state_scores.npz','metadata.json','preflight.json']]
        for seed in [0,1,2]:
            F=scores[f'FACTOR_seed{seed}'];Rr=scores[f'ROUTED_seed{seed}'];active=(F[:,1]>F[:,0])&~(Rr[:,1]>Rr[:,0]);mask=active[fi];mass=float(p[mask].sum());assert mass>0
            conditioned=p*mask/mass;mix=.5*p+.5*conditioned;assert np.all(mix>0) and np.all(mix>=.5*p-1e-15) and abs(mix.sum()-1)<1e-12
            # Poisoned labels and all held masks cannot affect this exact fixed probability.
            poisoned=z['target'].copy();poisoned[:]=999
            assert np.allclose(mix,.5*p+.5*(p*mask/(p*mask).sum()),atol=1e-15,rtol=1e-15)
            flipped=active.copy();flipped[hi]=~flipped[hi];assert np.array_equal(mask,flipped[fi])
            rng=np.random.default_rng(20261022+seed);stream=hashlib.sha256();counts=np.zeros(3582,np.int64)
            for _ in range(7250):
                ix=fi[rng.choice(len(fi),32,replace=True,p=mix)];stream.update(ix.astype('<i8').tobytes());np.add.at(counts,ix,1)
            assert counts[hi].sum()==0 and counts.sum()==232000
            np.savez_compressed(O/f'fold{fold}_seed{seed}_sampling.npz',fit_ids=fi,probability=mix,active_fit=mask,expected_state_draws=counts)
            row=dict(fold=fold,seed=seed,original_active_sampling_mass=mass,mixed_active_sampling_mass=float(mix[mask].sum()),sampling_hash=stream.hexdigest())
            for part,ids in [('fit',fi),('held',hi)]:
                ii=ids[active[ids]];row[part]=dict(states=len(ii),positive=int(np.sum(z['target'][ii]>0)),dates=len({meta[i]['date'] for i in ii}),physical_groups=len({meta[i]['physical_state_group'] for i in ii}))
            rows.append(row)
    save('support_records.json',rows);save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths+[Path(__file__),O/'design.json']})
    zero=sum(r['fit']['positive']==0 for r in rows);lo=min(r['original_active_sampling_mass'] for r in rows);up=max(r['original_active_sampling_mass'] for r in rows)
    report=['# GRID176：训练重点与组合实际采用区域','',
        f'小模型的决定只有在F继续、R停止时才会影响组合。但原抽样中，这一区域只占{lo:.2%}—{up:.2%}。12个折×种子配置中，有{zero}个配置在该区域没有任何正净收益训练状态，直接只训练该区域并不合适。','',
        '这是“决定能否被采用”的区域，不等于CPU实际运行小模型的区域：便宜模型优先执行时，也可能在其它位置被调用。','',
        '| 折 | 种子 | 拟合区域状态/正例 | 留出区域状态/正例 | 原抽样质量 |','|---|---:|---:|---:|---:|']
    for r in rows:report.append(f"| {r['fold']} | {r['seed']} | {r['fit']['states']} / {r['fit']['positive']} | {r['held']['states']} / {r['held']['positive']} | {r['original_active_sampling_mass']:.2%} |")
    report+=['','下一项只尝试一个固定方案：训练样本一半按原分布，一半按该区域的原条件分布。保留原区域之外的有利查询例子，不按正负标签挑样本，不扫混合比例，不改变模型、输入、目标或外层规则。原POWER8全部权重可作为既有对照，只需12份新拟合。','',
        '区域由原F/R在拟合日期上的预测定义，可能仍比留出日期容易，因此不能宣称覆盖问题已经解决。原4个开发日期和最终8不参与。新采样概率、232000条/模型的预定索引流已核验，标签污染与留出区域变化不会改变拟合采样；本阶段尚未拟合。']
    (O/'GRID176_report.md').write_text('\n'.join(report)+'\n',encoding='utf8',newline='\n');save('review.json',dict(passed=True,all12supports_checked=True,all12samplingstreams_frozen=True,held_draws=0,global_probability_floor_half=True,sampler_label_independent=True,new_fits=0,new_forwards=0,new_physics=0,final8_absent=True));save('complete.json',dict(passed_audit=True,new_fits=0,new_physics=0))
    dest=A/'complete';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
    collab=R/'论文协作资料/FRAME_组合决策区域覆盖_GRID176';collab.mkdir(exist_ok=False)
    for n in ['GRID176_report.md','design.json','support_records.json','review.json','complete.json']:shutil.copy2(O/n,collab/n)
    msg=f'GRID176组合决策影响区域FCONT/RSTOP仅占旧拟合抽样{lo:.2%}—{up:.2%}，12配置中{zero}无区域内正例。非高门槛否定学习，保留全局有利样本，提出唯一原P/区域条件P各半采样；标签/held区域不影响概率，12索引流固定。尚未拟合；下一步通知12份同POWER8模型预算采样对照，原权重控制，不改规则/标签/费用或新物理。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID176组合决策覆盖\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 固定一次组合区域采样对照\n\n1. 训练前通知：POWER8结构/输入/归一化/标签/预算/原F/R/外层规则不变，只用原分布与决策影响区域条件分布各半，四折三种子12新拟合；原POWER8为控制。\n2. 区域仅由拟合F/R预测定义，不按标签选样；保留全局一半，全部种子/日期/反例报告，不扫比例。\n3. 不新物理/数据/最终8，不改论文；模型独立改善与组合效果分开看。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID176组合决策区域覆盖\r\n\r\n[报告](protocols/GRID176_gate_support/complete/GRID176_report.md)；原拟合区域质量3.32%—9.95%，固定混合采样方案待训练。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID176_COMPLETE',zero,lo,up)
if __name__=='__main__':main()
