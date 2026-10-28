"""Audit conditional hold-group error collection without fitting calibration."""
from pathlib import Path
import json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
def main():
    C.scored();assert C.read(O/'review.json')['passed'] and not (O/'complete.json').exists();meta=C.read(C.META);receipts=C.read(O/'fit_receipts.json');traincounts=[];rows=0
    for r in receipts:
        pool=r['pool'];z=np.load(C.Q/'fit_inputs'/f'{pool}.npz');p=np.load(O/'models'/f'{pool}.npz');theta=p['theta'];fallback=bool(p['fallback_KEEP'][0]);v,g=C.objective(theta,z['X'],z['y'],z['weight'],z['b']);assert theta.shape==(43,)
        if not fallback:assert r['solver_success'] and abs(g).max()<1e-3
        fitlogit=np.ones(len(z['y'])) if fallback else z['b']+np.sum(z['X']*theta[:-1],axis=1)+theta[-1];mask=z['domain_STOP'];traincounts.append(dict(pool=pool,STOP=C.error_counts(z['ids'],fitlogit>=0,z['actual_NET'],mask,meta),CONT=C.error_counts(z['ids'],fitlogit>=0,z['actual_NET'],~mask,meta)))
        h=np.load(C.Q/'cal_public'/f'{pool}.npz');pred=np.load(O/'cal_predictions'/f'{pool}.npz');expected=np.ones(len(h['ids'])) if fallback else h['b']+np.sum(h['X']*theta[:-1],axis=1)+theta[-1];assert np.array_equal(expected,pred['logit']) and np.array_equal(expected>=0,pred['keep']) and np.array_equal(h['ids'],pred['ids']);rows+=len(expected)
    assert rows==3176 and sum(r['fitted'] for r in receipts)==11
    for phase,name,pubfile in [('trained','trained_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
        pub=C.read(O/pubfile);assert C.git('show',pub['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    assert all('/fit_inputs/' in r['path'] and not any(k in r['fields'] for k in ['cal4_y']) for r in C.read(O/'train_access.json'));assert len(C.read(O/'cal4_target_access.json'))==12 and all('_i0_' in r['path'] for r in C.read(O/'cal4_target_access.json'))
    errors=C.read(O/'cal4_error_rows.json');summary=C.read(O/'summary.json');miss=[r for r in errors if r['original_STOP'] and r['positive'] and not r['keep']];assert len(miss)==9 and len({r['call'] for r in miss})==5
    fit_summary={domain:{k:sum(r[domain][k] for r in traincounts) for k in ['rows','positive','positive_kept','positive_cancelled','nonpositive_cancelled','nonpositive_kept']} for domain in ['STOP','CONT']};C.save('fit8_error_counts.json',traincounts);C.save('fit8_error_summary.json',fit_summary);C.save('cal4_STOP_positive_errors.json',miss)
    seed_counts=[]
    for seed in range(3):
        rr=[r for r in errors if r['pool'].endswith(f'_s{seed}') and r['original_STOP']];seed_counts.append(dict(seed=seed,STOP_rows=len(rr),positive=sum(r['positive'] for r in rr),positive_cancelled=sum(r['positive'] and not r['keep'] for r in rr)))
    C.save('seed_summary.json',seed_counts);C.save('decision.json',dict(conditional_residual_error_collection_completed=True,no_claim_of_better_policy=True,no_new_calibration_fitted=True,cal4_STOP_error_support=dict(rows=9,unique_states=len({r['index'] for r in miss}),calls=5),next='GRID238 readonly assess fit8/cal4positiveconstraint support and error ancestry. Qualify at most one explicitlybounded paired-calibration study before anynewscalar;no newouter4scoresorpolicy inthiscompletedprobe.'))
    lines=['# GRID237：修正层留组错误已经收集','','本轮按既定方案，用8天记录拟合11个43参数线性修正器，另1个池保持常量继续。训练约0.045秒，全部数值资格通过；没有拟合步长、查询预算或二级选择器。','',
    '## 4天检查集上能看到哪些错误','',
    '4天公共输入共3176条，其中原STOP记录1644条、回本49条。未校准修正器保留40条回本记录，错停9条；同时取消710条不回本记录，另有885条不回本记录保持继续。9条回本误停涉及5个调用、8个不同状态，包含相关重复，不能当作9次独立失败。',
    '辅助CONT部分有1532条记录、702条回本，其中140条被未校准分数取消。这些停止前状态单独列出，不与真实STOP错误混成同一任务。','',
    '三个种子在原STOP上的回本误停记录分别为6、1、2条。没有挑出表现最好的种子；每条错误的分数、日期、调用和原始收益标签均已保存。','',
    '## 这次数据能用于什么','',
    '现在能具体查看：一个没有直接拟合这4天的修正器，会错停哪些回本机会。这提供了校准层可研究的错误记录，避免只在已经被保护规则强制保留的拟合正例上判断校准是否有效。',
    f"相同未校准规则在8天拟合STOP记录上，也错停{fit_summary['STOP']['positive_cancelled']}/{fit_summary['STOP']['positive']}条回本。不能把新4天的9条错误与旧871处分歧的0正例直接比较：人口、分数及保护条件均不同。旧0正例是保护约束造成，新结果也不是比旧方法更好或更差的证据。",'',
    '## 隔离与范围','',
    '所有公共先验先发布，再只连接8天拟合标签；本轮训练worker只读取fit_inputs。12个权重先发布、12个4天分数再发布，随后才在本轮诊断中解析inner0的4天目标。检查了全部3176分数、拟合目标梯度、访问字段与发布收据。',
    '基模型和原STOP总体仍在12天上学习，因此只能称固定上游条件下的修正层留组诊断；相关开发标签在历史分析中已暴露，不是盲测，也不是整条流程独立验证。',
    '没有做新的外层4天策略比较、整周仿真或时延测试；这些分类错误不能转换成整周成本结论。原230仍为主要内部候选，233的方向改善线索保留。',
    '下一步先核查8天与4天的正例约束支持，再资格固定一个有限的校准对照。本轮到错误收集为止，未根据这9条记录调整阈值或拟合新步长。']
    (O/'GRID237_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');C.scored();C.save('independent_audit.json',dict(passed=True,all11active43gradient_andsolver_qualifications_verified=True,all3176sealed_scores_reproduced=True,onlyfit8targets_used_for_residual_training=True,all12cal4target_sources_rolechecked=True,publicationbarriers_verified=True,all9STOPpositive_errors_and_seedcounts_preserved=True,conditional_not_fullpipeline_scope=True,new_scalar_fits=0,new_policy_evaluations=0));C.save('complete.json',dict(passed=True,new_linear_fits=11,constantKEEP=1,new_scalar_fits=0,new_NNfits=0,new_physics=0,cal4_predictions=3176,cal4_STOP_positive_errors=9,new_outer_policy_evaluations=0,final8_absent=True));C.archive('audited');dest=R/'论文协作资料/FRAME_修正层留组诊断_GRID237';dest.mkdir(exist_ok=False)
    for n in ['GRID237_report.md','summary.json','fit8_error_summary.json','seed_summary.json','cal4_STOP_positive_errors.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg='GRID236/237限定诊断完成。236同源先验22653行，fit8 3747(STOP2134/49正、CONT1613/561正)，cal4公共3176，24既有头复用，0NN/物理。237固定11个43参模型(.045s)、0标量，权重/分数先发后接cal4。cal4 STOP1644/49正，保留40误停9(5调用8状态)，取消710无利；CONT1532/702正误停140。种子STOP误停6/1/2全留。只条件修正层错误支持，不是全链独立或政策胜出；无新outer策略。下一步238只读正例约束支持及成对校准资格，暂不新拟合alpha。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID236至237条件修正层错误收集\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID237后\n\n1. 236同源再生成、237固定11模型及3176分数/标签错误收集均完成，不重做。原STOP误停9/49，涉及5调用8状态；0标量/新outer策略。\n2. 238仅只读检查fit8/cal4正例约束的分布与来源、9误停及其分组支持，资格明确唯一有限校准对照前不拟合新alpha、不试多个组合。校准数据是自身outer训练内部，标签已暴露；不称盲测或独立全链。\n3. 若提出新成对校准试验，明确它是新协议，不回改237；公开总拟合操作数<=22/阶段及600秒，不悄悄扩大旧探针，无本阶段新outer4策略分数/政策评估。先来源和可识别性资格再执行。\n4. 原230主候选、233局部方向改善保留。费用标签/控制器/排序别名/锁站/128/64/停滞2/.9不变；最终8/物理/电网/论文/FRAME不动。Git严格等helper完成，自主推进直到喊停。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID237修正层留组诊断\n\n[报告](protocols/'+A.name+'/audited/GRID237_report.md)；得到9条条件STOP回本误停记录，尚未校准或新做外层策略比较。\n\n').encode('utf8')+p.read_bytes());print('GRID237_AUDITED',json.dumps(dict(cal=summary,fit_STOP=fit_summary['STOP'],seeds=seed_counts)))
if __name__=='__main__':main()
