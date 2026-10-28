"""Read-only common misses, successful regions, and parent/prior alignment."""
from pathlib import Path
import collections,hashlib,json,shutil
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid240_common_bottleneck';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID240_common_bottleneck'
S=R/'outputs/grid239_paired_calibration';Q=R/'outputs/grid236_shared_prior_inputs';M=R/'outputs/grid237_residual_holdgroup';P=R/'outputs/grid220_full_prior';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    assert not O.exists() and read(S/'complete.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID240',scope='Read-onlyall610originalCONTrecords,7sharedprofitablemisses,twoseedsuccessfulcancellationsand131passthroughs. No newmodels/alphas/thirdarm/forecasts.',
        public='Compareexisting b8andoriginal208b12,rawendpointdirection and publishedFIT8/CAL4alpha. Check ownfit/calpositiveprior ranges andfixedEuclideanpublic41nearestpositive distance;diagnostic notOODclassifier.',
        mechanism='Controllerentryuses208twelve-datepriorbutcorrectionoffsetusesdifferentinner8prior. Quantifyall4681signagreements beforeusingoutcomes;doNOTevaluate substituting b12 intooldmodel.',
        future='Onlyqualify controller/priorsemantic alignment fromowntrainingandexistingpublicdata beforeanotherfinitelearningproposal. No held-informed bthreshold,seedselection,call exception orad hocfallbackextension.',
        limits='Sharedcallprefixesandseeds correlated. Exposedofflineproxy notfullweekoractualtime. Final8/physics/paper/FRAME untouched.'))
    paths=[W/'run.py',O/'design.json',S/'complete.json',S/'independent_audit.json',S/'gate_predictions.json',S/'shared_profitable_misses.json',S/'search_metrics.json',S/'calibration_receipts.json',S/'family_summary.json',S/'effect_decomposition.json',M/'cal4_error_rows.json',META,R/'work/grid239_paired_calibration/run.py',R/'work/grid236_shared_prior_inputs/run.py']
    for root,folder in [(S,'public_outer'),(S,'outer_predictions'),(S,'models'),(Q,'fit_inputs'),(Q,'cal_public'),(P,'public_held_inputs')]:paths+=list((root/folder).glob('*.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)});meta=read(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};pred=[r for r in read(S/'gate_predictions.json') if r['arm']=='CAL4' and r['base_continue']];shared={(r['seed'],r['sample_id']) for r in read(S/'shared_profitable_misses.json')};calrows=read(M/'cal4_error_rows.json');records=[];alignment=[]
    for f in range(4):
        for seed in range(3):
            pool=f'f{f}_s{seed}';h=np.load(S/'public_outer'/f'{pool}.npz');orig=np.load(P/'public_held_inputs'/f'{pool}.npz');fit=np.load(Q/'fit_inputs'/f'{pool}.npz');cal=np.load(Q/'cal_public'/f'{pool}.npz');assert np.array_equal(h['ids'],orig['ids']);posfit=(fit['y']>0)&fit['domain_STOP'];cr=[r for r in calrows if r['pool']==pool];byid={r['index']:r for r in cr};poscal=np.array([byid[int(i)]['positive'] for i in cal['ids']])&cal['original_STOP'];idx={int(i):k for k,i in enumerate(h['ids'])};models={a:np.load(S/'models'/f'{pool}_{a}.npz') for a in ['FIT8','CAL4']}
            alignment.append(dict(pool=pool,all_NETSTOP_rows=len(h['ids']),parentCONT_innerCONT=int((h['base_CONT']&h['inner_support']).sum()),parentCONT_innerSTOP=int((h['base_CONT']&~h['inner_support']).sum()),parentSTOP_innerCONT=int((~h['base_CONT']&h['inner_support']).sum()),parentSTOP_innerSTOP=int((~h['base_CONT']&~h['inner_support']).sum())))
            for r in pred:
                if r['fold']!=f or r['seed']!=seed:continue
                k=idx[lookup[r['sample_id'],r['position']]];x=h['X'][k];b8=float(h['b'][k]);b12=float(orig['b'][k]);t=models['CAL4']['theta'];direction=float(x@t[:-1]+t[-1]);scores={}
                for arm,p in models.items():
                    a=float(p['alpha'][0]);score=b8+a*direction;saved=np.load(S/'outer_predictions'/f'{pool}_{arm}.npz');assert abs(score-float(saved['logit'][k]))<1e-9;scores[arm]=dict(alpha=a,score=score,downward_over_prior=float(-a*direction/b8) if b8>0 and direction<0 else None)
                common=(seed,r['sample_id']) in shared;kind='passthrough' if r['passthrough'] else ('kept_' if r['continue_'] else 'cancelled_')+('profitable' if r['actual_next_NET']>0 else 'unprofitable');support={}
                for role,z,pos in [('fit8',fit,posfit),('cal4',cal,poscal)]:
                    if pos.any():
                        distance=np.linalg.norm(z['X'][pos,:41]-x[:41],axis=1);near=int(np.argmin(distance));ids=z['ids'][pos];support[role]=dict(positive_rows=int(pos.sum()),prior_min=float(z['b'][pos].min()),prior_max=float(z['b'][pos].max()),below_positive_prior_range=bool(b8<z['b'][pos].min()),nearest_public41_distance=float(distance[near]),nearest_positive_call=meta[int(ids[near])]['physical_state_group'])
                    else:support[role]=dict(positive_rows=0,prior_min=None,prior_max=None,below_positive_prior_range=None,nearest_public41_distance=None,nearest_positive_call=None)
                records.append(dict(pool=pool,seed=seed,sample_id=r['sample_id'],position=r['position'],call=r['call'],date=r['date'],kind=kind,shared_profitable_miss=common,actual_NET=r['actual_next_NET'],gain=r['actual_next_gain'],queries=r['actual_next_queries'],b8=b8,original_b12=b12,b8_minus_b12=b8-b12,endpoint_direction=direction,raw_endpoint=b8+direction,scores=scores,support=support,outside_fit_public41_column_ranges=int(((x[:41]<fit['X'][:,:41].min(axis=0))|(x[:41]>fit['X'][:,:41].max(axis=0))).sum())))
    assert len(records)==610 and sum(r['shared_profitable_miss'] for r in records)==7 and sum(r['kind']=='passthrough' for r in records)==131;summary_groups={}
    for name,rr in [('shared_miss',[r for r in records if r['shared_profitable_miss']]),('productive_cancellation',[r for r in records if r['kind']=='cancelled_unprofitable']),('kept_profitable',[r for r in records if r['kind']=='kept_profitable']),('passthrough',[r for r in records if r['kind']=='passthrough'])]:
        summary_groups[name]=dict(rows=len(rr),calls=len({r['call'] for r in rr}),b8_smaller_than_parent=sum(r['b8']<r['original_b12'] for r in rr),b8_below_fitpositive_range=sum(r['support']['fit8']['below_positive_prior_range'] is True for r in rr),b8_below_calpositive_range=sum(r['support']['cal4']['below_positive_prior_range'] is True for r in rr),outside_fit_public41_ranges=sum(r['outside_fit_public41_column_ranges']>0 for r in rr),nearest_fitpositive_distance_median=float(np.median([r['support']['fit8']['nearest_public41_distance'] for r in rr if r['support']['fit8']['nearest_public41_distance'] is not None])) if rr else None)
    metrics=read(S/'search_metrics.json');nc={r['date']:r['cases'] for r in metrics['GATE_seed0']['per_date']};decomp=[];calls=collections.defaultdict(float)
    for seed in range(3):
        rr=[r for r in records if r['seed']==seed and r['kind'].startswith('cancelled')];saving=sum(-r['actual_NET']/(16*nc[r['date']]) for r in rr if r['actual_NET']<=0);loss=sum(r['actual_NET']/(16*nc[r['date']]) for r in rr if r['actual_NET']>0);target=metrics[f'CAL4_seed{seed}']['macro']['mean_net_utility']-metrics[f'GATE_seed{seed}']['macro']['mean_net_utility'];assert abs(saving-loss-target)<1e-12;decomp.append(dict(seed=seed,saved_unprofitable=saving,lost_profitable=loss,total=target))
        for r in rr:
            if r['actual_NET']>0:calls[r['call']]+=r['actual_NET']/(3*16*nc[r['date']])
    aligned={k:sum(r[k] for r in alignment) for k in ['all_NETSTOP_rows','parentCONT_innerCONT','parentCONT_innerSTOP','parentSTOP_innerCONT','parentSTOP_innerSTOP']};assert aligned['all_NETSTOP_rows']==4681 and aligned['parentCONT_innerSTOP']==131;losses=sorted(calls.items(),key=lambda p:-p[1]);miss=[r for r in records if r['shared_profitable_miss']];assert sum(r['b8']<r['original_b12'] for r in miss)==5 and sum(r['support']['cal4']['below_positive_prior_range'] for r in miss)==6
    summary=dict(groups=summary_groups,entry_prior_alignment=aligned,loss_by_call=dict(losses),largest_loss_call_share=losses[0][1]/sum(calls.values()),per_seed=decomp,no_new_models_or_scalars=True,no_new_policy_scores=True)
    for n,v in [('cohort_records.json',records),('shared_miss_details.json',miss),('entry_prior_alignment.json',alignment),('summary.json',summary)]:save(n,v)
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    save('review.json',dict(passed=True,all610baseCONTrecords_and7sharedmisses_checked=True,all4681existingNETSTOPprior_signs_counted=True,published_scores_reconstructed_without_new_forecasts=True,saving_loss_identity_checked_allseeds=True,fit_cal_positive_support_kept_separate=True,no_new_arm_oralpha=True));save('complete.json',dict(passed=True,read_only=True,new_fits=0,new_forecasts=0,new_policy_evaluations=0,new_physics=0,final8_absent=True))
    lines=['# GRID240：共同误停与父方法先验的一致性','','本轮只读检查610条原继续记录、两组共同的7条回本误停、两个种子的省查询区域及131条支持域回退，没有新增模型、步长或第三组策略。','',
    '## 共同误停不是同一种现象','',
    '种子2的1月5日3条大收益记录上，8天先验裕度仅约0.037—0.047，而原208的12天先验约0.397—0.531；对应修正方向约为−1。两种已发表步长都会使分数变负，单纯把拟合正例换成留组正例没有补回这类机会。',
    '另外4条共同误停中，2月16日的8天先验也比父方法弱；6月1日和11月16日的8天先验反而更强。因此不能用“8天先验都偏低”解释全部失败。后两条分数接近0，但本轮没有据此加容差或换阈值。',
    f"共同误停中6/7的8天先验低于本池校准正例的最低裕度，却都落在拟合正例的裕度范围内。有效的不回本取消中，也有{summary_groups['productive_cancellation']['b8_below_calpositive_range']}条低于校准正例范围。这只是支持描述，不能直接变成一个域外回退阈值。",'',
    '## 更值得先核对的是先验语义','',
    '外层入口仍由原208的12天先验决定，额外修正器却把另一套8天先验当作偏置和输入。两者不只是置信度数值不同，继续资格也会不同。',
    f"全部4681个既有NET停止点中，父方法继续而8天头停止的有{aligned['parentCONT_innerSTOP']}条，父方法停止而8天头继续的有{aligned['parentSTOP_innerCONT']}条。前者按239共同规则保留父决定，后者没有新增入口。改换先验来源引入的这个不一致，独立于具体收益标签就存在。",
    '这不证明它导致了所有误停，也不意味着只需把旧分数里的b直接替换就会改善。若要保持父方法先验，应在拟合数据、偏置、输入和实际入口上统一来源，再单独资格和训练；不能拿已拟合权重临时换先验后挑结果。','',
    '## 保留有效区域','',
    '种子0、1的省查询区域仍完整保留：CAL4与共同入口的即时增益、动作一致率相同，取消的是不回本追加。种子2的大损失不能掩盖这些线索，也不能靠只保留成功种子处理。',
    f"当前回本损失中，1月5日同调用3条相关状态占日期宏平均损失的{summary['largest_loss_call_share']:.2%}。这是一个需要理解的高价值区域，不是3个独立失败，也不能据此设置调用例外。",'',
    '下一步仅资格核对“保留原208先验再学修正”的条件方案：追踪既有208头及标准化，明确如何在所有训练/检查/部署记录上使用同一父先验，列出改变的资格和祖先关系，再固定一个有限学习对照。先不生成新分数或修改当前239。',
    '当前均为已暴露离线开发数据。最近邻距离和范围检查不证明域外或因果，代理净收益不等整周成本或实际时延。最终8、物理仿真、论文和FRAME比较均未变化。']
    (O/'GRID240_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');dest=R/'论文协作资料/FRAME_共同误停与先验_GRID240';dest.mkdir(exist_ok=False)
    for n in ['GRID240_report.md','summary.json','shared_miss_details.json','entry_prior_alignment.json','review.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg=f"GRID240只读610/7共漏和4681入口先验分歧完成。共漏5/7 b8低于原b12、6/7低于cal正例最小裕度但全在fit正例范围；非普适OOD解释。Jan5三相关大损失b8 .037-.047而b12 .397-.531，损失占{summary['largest_loss_call_share']:.2%}；另Jun/Nov b8更强。原208入口与inner8偏置语义不同，131父继续/inner停、{aligned['parentSTOP_innerCONT']}父停/inner继续。保留种子0/1省查询区域，不按外层设阈值/调用例外；下一步241仅资格同源父先验方案，0新拟合/分数/规则。"
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID240共同误停与先验语义\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID240后\n\n1. 240只读完成，不重做239。保留种子0/1省查询区域，种子2共享7误停不能统一归b低或域外。\n2. 241仅资格父先验一致方案：确认208固定event/amount头、124维标准化和原CONT入口，评估如何训练/检查/部署都沿用同一12天父先验，明确样本资格/标签祖先及标准化变更，不立即替换现有模型b或生成新策略分数。\n3. 依据自身训练标签/既有公开支持固定唯一有限新学习对照后再执行；不按外层给Jan5阈值/例外或择种子，不追加239第三臂。每阶段<=22拟合/校准、600秒，源码/模型/预测先发后连接结果。\n4. 原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保留，最终8/物理/电网/论文/FRAME不动，开发代理不等整周质量/时延。Git严格等helper完成，自主推进直到喊停。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID240共同误停与先验语义\n\n[报告](protocols/'+A.name+'/GRID240_report.md)；只读定位入口与修正先验来源不一致，尚未修改模型。\n\n').encode('utf8')+p.read_bytes());print('GRID240_COMPLETE',json.dumps(summary))
if __name__=='__main__':main()
