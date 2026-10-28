"""Independently check saved losses and public-call neighbors, then report."""
from pathlib import Path
import collections,json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO

def main():
    C.verify();assert not (O/'complete.json').exists();rows=C.read(O/'row_support.json');meta=C.read(C.META);lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};original={(r['seed'],r['sample_id']):r for r in C.read(C.S/'gate_predictions.json') if r['arm']=='ANCHORED'};group=collections.defaultdict(list)
    for r in rows:group[r['pool']].append(r)
    checks=0
    for pool,rr in group.items():
        f,s=[int(v[1:]) for v in pool.split('_')];fit=np.load(C.P/'fit_inputs'/f'{pool}.npz');held=np.load(C.P/'public_held_inputs'/f'{pool}.npz');fi={int(i):k for k,i in enumerate(fit['ids'])};hi={int(i):k for k,i in enumerate(held['ids'])};calls=[meta[int(i)]['physical_state_group'] for i in fit['ids']];positive=fit['actual_NET']>0
        for r in rr:
            source=original[s,r['sample_id']];assert r['index']==lookup[r['sample_id'],r['position']] and r['net']==source['actual_next_NET'];assert source['base_continue'];k=hi[r['index']]
            for metric,sl in [('public41',slice(83,124)),('public124',slice(0,124))]:
                x=held['X'][k,sl];squared=np.array([sum((float(v)-float(w))**2 for v,w in zip(row[sl],x)) for row in fit['X']]);d=np.sqrt(squared);near=r['neighbors'][metric];keys={v['call'] for v in near['nearest3']};assert len(keys)==3;threshold=near['nearest3'][-1]['distance']
                for n in near['nearest3']:
                    j=fi[n['index']];assert abs(d[j]-n['distance'])<1e-10;assert calls[j]==n['call'];assert d[j]<=min(d[t] for t in range(len(d)) if calls[t]==n['call'])+1e-10;assert bool(positive[j])==n['nearest_row_positive'];assert n['any_positive_in_call']==any(positive[t] for t in range(len(d)) if calls[t]==n['call'])
                for j,call in enumerate(calls):
                    if call not in keys:assert d[j]>=threshold-1e-10
                if positive.any():assert abs(min(d[positive])-near['nearest_positive_distance'])<1e-10 and positive[fi[near['nearest_positive_index']]]
                else:assert near['nearest_positive_distance'] is None
                checks+=1
    assert len(rows)==610 and checks==1220;summary=C.read(O/'summary.json');miss=[r for r in rows if r['kind']=='removed_profitable'];assert len(miss)==16 and len({r['index'] for r in miss})==11 and len({r['call'] for r in miss})==7
    metrics=C.read(C.S/'search_metrics.json');ncases={d['date']:d['cases'] for d in metrics['GATE_seed0']['per_date']};saving=loss=0.
    for r in rows:
        if not r['kind'].startswith('removed'):continue
        delta=-r['net']/(3*16*ncases[r['date']]);saving+=max(delta,0);loss+=max(-delta,0)
    assert abs(loss-summary['macro_loss_profitable'])<1e-12 and abs(saving-summary['macro_saving_unprofitable'])<1e-12;limits=C.read(O/'limiting_fit_support.json');assert len(limits)==22 and len({r['index'] for r in limits})==7 and len({r['call'] for r in limits})==6
    masks=[r for r in rows if r['kind']=='removed_unprofitable'];decision=dict(retain_GRID208_and222_tradeoff=True,no_rule_change_from_held_failures=True,major_loss_concentrates_one_call=summary['top_loss_call_share'],all16_loss_rows_are11_states7calls=True,no_simple_distance_OOD_explanation=True,no_single_weak_prior_rescue_rule_justified=True,empirical_step_depends_on6_training_calls=True,next_candidate='Qualify available preSTOPCONT auxiliary support for same public residual learner, as a fixed within-training exposure comparison. Keep own-source date graph and deploy originalSTOPonly;sameclasslabels/fee/meanrisk/fullprior/unitpenalty/fit-positive calibration. No fit until sealed.',motivation='Current correction training conditions on both originalNETSTOP and innerheadCONT;prior206 previouslyused mixedpreSTOP support. Existing214 contains preSTOPCONT outcomes fromsameownouter training dates,notnewphysics. Confirm selectedpublicb>0 rows and mixedlabels/calls first;do not infer effect before fitting.',new_fits=0);C.save('decision.json',decision)
    lines=['# GRID223：保护校准剩余误停与训练支持','','本轮只读GRID222/220保存记录，核查全部610个原准备追加状态及22个步长限制点，没有重新拟合、改步长、评分新模型或重放新策略。','', '## 剩余损失集中在哪','','16条误删回本记录对应11个不同状态、7个物理调用。2037-01-19_101的call005占误删净收益损失58.94%，与第二个调用合计73.81%；该调用在种子1和2均失去回本补查，但同调用多个前缀不独立。','', '|物理调用|记录/状态|种子|宏平均净收益损失贡献|','|---|---:|---|---:|']
    for r in summary['loss_calls']:lines.append(f"|{r['call']}|{r['rows']}/{r['unique_sample_ids']}|{','.join(map(str,r['seeds']))}|{r['loss_contribution']:.9f}|")
    lines+=['',f"取消不回本追加贡献+{saving:.9f}，误删回本贡献-{loss:.9f}，合计{saving-loss:.9f}，与GRID222相对208的日期/种子宏平均差精确一致。这是原指标的分解，没有执行事后补救策略。",'', '## 为什么拟合内保护没有直接迁移','','22个闭式步长由7个训练状态、6个训练调用限制；每个池的最小约束通常只由1个状态决定。少数调用因此决定了全池步长。它们在自身拟合池保护了全部正例，没有覆盖外层所有回本模式。','', '在16条误删中，有9条原判断b低于自身拟合正例的最小值。但111条成功取消的不回本记录中也有63条满足同样条件，两者比例相近。不能据此临时增加“低于正例最小值就保留”的规则，再把同一批外层改善当作确认。','', '误删的16条中，有5条查询量高于本池拟合正例最大值；无利取消中也有24条。弱先验或查询范围能够描述部分样本，不构成足够的触发条件。','', '这些误删点沿既定方向的零分交叉位置，相当于原拟合步长的0.0068—0.9011倍，中位0.3594倍。交叉位置仅用于解释边界；没有据外层标签回调步长。','', '## 公开邻域不支持简单的“更远所以失败”','','固定原208标准化，分别检查原41维公共摘要及124维公共融合输入。对所有610状态报告最近3个不同训练调用，区分最近行本身为正和该调用中其他行有正例，所有邻居严格来自自身外层训练日期。','', '|对象|记录|41维最近正例距离中位数|124维最近正例距离中位数|最近3调用有正例（41/124维）|','|---|---:|---:|---:|---:|']
    for k,label in [('removed_profitable','误删回本'),('kept_profitable','保留回本'),('removed_unprofitable','取消无利'),('kept_unprofitable','保留无利')]:
        z=summary['cohort_support'][k];p,q=z['metrics']['public41'],z['metrics']['public124'];lines.append(f"|{label}|{z['rows']}|{p['nearest_positive_distance_median']:.4f}|{q['nearest_positive_distance_median']:.4f}|{p['any_of3_call_has_positive']}/{q['any_of3_call_has_positive']}|")
    lines+=['', '误删回本到最近训练正例的距离反而小于保留回本的中位距离，不能把这轮错误笼统归为公共空间距离更远。该距离依赖原编码和标准化，只是支持描述，不是不可学习或域外的证明。保留回本中3条来自无正例常量KEEP池，距离统计中单列缺失，没有伪造正邻居。','', '有些调用在别的外层划分属于训练、在当前划分属于评估。不能借用其他外层的标签或模型来保护当前评估点；所有当前邻居和限制点均检查自身日期隔离。','', '## 后续限定方向','','当前修正训练入口同时受原NET停止和内层头继续筛选，正收益支持仍稀少。原208的事件头曾使用停止前CONT支持，而新修正模型只用停止入口。下一步先资格现有GRID214中停止前、且公共原判断b>0的辅助行；所有正负标签都保留，不按收益选样本。','', '候选对照将保持公共输入、费用、标签、平均风险、完整先验、单位惩罚及部署入口不变，只改变拟合支持来源。保护步长仍只由原STOP拟合正例确定。辅助域与部署域不同、总体样本暴露更多必须报告；不能称同样的数据量或已校准的STOP概率。资格完成并封存后才拟合；未证明辅助支持有效，不扩大物理或最终8。','', '当前保留208主参考和222成本质量折中，全部反例保留。没有新物理、最终8、论文或原FRAME对照。']
    (O/'GRID223_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.verify();C.save('independent_audit.json',dict(passed=True,all610_held_rows_source_joined=True,all1220_public_neighbor_sets_and_positive_distances_verified=True,all16_lossrows11states7calls_verified=True,macro_saving_loss_reconciled_with_saved_metrics=True,all22_limitingfit_rows_in_own_fit_date_graph=True,source_freezes_unchanged=True,new_fits=0,new_NN_forwards=0,new_held_scores=0));C.save('complete.json',dict(passed=True,scope='saved_data_remaining_loss_and_support_only',cohort_rows=610,loss_rows=16,loss_states=11,loss_calls=7,neighbor_checks=1220,new_fits=0,new_policy_evaluations=0));A.mkdir(parents=True,exist_ok=False)
    for p in [*Path(__file__).parent.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n');dest=R/'论文协作资料/FRAME_剩余误停与支持_GRID223';dest.mkdir(exist_ok=False)
    for name in ['GRID223_report.md','summary.json','loss_calls.json','limiting_fit_support.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/name,dest/name)
    msg='GRID223只读核查610原追加状态及22步长限制点完成：16误删=11状态7调用，Jan19_101call005占误删价值58.94%，前2调用73.81%。无利节省+.00034159/回本损失-.00050730。22步长受7训练状态6调用限制；9/16误删b低于拟合正例min，但无利取消也63/111，非单特征补救依据。误删到正例公开距离不比保留更远，不能笼统归域外。保留208与222，下一步资格原有preSTOPCONT辅助支持，不按已看外层调步长/阈值。零拟合/新评分/物理。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID223剩余误停与支持\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID223后\n\n1. 保留208与222成本质量折中；剩余16误删主要集中Jan19_101call005，不针对该评估调用造触发器。\n2. GRID224先资格现有214的preSTOPCONT且公共b>0辅助行，正负全收。原STOP约束与部署入口不变，只比较拟合支持；总体样本暴露不同需报告。\n3. 固定均等STOP/CONT结构域权重后仍乘原absNET；平均风险/完整先验/单位惩罚/原费用标签维持。新辅助模型最多11拟合+11原STOP正例步长标量，共22，一次600秒；旧222控制复用不重训。先封存再执行，不按外层调参。\n4. 用户授权持续自主推进；仅阶段发现汇报。不新物理/最终8/论文/FRAME，不借跨外层OOF。\n',encoding='utf8',newline='\n');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID223剩余误停与支持核查\n\n[报告](protocols/'+A.name+'/GRID223_report.md)；一个调用占约59%误删损失，零新拟合。\n\n').encode('utf8')+p.read_bytes());print('GRID223_AUDITED',checks)
if __name__=='__main__':main()
