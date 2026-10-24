"""Explain the already-frozen outcome; no refit, new scores or cutoff change."""
from base import *
def main():
    verify_trained();assert read(O/'review.json')['passed'];assert not (O/'failure_diagnosis.json').exists()
    records=read(O/'all_search_replays.json');sums=read(O/'summary.json');s=lambda name:next(r for r in sums if r['evaluation_partition']=='validation' and r['method']==name)
    errors=[r for r in records if r['method']=='RF142_CALIBRATED' and r['evaluation_partition']=='validation' and not r['action_match']];assert len(errors)==1
    error=errors[0];ref=next(r for r in refs() if r['call_id']==error['call_id']);c=B.call_at(ref['source']);truth={v['pool_id']:v for v in c['outcomes']};trace=P.manual_guard(c,B.subids());cut=read(O/'cutoff.json')['primary_threshold'];oldcut=read(PREV/'cutoff.json')['primary_threshold']
    rows=eligible_data('validation');idx=next(i for i,r in enumerate(rows) if r['meta']['call_id']==error['call_id'] and r['meta']['visible_count']==error['global_queries']);oldscore=float(np.load(PREV/'validation_scores.npz')['scores'][idx]);newscore=float(np.load(O/'validation_scores.npz')['scores'][idx]);assert newscore==error['stopped_score']
    capcall=next(r for r in records if r['method']=='CAP256' and r['call_id']==error['call_id']);delta=P.key(truth[error['original_selected']])[0]-P.key(truth[error['selected']])[0]
    points=[]
    for n in trace['visited']:
        b=P.best(c['outcomes'][:n]);points.append(dict(queries=n,pool_id=P.bid(b),reward=P.key(b)[0] if b else None,rho=b['rho'] if b else None,station=B.subids()[b['pool_id']] if b else None))
    static=dict(sample_id=rows[idx]['meta']['sample_id'],call_id=error['call_id'],physical_step=ref['step'],public_at_stop=rows[idx]['features'],new_score=newscore,new_cutoff=cut,new_margin=newscore-cut,old_score=oldscore,old_cutoff=oldcut,
        changed_action_from=error['original_selected'],changed_action_to=error['selected'],signed_immediate_reward_gap=delta,new_queries=error['queries'],original_queries=error['original_queries'],cap256_queries=capcall['queries'],cap256_action=capcall['selected'],
        hindsight_trace=dict(not_online_features=True,original_global_checkpoints=points,global_choice=P.bid(trace['pre']),local_completion_queries=len(trace['local']),final_choice=P.bid(trace['after'])))
    default_equal={}
    for part in ['fit','calibration','validation']:
        a={r['call_id']:r for r in records if r['evaluation_partition']==part and r['method']=='RF142_FIXED_0.5'};b={r['call_id']:r for r in records if r['evaluation_partition']==part and r['method']=='CAP128'}
        default_equal[part]=all(a[k]['selected']==b[k]['selected'] and a[k]['queries']==b[k]['queries'] for k in a)
    total=[]
    for name in ['RF140_CALIBRATED','RF142_CALIBRATED','CAP256']:
        r=s(name);total.append(dict(method=name,queries=r['queries'],queries_saved=r['original_queries']-r['queries'],pooled_query_reduction=1-r['queries']/r['original_queries'],date_macro_query_reduction=r['date_macro_query_reduction'],action_differences=r['action_differences']))
    result=dict(scope='Description of fixed development results;no new threshold/model/fit.',error=static,validation_query_metrics=total,fixed05_equals_cap128=default_equal,new_physical=0,new_fits=0,new_predictions=0,old_model_retained=True)
    save('failure_diagnosis.json',result)
    lines=['# GRID142结果解释：新增批次统计没有形成稳定增量','',
        '本轮已完成一次新拟合和全部审计，没有第二次拟合或验证后修改阈值。保留旧模型，新增特征方案不采用。','',
        '## 不能只看查询总量','',
        '新模型验证查询从23988降到23263，总计少查725，即3.02%；旧模型少查256，即1.07%。但事前固定的指标是日期宏平均，新模型为2.79%，旧模型4.14%。这是因为新模型节省的位置发生变化，三月、八月和十月比旧模型少查，五月反而多查64；各日期在宏平均中权重相同。总量和宏平均均完整保留，不在看到结果后选择更有利的口径。','',
        '新模型有1次改变原动作，旧模型和CAP256均没有，因此本轮也没有达到动作保持要求。新模型总节省中216个查询来自这次不保持动作的提前停止，不能把它当作同质量计算收益。没有运行该动作的真实闭环，不能把这次动作差异称为整周成本失败或电网事故。','',
        '## 一次新误停的具体路径','',
        f"{error['call_id']}对应物理步{ref['step']}。128与192个查询后的最好候选均为59，预测最大负载率约0.88087。新模型在192处分数{newscore:.12f}，略高于已冻结阈值{cut:.12f}，超出{newscore-cut:.8f}，因此停止。原模型对此点分数{oldscore:.12f}，低于其自己的阈值{oldcut:.12f}，会继续。",'',
        '保存的后续反馈显示，下一批找到候选195。原FRAME继续到384个全局查询，随后补查同站的24个未见候选，最终选中444。固定256带相同补查也选中444。新模型提前停止时尚未换掉初始候选，因此不触发原补查，错过了这个后续路径。以上后续反馈仅作错误解释，不进入特征。','',
        f"新旧动作的一步奖励差为{delta:.9f}。这个差较小，但既定任务要求保持原动作，本轮不为消除反例而更改标签或容差，也不把它直接换算成整周经济损失。",'',
        '## 当前能得出的结论','',
        '新增六项统计后，拟合AUC从0.8649升至0.8775，校准从0.5988升至0.6128，但开发验证从0.7206降至0.6913。局部案例中存在信息差，不意味着这组摘要能稳定改善跨日期排序。', '',
        '默认0.5仍在三个分区与CAP128完全相同；校准后放行更多前缀，却出现一个略高于校准最大错误分数的新反例。这说明四个校准日期零错误只是经验结果，不能保证下一个日期没有更高分的错误。不能通过在本验证集上再提高一次阈值宣称问题解决。','',
        '下一步应先区分两件事：学习器是否能把必须继续的少数情形排在正确位置，以及有限日期的经验阈值能否泛化。只继续堆批次统计并未得到支持。若后续研究类别权重或训练目标，必须作为另一项单独冻结的干预，原任务、标签和旧结果继续保留。本轮到此完成，不自动追加训练、扩大网络或运行物理回放。']
    (O/'结果解释与后续判断.md').write_text('\n'.join(lines)+'\n',encoding='utf8');verify_trained();archive('diagnosis')
    collab=R/'论文协作资料/FRAME_批次信息训练_GRID142'
    for n in ['结果解释与后续判断.md','failure_diagnosis.json']:shutil.copy2(O/n,collab/n)
    entry=f'\n\nGRID142结果解释：新查询总量少725(3.02%)优于旧少256(1.07%)，但预定日期宏平均2.79%低于4.14%，五月丢失一次64节省；不更换汇总口径。唯一动作差异{error["call_id"]}在192处分数{newscore:.9f}略超校准阈值{cut:.9f}；下一批找到195、原384全局加24同站最终444，CAP256也返回444。提前停返回59，一步奖励差{delta:.9f}，没有物理成本失败结论。新增表示训练/校准AUC略升、开发验证下降，默认0.5仍等于CAP128；不在验证上抬阈值救结果。保留旧模型，下一路线需先区分少数继续样本排序和经验校准泛化，无自动新拟合。\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+entry,encoding='utf8')
    p=REPO/'docs/NEXT_STEPS.md';p.write_text(p.read_text(encoding='utf8')+'\n5. 新误停为10月call005，192处分数比校准阈值仅高0.00003098，错过下一批候选195和后续站内444；不抬验证阈值修结果。下一步先分清少数继续类的排序与有限日期校准泛化，再决定单独训练干预，不继续加摘要。新总量更省但日期宏平均更差，全部指标保留。\n',encoding='utf8')
    files=0
    for d in A.iterdir():
        if d.is_dir():
            for n,h in read(d/'archive_manifest.json').items():assert sha(d/n)==h;files+=1
    for n in ['GRID142_report.md','结果解释与后续判断.md','failure_diagnosis.json']:assert sha(collab/n)==sha(O/n)
    save('post_diagnosis_verification.json',dict(passed=True,archive_files_verified=files,source_model_labels_unchanged=True,new_fits_total=1,second_fit=False,new_physical=0,final_tests_absent=True))
    d=A/'post_verification';d.mkdir(exist_ok=False);shutil.copy2(O/'post_diagnosis_verification.json',d/'post_diagnosis_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'post_diagnosis_verification.json':sha(d/'post_diagnosis_verification.json')},indent=2)+'\n',encoding='utf8')
    print('GRID142_DIAGNOSIS_COMPLETE',json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
