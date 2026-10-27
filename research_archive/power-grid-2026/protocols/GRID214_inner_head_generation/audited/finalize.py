"""Audit head-only sample generation and report coverage, not policy performance."""
from pathlib import Path
import collections,json,pickle,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def main():
    C.scored();assert not (O/'complete.json').exists();assert (O/'labelled_freeze.json').exists()
    for p,h in read(O/'labelled_freeze.json').items():assert C.sha(O/p)==h,p
    jobs=read(O/'jobs.json');receipts=read(O/'fit_receipts.json');assert len(jobs)==len(receipts)==108;fitchecks=0
    for job,rec in zip(jobs,receipts):
        assert (job['ordinal'],job['bundle'],job['head'])==(rec['ordinal'],rec['bundle'],rec['head']);bundle,head=job['bundle'],job['head'];x,_,_,q=C.matrix(bundle,'fit');target=np.load(C.PLAN/'fit_targets'/f'{bundle}.npz');event=head=='event';xx=x if event else x[target['amount_positive_positions']];y=target['event_y'] if event else target['amount_y'];w=q['event_weights'] if event else target['amount_weights'];model=pickle.loads((O/'models'/f'{bundle}_{head}.pkl').read_bytes());params=read(O/'configs.json')[f"f{job['outer_fold']}_s{job['seed']}"][head];assert model.get_params()==params and model.n_features_in_==124;nodes=leaves=0;worst=0.;manual=[]
        for tree in model.estimators_:
            leaf=tree.apply(xx);v=tree.tree_.value[leaf,0,:];manual.append(v[:,1]/v.sum(1) if event else v[:,0]);assert tree.tree_.max_depth<=params['max_depth'];nodes+=tree.tree_.node_count;leaves+=int(np.sum(tree.tree_.children_left==-1))
            for j in np.unique(leaf):
                m=leaf==j;assert m.sum()>=10;truth=float(w[m]@y[m]/w[m].sum());v=tree.tree_.value[j,0,:];value=float(v[1]/v.sum()) if event else float(v[0]);worst=max(worst,abs(value-truth));assert abs(value-truth)<1e-8
        guess=np.mean(manual,axis=0);assert abs(float(w@((guess-y)**2)/w.sum())-rec['fit_weighted_error'])<1e-9;assert nodes==rec['nodes'] and leaves==rec['leaves'];fitchecks+=1
    for phase in ['qualify','train','score']:assert not any(r['kind']=='score_targets' for r in read(O/f'{phase}_access.json'))
    assert sum(r['kind']=='score_targets' for r in read(O/'join_access.json'))==36
    pub=read(O/'pre_score_publication.json');joined=read(O/'pre_join_publication.json');assert git('show',pub['git_sha']+':'+(C.A/'trained/trained_freeze.json').relative_to(C.REPO).as_posix())==(O/'trained_freeze.json').read_bytes();assert git('show',joined['git_sha']+':'+(C.A/'scored/scored_freeze.json').relative_to(C.REPO).as_posix())==(O/'scored_freeze.json').read_bytes()
    meta=read(C.META);coverage=collections.defaultdict(list);poolcoverage=collections.defaultdict(lambda:collections.defaultdict(list));identity=collections.defaultdict(list);allrecords=[];scorechecks=0
    for r in read(O/'bundles.json'):
        bundle=r['bundle'];p=np.load(O/'predictions'/f'{bundle}.npz');label=np.load(O/'labelled_samples'/f'{bundle}.npz');target=np.load(C.PLAN/'score_targets'/f'{bundle}.npz');x,ids,charge,q=C.matrix(bundle,'score');values={}
        for head,field in [('event','event_p'),('amount_depth4','mu4'),('amount_depth5','mu5')]:
            model=pickle.loads((O/'models'/f'{bundle}_{head}.pkl').read_bytes());manual=[]
            for tree in model.estimators_:v=tree.tree_.value[tree.apply(x),0,:];manual.append(v[:,1]/v.sum(1) if head=='event' else v[:,0])
            values[field]=np.mean(manual,axis=0);assert np.allclose(values[field],p[field],atol=1e-10,rtol=1e-10)
        assert np.array_equal(ids,p['meta_ids']) and np.array_equal(ids,label['meta_ids']);assert np.array_equal(charge,p['public_charge']) and np.array_equal(charge,target['public_charge']);assert np.array_equal(p['continue4'],values['event_p']*values['mu4']>charge) and np.array_equal(p['continue5'],values['event_p']*values['mu5']>charge);assert np.array_equal(p['candidate_cancel'],p['continue4']&~p['continue5'])
        for field in ['event_p','mu4','mu5','public_charge','continue4','continue5','candidate_cancel']:assert np.array_equal(label[field],p[field])
        for field in ['event_y','gain_normalized','actual_NET','original_STOP_mask']:assert np.array_equal(label[field],target[field])
        assert not {meta[int(i)]['date'] for i in ids}&set(r['outer_evaluation_dates']);assert {meta[int(i)]['date'] for i in ids}==set(r['score_dates']);identity[r['outer_fold'],r['seed']]+=ids.tolist();a,b=p['continue4'],p['continue5'];stop=target['original_STOP_mask'];pool=f"f{r['outer_fold']}_s{r['seed']}";masks={'actual_STOP_all':stop,'actual_STOP_base4_continue':stop&a,'actual_STOP_both_continue':stop&a&b,'actual_STOP_candidate_cancel':stop&a&~b,'actual_STOP_both_stop':stop&~a&~b,'actual_STOP_only5_continue':stop&~a&b,'preSTOP_CONT_all':~stop,'preSTOP_CONT_candidate_cancel':~stop&a&~b}
        for k,i in enumerate(ids):allrecords.append(dict(outer_fold=r['outer_fold'],inner_fold=r['inner_fold'],seed=r['seed'],index=int(i),date=meta[int(i)]['date'],call=meta[int(i)]['physical_state_group'],profitable=bool(target['actual_NET'][k]>0),actual_NET=float(target['actual_NET'][k]),gain_normalized=float(target['gain_normalized'][k]),original_STOP=bool(stop[k]),base4_continue=bool(a[k]),depth5_continue=bool(b[k])))
        for name,mask in masks.items():
            for k in np.flatnonzero(mask):
                item=dict(index=int(ids[k]),call=meta[int(ids[k])]['physical_state_group'],date=meta[int(ids[k])]['date'],profitable=bool(target['actual_NET'][k]>0),zero_gain=bool(target['gain_normalized'][k]==0));coverage[name].append(item);poolcoverage[pool][name].append(item)
        scorechecks+=1
    for (f,s),ids in identity.items():
        source=np.load(C.FIT/'fit_features'/f'f{f}_s{s}.npz');assert len(ids)==len(set(ids))==len(source['ids']) and set(ids)==set(source['ids'].tolist())
    assert scorechecks==36 and len(identity)==12 and len(allrecords)==22653
    def summarize(rows):
        positive=[r for r in rows if r['profitable']]
        return dict(model_state_rows=len(rows),unique_state_indices=len({r['index'] for r in rows}),profitable_rows=len(positive),profitable_unique_states=len({r['index'] for r in positive}),profitable_calls=len({r['call'] for r in positive}),profitable_dates=len({r['date'] for r in positive}),unprofitable_rows=len(rows)-len(positive),zero_gain_rows=sum(r['zero_gain'] for r in rows))
    total={k:summarize(v) for k,v in coverage.items()};perpool={pool:{k:summarize(v) for k,v in groups.items()} for pool,groups in poolcoverage.items()};allnames=list(total)
    for name in allnames:
        total[name]['pools_with_profitable']=sum(perpool[pool].get(name,{}).get('profitable_rows',0)>0 for pool in perpool)
        total[name]['pools_with_both_labels']=sum(perpool[pool].get(name,{}).get('profitable_rows',0)>0 and perpool[pool].get(name,{}).get('unprofitable_rows',0)>0 for pool in perpool)
    old=read(R/'outputs/grid212_cancellation_support/summary.json')['fit_domains'];fits=read(O/'fits.json');summary=dict(total_score_model_state_rows=len(allrecords),coverage=total,per_outer_seed=perpool,previous_in_sample_domain_counts=old,actual_forests=108,actual_trees=6912,fit_and_native_check_wall_s=fits['total_fit_and_native_check_wall_s'],new_NN_fits=0,new_NN_forwards=0,new_physics=0,router_fits=0,policy_evaluations=0);C.save('coverage_summary.json',summary);C.save('all_score_row_identities.json',allrecords)
    cancel=total['actual_STOP_candidate_cancel'];base=total['actual_STOP_base4_continue'];whole=total['actual_STOP_all'];decision=dict(generation_complete=True,all_108_fixed_fits_and36_scores_finished=True,no_router_or_combined_policy_trained=True,candidate_cancel_protection_rows=cancel['profitable_rows'],candidate_cancel_pools_with_protection=cancel['pools_with_profitable'],broader_base4_continue_protection_rows=base['profitable_rows'],broader_STOP_protection_rows=whole['profitable_rows'],counts_are_correlated_model_state_rows=True,retain_GRID208=True,no_automatic_extra_fitting_or_thresholdchanges=True);C.save('decision.json',decision)
    lines=['# GRID214：内层读出分歧样本生成完成','','本轮按GRID213固定方案完成108个森林的训练、36组内层预测及标签连接。没有训练取消开关、执行组合策略或评分外层日期。','', '## 执行和隔离','', f"108个森林共6912棵树，训练与内置逐叶核验耗时{fits['total_fit_and_native_check_wall_s']:.3f}秒，低于固定600秒上限；该计时不含后续评分、归档和独立复核，也不是线上延迟。所有权重在内层评分前已发布，36个预测文件在连接评分标签前已发布。",'', '每组用8个日期拟合读出，在另4个日期生成样本，始终处于对应外层12个训练日期内。输入、缩放、权重及模型参数逐项消费资格文件。评分目标在资格、拟合和评分阶段未解析；标签只在预测封存后的连接阶段读取。全部36组目标连接和22653条模型—状态评分记录均保留。','', '这些是读出头未参与自身拟合的日期预测。冻结的基础NET和停止状态筛选仍来自外层12日期的拟合，不能称整个流水线完全交叉拟合。外层4个日期始终未用于这轮训练、缩放或样本挑选。','', '## 取消分歧的覆盖结果','', '|类别|模型—状态行|不同状态|回本行|回本不同状态|回本调用/日期|有回本的模型池|','|---|---:|---:|---:|---:|---:|---:|']
    for name in allnames:
        z=total[name];lines.append(f"|{name}|{z['model_state_rows']}|{z['unique_state_indices']}|{z['profitable_rows']}|{z['profitable_unique_states']}|{z['profitable_calls']}/{z['profitable_dates']}|{z['pools_with_profitable']}/12|")
    lines+=['',f"与GRID212拟合内筛选相比，实际STOP域取消分歧由{old['actual_STOP']['repeated_model_state_rows']}行变为{cancel['model_state_rows']}行，回本行仍为4；有回本例的模型池由2变为{cancel['pools_with_profitable']}。头级交叉生成改善了样本生成规范，但没有明显增加这个狭窄分歧子集的保护例。训练日期规模与缩放也随内层划分改变，不能将计数差异全归于去除拟合内选择效应。",'', '模型—状态行会跨外折或种子重复，同一调用也有多个前缀；不同状态数量同样不是独立场景数量。不能将这些行计数当独立新实验。','', '## 保护例是否在更完整的样本中','', f"全部实际停止域有{whole['profitable_rows']}条回本记录；深4原本会继续的状态有{base['profitable_rows']}条回本记录，分布于{base['pools_with_profitable']}/12个模型池。两棵树都愿意继续的状态包含{total['actual_STOP_both_continue']['profitable_rows']}条。因此，许多保护例存在于本轮保存的数据里，只是被‘两树必须分歧’这一入口排除了。",'', f"深4继续域仍有{12-base['pools_with_profitable']}个模型池没有回本例，不能据90条总记录就声称各个模型都已具备足够训练支持。全部实际停止域在{whole['pools_with_profitable']}/12池有回本例，但扩大训练域会改变学习问题与部署条件，需要另定训练方案。",'', '这些覆盖统计只描述已封存预测与标签，未执行“取消全部分歧”或其他策略，也未计算事后最优策略成绩。保留连续净收益和费用，未来不能仅按分类数量决定保护价值。','', '## 各模型池实际STOP覆盖','', '|外层/种子|全部STOP回本|原深4继续行/回本|取消分歧行/回本|','|---|---:|---:|---:|']
    for pool in sorted(perpool):
        rr=perpool[pool];a=rr['actual_STOP_all'];b=rr.get('actual_STOP_base4_continue',{});c=rr.get('actual_STOP_candidate_cancel',{});lines.append(f"|{pool}|{a['profitable_rows']}|{b.get('model_state_rows',0)}/{b.get('profitable_rows',0)}|{c.get('model_state_rows',0)}/{c.get('profitable_rows',0)}|")
    lines+=['','## 当前结论与后续','', '固定样本生成已完成，数据与全部模型保留。208继续作为当前较均衡候选，不依据这轮覆盖数字改变线上行为。下一步可利用已保存的全部评分行，先研究训练域应覆盖原本会查的状态还是整个实际STOP域，避免只筛两树分歧而丢失保护例；先冻结学习目标和日期依赖，再训练取消模块。','', '本轮不追加更多森林、不扩大网络、不改变费用或阈值，也不自动训练路由。无新物理、最终8、论文或原FRAME比较；样本覆盖不等于完整周质量或控制时延改善。']
    (O/'GRID214_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.scored();C.save('independent_audit.json',dict(passed=True,all108_params_positive_leafmeans_and_fiterrors_reconstructed=True,all36_prediction_files_reconstructed_from_native_leaves=True,all22653_labels_dates_and_row_identities_checked=True,all12_outer_fit_state_sets_scored_exactly_once_perseed=True,weight_and_prediction_git_barriers_verified=True,no_score_target_parses_in_qualify_fit_or_score=True,router_fits=0,new_NN_forwards=0,new_physics=0));C.save('complete.json',dict(passed=True,scope='inner_head_sample_generation_only',forests_fitted=108,trees_fitted=6912,score_bundles=36,score_model_state_rows=22653,router_fits=0,new_NN_fits=0,new_NN_forwards=0,new_physics=0,final8_absent=True));C.archive('audited');dest=R/'论文协作资料/FRAME_内层分歧样本_GRID214';dest.mkdir(exist_ok=False)
    for n in ['GRID214_report.md','coverage_summary.json','decision.json','independent_audit.json','complete.json','fits.json','pre_score_publication.json','pre_join_publication.json']:shutil.copy2(O/n,dest/n)
    msg=f"GRID214按授权完成108森林/6912树，36组内层评分22653模型状态行；权重先发布、预测先发布后连标签，零路由/NN/物理。实际STOP分歧407行仍仅4回本(4/12池有)，未明显扩充狭窄分歧保护例；原深4继续域3466行含90回本(11/12池)，全STOP含199(12/12池)。重复记录非独立场景，保留全部评分数据，下一步先定训练域与目标，不自动训练取消开关。"
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-27 — GRID214内层分歧样本生成完成\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID214后\n\n1. 固定108头样本生成已完成；208继续主候选，不重复拟合或改变已封存预测。\n2. 利用已保存全部评分行，先定取消学习的训练域/目标和日期依赖；仅两树分歧仍缺保护例，原深4继续域与全STOP有更多相关保护记录，不能直接把计数当效果。\n3. 未训练路由或组合策略。新训练先通知，不追加生成/调费用阈值/扩物理/最终8/论文/原FRAME。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID214内层样本生成完成\n\n[报告](protocols/'+C.A.name+'/audited/GRID214_report.md)；108森林与36组预测全部封存，未训练取消开关。\n\n').encode('utf8')+p.read_bytes());print('GRID214_COMPLETE',json.dumps(dict(coverage=total,decision=decision),ensure_ascii=False))
if __name__=='__main__':main()
