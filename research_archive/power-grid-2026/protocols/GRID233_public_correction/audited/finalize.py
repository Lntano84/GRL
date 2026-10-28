"""Audit one restricted-residual experiment, including all wins and adverse cases."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
def main():
    C.scored();assert C.read(O/'review.json')['passed'] and not (O/'complete.json').exists();recs=C.read(O/'fit_receipts.json');positive=0;scored=0
    for r in recs:
        z=np.load(C.Q/'qualified_inputs'/f"{r['pool']}.npz");p=np.load(O/'models'/f"{r['pool']}.npz");theta=p['theta'];a=float(p['alpha'][0]);fallback=bool(p['fallback_KEEP'][0]);assert theta.shape==(43,);v,g=C.objective(theta,z['X'],z['y'],z['weight'],z['b'])
        if not fallback:assert abs(g).max()<1e-3 and r['solver_success']
        b=z['calibration_b'];d=z['calibration_X']@theta[:-1]+theta[-1];pos=z['calibration_y']>0;mass=z['query_mass'];budget=float(z['parent_budget'][0]);assert np.all((b+a*d)[pos]>=0) and C.PLAN.BUDGET.saved_mass(b,d,a,mass)<=budget;positive+=int(pos.sum())
        if not fallback:
            detail=r['joint_calibration'];bound=min(1.,detail['positive_bound'],detail['budget_constraint']['analytic_bound']);assert a<=bound+1e-14
            if bound<1:
                witness=.5*(bound+1.);assert np.any((b+witness*d)[pos]<0) or C.PLAN.BUDGET.saved_mass(b,d,witness,mass)>budget
        h=np.load(C.P/'public_held_inputs'/f"{r['pool']}.npz");s=np.load(O/'held_predictions'/f"{r['pool']}.npz");expected=h['b']+a*(np.sum(h['X'][:,83:125]*theta[:-1],axis=1)+theta[-1]);assert np.array_equal(expected,s['logit']) and np.array_equal(expected>=0,s['keep']);scored+=len(expected)
    assert positive==90 and scored==4681 and sum(r['fitted'] for r in recs)==11
    for phase,name,pubname in [('trained','trained_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
        pub=C.read(O/pubname);assert C.git('show',pub['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];tr=C.read(O/'oof_trajectories.json')
    for r in tr:
        c=cases[r['sample_id']];j=r['baseline_stop']+int(r['append']);assert r['gain']==float(c['gains'][:j].sum()) and r['queries']==int(c['queries'][:j].sum()) and r['chosen_pool_id']==c['chosen'][j] and r['actions']==[1]*j+([0] if j<len(c['states']) else []);assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
    assert len(tr)==28536;changes=C.read(O/'changed_choices.json');family=C.read(O/'family_summary.json');pairs=C.read(O/'paired_summary.json');counts=C.read(O/'gate_counts.json');metrics=C.read(O/'search_metrics.json');n={r['date']:r['cases'] for r in metrics['GATE_seed0']['per_date']};calls=collections.defaultdict(float);decomp=[]
    for r in changes:calls[r['physical_state_group']]+=r['NET_delta']/(3*16*n[r['date']])
    for seed in range(3):
        rr=[r for r in changes if r['seed']==seed];good=sum(max(r['NET_delta'],0)/(16*n[r['date']]) for r in rr);bad=sum(min(r['NET_delta'],0)/(16*n[r['date']]) for r in rr);expected=next(r for r in pairs if r['base']=='BUDGET' and r['seed']==seed)['macro_delta']['mean_net_utility'];assert abs(good+bad-expected)<1e-12
        decomp.append(dict(seed=seed,changed=len(rr),positive_contribution=good,negative_contribution=bad,total=expected,recovered_profitable=sum(r['public_append'] and r['actual_next_NET']>0 for r in rr),lost_profitable=sum(not r['public_append'] and r['actual_next_NET']>0 for r in rr),raw_query_delta=sum(r['query_delta'] for r in rr)))
    recovered=[r for r in changes if r['public_append'] and r['actual_next_NET']>0];lost=[r for r in changes if not r['public_append'] and r['actual_next_NET']>0];assert len(recovered)==8 and len(lost)==6 and counts['retained_profitable']==41
    pred=C.read(O/'gate_predictions.json');repeat=[]
    for sid in ['077c228d0254d5b751b98a62','539b1f1919b80a68cb2264cb']:
        rr=[r for r in pred if r['sample_id']==sid and r['seed'] in [1,2]];assert len(rr)==2;repeat.append(dict(sample_id=sid,rows=[dict(seed=r['seed'],parent_continue=r['parent_budget_continue'],public_continue=r['public_continue'],actual_NET=r['actual_next_NET']) for r in rr]))
    negative=sorted(calls.items(),key=lambda p:p[1]);positivecalls=sorted(calls.items(),key=lambda p:-p[1]);C.save('effect_decomposition.json',dict(per_seed=decomp,recovered_profitable=recovered,lost_profitable=lost,call_contributions=dict(negative),positive_calls=positivecalls,previous_repeated_savings=repeat,correlated_prefixes_not_independent=True));C.save('decision.json',dict(retain230_as_primary_internal_candidate=True,do_not_replace_by233=True,retain233_positive_pockets=True,reason='Restrictedresidualrecovers8andloses6profitable prefixes,mainnewloss4correlatedstatesofJan5;seed2recoversJan19andJun1butseed0lossdominates. Morequeryonaverage,notstronger overall.',next='GRID234readonlycompare8recoveries6losses,Jan5fourcorrelatedstates,Jan19/Jun1recoveriesandlostrepeatedsavings. Separate endpointvsjointstepandowntrainingsupport beforeanyfinitelearningproposal;noheldthreshold/seedselection.'))
    lines=['# GRID233：公共摘要修正器的固定训练结果','','一次固定的11个线性模型和11个联合标量训练已经完成，共22拟合/校准，用时约0.058秒；所有模型的数值资格通过。强模型先验、原风险权重和查询费用保持不变，额外修正器由126参数限制为43参数。','',
    '## 找回了哪些机会，又丢掉了什么','',
    '相对230，本轮恢复8条回本补查、误停6条，回本保留数从39/55增至41/55。种子2恢复了1月19日同一调用的3个相关前缀，以及6月1日一条此前漏掉的机会，代理净效用提高。这些是值得保留的正向线索。',
    '但种子0在1月5日同一调用的4条相关状态上再次误停，损失幅度很大；不能把它们说成4个独立失败。此前两处在种子1、2重复省查询的状态也全部恢复成了无利补查。因此，回本数量增多并没有带来总体收益提高，输入限制方案不能整体替换230。','',
    '## 所有种子的结果','']
    for r in pairs:
        if r['base']=='BUDGET':m=r['macro_delta'];lines.append(f"种子{r['seed']}相对230：代理净效用{m['mean_net_utility']:+.9f}，候选查询{m['mean_queries']:+.6f}，即时增益{m['mean_gain']:+.10f}。")
    lines +=['',f"三种子日期宏平均：230代理净效用{family['BUDGET']['mean_net_utility']:.9f}，新方案{family['PUBLIC']['mean_net_utility']:.9f}；候选查询{family['BUDGET']['mean_queries']:.6f}变为{family['PUBLIC']['mean_queries']:.6f}；即时增益{family['BUDGET']['mean_gain']:.10f}变为{family['PUBLIC']['mean_gain']:.10f}。",'',
    '新方案在三个种子上的平均查询都增加。虽然补查追加次数减少了，但被保留的批次大小不同，次数少不等于候选查询少。完整局部搜索动作一致率也没有总体提高。',
    '种子2的改善同时增加查询；不能称等预算的排序优势。种子1新增变化没有恢复回本机会，代理净效用略降；种子0的大损失压过了其他改善。全部日期和种子都保留，不选出种子2作为最终结果。','',
    '## 验证与工程记录','',
    '核对全部43维目标与梯度、90条原STOP训练正例、联合训练查询代价约束、4681个已发布预测和28536条四臂保存序列；原230、222、208控制完整复现。全部模型和标量发布后才评分，全部预测发布后才连接结果。',
    '发布模型时，我在原上传进程结束前过早检查远端；屏障拒绝继续，预测未开始。另一个上传调用因共享临时输出失败。随后等待原上传确认成功，保存错误记录，再首次评分。没有重训、修改模型或放宽发布条件。后续上传已严格串行等待完成。','',
    '下一步只读分析8条恢复和6条误停：区分方向变化、联合步长变化与训练支持，重点保留1月19日和6月1日的改善线索，也检查重复省查询为何丢失。不对1月5日设置例外，不根据外层结果调阈值、正则或择种子。',
    '这是已暴露开发数据上的局部代理比较。输入限制改变了容量，不是同容量消融，也不保证外层等预算；未新增物理仿真、读取最终8、修改论文或开展新FRAME比较。']
    (O/'GRID233_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');C.scored();C.save('independent_audit.json',dict(passed=True,all11active43gradient_andsolver_qualifications_verified=True,all90fitpositive_and_querymass_constraints_checked=True,joint_maximality_checked_without_recalibration=True,all4681_published_scores_reconstructed=True,all28536_saved_sequences_and_costs_verified=True,controls_reproduced=True,publication_barriers_verified=True,all8recoveries6losses_and_lostrepeatedsavings_retained=True));C.save('complete.json',dict(passed=True,new_linear_fits=11,new_joint_scalar_fits=11,constantKEEP=1,new_NNfits=0,held_predictions=4681,saved_trajectories=28536,new_physics=0,final8_absent=True));C.archive('audited')
    dest=R/'论文协作资料/FRAME_公共修正器对照_GRID233';dest.mkdir(exist_ok=False)
    for name in ['GRID233_report.md','family_summary.json','paired_summary.json','gate_counts.json','effect_decomposition.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/name,dest/name)
    msg='GRID233固定公共41+b修正器完成11线性+11联合标量(.058s)，全部数值资格通过。对230恢复8回本/误停6，保留39→41/55；种子2找回Jan19三相关前缀与Jun1机会，净效用+.000505086，但多查询+.128303。种子0Jan5同调用4相关状态误停主导损失，净效用-.007654570；种子1-.000039305。总体NET.095926664→.093530401、查询13.171604→13.234543，之前两处重复节省也丢失。保留233局部改善，不整体替换230，不择种子。模型上传未完过早检查被屏障阻止，0预测/0重训，错误保留后严格串行完成。下一步234只读分解8恢复6误停及训练支持，不按外层调参。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID233公共修正器对照\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID233后\n\n1. 233固定22拟合和评分审计完成，不重做。保留230为主要内部候选，233保留种子2恢复Jan19/Jun1的改善线索，不能挑种子替换。\n2. 234只读：分解8回本恢复/6误停、Jan5四相关状态、新丢失的两处重复省查询，核对方向、联合标量和自身拟合支持。不按外层选择阈值/步长/正则/种子，不给调用设例外。\n3. 根据自身训练支持固定下一有限学习方案前，不新增拟合或恢复旧运行器。各阶段最多22拟合/校准、600秒；上传必须串行等原helper完成后才进行发布屏障或下一次上传。\n4. 最终8/物理/电网/论文/新FRAME不动，原控制器/费用/标签/排序别名/锁站/128/64/停滞2/.9保留。局部代理/查询不等整周质量/时延。用户持续授权直到喊停。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID233公共修正器对照\n\n[报告](protocols/'+A.name+'/audited/GRID233_report.md)；回本数量改善但大损失主导，保留局部线索与全部反例。\n\n').encode('utf8')+p.read_bytes());print('GRID233_AUDITED',json.dumps(dict(counts=counts,decomp=decomp)))
if __name__=='__main__':main()
