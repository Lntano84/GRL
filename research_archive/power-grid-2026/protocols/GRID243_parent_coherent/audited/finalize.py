"""Audit the combined parent-coherent candidate and retain unsuccessful results."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
def main():
    C.scored();assert C.read(O/'review.json')['passed'] and not (O/'complete.json').exists();fitrec=C.read(O/'fit_receipts.json');positives=0;scored=0
    for r in fitrec:
        pool=r['pool'];z=np.load(C.Q/'fit_inputs'/f'{pool}.npz');c=np.load(C.Q/'calibration_inputs'/f'{pool}.npz');p=np.load(O/'models'/f'{pool}.npz');theta=p['theta'];a=float(p['alpha'][0]);fallback=bool(p['fallback_KEEP'][0]);v,g=C.objective(theta,z['X'],z['y'],z['weight'],z['b'])
        if not fallback:assert r['solver_success'] and abs(g).max()<1e-3
        end=c['b']+np.sum(c['X']*theta[:-1],axis=1)+theta[-1];pos=c['y']>0;assert np.all((c['b']+a*(end-c['b']))[pos]>=0);positives+=int(pos.sum())
        if r['fitted'] and r['analytic_bound']<1:
            witness=.5*(r['analytic_bound']+1.);assert np.any((c['b']+witness*(end-c['b']))[pos]<0)
        h=np.load(C.Q/'outer_parent_public'/f'{pool}.npz');pred=np.load(O/'outer_predictions'/f'{pool}.npz');end=h['b']+np.sum(h['X']*theta[:-1],axis=1)+theta[-1];expected=h['b']+a*(end-h['b']);assert np.array_equal(expected,pred['logit']) and np.array_equal(h['base_CONT']&(expected>=0),pred['continue_']);scored+=len(expected)
    assert positives==85 and scored==4681 and sum(r['fitted'] for r in fitrec)==11
    for phase,name,pubfile in [('trained','trained_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
        pub=C.read(O/pubfile);assert C.git('show',pub['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];tr=C.read(O/'oof_trajectories.json')
    for r in tr:
        c=cases[r['sample_id']];j=r['baseline_stop']+int(r['append']);assert r['gain']==float(c['gains'][:j].sum()) and r['queries']==int(c['queries'][:j].sum()) and r['chosen_pool_id']==c['chosen'][j] and r['actions']==[1]*j+([0] if j<len(c['states']) else []);assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
    assert len(tr)==14268;pred=C.read(O/'gate_predictions.json');metrics=C.read(O/'search_metrics.json');family=C.read(O/'family_summary.json');pairs=C.read(O/'paired_summary.json');counts=C.read(O/'gate_counts.json');n={r['date']:r['cases'] for r in metrics['GATE_seed0']['per_date']};decomp=[];calls=collections.defaultdict(float);miss=[]
    for seed in range(3):
        removed=[r for r in pred if r['seed']==seed and r['base_continue'] and not r['continue_']];saving=sum(-r['actual_next_NET']/(16*n[r['date']]) for r in removed if r['actual_next_NET']<=0);loss=sum(r['actual_next_NET']/(16*n[r['date']]) for r in removed if r['actual_next_NET']>0);expected=next(r for r in pairs if r['seed']==seed)['macro_delta']['mean_net_utility'];assert abs(saving-loss-expected)<1e-12;decomp.append(dict(seed=seed,saved_unprofitable=saving,lost_profitable=loss,total=expected,removed_profitable=sum(r['actual_next_NET']>0 for r in removed),removed_unprofitable=sum(r['actual_next_NET']<=0 for r in removed)))
        for r in removed:
            if r['actual_next_NET']>0:calls[r['call']]+=r['actual_next_NET']/(3*16*n[r['date']]);miss.append(r)
    assert len(miss)==22;oldshared=C.read(R/'outputs/grid239_paired_calibration/shared_profitable_misses.json');lookup={(r['seed'],r['sample_id']):r for r in pred};oldstatus=[dict(seed=r['seed'],sample_id=r['sample_id'],call=r['call'],old239_shared_miss=True,new243_continue=lookup[r['seed'],r['sample_id']]['continue_']) for r in oldshared]
    C.save('profitable_misses.json',miss);C.save('effect_decomposition.json',dict(per_seed=decomp,loss_by_call=dict(sorted(calls.items(),key=lambda p:-p[1])),old239_shared_miss_status=oldstatus,correlated_prefixes_and_seeds_not_independent=True));C.save('decision.json',dict(do_not_replace230_by243=True,retain_seed1_cost_quality_tradeoff=True,do_not_claim_prior_alignment_sufficient=True,do_not_claim_pure_prior_ablation=True,reason='243savesqueriesbutretains33/55profitable,globalNETlower. Seed1NETpositivewithsmallgainloss;seed2large lossdominates.',next='GRID244readonlypairednative12vsinner8prior distributions on SAMEfit8/cal4trainingstates andnewoutermisses;checkin-sampleparentreadoutoptimismandsupport,notanotheralphaswap. No newforecasts/thresholdsorcall exceptions.'))
    lines=['# GRID243：父先验一致候选的固定拟合结果','','本轮按新方案完成11个43参数修正器和11个正例保护步长，约0.052秒；所有数值资格通过。修正器只在8天拟合，步长保护自身12天的85条已选STOP回本记录。原208父先验在输入、偏置与入口上保持一致。','',
    '## 结果','',
    f"相对共同GATE入口，原610次追加变为{counts['appends']}次；保留的回本补查为{counts['profitable_retained']}/55，漏掉{counts['profitable_removed']}条，同时取消{counts['unprofitable_removed']}条不回本追加。",
    f"日期宏平均后对三种子平均：候选查询{family['GATE']['mean_queries']:.6f}降至{family['ALIGNED']['mean_queries']:.6f}，即时增益{family['GATE']['mean_gain']:.10f}降至{family['ALIGNED']['mean_gain']:.10f}，代理净效用{family['GATE']['mean_net_utility']:.9f}降至{family['ALIGNED']['mean_net_utility']:.9f}。动作一致率也下降。",'']
    for r in pairs:lines.append(f"种子{r['seed']}减共同GATE：代理净效用{r['macro_delta']['mean_net_utility']:+.9f}，候选查询{r['macro_delta']['mean_queries']:+.6f}，即时增益{r['macro_delta']['mean_gain']:+.10f}。")
    lines+=['','种子1的代理净效用有小幅提高，查询减少，但即时增益也略降，这是保留研究的成本质量折中。种子0、2的损失更大，尤其1月5日同一调用的相关记录主导总体损失。全部种子和反例保留，当前不替换230。','',
    '## 对原假设的判断','',
    '先验接口一致性已核对，但这个联合候选没有因此获得更好的总体表现。现有结果不足以把此前失败主要归因于两套先验不一致。',
    '本候选同时改变了先验来源、筛选后的样本和正例保护集合，不是纯先验消融。另一方面，父模型对内部12天记录的判断已参与过这些标签的学习，修正器却必须处理原外层4天的新记录；训练和校准看到的父判断可能更乐观。这个分布问题应先核查，不能继续假定只换步长来源就能解决。',
    '242与236已经保存相同状态上的父12天先验和inner8先验，下一步可以直接比较拟合8天与检查4天上的差异，不必训练或推理新模型。这个比较仍同时包含训练规模与标签参与差别，只作来源与支持诊断，不宣称单一因果。','',
    '## 验证与范围','',
    '原父预测4681条重现误差为0；本轮85条经验正例保护、11模型梯度与求解状态、4681个发布预测和14268条保存前缀的动作、成本和查询全部核对。权重和步长先发布、预测再发布、随后连接结果；原共同GATE完整复现。',
    '原控制器和候选入口未改变，没有额外8天收益头或支持域回退。额外公共特征与线性推理未做在线计时，查询减少不等于实际控制加速。',
    '所有数据均为已暴露开发记录。内部12天正例保护不是独立验证，原外层标签祖先隔离仍保持；没有新增物理仿真、最终8、论文或FRAME比较。']
    (O/'GRID243_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');C.scored();C.save('independent_audit.json',dict(passed=True,all11gradient_solver_checks=True,all85empirical_positiveconstraints_and_maximality_checked=True,all4681published_scores_reproduced=True,all14268savedsequences_costs_queries_checked=True,commonGATEreproduced=True,allseeds_counterexamples_retained=True,publication_barriers_verified=True,no_new_physicalsolver=True));C.save('complete.json',dict(passed=True,new_linear_fits=11,new_scalar_fits=11,constantKEEP=1,new_NNfits=0,new_physics=0,published_predictions=4681,saved_trajectories=14268,final8_absent=True));C.archive('audited');dest=R/'论文协作资料/FRAME_父先验一致对照_GRID243';dest.mkdir(exist_ok=False)
    for n in ['GRID243_report.md','family_summary.json','paired_summary.json','gate_counts.json','effect_decomposition.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg='GRID242/243父先验一致联合候选完成，11线性+11标量(.052s)、85经验正例保护及全部数值资格通过。对共享GATE回本保留33/55、漏22、取消243无利；查询13.440525→13.203561，NET .096092524→.092921661。种子NET-.000052505/+.000160804/-.009620889，种子1小折中保留，整体不替换230，不能将先验一致当充分解释。下一步244只读同状态12天父先验与8天头在fit8/cal4上的差异、拟合乐观性/正例支持与新22漏查，0新模型/alpha/预测，不再盲换校准来源。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID243父先验一致联合候选\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID243后\n\n1. 242输入与243固定22拟合/评分审计完成，不重做。243整体不替换230，保留种子1查询/收益折中。\n2. 244仅只读：对齐242与236同状态公共先验，在固定fit8/cal4和原STOP/CONT、正负标签分别比较b12/b8/裕度/资格分布；检查父12天读出已参与内部标签学习而外层未参与的支持差别。比较含样本量差别，不宣称单一因果。\n3. 同时核对243的22漏查与85保护正例范围、旧239共同错停状态；不新增alpha/阈值/第三臂或按Jan5例外。数据支持明确后再资格固定唯一有限方案。\n4. 每阶段<=22拟合/校准、600秒；所有源码/权重/预测先发后接结果。原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保持，最终8/物理/电网/论文/FRAME不动。Git严格等helper完成，自主推进至喊停。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID243父先验一致候选\n\n[报告](protocols/'+A.name+'/audited/GRID243_report.md)；候选整体退步，保留种子1折中并核查读出分布。\n\n').encode('utf8')+p.read_bytes());print('GRID243_AUDITED',json.dumps(dict(counts=counts,decomposition=decomp)))
if __name__=='__main__':main()
