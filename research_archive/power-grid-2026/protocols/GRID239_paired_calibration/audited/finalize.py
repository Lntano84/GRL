"""Audit the one matched calibration-source pair and preserve positive pockets."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
def main():
    C.scored();assert C.read(O/'review.json')['passed'] and not (O/'complete.json').exists();records=C.read(O/'calibration_receipts.json');checked=0
    for r in records:
        pool,arm=r['pool'],r['arm'];p=np.load(O/'models'/f'{pool}_{arm}.npz');parent=np.load(C.S/'models'/f'{pool}.npz');assert np.array_equal(p['theta'],parent['theta']);x,b,pos,ids=C.calibration_data(pool,arm);d=np.sum(x*p['theta'][:-1],axis=1)+p['theta'][-1];a=float(p['alpha'][0]);assert np.all((b+a*d)[pos]>=0)
        if r['calibrated'] and r['analytic_bound']<1:
            witness=.5*(r['analytic_bound']+1.);assert np.any((b+witness*d)[pos]<0)
        z=np.load(O/'public_outer'/f'{pool}.npz');pred=np.load(O/'outer_predictions'/f'{pool}_{arm}.npz');expected=z['b']+a*(np.sum(z['X']*p['theta'][:-1],axis=1)+p['theta'][-1]);cont=z['base_CONT']&(~z['inner_support']|(expected>=0));assert np.array_equal(expected,pred['logit']) and np.array_equal(cont,pred['continue_']) and np.array_equal(pred['passthrough'],z['base_CONT']&(z['b']<=0));assert np.all(np.isfinite(expected));checked+=len(expected)
        b8=(z['event_p']*z['mu4']-z['charge'])/(z['event_p']*z['mu4']+z['charge']);assert np.array_equal(b8,z['b']);original=np.load(R/'outputs/grid220_full_prior/public_held_inputs'/f'{pool}.npz');assert np.array_equal(original['ids'],z['ids']) and np.array_equal(original['base_CONT'],z['base_CONT'])
    assert checked==9362 and sum(r['calibrated'] for r in records)==22
    for phase,name,pubfile in [('calibrated','calibrated_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
        pub=C.read(O/pubfile);assert C.git('show',pub['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];tr=C.read(O/'oof_trajectories.json')
    for r in tr:
        c=cases[r['sample_id']];j=r['baseline_stop']+int(r['append']);assert r['gain']==float(c['gains'][:j].sum()) and r['queries']==int(c['queries'][:j].sum()) and r['chosen_pool_id']==c['chosen'][j] and r['actions']==[1]*j+([0] if j<len(c['states']) else []);assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
    assert len(tr)==21402;pred=C.read(O/'gate_predictions.json');index={(r['seed'],r['sample_id'],r['arm']):r for r in pred};metrics=C.read(O/'search_metrics.json');n={r['date']:r['cases'] for r in metrics['GATE_seed0']['per_date']};changes=[];shared=[];paired=C.read(O/'paired_summary.json');family=C.read(O/'family_summary.json');counts=C.read(O/'gate_counts.json')
    for old in [r for r in pred if r['arm']=='FIT8']:
        new=index[old['seed'],old['sample_id'],'CAL4'];assert old['base_continue']==new['base_continue'] and old['passthrough']==new['passthrough']
        if old['continue_']!=new['continue_']:
            sign=int(new['continue_'])-int(old['continue_']);changes.append(dict(seed=old['seed'],date=old['date'],call=old['call'],sample_id=old['sample_id'],position=old['position'],FIT8_continue=old['continue_'],CAL4_continue=new['continue_'],actual_NET=old['actual_next_NET'],NET_delta=sign*old['actual_next_NET'],query_delta=sign*old['actual_next_queries'],gain_delta=sign*old['actual_next_gain']))
        if old['base_continue'] and old['actual_next_NET']>0 and not old['continue_'] and not new['continue_']:shared.append(old)
    assert len(changes)==32 and sum(r['actual_NET']>0 for r in changes)==1 and len(shared)==7
    decomp=[]
    for seed in range(3):
        rr=[r for r in changes if r['seed']==seed];positive=sum(max(r['NET_delta'],0)/(16*n[r['date']]) for r in rr);negative=sum(min(r['NET_delta'],0)/(16*n[r['date']]) for r in rr);expected=next(r for r in paired if r['seed']==seed)['macro_delta']['mean_net_utility'];assert abs(positive+negative-expected)<1e-12;decomp.append(dict(seed=seed,changed=len(rr),positive_contribution=positive,negative_contribution=negative,total=expected))
    context=[]
    for seed in range(3):
        delta={k:metrics[f'CAL4_seed{seed}']['macro'][k]-metrics[f'GATE_seed{seed}']['macro'][k] for k in ['mean_net_utility','mean_gain','mean_queries','action_match_rate']};context.append(dict(seed=seed,delta_vs_common_Gate=delta,context_not_primary_matched_comparison=True))
    passprofits=sum(r['passthrough'] and r['actual_next_NET']>0 for r in pred if r['arm']=='CAL4');assert counts['FIT8']['passthrough']==counts['CAL4']['passthrough']==131 and passprofits==3
    C.save('changed_choices.json',changes);C.save('shared_profitable_misses.json',shared);C.save('effect_decomposition.json',dict(per_seed=decomp,common_gate_context=context,shared_profitable_misses=7,shared_missed_calls=sorted({r['call'] for r in shared}),passthrough_records_perarm=131,passthrough_profitable_records_perarm=3,all_prefix_seed_correlations_retained=True));C.save('decision.json',dict(do_not_claim_CAL4_superior_to_FIT8=True,primary_NET_signs=[-1,0,-1],preserve_CAL4_two_seed_common_gate_positive_pockets=True,do_not_pick_successful_seeds=True,reason='CAL4restoresoneprofitableprefixbutaddsqueriesanddoesnotimproveaveragenet;botharmsretaincommondominantseed2misses.',next='GRID240readonlysharedseed2missesandtwoseedpositivepockets;separateendpoint/constraintcoverage/supportfallback. No thirdarm,newthresholdorrefitwithoutnewqualification.'))
    lines=['# GRID239：拟合正例校准与留组正例校准的成对比较','','本轮是独立预注册的新离线开发实验，复用237所有修正器，只比较校准正例来自8天拟合集还是4天修正层未拟合组。两组都使用原最大正例保护规则，22个标量约0.100秒，0个新模型。237错误收集方案与结果保持原样。','',
    '## 校准来源的直接比较','',
    f"拟合组FIT8保留47/55条原回本补查，留组CAL4保留48/55条。日期宏平均后再对三种子平均，代理净效用{family['FIT8']['mean_net_utility']:.9f}变为{family['CAL4']['mean_net_utility']:.9f}，候选查询{family['FIT8']['mean_queries']:.6f}变为{family['CAL4']['mean_queries']:.6f}。", 
    '两组共32处决定不同，仅1处是回本机会的恢复，来自3月16日、种子2；其余31处不回本。留组校准即时增益略增，但查询增加，三个种子的净效用变化为两负、一相同。当前结果不能说明换成留组正例校准就能提高平均收益。']
    for r in paired:lines.append(f"种子{r['seed']}，CAL4减FIT8：代理净效用{r['macro_delta']['mean_net_utility']:+.9f}，查询{r['macro_delta']['mean_queries']:+.6f}，即时增益{r['macro_delta']['mean_gain']:+.10f}。")
    lines+=['','## 保留值得研究的小改善','',
    '以共同208入口作上下文，CAL4在种子0、1上保留了相同即时增益及相同动作一致率，同时减少候选查询；两组在种子2上却共享7条回本误停。种子0、1的平均查询分别减少约0.200、0.078，代理净效用分别增加约0.000288、0.000112。这个区域性改善保留，但不能挑出两个成功种子作为最终方法。',
    '共同的7条误停全部发生在种子2，涉及1月5日、2月16日、6月1日和11月16日的4个调用；其中1月5日有3条相关状态，不能当独立失败。改变正例校准来源尚未解决这一共同问题。','',
    '## 共同入口与代价','',
    '两组都只修正原208本来要继续的610条记录，没有新增候选入口或救回原208已停止的动作。8天先验b8不为正的131条记录统一保持父方法决定，其中3条回本。这些保留不能记作新校准器学出来的效果。该支持域回退是两组共同的预注册条件。',
    '旧230、233的训练日期、先验和约束不同，不作为校准来源的纯对照。额外收益头推理成本未做在线计时，离线候选查询减少不等于控制时间同比减少。','',
    '## 核验与后续','',
    '全部237权重逐项未变；每个步长的来源正例保护与最大性已核对。24个有效模型与步长先发布，24组外层预测再发布，随后连接结果；9362个预测与21402条保存前缀的动作、成本、查询全部核对。原208控制复现，全部日期和种子保留。',
    '这是旧暴露外层日期上的离线开发比较，上游模型在外层训练12天、没有使用自身外层4天标签；内部校准仍仅为条件修正层隔离，不称全链独立或盲测。没有新增物理仿真、最终8、论文或FRAME比较。',
    '下一步只读分析共同的种子2误停及两个成功种子的省查询区域，检查端点、约束覆盖和支持域回退的作用。当前不追加第三种校准规则，不按1月5日调阈值或择种子。']
    (O/'GRID239_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');C.scored();C.save('independent_audit.json',dict(passed=True,all24theta_equal237=True,allsourcepositiveconstraints_and_maximality_checked_without_refitting=True,all9362public_predictions_entry_andfallback_reproduced=True,all21402saved_sequences_costs_queries_verified=True,commonGATEcontrols_reproduced=True,publication_barriers_verified=True,all32changes_shared7misses_and_positive_context_retained=True,no_new_physical_solver=True));C.save('complete.json',dict(passed=True,new_scalar_fits=22,new_model_fits=0,new_NNfits=0,new_forestfits=0,outer_predictions=9362,saved_trajectories=21402,new_physics=0,final8_absent=True));C.archive('audited');dest=R/'论文协作资料/FRAME_成对校准对照_GRID239';dest.mkdir(exist_ok=False)
    for n in ['GRID239_report.md','family_summary.json','paired_summary.json','gate_counts.json','effect_decomposition.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg='GRID239独立新协议成对校准完成22标量/0新模型(.100s)。同237端点+inner8先验，FIT8vsCAL4只改正例来源。保留47→48/55，CAL4 NET.094299228低于FIT8 .094323443，查询13.283551高于13.264926；32分歧仅1回本恢复，三种子NET负/同/负。共同入口回退131含3回本单列。保留CAL4相对共同GATE在种子0/1即时增益和动作一致、查询-.200/-.078的小改善；种子2共享7误停4调用，Jan5三相关状态主导，不择种子或声称CAL4优越。下一步240只读共同失败和省查询区域，0新规则。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID239成对校准来源对照\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID239后\n\n1. 238资格和239成对22标量/离线评估均完成，不重做；237旧探针不回改。CAL4不优于FIT8平均NET，但保留相对共同GATE两个种子的省查询区域。\n2. 240只读共享7回本误停(种子2/4调用，Jan5三相关状态)、两个成功种子的省查询区域、131域外回退及共同端点/正例覆盖；不新增第三臂、阈值、alpha或模型。\n3. 用自身训练标签与既有公开支持固定下一有限方案前不继续拟合。不能按外层给Jan5例外或择种子。每阶段<=22拟合/校准、600秒，原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保留。\n4. 最终8/物理/电网/论文/FRAME不动。已暴露离线代理不等整周质量/实际时延，额外头推理未计时。模型/预测先发后连标签，Git严格等helper完成，自主推进至喊停。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID239成对校准来源比较\n\n[报告](protocols/'+A.name+'/audited/GRID239_report.md)；整体未改善，保留两种子省查询线索与共享失误。\n\n').encode('utf8')+p.read_bytes());print('GRID239_AUDITED',json.dumps(dict(counts=counts,changes=len(changes),decomposition=decomp,context=context)))
if __name__=='__main__':main()
