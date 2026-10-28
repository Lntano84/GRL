"""Audit fit-only maximal steps, sealed interpolations and full saved replay."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,R,A,REPO=C.O,C.R,C.A,C.REPO

def main():
    C.scored();assert not (O/'complete.json').exists() and C.read(O/'review.json')['passed'];checked=0;fitchecks=0;calib=C.read(O/'calibration_receipts.json');alpha_rows=[]
    for r in calib:
        pool,arm=r['pool'],r['arm'];z=np.load(C.PARENT/'fit_inputs'/f'{pool}.npz');source=np.load(C.PARENT/'models'/f'{pool}_{arm}.npz');model=np.load(O/'models'/f'{pool}_{arm}.npz');a=float(model['alpha'][0]);theta=source['theta'];b=z['b'];end=z['X']@theta[:-1]+theta[-1]+(b if arm=='ANCHORED' else 0);positive=z['actual_NET']>0;assert np.array_equal(positive,z['y']>0);delta=end-b;mask=positive&(delta<0);bound=min(1.,float(np.min(b[mask]/(-delta[mask])))) if mask.any() else 1.;constant=bool(source['fallback_KEEP'][0]);score=b+a*delta
        if constant:assert pool=='f0_s1' and a==0 and not positive.any()
        else:
            assert 0<=a<=bound and bound-a<=1e-13 and np.all(score[positive]>=0)
            if bound<1:assert np.any((b+(bound+1e-8)*delta)[positive]<0)
            fitchecks+=1
        assert str(model['source_theta_sha256'][0])==C.sha(C.PARENT/'models'/f'{pool}_{arm}.npz');h=np.load(C.PARENT/'public_held_inputs'/f'{pool}.npz');old=np.load(C.PARENT/'held_predictions'/f'{pool}_{arm}.npz');p=np.load(O/'held_predictions'/f'{pool}_{arm}.npz');expected=h['b']+a*(old['logit']-h['b']);assert np.array_equal(p['logit'],expected) and np.array_equal(p['keep'],expected>=0) and np.array_equal(p['ids'],h['ids']);assert np.all(~(old['keep']&h['base_CONT'])|(p['keep']&h['base_CONT']));checked+=len(expected);alpha_rows.append(dict(pool=pool,arm=arm,alpha=a,fit_positive=int(positive.sum()),limiting_ids=r['limiting_fit_ids']))
    assert fitchecks==22 and checked==9362
    for r in C.read(O/'fit_access.json'):assert r['kind'] in ['own_fit_inputs','frozen_parent_model'] and 'held' not in r['path']
    for name,phase,pubname in [('calibrated_freeze.json','calibrated','pre_score_publication.json'),('scored_freeze.json','scored','pre_join_publication.json')]:
        pub=C.read(O/pubname);assert C.git('show',pub['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    cases={v['sample_id']:v for v in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];tr=C.read(O/'oof_trajectories.json');base={(r['method'],r['sample_id']):r for r in tr if r['method'].startswith('GATE')};parentmap={(r['method'],r['sample_id']):r for r in C.read(C.PARENT/'oof_trajectories.json')}
    for r in tr:
        c=cases[r['sample_id']];end=r['baseline_stop']+int(r['append']);assert r['gain']==float(c['gains'][:end].sum()) and r['queries']==int(c['queries'][:end].sum()) and r['chosen_pool_id']==c['chosen'][end];assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
        if not r['method'].startswith('GATE'):
            seed=int(r['method'].split('seed')[-1]);old=parentmap[r['method'],r['sample_id']];full=base[f'GATE_seed{seed}',r['sample_id']];assert old['queries']<=r['queries']<=full['queries'] and old['gain']<=r['gain']<=full['gain']
    assert len(tr)==21402;family=C.read(O/'family_summary.json');parent=C.read(C.PARENT/'family_summary.json');pairs=C.read(O/'paired_summary.json');counts=C.read(O/'gate_counts.json');source_metrics=C.read(C.PARENT/'search_metrics.json');metrics=C.read(O/'search_metrics.json');vsparent=[]
    for arm in C.ARMS:
        for seed in range(3):
            old,new=source_metrics[f'{arm}_seed{seed}']['macro'],metrics[f'{arm}_seed{seed}']['macro'];d={k:new[k]-old[k] for k in old};vsparent.append(dict(arm=arm,seed=seed,macro_delta=d));assert d['mean_gain']>=0 and d['mean_queries']>=0
    C.save('vs_source220.json',vsparent);decision=dict(retain_GRID208=True,retain222_as_quality_cost_tradeoff=True,not_claim_superiority_over208=True,fitonly_step_substantially_improves_source220_protection=True,ANCHOR_profitable_retained=39,source220_ANCHOR_retained=21,original208_profitable=55,remaining_profitable_cancellations=16,NET_vs208_one_positive_two_negative=True,not_equal_query_budget_with_source220=True,next='Read-only group the16remainingprofitablecancellations and22calibration limiting calls; compare publicly available fit support without retuning steps on held labels.',no_auto_step_or_threshold_search=True);C.save('decision.json',decision)
    qreduction=100*(1-family['ANCHORED']['mean_queries']/family['GATE']['mean_queries']);netgap=100*(family['ANCHORED']['mean_net_utility']/family['GATE']['mean_net_utility']-1);lines=['# GRID222：拟合正收益保护的闭式步长校准','','主线性模型沿用GRID220冻结权重，完成22个仅用拟合正例的闭式标量校准，另2个无正例池KEEP。不是重新训练22个主模型。所有标量发布后评分，全部24份预测发布后连接结果；没有用外层结果选择缩放系数。','', '## 主要结果','','|指标|原GRID208|校准直接读出|校准原判断偏置|','|---|---:|---:|---:|']
    for key,label in [('mean_queries','候选查询'),('mean_gain','即时增益'),('mean_net_utility','固定费代理净效用'),('gain_capture','局部收益保留比例'),('action_match_rate','完整局部搜索动作一致率')]:lines.append(f"|{label}|{family['GATE'][key]:.9f}|{family['DIRECT'][key]:.9f}|{family['ANCHORED'][key]:.9f}|")
    lines+=['',f"偏置一组相对原208平均查询减少{qreduction:.3f}%，代理净效用变化{netgap:.3f}%。原55条回本追加保留39条，仍误删16条（7个调用）。不称优于原模型，保留为查询较少的质量折中点。",'', '与源GRID220相比，偏置组回本保留21→39，总查询13.0093→13.1747；直接组11→37。保护有所提高，也增加了查询，不能声称等预算更好。','', '|种子|校准偏置相对原208：净效用|查询|动作一致率（百分点）|','|---|---:|---:|---:|']
    for d in [r for r in pairs if r['base']=='GATE' and r['new']=='ANCHORED']:m=d['macro_delta'];lines.append(f"|{d['seed']}|{m['mean_net_utility']:.9f}|{m['mean_queries']:.6f}|{100*m['action_match_rate']:.6f}|")
    lines+=['','## 校准做了什么','','沿原判断分数b到学习终点s的线段，选择最大步长a，使拟合集所有z>0记录仍满足b+a(s-b)>=0。上界完全由各自外层训练池的正例计算；只因浮点边界向下舍入，不搜索验证阈值。每臂90条相关拟合正例全部保留。','', '它增加了经验正收益保护约束，并不等价于原无约束损失；这是明确改变决策学习的约束，没有改变查询费、收益标签或仅挑选大收益正例。拟合内约束没有提供外层保证，仍然误删16条就是其边界。','', '直接与偏置两臂各自都有一个由训练数据求出的步长。两臂之间除模型正则中心外，校准数值也随各自源终点变化；当前评估的是完整校准流程，不将结果拆成未验证的单个因果作用。','', '## 保留全部情况','','相对原208，净效用为1个种子提高、2个下降；相对源220，质量明显恢复但查询更多。没有删掉反向日期或挑种子。原模型仍作为当前主要参考，校准版用于下一阶段分析剩余错误；不因未全赢丢弃小幅查询收益，也不把平均差距小当作安全或整周质量保证。','', '## 验证','','独立重算22个步长的最大可行界和180条跨臂拟合正例约束，核对9362个插值输出、21402条保存序列及所有发布屏障。所有源220权重和输入哈希不变。校准进程只解析拟合输入与冻结模型，评分进程使用公共特征和已封存源分数。','', '复用的重放器保留旧控制台前缀，实际输出路径和实验身份由GRID222模块决定。原外层数据已暴露，本次为开发研究；无新主模型或NN训练、无新物理/最终8/论文/新FRAME。查询和即时收益不代表完整控制时间或整周成本。']
    (O/'GRID222_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.scored();C.save('independent_audit.json',dict(passed=True,all22_maximal_step_bounds_and180_fitpositive_constraints_checked=True,all9362_sealed_interpolations_checked=True,all21402_sequences_and_query_gain_between_source_and208_checked=True,fit_access_excludes_held_inputs_and_outcomes=True,publication_barriers_and_source_hashes_verified=True));C.save('complete.json',dict(passed=True,closed_form_scalar_fits=22,constant_KEEP=2,new_main_model_fits=0,new_NN_fits=0,predictions=9362,trajectories=21402,retain_GRID208=True,new_physics=0,final8_absent=True));C.archive('audited');dest=R/'论文协作资料/FRAME_训练内保护步长_GRID222';dest.mkdir(exist_ok=False)
    for name in ['GRID222_report.md','family_summary.json','paired_summary.json','vs_source220.json','calibration_receipts.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/name,dest/name)
    plan=R/'论文协作资料/FRAME_保护步长资格_GRID221';plan.mkdir(exist_ok=False)
    for name in ['GRID221_plan.md','decomposition.json','synthetic_qualification.json','complete.json']:shutil.copy2(R/'outputs/grid221_step_plan'/name,plan/name)
    msg='GRID221分解与合成资格完成；GRID222固定22训练内正收益保护步长+2KEEP，源220模型不重训。偏置回本保留21→39/55（仍16损失7调用）；对208查询13.4405→13.1747，代理净效用.09609252→.09592681，约-0.172%，三种子1正2负。相对源220质量提高但查询更多，不称等预算优势。保留208主参考与222成本质量折中，不全赢否定。下步只读核查16损失和步长限制调用，不按外层结果改步长。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID221分解与GRID222保护步长\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID222后\n\n1. 保留208及222成本质量折中：查询约-1.98%、代理净效用约-0.172%，1正2负；回本39/55。\n2. 只读核对16回本损失（7调用）、22步长的限制拟合调用和公开支持；不按已看外层标签挑缩放或阈值。\n3. 源220权重保持，不自动扫正则/步长；用户授权自主推进，形成明确新方案再执行并汇报。\n4. 不新物理/最终8/论文/新FRAME；完整控制时间与整周质量未评。\n',encoding='utf8',newline='\n');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID222训练内保护步长\n\n[报告](protocols/'+A.name+'/audited/GRID222_report.md)；回本保留39/55，查询约少1.98%，保留成本质量折中及反例。\n\n').encode('utf8')+p.read_bytes());print('GRID222_AUDITED',json.dumps(dict(calibrations=fitchecks,query_reduction_percent=qreduction,proxy_NET_percent=netgap,counts=counts),ensure_ascii=False))
if __name__=='__main__':main()
