"""Audit full-prior matched endpoints, retain improvements and reverse cases."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,R,A,REPO=C.O,C.R,C.A,C.REPO

def main():
    C.scored();assert not (O/'complete.json').exists() and C.read(O/'review.json')['passed'];fitted=scored=0
    for rec in C.read(O/'fit_receipts.json'):
        pool,arm=rec['pool'],rec['arm'];z=np.load(O/'fit_inputs'/f'{pool}.npz');cp=np.load(O/'models'/f'{pool}_{arm}.npz');theta=cp['theta'];constant=bool(cp['fallback_KEEP'][0]);off=z['b'] if arm=='ANCHORED' else np.zeros(len(z['b']));assert constant==(pool=='f0_s1')
        if not constant:
            logit=z['X']@theta[:-1]+theta[-1]+off;p=np.exp(-np.logaddexp(0,-logit));grad=np.array(theta,copy=True);risk=0.
            for xx,yy,ww,pp,ll in zip(z['X'],z['y'],z['weight'],p,logit):grad+=ww/z['weight'].sum()*(pp-yy)*np.r_[xx,1.];risk+=ww/z['weight'].sum()*(np.logaddexp(0,ll)-yy*ll)
            loss=float(risk+.5*sum(theta**2));assert abs(loss-rec['final_loss'])<1e-10 and abs(grad).max()<1e-3 and rec['solver_success'];fitted+=1
        h=np.load(O/'public_held_inputs'/f'{pool}.npz');pred=np.load(O/'held_predictions'/f'{pool}_{arm}.npz');expected=np.ones(len(h['ids'])) if constant else np.sum(h['X']*theta[:-1],axis=1)+theta[-1]+(h['b'] if arm=='ANCHORED' else 0);assert np.allclose(expected,pred['logit'],atol=1e-10,rtol=1e-10);assert np.array_equal(pred['keep'],expected>=0);scored+=len(expected)
        old=np.load(R/'outputs/grid218_anchored_logit/fit_inputs'/f'{pool}.npz')
        for k in z.files:assert np.array_equal(z[k],old[k])
    assert fitted==22 and scored==9362
    for name,phase,pubfile in [('trained_freeze.json','trained','pre_score_publication.json'),('scored_freeze.json','scored','pre_join_publication.json')]:
        pub=C.read(O/pubfile);assert C.git('show',pub['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    cases={v['sample_id']:v for v in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];tr=C.read(O/'oof_trajectories.json');base={(r['method'],r['sample_id']):r for r in tr if r['method'].startswith('GATE')}
    for r in tr:
        c=cases[r['sample_id']];end=r['baseline_stop']+int(r['append']);assert r['gain']==float(c['gains'][:end].sum()) and r['queries']==int(c['queries'][:end].sum()) and r['chosen_pool_id']==c['chosen'][end];assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
        if not r['method'].startswith('GATE'):s=int(r['method'].split('seed')[-1]);assert not r['append'] or base[f'GATE_seed{s}',r['sample_id']]['append']
    assert len(tr)==21402;changed=C.read(O/'changed_between_arms.json');assert len(changed)==62 and all(r['ANCHORED_append'] and not r['DIRECT_append'] for r in changed);pos=[r for r in changed if r['actual_next_NET']>0];assert len(pos)==10;groups=collections.defaultdict(list)
    for r in pos:groups[r['physical_state_group']].append(r)
    restored=[dict(call=k,dates=sorted({r['date'] for r in rr}),seeds=sorted({r['seed'] for r in rr}),rows=len(rr),sample_ids=[r['sample_id'] for r in rr],NET_values=[r['actual_next_NET'] for r in rr]) for k,rr in groups.items()];assert len(restored)==4;C.save('restored_profitable_calls.json',restored)
    fam=C.read(O/'family_summary.json');counts=C.read(O/'gate_counts.json');paired=C.read(O/'paired_summary.json');delta=[r for r in paired if r['base']=='DIRECT'];assert len(delta)==3 and all(r['macro_delta']['mean_gain']>0 and r['macro_delta']['action_match_rate']>0 for r in delta);decision=dict(retain_GRID208=True,adopt220=False,preserve_small_repeated_quality_improvement=True,matched_query_cost_not_equal=True,restored_profit_rows=10,restored_distinct_calls=4,one_call_two_seeds=True,all3_gain_action_improve_vsDIRECT=True,NET_signs=[1,-1,-1],remaining_profitable_cancellations=34,next='Saved-data decomposition then qualify a training-only maximal safe correction step; do not choose shrinkage by held outcomes or scan penalty/threshold.',new_physics=False);C.save('decision.json',decision)
    lines=['# GRID220：平均风险与完整先验下的同容量对照','','GRID219资格通过后，按固定方案完成22次线性拟合及2个KEEP常量。输入、标签、原费用、求解器和数据与GRID218逐项相同；新目标使用加权平均损失和完整126参数L2。单位惩罚事先固定，没有扫描惩罚或阈值。模型先发布、预测先发布后连接结果。','', '## 主要结果','','|指标|原GRID208|直接读出|原判断偏置|','|---|---:|---:|---:|']
    for key,label in [('mean_queries','候选查询'),('mean_gain','即时增益'),('mean_net_utility','固定费代理净效用'),('gain_capture','局部收益保留比例'),('action_match_rate','完整局部搜索动作一致率')]:lines.append(f"|{label}|{fam['GATE'][key]:.9f}|{fam['DIRECT'][key]:.9f}|{fam['ANCHORED'][key]:.9f}|")
    lines+=['', '同配置比较中，原判断偏置比直接读出多保留62次追加，其中10次回本、52次不回本；回本保留从11/55增至21/55。全部3种子的即时增益和动作一致率提高，净效用1正2负，宏平均仅有很小正差。多查付出的成本必须同时报告，不能把增加查询后的质量改善说成等预算优势。','', '|种子|偏置减直接：净效用|查询|动作一致率（百分点）|','|---|---:|---:|---:|']
    for d in delta:m=d['macro_delta'];lines.append(f"|{d['seed']}|{m['mean_net_utility']:.9f}|{m['mean_queries']:.6f}|{100*m['action_match_rate']:.6f}|")
    lines+=['','## 值得保留的局部改善','','10条回本记录来自4个物理调用，其中2037-04-06_101的call000在种子0和2均恢复了回本追加，其余属于单种子或同调用多个前缀。不能把10条当10次独立确认，也不因此挑选种子。','', '|物理调用|种子|恢复记录数|','|---|---|---:|']
    for r in restored:lines.append(f"|{r['call']}|{','.join(map(str,r['seeds']))}|{r['rows']}|")
    lines+=['', '该线索比GRID218仅取消1条零收益记录更接近“保留有价值补查”的目标，应继续研究。当前偏置组仍误删34条原本回本的追加，离原208的55条保留还有差距，故不替换208。','', '## 能归因到什么','','主比较是在相同新目标下改变原判断的正则化中心，输入、容量、标签、费用和优化预算一致。相对GRID218同时改变了损失归一化和截距惩罚，跨轮改善不能算纯单因素消融。新目标的复制记录与整体权重缩放不变性、完整梯度和Hessian资格由GRID219保存；这些数值性质不保证外层质量。','', '两个线性读出都能在训练中学习与原判断不同的函数，零初始化只保证起点。下一步先在现有记录上分解恢复的10条和额外52条成本，再研究能否从训练正收益记录确定修正步长，避免按已看外层结果手选缩放系数。尚未计算或评估该步长策略。','', '## 验证与边界','','22个固定终点全部收敛；独立重算全部目标和梯度、9362个线性预测、21402条保存序列重放。复用218重放器的源码字节相同，实际输出目录由220模块绑定；控制台旧前缀不作为实验身份依据。原218和219源码、原失败的源文件生成检查、全部结果均保留；训练前输入等价校验通过，失败准备未产生拟合。','', '原历史外层结果已暴露，这是旧教师路径上的开发比较，非新独立测试；无新NN拟合或物理调用，最终8、论文和新FRAME未动。候选查询与即时收益不等于完整控制时间或整周成本。']
    (O/'GRID220_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.scored();C.save('independent_audit.json',dict(passed=True,all22_meanrisk_fullnorm_objectives_and_stationarygradients_reconstructed=True,all9362_outputs_reconstructed=True,all21402_sequences_reconstructed=True,all_input_arrays_identical_to218=True,all62_betweenarm_changes_verified=True,restored10_profitrows_in4calls=True,all_publication_barriers_and_source_hashes_verified=True));C.save('complete.json',dict(passed=True,fits=22,constant_KEEP=2,held_predictions=9362,trajectories=21402,restored_profit_vsDIRECT=10,retain_GRID208=True,new_NN_fits=0,new_physics=0,final8_absent=True));C.archive('audited');dest=R/'论文协作资料/FRAME_完整先验对照_GRID220';dest.mkdir(exist_ok=False)
    for name in ['GRID220_report.md','family_summary.json','paired_summary.json','gate_counts.json','restored_profitable_calls.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/name,dest/name)
    qdest=R/'论文协作资料/FRAME_先验目标资格_GRID219';qdest.mkdir(exist_ok=False)
    for name in ['GRID219_plan.md','qualification.json','complete.json']:shutil.copy2(R/'outputs/grid219_prior_objective'/name,qdest/name)
    msg='GRID219资格零拟合；GRID220固定22平均风险+完整参数先验线性对照完成。偏置对直接回本保留11→21/55，10恢复来自4调用，April101call000在2种子重复；三种子增益/动作均正，但多62追加含52无利，代理净效用1正2负。宏净效用直接.08969034/偏置.08969446，低208 .09609252，偏置仍误删34回本。不全赢否决小改善，不冒充等预算。保留208及线索，接下来只读分解并资格训练内确定修正步长，不按外层结果选缩放。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID219资格与GRID220完整先验对照\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID220后\n\n1. 保留208及220三种子质量改善：偏置比直接多10回本/4调用，1调用2种子；多52无利，净效用1正2负，非等预算优势。\n2. 只读分解恢复与额外成本；随后资格一个只用拟合正例确定的最大保护步长，不按外层标签调缩放或阈值。零修正复现208，经验拟合保护不保证泛化。\n3. 用户授权自主训练，明确阶段方案后执行固定对照，保留全部种子和反例，不扫惩罚、不追加物理/最终8/论文/新FRAME。\n',encoding='utf8',newline='\n');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID220平均风险与完整先验对照\n\n[报告](protocols/'+A.name+'/audited/GRID220_report.md)；回本保留11→21，三种子质量改善但额外查询抵消部分收益。\n\n').encode('utf8')+p.read_bytes());print('GRID220_AUDITED',json.dumps(dict(fits=fitted,scored=scored,restored_calls=restored),ensure_ascii=False))
if __name__=='__main__':main()
