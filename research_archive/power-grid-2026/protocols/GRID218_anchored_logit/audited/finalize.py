"""Verify matched convex endpoints, sealed predictions and saved replay."""
from pathlib import Path
import json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,R,A,REPO=C.O,C.R,C.A,C.REPO

def main():
    C.scored();assert not (O/'complete.json').exists() and C.read(O/'review.json')['passed'];records=C.read(O/'fit_receipts.json');geometry=[];fitted=0;scored=0
    for rec in records:
        pool,arm=rec['pool'],rec['arm'];z=np.load(O/'fit_inputs'/f'{pool}.npz');cp=np.load(O/'models'/f'{pool}_{arm}.npz');theta=cp['theta'];offset=z['b'] if arm=='ANCHORED' else np.zeros(len(z['b']));constant=bool(cp['fallback_KEEP'][0]);assert theta.shape==(126,) and constant==(pool=='f0_s1')
        if not constant:
            logit=z['X']@theta[:-1]+theta[-1]+offset;p=np.exp(-np.logaddexp(0,-logit));g=np.zeros(126)
            for xx,yy,ww,pp in zip(z['X'],z['y'],z['weight'],p):g[:-1]+=ww*(pp-yy)*xx;g[-1]+=ww*(pp-yy)
            g[:-1]+=theta[:-1];loss=float(sum(ww*(np.logaddexp(0,ll)-yy*ll) for ww,yy,ll in zip(z['weight'],z['y'],logit))+.5*sum(theta[:-1]**2));assert abs(loss-rec['final_loss'])<1e-8 and abs(g).max()<1e-3 and rec['solver_success'];fitted+=1
        else:assert not z['y'].any() and not rec['fitted'] and np.array_equal(theta,np.zeros(126))
        h=np.load(O/'public_held_inputs'/f'{pool}.npz');p=np.load(O/'held_predictions'/f'{pool}_{arm}.npz');expected=np.ones(len(h['ids'])) if constant else np.sum(h['X']*theta[:-1],axis=1)+theta[-1]+(h['b'] if arm=='ANCHORED' else 0);assert np.allclose(expected,p['logit'],atol=1e-10,rtol=1e-10) and np.array_equal(p['keep'],expected>=0);scored+=len(expected)
        selected=h['base_CONT'];delta=np.zeros(selected.sum()) if constant else np.sum(h['X'][selected]*theta[:-1],axis=1)+theta[-1];prior=h['b'][selected];geometry.append(dict(pool=pool,arm=arm,constant=constant,correction_five_quantiles=np.quantile(delta,[0,.25,.5,.75,1]).tolist() if len(delta) else [],prior_five_quantiles=np.quantile(prior,[0,.25,.5,.75,1]).tolist() if len(prior) else [],kept=int((p['keep']&selected).sum()),unpenalized_intercept=float(theta[-1]),prior_feature_coefficient=float(theta[-2]),effective_prior_coefficient=float(theta[-2]+(arm=='ANCHORED'))))
    assert fitted==22 and scored==9362
    for name,phase,pubfile in [('trained_freeze.json','trained','pre_score_publication.json'),('scored_freeze.json','scored','pre_join_publication.json')]:
        p=C.read(O/pubfile);assert C.git('show',p['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    changed=C.read(O/'changed_between_arms.json');assert len(changed)==1 and changed[0]['gain_delta']==0 and changed[0]['query_delta']==-40 and changed[0]['actual_next_NET']<0;counts=C.read(O/'gate_counts.json');assert all(counts[a]['removed_profitable']==52 for a in C.ARMS)
    fam=C.read(O/'family_summary.json');paired=C.read(O/'paired_summary.json');C.save('correction_geometry.json',geometry);decision=dict(retain_GRID208=True,adopt_new_gate=False,small_positive_signal='One seed0 zero-gain batch removed relativeDIRECT,40queries saved;other2seeds same. Preserve,notstablegeneral improvement.',major_loss='Both arms delete52/55 profitable original appends.',mechanism='Offset is within same inputspan;only L2center changes. Prior stays explicit but total learned correction overcomes it. Sum loss scales with selectedrowweight,interceptunpenalized;doesnotestablishallbaselineanchoringineffective.',next='Qualify one weighted-mean-loss and full-correction-norm matched probe; preservefee/labels/data/threshold and expose change in regularizationstrength and intercept penalty.',no_thresholdsearch=True,no_new_data=True);C.save('decision.json',decision)
    lines=['# GRID218：同容量线性直接读出与原判断偏置对照','','完成22次固定线性拟合及2个KEEP常量。125个输入、126个参数、初始化、数据、费用、损失及求解预算均匹配，主比较只改变是否加入固定原判断偏置。全部模型先发布，24份预测先发布后连接结果。外层历史已暴露，本次为开发比较。','', '## 结果','','|指标|原GRID208|直接读出|保留原判断偏置|','|---|---:|---:|---:|']
    for key,label in [('mean_queries','候选查询'),('mean_net_utility','固定费代理净效用'),('gain_capture','局部收益保留比例'),('action_match_rate','完整局部搜索动作一致率')]:lines.append(f"|{label}|{fam['GATE'][key]:.9f}|{fam['DIRECT'][key]:.9f}|{fam['ANCHORED'][key]:.9f}|")
    lines+=['', '保留原判断偏置相对直接读出，只在种子0少查1个零收益批次，节省40个候选查询、收益和动作不变；其他两种子完全相同。该小收益保留，但只有一个相关前缀实例，不能称稳定改善。两个读出均误删原55条回本追加中的52条，余下3条由常量KEEP保留；当前不替换208。','', '## 这次真正检验了什么','','两组初始权重都为零，只在原模型已准备追加的入口运行。直接读出在零分数按固定平局保留，因此也初始复现208；不能把起点相同当作偏置组独有的优点。','', '原判断b已经作为第125维输入，固定加上b只等价于将这一维的有效系数中心从0移到1。因此两者是相同线性函数类，区别为L2正则化中心，没有新增表达容量。结果只说明当前软约束很弱，不能推论任何继承原模型的办法都无效。','', '原目标为加权损失求和加系数L2，截距未惩罚。有效数据权重和等于所选记录条数；学习后的总修正可以压过0到1范围内的原判断。截距坐标依赖输入原点，不能单凭其负值认定唯一原因；保存了实际修正与原判断分布，供后续核查。','', '下一步先核验按总权重取平均的损失、并对全部修正参数（含截距）施加统一L2的匹配对照。该目标对复制相同记录或整体缩放权重保持不变；它同时改变有效正则化强度和截距惩罚，不能把与本轮的差异说成单因素消融。原费、标签和部署入口不改，不扫描惩罚系数或判定阈值。','', '## 验证与边界','','22个固定终点全部收敛；独立重算目标及梯度，复核9362个线性输出、三臂21402条教师序列重放和原NET/208控制。全部种子、日期和反例保留。没有新NN拟合或物理调用，未触及最终8、论文或新FRAME比较。离线候选查询和即时收益不代表整周质量或完整控制时间。']
    (O/'GRID218_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.scored();C.save('independent_audit.json',dict(passed=True,all22_stationary_objectives_and_gradients_recomputed=True,all9362_linear_outputs_reconstructed=True,all24_publication_barriers_checked=True,all21402_replays_reviewed=True,one_changed_zero_batch_verified=True,all_models_and_sources_unchanged=True));C.save('complete.json',dict(passed=True,fits=22,constant_KEEP=2,scored_rows=9362,trajectories=21402,retain_GRID208=True,new_NN_fits=0,new_physics=0,final8_absent=True));C.archive('audited');dest=R/'论文协作资料/FRAME_原判断偏置对照_GRID218';dest.mkdir(exist_ok=False)
    for name in ['GRID218_report.md','family_summary.json','paired_summary.json','decision.json','independent_audit.json','complete.json','correction_geometry.json']:shutil.copy2(O/name,dest/name)
    msg='GRID218完成22固定线性拟合+2KEEP：125输入126参数/同损失预算，只比较是否加入原判断b偏置，同函数类仅正则中心不同。偏置相对直接读出种子0少1零收益批次/40查询，另2种子同；两臂都误删52/55回本，净效用.08901554/.08902320仍低208 .09609252。保留微小正面与全部反例。继续核验加权平均损失+完整修正范数（含截距）的匹配对照，不改费/标签/阈值或扫描参数。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID218同容量原判断偏置对照\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID218后\n\n1. 保留208及218一个零收益取消的小线索；当前偏置与直接都误删52/55回本，未达到主目标。\n2. 只读资格核验：损失按总权重取平均，完整126参数L2（含截距），检验整体权重缩放/复制记录不改变目标；两臂仍同输入容量预算。\n3. 显式报告相对218改变有效正则强度及截距惩罚，主比较仍在新目标下偏置vs直接。用户已授权自主训练，无需逐次确认；阶段汇报后执行固定一轮，不扫描正则或阈值。\n4. 不扩物理/最终8/论文/新FRAME，保留全部种子反例。\n',encoding='utf8',newline='\n');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID218同容量原判断偏置对照\n\n[报告](protocols/'+A.name+'/audited/GRID218_report.md)；一个零收益取消小线索，主损失仍在。\n\n').encode('utf8')+p.read_bytes());print('GRID218_AUDITED',fitted,scored)
if __name__=='__main__':main()
