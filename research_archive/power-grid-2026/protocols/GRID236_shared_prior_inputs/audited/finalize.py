"""Audit public regeneration and fit8-only inputs; no extra forecasts or targets."""
from pathlib import Path
import json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
def main():
    C.generated();assert C.read(O/'fit_qualification.json')['passed'] and not (O/'complete.json').exists();q=C.read(O/'fit_qualification.json');gens=C.read(O/'generation_receipts.json');pub=C.read(O/'pre_fit_label_publication.json');assert C.git('show',pub['git_sha']+':'+(A/'generated/generated_freeze.json').relative_to(REPO).as_posix())==(O/'generated_freeze.json').read_bytes()
    for r in q['pools']:
        pool=r['pool'];p=np.load(O/'public_full'/f'{pool}.npz');assert np.array_equal((p['event_p']*p['mu4']-p['charge'])/(p['event_p']*p['mu4']+p['charge']),p['b']);assert np.array_equal(p['head_CONT'],p['b']>0) and np.all(p['fit_role']^p['cal_role']);fit=np.load(O/'fit_public'/f'{pool}.npz');z=np.load(O/'fit_inputs'/f'{pool}.npz');assert np.array_equal(fit['ids'],z['ids']) and np.array_equal(fit['X'],z['X']) and np.array_equal(z['y'],z['actual_NET']>0)
        if not r['fallback_KEEP']:
            w=z['weight']/z['weight'].sum();assert abs(w[z['domain_STOP']].sum()-.5)<1e-12 and abs(w[~z['domain_STOP']].sum()-.5)<1e-12 and np.all(w>0)
    accesses=C.read(O/'fit_label_access.json');assert len(accesses)==24 and all(('_i1_' in v['path'] or '_i2_' in v['path']) and 'actual_NET' in v['fields'] for v in accesses);assert not any(set(v['fields'])&{'y','actual_NET','event_y','gain_normalized','original_source_mass'} for v in C.read(O/'public_access.json'))
    counts=dict(pools=12,source_public_rows=sum(r['source_rows'] for r in gens),fit8_rows=sum(r['fit_rows'] for r in q['pools']),fit8_STOP_rows=sum(r['STOP_rows'] for r in q['pools']),fit8_STOP_positive=sum(r['STOP_positive'] for r in q['pools']),fit8_CONT_rows=sum(r['CONT_rows'] for r in q['pools']),fit8_CONT_positive=sum(r['CONT_positive'] for r in q['pools']),cal4_public_rows=sum(r['cal4_selected'] for r in gens),cal4_public_STOP_rows=sum(r['cal4_STOP'] for r in gens),new_linear_models_planned=q['new_linear_fits_planned'],new_fits=0,cal4_targets_parsed=0)
    C.save('summary.json',counts);C.save('review.json',dict(passed=True,all24existingheadweights_published_and_unchanged=True,all12oldcal_public_predictions_and_matrixhashes_reproduced=True,public_exports_contain_no_outcomes=True,all24targetfiles_scoredates_withinfit8=True,no_inner0cal4_targets_parsed=True,all_fit8_risks_domainnormalized=True,public_prediction_publication_barrier_verified=True));C.save('complete.json',dict(passed=True,new_fits=0,new_NNforwards=0,new_physics=0,cal4_targets_parsed=0,new_linear_models_qualified=11,new_scalar_fits_planned=0,final8_absent=True))
    lines=['# GRID236：同源先验与修正层训练输入已就绪','','本轮复用24个已发表的森林头，为每个池的12天记录生成同一来源的先验。没有重新训练森林或神经网络，没有运行物理仿真。','',
    f"共导出{counts['source_public_rows']}条公共记录；按新先验筛选后，8天拟合记录{counts['fit8_rows']}条，其中原STOP {counts['fit8_STOP_rows']}条、回本正例{counts['fit8_STOP_positive']}条；辅助CONT {counts['fit8_CONT_rows']}条、正例{counts['fit8_CONT_positive']}条。不同池、种子和前缀相关，不能把行数当独立场景数。",
    f"另外4天的公共输入共{counts['cal4_public_rows']}条，其中原STOP {counts['cal4_public_STOP_rows']}条。这4天的结果标签尚未解析，将在修正器权重及其预测发布后才连接。",'',
    '12个池的既有4天公共预测及矩阵哈希全部重现，头权重未变。重新生成先验后重新筛选，没有沿用旧跨头资格。拟合风险只使用8天标签，在STOP和辅助CONT内分别归一化后各占一半。',
    '预定11个43参数线性修正器均具备拟合支持，另1个池按原方案保持常量继续；不会因为重新生成后该池出现了正例而临时激活。下一步只拟合这11个修正器，600秒窗口，标量拟合0。','',
    '本轮先发布全部公共预测，再只解析另外两个inner分组、其日期完全属于拟合8天的既有标签文件。未解析inner0的4天标签文件。原收益、费用和标签均复用，没有重新仿真。',
    '这仍是冻结上游基模型与STOP总体后的条件修正层诊断，不是整条流程独立验证；不会进行新的外层策略比较。']
    (O/'GRID236_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');C.archive('audited');dest=R/'论文协作资料/FRAME_同源先验输入_GRID236';dest.mkdir(exist_ok=False)
    for n in ['GRID236_report.md','summary.json','fit_qualification.json','generation_receipts.json','review.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg=f"GRID236完成0拟合/0NN/物理：复用24既有头同源生成{counts['source_public_rows']}公共行，旧4天预测/矩阵均重现。新fit8 {counts['fit8_rows']}行，STOP {counts['fit8_STOP_rows']}/{counts['fit8_STOP_positive']}正、CONT {counts['fit8_CONT_rows']}/{counts['fit8_CONT_positive']}正；cal4公共{counts['cal4_public_rows']}行、目标未解析。全部公共预测先发后只接fit8标签，各域半半风险独立归一化。237资格11个43参模型+1预定KEEP，标量0、600秒，只收集条件层4天错误不做新outer策略。"
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID236同源先验与拟合输入\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID236后\n\n1. 236同源先验/fit8输入资格已完成，不重复头预测或标签连接。cal4目标未解析。\n2. 实施237：仅消费236fit_inputs及qualified_fit_freeze，11个43参线性修正器+1KEEP、0标量，原219均值NLL+全43单位L2，原L-BFGS-B设置，零初始化、600秒总窗口，全部success与rawgradinf<.001。\n3. 权重先发布再读取236cal_public评分；全部4天公共分数发布后才按236future_cal_target_sources读取inner0标签，统计真实STOP/CONT条件错误。训练worker不解析cal4标签，未校准分数b+Xtheta，无alpha/预算/二级选择器，不做新outer4策略对比。\n4. 上游baseNET/STOP仍outer12，明确局部条件诊断，不称独立OOF/全链嵌套或竞争力提升。原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保留，最终8/物理/电网/论文/FRAME不动。Git严格等待helper完成，自主推进至喊停。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID236同源先验输入\n\n[报告](protocols/'+A.name+'/audited/GRID236_report.md)；4天标签未解析，下一步固定11修正器诊断。\n\n').encode('utf8')+p.read_bytes());print('GRID236_AUDITED',json.dumps(counts))
if __name__=='__main__':main()
