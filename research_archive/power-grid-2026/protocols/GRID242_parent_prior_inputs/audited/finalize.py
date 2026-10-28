"""Hash/formula/role audit for parent-coherent fitting inputs."""
from pathlib import Path
import json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
def main():
    C.generated();assert C.read(O/'fit_qualification.json')['passed'] and not (O/'complete.json').exists();q=C.read(O/'fit_qualification.json');roles={r['pool']:r for r in C.read(C.PLAN/'qualified_pools.json')};meta=C.read(C.META);checks=0
    for r in q['pools']:
        pool=r['pool'];pub=np.load(O/'public_full'/f'{pool}.npz');expected=(pub['event_p']*pub['mu']-pub['charge'])/(pub['event_p']*pub['mu']+pub['charge']);assert np.array_equal(expected,pub['b']) and np.array_equal(pub['head_CONT'],pub['b']>0);z=np.load(O/'fit_inputs'/f'{pool}.npz');c=np.load(O/'calibration_inputs'/f'{pool}.npz');assert {meta[int(i)]['date'] for i in z['ids']}<=set(roles[pool]['fit_dates']);assert {meta[int(i)]['date'] for i in c['ids']}<=set(roles[pool]['fit_dates'])|set(roles[pool]['cal4_dates']);assert np.array_equal(z['y'],z['actual_NET']>0) and np.array_equal(c['y'],c['actual_NET']>0) and (z['b']>0).all() and (c['b']>0).all()
        if not r['fallback_KEEP']:
            weight=z['weight']/z['weight'].sum();assert abs(weight[z['domain_STOP']].sum()-.5)<1e-12 and abs(weight[~z['domain_STOP']].sum()-.5)<1e-12
        outer=np.load(O/'outer_parent_public'/f'{pool}.npz');old=np.load(R/'outputs/grid220_full_prior/public_held_inputs'/f'{pool}.npz');assert np.array_equal(outer['b'],old['b']) and np.array_equal(outer['base_CONT'],old['base_CONT']);checks+=len(outer['ids'])
    assert checks==4681;assert not any(set(r['fields'])&{'actual_NET','y','original_source_mass'} for r in C.read(O/'public_access.json'));assert len(C.read(O/'target_access.json'))==36
    pub=C.read(O/'pre_target_publication.json');assert C.git('show',pub['git_sha']+':'+(A/'generated/generated_freeze.json').relative_to(REPO).as_posix())==(O/'generated_freeze.json').read_bytes()
    summary=dict(fit8_input_rows=sum(r['fit8_rows'] for r in q['pools']),fit8_STOP_rows=sum(r['fit8_STOP'] for r in q['pools']),fit8_STOP_positive=sum(r['fit8_STOP_positive'] for r in q['pools']),fit8_CONT_rows=sum(r['fit8_CONT'] for r in q['pools']),fit8_CONT_positive=sum(r['fit8_CONT_positive'] for r in q['pools']),union12_STOP_rows=sum(r['union12_STOP'] for r in q['pools']),union12_STOP_positive=sum(r['union12_STOP_positive'] for r in q['pools']),new_linear_fits_planned=q['linear_fits_planned'],new_scalar_fits_planned=q['scalar_fits_planned'],published_parent_outer_rows_reproduced=4681,new_fits=0)
    C.save('summary.json',summary);C.save('review.json',dict(passed=True,all12parentprior_formula_and_entry_reproduced=True,all4681existingparent_outer_margins_bitidentical=True,onlyfit8_risk_normalization=True,union12_empirical_calibration_sources_checked=True,public_outputs_before_target_join_verified=True,public_target_fields_separated=True,no_new_policy_scores=True));C.save('complete.json',dict(passed=True,new_fits=0,new_NNforwards=0,new_physics=0,new_candidate_scores=0,linear_fits_planned=11,scalar_fits_planned=11,final8_absent=True))
    lines=['# GRID242：父先验一致输入已经重建','','原208父方法的24个固定收益头已用于重建训练和检查输入，修正模块的偏置与原入口现在来自同一套判断。没有训练新模型或步长。','',
    f"准备的8天输入共{summary['fit8_input_rows']}条，其中原STOP {summary['fit8_STOP_rows']}条、回本正例{summary['fit8_STOP_positive']}条；辅助CONT {summary['fit8_CONT_rows']}条、正例{summary['fit8_CONT_positive']}条。12天联合原STOP保护集合共{summary['union12_STOP_rows']}条，其中回本正例{summary['union12_STOP_positive']}条。输入行包含预定KEEP池，不能全部算作实际模型拟合记录。", 
    '原来4681个已发表外层停止点的父先验和入口全部重现，裕度最大误差为0；未来直接复用这些原父值，不额外调用8天收益头。公共41维修正特征仍使用固定inner8标准化。',
    '所有公共输出先封存发布，之后才连接既有标签。线性修正器的风险权重只在8天内分别归一化两域，各占一半；12天正例文件只用于预先声明的经验保护步长，不进入线性模型拟合。',
    '11个学习池均通过拟合支持资格，另1池保持预定继续。下一阶段按唯一方案进行11个43参数模型和11个标量的一次拟合，600秒总窗口，不另选日期或参数。',
    '当前尚无新候选效果。新候选改变先验来源、样本资格及保护集合，不能称纯先验消融或内部4天独立校准。没有新物理仿真、最终8、论文或FRAME比较。']
    (O/'GRID242_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');C.archive('audited');dest=R/'论文协作资料/FRAME_父先验一致输入_GRID242';dest.mkdir(exist_ok=False)
    for n in ['GRID242_report.md','summary.json','fit_qualification.json','generation_receipts.json','review.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg=f"GRID242父先验一致输入完成0拟合/0新候选分数。24原头重建同一b12，4681旧外层父裕度/入口逐项一致最大误差0。fit8输入{summary['fit8_input_rows']}，STOP{summary['fit8_STOP_rows']}/{summary['fit8_STOP_positive']}正、CONT{summary['fit8_CONT_rows']}/{summary['fit8_CONT_positive']}正；union12 STOP{summary['union12_STOP_rows']}/{summary['union12_STOP_positive']}正。公共先发后接标签，fit8风险半半独立归一化；243资格11线性+11标量/600秒、f0_s1KEEP，当前无效果结果。"
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID242父先验一致输入\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID242后\n\n1. 242同源父先验与输入资格完成，不重复父预测/标签连接。\n2. 执行243：消费242fit_inputs与calibration_inputs/qualified_fit_freeze，11个43参线性+11原221正例保护标量，600秒总窗口；theta只fit8、alpha保护union12 STOP正例，f0_s1KEEP。原meanNLL+全43单位L2/原L-BFGS-B，不调正则/阈值。\n3. 模型步长全部先发，读取242outer_parent_public评分后全部预测先发再接既有结果。共享GATE为主要对照，230仅历史参考；不称纯先验消融或cal4独立。\n4. 无额外inner8推理/第二支持域，入口原208CONT不变，无新物理/最终8/论文/FRAME。原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保持，Git严格等待helper。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID242父先验一致输入\n\n[报告](protocols/'+A.name+'/audited/GRID242_report.md)；候选尚未训练。\n\n').encode('utf8')+p.read_bytes());print('GRID242_AUDITED',json.dumps(summary))
if __name__=='__main__':main()
