"""Read-only remaining misses, unnecessary restorations and training-domain support."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,hashlib,json,shutil
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid231_residual_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID231_residual_support'
S=R/'outputs/grid230_training_budget';D=R/'outputs/grid228_context_diagnosis';Q=R/'outputs/grid224_context_support_plan/revision02';P=R/'outputs/grid220_full_prior';T=R/'outputs/grid222_protected_step';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
def cosine(x,y):
    den=np.linalg.norm(x)*np.linalg.norm(y);return float(x@y/den) if den else None
def main():
    assert not O.exists() and read(S/'complete.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID231',scope='Read-only residuals and owntrainingdomain support. No fits/scalars,newmodelpredictions,policyreplay,heldthresholdselection.',
        questions='Which profitable misses remain from222,which switch bothways? How much proxyNET is saved/lost? Eleven unnecessaryrestorations and two repeat savings kept. Decompose fixedscore blocks and fixednearest support.',
        training_only='For eachownpool record STOP/auxCONT weight and class mass,inputrange departures and fixed zero-correction gradient block angle. Candidate42input restriction alreadyproposed beforethisstage;qualify fromtraining only,notwhichheldcases itwouldfix.',
        limits='Feature distances/support/gradient alignment descriptive,not proof of causality orgeneralization. Correlatedprefix/seed records notindependent. No heldtargettuning. Originalfixed data/controllers/fees/labels remain;final8/physics/FRAME/paper untouched.'))
    paths=[W/'run.py',O/'design.json',S/'complete.json',S/'independent_audit.json',S/'gate_predictions.json',S/'changed_choices.json',S/'calibration_receipts.json',S/'search_metrics.json',S/'family_summary.json',S/'paired_summary.json',D/'rows.json',D/'complete.json',R/'outputs/grid223_remaining_support/row_support.json',META]
    for root,folder in [(Q,'qualified_inputs'),(P,'fit_inputs'),(S,'models')]:paths+=list((root/folder).glob('*.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify()
    diag={(r['seed'],r['sample_id'],r['position']):r for r in read(D/'rows.json')};near={(r['seed'],r['sample_id'],r['position']):r for r in read(R/'outputs/grid223_remaining_support/row_support.json')};pred=[r for r in read(S/'gate_predictions.json') if r['base_continue']];metrics=read(S/'search_metrics.json');n={v['date']:v['cases'] for v in metrics['GATE_seed0']['per_date']};records=[];loss_calls=collections.defaultdict(lambda:dict(rows=0,weighted_loss=0.,seeds=set(),sample_ids=set()));decomp=[]
    for r in pred:
        key=r['seed'],r['sample_id'],r['position'];d=diag[key];positive=r['actual_next_NET']>0;kept=r['budget_continue'];oldkept=r['stop_only_continue'];kind=('kept_' if kept else 'missed_')+('profitable' if positive else 'unprofitable');weight=1/(3*16*n[r['date']]);x=dict(seed=r['seed'],fold=r['fold'],sample_id=r['sample_id'],position=r['position'],call=r['physical_state_group'],date=r['date'],actual_NET=r['actual_next_NET'],actual_gain=r['actual_next_gain'],queries=r['actual_next_queries'],kind=kind,old222_kept=oldkept,new230_kept=kept,score=r['logit'],old222_score=d['arms']['STOP_ONLY']['score'],prior=d['b'],date_weight=weight,score_margin_below_zero=-r['logit'] if not kept else None,originalSTOPsupport=near[key]['neighbors'])
        records.append(x)
        if positive and not kept:
            c=loss_calls[x['call']];c['rows']+=1;c['weighted_loss']+=weight*r['actual_next_NET'];c['seeds'].add(r['seed']);c['sample_ids'].add(r['sample_id'])
    assert len(records)==610;prof=[r for r in records if r['actual_NET']>0];lost=[r for r in prof if not r['new230_kept']];assert len(prof)==55 and len(lost)==16
    stablelost=[r for r in lost if not r['old222_kept']];newlost=[r for r in lost if r['old222_kept']];restored=[r for r in prof if r['new230_kept'] and not r['old222_kept']];assert (len(stablelost),len(newlost),len(restored))==(14,2,2)
    call_rows=[dict(call=k,rows=v['rows'],weighted_loss=v['weighted_loss'],seeds=sorted(v['seeds']),unique_states=len(v['sample_ids'])) for k,v in sorted(loss_calls.items(),key=lambda kv:-kv[1]['weighted_loss'])]
    for seed in range(3):
        rr=[r for r in records if r['seed']==seed and not r['new230_kept']];saving=sum(-r['actual_NET']/(16*n[r['date']]) for r in rr if r['actual_NET']<=0);loss=sum(r['actual_NET']/(16*n[r['date']]) for r in rr if r['actual_NET']>0);actual=metrics[f'BUDGET_seed{seed}']['macro']['mean_net_utility']-metrics[f'GATE_seed{seed}']['macro']['mean_net_utility'];assert abs(saving-loss-actual)<1e-12;decomp.append(dict(seed=seed,saving=saving,lost=loss,total=actual))
    meta=read(META);train=[]
    for f in range(4):
        for seed in range(3):
            pool=f'f{f}_s{seed}';q=np.load(Q/'qualified_inputs'/f'{pool}.npz');stop=q['domain_STOP'];z=np.load(P/'fit_inputs'/f'{pool}.npz');assert np.array_equal(q['calibration_X'],z['X']);b=q['b'];prob=np.exp(-np.logaddexp(0,-b));w=q['weight']/q['weight'].sum();block={}
            for name,sl in [('latent83',slice(0,83)),('public41',slice(83,124)),('prior1',slice(124,125))]:
                x=q['X'][:,sl];s=x[stop];aux=x[~stop];outside=((aux<s.min(axis=0))|(aux>s.max(axis=0))).any(axis=1);gw=w*(prob-q['y']);g0=x[stop].T@gw[stop];g1=x[~stop].T@gw[~stop];block[name]=dict(STOP_max_abs=float(abs(s).max()),CONT_max_abs=float(abs(aux).max()) if len(aux) else None,CONT_rows_outside_anySTOP_column_range=int(outside.sum()),CONT_rows=len(aux),CONT_risk_mass_outside_anySTOP_range=float(w[~stop][outside].sum()/w[~stop].sum()) if len(aux) and w[~stop].sum()>0 else None,zero_correction_gradient_STOP_norm=float(np.linalg.norm(g0)),zero_correction_gradient_CONT_norm=float(np.linalg.norm(g1)),zero_correction_gradient_cosine=cosine(g0,g1))
            train.append(dict(pool=pool,active=bool(q['active_fit'][0]),STOP_rows=int(stop.sum()),CONT_rows=int((~stop).sum()),STOP_positive=int(q['y'][stop].sum()),CONT_positive=int(q['y'][~stop].sum()),STOP_risk_weight=float(w[stop].sum()),CONT_risk_weight=float(w[~stop].sum()),blocks=block))
    current=read(S/'changed_choices.json');unnecessary=[r for r in current if r['actual_next_NET']<=0];assert len(unnecessary)==11
    losses=sum(v['weighted_loss'] for v in call_rows);summary=dict(cohort=610,original_profitable=55,retained_profitable=39,missed_profitable=16,missed_calls=len(call_rows),shared_with222_misses=14,new_misses_vs222=2,restored_vs222=2,top_loss_call=call_rows[0]['call'],top_loss_call_share=call_rows[0]['weighted_loss']/losses,total_macro_lost_profitable=losses,per_seed=decomp,new_fits=0,new_scalars=0,new_policy_evaluations=0)
    for name,v in [('cohort_records.json',records),('missed_profitable.json',lost),('changed_profitable_vs222.json',dict(newly_missed=newlost,restored=restored)),('loss_by_call.json',call_rows),('training_domain_support.json',train),('unnecessary_restorations.json',unnecessary),('summary.json',summary)]:save(name,v)
    verify();save('review.json',dict(passed=True,all610_sealed_records_and_old_support_joined=True,all55_profitable_records_retained=True,saving_loss_identity_checked_allseeds=True,all12owntrainingdomain_summaries_checked=True,new_fits=0,new_NNforwards=0,new_policy_evaluations=0));save('complete.json',dict(passed=True,read_only=True,cohort=610,profitable=55,new_fits=0,new_scalars=0,new_physics=0,final8_absent=True))
    lines=['# GRID231：剩余误停与训练支持','','这轮只核查已有记录，没有重新训练或试新的停止规则。','',
    f"原208会补查的55条回本记录中，新230保留39条、漏掉16条，涉及{len(call_rows)}个调用。与旧222相比，14条漏查相同，另外漏掉2条、恢复2条。说明上一轮控制力度修复了最大的新增损失，但原先存在的漏查并没有随之解决。",
    f"当前漏查损失最多的调用是{call_rows[0]['call']}，占日期宏平均回本损失的{100*summary['top_loss_call_share']:.2f}%。这些记录含同调用前缀和不同种子，不能按16次独立失败理解。",'',
    '相对222新增的两条漏查分别来自5月18日与10月5日、种子0；恢复的两条来自11月16日、种子2。两处跨种子重复省查询仍保留。相对227恢复补查的11条不回本记录也全部列出，不能只报告大收益恢复。','',
    '## 训练内的支持差异','',
    '已分别记录各池STOP与辅助CONT的风险权重、正负样本、输入范围，以及零修正点的梯度方向。辅助CONT虽按半半风险混合，却来自停止前状态；增加这些样本并不等于增加了同样多的真实停止决策正例。',
    '原83维派生/缓存表示块在辅助状态中出现远大于STOP的标准化值；公共41维摘要的极值较小。这支持资格核对较简单的修正器，但不能据此宣称公共摘要更可预测。原强模型先验仍保留全部已有信息，候选限制只作用于额外修正器。',
    '下一步按此前已提出的公共41维+b方案做训练内资格：固定42输入、43参数，继续使用原半半风险及完整单位L2；同时沿用230训练查询代价约束和STOP正例保护。它有意限制修正器容量，既不是同容量消融，也不是为了某个外层失败例子调特征。资格通过后才固定一次11模型+11联合标量的有限训练。',
    '当前分析不能证明输入范围差异导致误停，邻居支持也不能证明可泛化。所有外层已暴露，只称开发比较；费用、标签、控制器和最终8等边界不变。']
    (O/'GRID231_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');dest=R/'论文协作资料/FRAME_残余误停分析_GRID231';dest.mkdir(exist_ok=False)
    for n in ['GRID231_report.md','summary.json','loss_by_call.json','training_domain_support.json','review.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg=f"GRID231只读610/55回本记录完成：230保留39、漏16/{len(call_rows)}调用，与222共漏14、新漏2、恢复2。主要残余损失{call_rows[0]['call']}占{summary['top_loss_call_share']:.2%}；训练12池STOP/CONT风险/尺度/零点梯度已核对，不将域支持描述当因果。下一步232训练内资格固定公共41+b的42输入43参数修正器，沿用原风险/先验及230联合查询代价+正例约束，资格前不拟合。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID231残余误停与拟合支持\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':first,rest=t.split('\n',1);t=first+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID231后\n\n1. 231只读完成，230/231及旧完整项不重做。16漏查与222共享14条，主残余问题不是刚新增的alpha放大。\n2. 232仅训练内资格：公共41+b、42输入43参数固定修正器，原半半风险/完整单位L2/强先验；联合原STOP正例与230取消查询代价约束。用原训练及固定合成问题资格，不读取外层新策略结果。\n3. 资格后233最多11模型+11联合标量、600秒，旧230/222/208复用；全部模型及标量先发、全部预测先发后连标签。只固定一方案，不按已看外层扫阈值/正则/种子或挑特征。\n4. 最终8/物理/电网/论文/新FRAME不动，原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保留。用户持续授权直至喊停。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID231残余误停分析\n\n[报告](protocols/'+A.name+'/GRID231_report.md)；下一步只做固定输入限制的训练内资格。\n\n').encode('utf8')+p.read_bytes());print('GRID231_COMPLETE',json.dumps(summary))
if __name__=='__main__':main()
