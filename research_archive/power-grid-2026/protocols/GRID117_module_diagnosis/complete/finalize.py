"""Independent NumPy component checks and publication of read-only diagnosis."""
from analyze import *
import shutil,math
REPO=R/'work/GRL-results-integration'
def numpy_variants(w,r,arm):
    def lin(x,k):return x@w[k+'.weight'].T+(w[k+'.bias'] if k+'.bias' in w else 0)
    def enc(x,k):return np.maximum(lin(np.maximum(lin(x,k+'.0'),0),k+'.2'),0)
    a=enc(r['context_action'],'delta.A');f=enc(r['context_feedback'],'delta.F');q=enc(r['next_action'],'delta.A');anchor=enc(r['anchor_action'],'delta.A')
    vj=enc(np.concatenate([a,f],-1),'delta.V');vf=enc(np.concatenate([a,np.broadcast_to(f.mean(0),f.shape)],-1),'delta.V')
    score=lin(q,'delta.Q')@lin(a,'delta.K').T/math.sqrt(8);ex=np.exp(score-score.max(1,keepdims=True));alpha=ex/ex.sum(1,keepdims=True);unif=np.ones_like(alpha)/len(a)
    g=enc(np.concatenate([r['global41'],anchor,r['known_query_cost']]),'delta.G')
    base=r['global41'].astype(float)
    for i in [0,2,4]:
        base=lin(base,f'base.q_net.{i}')
        if i!=4:base=np.maximum(base,0)
    out=dict(base=base)
    for k,att,v in [('actual',alpha,vj if arm=='BASE_JOINT' else vf),('factorized_same_weights',alpha,vf),('uniform_attention',unif,vj if arm=='BASE_JOINT' else vf),('both',unif,vf)]:
        ctx=att@v;hh=enc(np.concatenate([q,ctx,np.broadcast_to(g,(len(q),16))],-1),'delta.H');h=np.concatenate([hh.mean(0),hh.max(0),g])
        delta=lin(np.maximum(lin(h,'delta.head.0'),0),'delta.head.2');out[k]=base+delta;out['delta_'+k]=delta
    return out
def audit():
    assert not (O/'review.json').exists();configure();verify116();A=np.load(O/'component_forward_outputs.npz');meta={p:read(S113/f'{p}_metadata.json') for p in ['train','validation']};rows={p:pickle.loads((S113/f'{p}_public.pkl').read_bytes()) for p in meta};div=read(O/'first_divergences.json');count=0;maximum=0.;near=[]
    for rec in read(S116/'fits.json')['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";cp=torch.load(S116/'models'/f'{name}.pt',map_location='cpu',weights_only=False);w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()}
        for part in rows:
            y=np.load(S113/f'{part}_targets.npz')['target'];lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta[part])}
            indices=set([0,len(y)-1]+list(range(0,len(y),151))+list(np.flatnonzero(y>0)))
            if part=='validation':indices.update(lookup[r['sample_id'],r['position']] for r in div if r['seed']==rec['seed'])
            for i in sorted(indices):
                out=numpy_variants(w,rows[part][i],rec['arm'])
                for k,q in out.items():
                    original=A[name+'__'+part+'__'+k][i];e=float(np.max(abs(q-original)));maximum=max(maximum,e)
                    assert np.allclose(q,original,atol=3e-5,rtol=3e-5),(name,part,i,k,e)
                    if k in ['actual','factorized_same_weights','uniform_attention','both']:
                        if abs(float(original[1]-original[0]))>3e-5:assert bool(q[1]>q[0])==bool(original[1]>original[0])
                        else:near.append(dict(method=name,partition=part,index=i,variant=k,original_margin=float(original[1]-original[0]),numpy_margin=float(q[1]-q[0])))
                    count+=1
    # Independently locate first disagreement before either stop, using original saved scores.
    Q=np.load(S116/'state_predictions.npz');cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta['validation'])};expected={}
    for seed in range(3):
        for c in cases:
            if c['partition']!='validation':continue
            for j in range(len(c['states'])):
                i=lookup[c['sample_id'],j];jq=Q[f'BASE_JOINT_seed{seed}__validation'][i];fq=Q[f'BASE_FACTOR_seed{seed}__validation'][i];ja=jq[1]>jq[0];fa=fq[1]>fq[0]
                if ja!=fa:expected[seed,c['sample_id']]=j;break
                if not ja:break
    assert len(expected)==len(div)
    fee=read(S116/'design.json')['unchanged']['cost_per_query'];scale=read(S116/'design.json')['unchanged']['gain_scale'];cm={c['sample_id']:c for c in cases}
    for r in div:
        assert expected[r['seed'],r['sample_id']]==r['position'];c=cm[r['sample_id']];j=r['position'];assert abs((c['gains'][j]-fee*c['queries'][j])/scale-r['next_batch_net'])<1e-12
    calls=read(O/'physical_call_contributions.json');tra=read(S116/'validation_trajectories.json');tm={(r['method'],r['sample_id']):r for r in tra};dc=collections.Counter(c['date'] for c in cases if c['partition']=='validation');recompute=collections.defaultdict(float)
    for seed in range(3):
        for c in cases:
            if c['partition']!='validation':continue
            j=tm[f'BASE_JOINT_seed{seed}',c['sample_id']];f=tm[f'BASE_FACTOR_seed{seed}',c['sample_id']]
            recompute[c['physical_state_group']]+=(j['net_utility']-f['net_utility'])/(3*len(dc)*dc[c['date']])
    for r in calls:assert abs(recompute[r['physical_state_group']]-r['net_contribution'])<1e-12
    for r in read(O/'additive_identifiability_check.json')['records']:assert r['total_decisions_unchanged'] and r['max_total_score_error']<1e-12
    assert all(sha(R/n)==h for n,h in read(O/'input_manifest.json').items());verify116();sealed()
    save(O/'near_zero_audit_scores.json',near)
    save(O/'review.json',dict(passed=True,independent_numpy_component_vectors=count,max_abs_error=maximum,near_zero_variant_scores_retained=len(near),
        first_actual_common_prefix_divergences_independently_checked=len(div),physical_call_weights_recomputed=True,additive_identity_checked=True,
        all_saved_weights_and_frozen_sources_unchanged=True,new_fits=0,backward_calls=0,optimizer_instances=0,new_physical_calls=0,final_test_materialized=False,
        scope='Endpoint function/decision accounting, not independent training-cause or physical AC verification.'))
def publish():
    assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();s=read(O/'component_summary.json');a=read(O/'accounting_summary.json');overview=read(O/'divergence_overview.json');ids=read(O/'additive_identifiability_check.json')
    lines=['# GRID117：配对版为什么多查、却没有多赚','',
        '**目前最明确的原因是：整套模型对不少无利补查给出了过高的继续价值，尤其种子1；而当前配对编码和注意力对这些决定影响很小。错误并不是把推理时的配对关系去掉就能修复。主干和修正共同训练，当前设计也不能唯一判定是哪一部分造成了训练偏差。**','',
        '本轮读取既有六份固定权重、3582训练/856验证状态和完整搜索序列，核对第一次实际分歧、计算分支输出，并做三项固定权重的函数诊断。无训练、反向传播、优化器更新、电网预测或新回放，最终测试保持封存。','',
        '## 1. 多花的查询确实集中在无利批次','',
        'JOINT相对FACTOR的平均净效用差为−0.00250819。收益多得了+0.00029344，但多付查询费用−0.00280163，收益不足以抵消费用。按原四日期、种子等权记账，种子1贡献−0.00216331，约占净差的86.25%；其余种子有正有负，全部保留。',
        '对每条起始前缀，只检查两组都还在搜索时的第一次不同决定，避免拿已经停止的一方与另一方后续状态强行比较：','']
    for seed in range(3):
        r=overview[str(seed)];lines.append(f"- 种子{seed}：JOINT继续、FACTOR停止有{r['joint_continues_factor_stops']}条前缀；其中下一批无正净收益{r['those_next_batch_unprofitable']}条，来自{r['unprofitable_distinct_calls']}次物理搜索。")
    lines += ['',
        '种子1的99条来自31次搜索，不是99个独立电网场景。每条前缀都有自己的可见反馈与剩余集合，已完整记录实际下一批收益、查询和分支分数。未来批次反馈只用于事后核对，从未输入模型。','',
        '按物理搜索汇总，最大的几个负贡献如下；金额均为归一化搜索净效用对家族均值的贡献，不是整周成本：','']
    for r in a['most_negative_calls'][:4]:lines.append(f"- {r['physical_state_group']}，实际控制步{r['steps']}：贡献{r['net_contribution']:.8f}，其中查询费部分{r['fee_contribution']:.8f}，收益部分{r['gain_contribution']:.8f}。")
    lines += ['',
        '完整56次物理搜索中，38次JOINT贡献为负、14次为正、4次相同；正反清单均保留。上述是同一保存路径上的精确记账，不是物理运行的因果干预。','',
        '## 2. 配对关系和注意力不是这些错误的直接开关','',
        '固定所有已训练参数，分别只做三项函数诊断：把V编码改为先汇总反馈、把注意力换成均匀权重、同时做两项。其他公开输入、主干、候选与阈值均不变。这些是假设性函数检查，不是重新训练的对照，也不按它们的效用挑选新部署策略。','']
    for seed in range(3):
        e=s[f'BASE_JOINT_seed{seed}']['validation'];v=e['variants'];lines.append(f"- 种子{seed}：856个验证状态中，临时先汇总改变{v['factorized_same_weights']['decision_sign_changes']}个决定（正收益{v['factorized_same_weights']['positive_sign_changes']}个）；均匀注意力改变{v['uniform_attention']['decision_sign_changes']}个；同时处理改变{v['both']['decision_sign_changes']}个。")
    lines += ['',
        '尤其种子1那99条额外无利查询，三项诊断都没有把任何一条改成停止。单独改变推理时的配对/注意力路径，不能解释或修复其大部分成本差；差异已经体现在两套训练权重所形成的完整停止边界中。这不排除训练过程中聚合方式影响了梯度和最后的权重，当前只有终点权重，无法重建完整形成过程。',
        '三个JOINT模型在多记录验证上下文中的注意力归一化熵中位数约0.99952、0.99967、0.99980，接近均匀；更直接的证据是，改成完全均匀后，三个种子的验证决定都没有变。但其输出数值会略变，不能说注意力严格为零或所有任务里都无用。',
        '也核对了“配对编码是否完全线性化”的解释：满足共同ReLU区域这一足够条件的训练状态只有39/40/40个，主要是单记录上下文；不能声称配对信息在数学上全部被线性平均抹掉。相反，多数上下文仍有非线性，只是对最终继续/停止的作用较弱。','',
        '## 3. 为什么不能简单说是主干错了，或修正项错了','',
        '按当前保存参数分工，种子1有69条额外无利前缀，是修正项把主干原本的停止翻成继续；种子2则相反，修正项主要抑制查询，在50条差异前缀里都是汇总版进一步关掉了查询。不同种子的模块角色并不一致。',
        '不过，“哪一项翻转了决定”不是唯一的训练原因。损失只约束两项相加后的价值；主干多报一个常数、修正项少报同一个常数，总预测与所有决定完全不变。两边最后都有可调偏置，模型设计允许这种重新分工。',
        '本轮只在保存输出上验证该代数恒等式，没有改动模型：把继续分数中的0.25在两项间对消，完整输出不变，但“修正项是否翻转了主干”的计数会变化。因此，不能把主干单独拿出来当一份已独立训练的BASE，也不能把修正值直接解释为真实额外收益。',
        '当前修正分支还直接接收原41维汇总，与主干有重复信息来源；优化器可以利用它拟合普通状态统计，而不必学会动作—反馈的精细关系。这是代码允许的路径，诊断也显示关系通道的决定作用有限；但尚未独立证明这个路径是全部退步的唯一原因。','',
        '## 4. 能下什么结论，下一步应解决什么','',
        '可以确定：差距主要是无利查询；种子1占主要份额；在这些错误位置，推理时改聚合或去掉非均匀注意力，几乎不起作用。可以排除“注意力有一点偏差，所以直接买错了一大批查询”这种过于简单的解释。',
        '还不能确定：训练动态为什么让这套关系模型形成更差的停止边界；现有终点无法把共同训练后的主干与修正唯一分开。也不能据此断言公开结构信息不可预测，或者只要再训练更久就会好。GRID108的长训练/MSE反例仍保留。',
        '我的判断是先停止增加网络复杂度。如果继续研究模块，先把“相对哪份稳定基础预测、究竟要修正什么”定义清楚，使模块目标与基础能力能分开检查；同时检查整体继续价值对无利批次的高估。固定基础网络后学习校正可以作为下一方案考虑，但需要重新匹配预训练、额外更新和对照预算，本轮没有启动它。',
        '全部结论来自同一电网、四个已暴露验证日期、保存的FULL教师搜索路径，不代表实际控制时延、整周质量或跨网结论。','',
        '## 复核资料','',
        '- first_divergences.json：所有301条第一次共同前缀分歧及分支分数。',
        '- physical_call_contributions.json：56次物理搜索的正反贡献及相关前缀数量。',
        '- component_summary.json、component_forward_outputs.npz：完整训练/验证分支与三项固定权重函数诊断。',
        '- accounting_summary.json：多查位置、成本贡献和种子1份额。',
        '- additive_identifiability_check.json：总输出不变、分支分工可变的代数核验。',
        '- review.json、input_manifest.json：独立NumPy前向、第一次分歧和物理调用权重核验；所有旧权重与封存源保持不变。']
    report=O/'GRID117_report.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    decision=dict(extra_unprofitable_continuation_localized=True,seed1_share_of_net_gap=a['seed1_share_of_net_gap'],
        seed1_bad_extra_prefixes=99,seed1_bad_extra_distinct_calls=31,inference_factorization_or_uniform_attention_repairs_seed1_bad_extras=0,
        backbone_vs_residual_training_cause_not_identified=True,all_encoder_affine_collapse_not_supported=True,
        no_new_fits=True,next='Define interpretable correction relative to a stable base before new fitting; separately match pretraining/extra-update budget and controls.',ready_to_scale=False)
    save(O/'decision.json',decision)
    arc=REPO/'research_archive/power-grid-2026/protocols/GRID117_module_diagnosis/complete';collab=R/'论文协作资料/FRAME_模块原因核查_GRID117';arc.mkdir(parents=True,exist_ok=False);collab.mkdir(exist_ok=False)
    save(O/'complete.json',dict(passed=True,scope='Read-only module/cost diagnosis, not a new trained improvement.',report=str(report),archive=str(arc),collaboration=str(collab),new_fits=0,new_physical_calls=0,ready_to_scale=False,final_test_materialized=False))
    for p in O.iterdir():
        if p.is_file():shutil.copy2(p,arc/p.name)
    for p in W.glob('*.py'):shutil.copy2(p,arc/p.name)
    for n in ['GRID117_report.md','decision.json','review.json','complete.json','accounting_summary.json','divergence_overview.json']:shutil.copy2(O/n,collab/n)
    save(arc/'archive_manifest.json',{p.name:sha(p) for p in arc.iterdir() if p.is_file()})
    note='## 2026-10-23 — GRID117 frozen module/cost diagnosis complete, no fits\n\nAll6GRID116weights fixed.301firstactualcommon-prefixdivergences checked;extra Jcontinue/Fstop next-unprofitable56/99/50prefixes from26/31/20calls. Seed1accounts86.25%ofnetJFgap. TopnegativeOctober19_100step702call003. At fixedJOINTweights factorizedV changes2/1/6validationdecisions,uniformattention0/0/0;none repairsseed1bad99extras. Attentionnearuniform butaffineVcollapse sufficientconditiononly39/40/40trainstates,mostlysingletons,so notliteralcompleteaffinecollapse. Storedbase+delta69seed1badextrasflipbycorrection,seed2rolesopposite; additivegauge qbase+c,delta-c keepsalltotaloutputs/decisions while changingcomponentattribution,so branchtrainingcause notidentified. Duplicate41path allowsordinarysummaryfitting;notprovenuniquefailurecause.IndependentNumPyvariants/divergences/callweightauditpasses;allweights/sourcehashesunchanged,zerooptimizer/backward/fits/physical/test. Next definecorrectionrelativestablebase andmatchpretraining/updatebudgetbeforeanyfit;noautonewnetwork/scaling.'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+note+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID117模块原因核查完成，未新增训练\n\n1. 汇报多查定位、推理配对/注意力弱作用及主干—修正不可唯一归因的限制。\n2. 若继续，先定义相对稳定基础预测的校正目标；固定主干等方案需另行匹配预训练、额外更新与对照预算，训练前给具体方案。\n3. 不自动训练、扩网络、改费用/门槛、上电网闭环或读取最终测试。\n\n报告：research_archive/power-grid-2026/protocols/GRID117_module_diagnosis/complete/GRID117_report.md\n',encoding='utf8')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID117模块原因核查](protocols/GRID117_module_diagnosis/complete/GRID117_report.md)：零训练，成本位置、首次分歧、固定权重函数诊断与加法分工不可辨识核验。\n')
    print('GRID117_AUDITED_PUBLISHED_NO_FITS',flush=True)
if __name__=='__main__':audit();publish()
