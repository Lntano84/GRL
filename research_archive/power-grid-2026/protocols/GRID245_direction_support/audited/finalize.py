from pathlib import Path
import sys,json,shutil,collections
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO;C.verify();assert C.read(O/'complete.json')['passed'] and not (O/'independent_audit.json').exists();rows=C.read(O/'held_contributions.json');names=C.read(O/'feature_names.json');train=C.read(O/'fit_contributions.json');protected=C.read(O/'protected_contributions.json');cats=C.read(O/'category_summary.json');checks=0
for source,folder in [(train,'fit_inputs'),(protected,'calibration_inputs')]:
 for pool in sorted({r['pool'] for r in source}):
  z=np.load(C.Q/folder/f'{pool}.npz');p=np.load(C.P/'models'/f'{pool}.npz');lookup={int(i):j for j,i in enumerate(z['ids'])}
  for r in [r for r in source if r['pool']==pool]:
   j=lookup[r['id']];terms=np.r_[z['X'][j]*p['theta'][:-1],p['theta'][-1]];assert np.array_equal(terms,np.array(r['contributions']));checks+=1
for r in rows:
 assert len(r['contributions'])==43 and abs(r['b12']+r['alpha']*sum(r['contributions'])-r['score'])<1e-12
assert checks==2936 and len(rows)==610
rawpath=R/'outputs/grid214_inner_head_generation/public_raw41.npz';before=C.sha(rawpath);raw=np.load(rawpath)['values'];sem=dict(public_source=rawpath.relative_to(R).as_posix(),sha256=before,rows=len(raw),mean_reward_plus_mean_rho_min=float((raw[:,18]+raw[:,20]).min()),mean_reward_plus_mean_rho_max=float((raw[:,18]+raw[:,20]).max()),mean_reward_plus_mean_rho_distance_from2=float(abs(raw[:,18]+raw[:,20]-2).max()),rewardstd_rhostd_maxdifference=float(abs(raw[:,19]-raw[:,21]).max()),descriptive_not_feature_selection=True);assert C.sha(rawpath)==before;C.save('public_semantics_check.json',sem)
miss=[r for r in rows if r['kind']=='profitable_cancel'];cancel=[r for r in rows if r['kind']=='nonprofitable_cancel'];jan=[r for r in miss if r['seed']==2 and r['call']=='2037-01-05_101|call001'];extra=dict(misses22=len(miss),misses_any_training_range_departure=sum(bool(r['outside_fit_features']) for r in miss),misses_top5contains_training_range_departure=sum(any(t['outside_fit'] for t in r['top_negative']) for r in miss),misses_largest_negative_outsidefit=sum(r['top_negative'][0]['outside_fit'] for r in miss),nonprofitable_cancellations243=len(cancel),cancellations_any_training_range_departure=sum(bool(r['outside_fit_features']) for r in cancel),all55profitable_outside_atleastone_selectedpositivefeature_range=all(r['outside_protected_features'] for r in rows if r['actual_next_NET']>0),jan5five_public_rewardmean_range=sorted({next(t['value'] for t in r['top_negative'] if t['feature']=='admissible_reward_mean') for r in jan}),jan5_rewardmean_training_min=next(t['fit_min'] for t in jan[0]['top_negative'] if t['feature']=='admissible_reward_mean'))
C.save('interpretation_support.json',extra);C.save('decision.json',dict(retain230main=True,retain243seed1tradeoff=True,retain243learned_nonprofitable_cancellation_region=True,no_simple_OOD_gate=True,no_featuredrop_fromheld=True,no_claim_duplicatefeature_causesfailure=True,next='GRID246qualificationonly: all41 public semantic identities, ownfit8 rank/scales and risk/objective/offsetunits, no newouterpolicy/model. Useonlyowntraininglabels/publicdefinitionstodetermineoneboundedmatchedcandidate,not sweepfeature subsets or clip thresholds.'))
lines=['# GRID245：固定修正器为什么取消补查','','本轮将已发布修正器的43个项逐项展开：41个公共特征、父先验修正项和截距。核对全部610次原GATE继续决定，以及2851条fit8输入和85条经验保护正例。0拟合、0新收益头推理、0新策略评分。','','## 取消成功与失败用的是哪些信号','','| 既有决定类别 | 记录数 | 修正方向中位数 | 至少一项超出自身fit8输入范围 |','|---|---:|---:|---:|']
labels={'profitable_keep':'回本，保留','profitable_cancel':'回本，误停','nonprofitable_cancel':'不回本，取消','nonprofitable_keep':'不回本，保留'}
for r in cats:lines.append(f"| {labels[r['kind']]} | {r['rows']} | {r['direction']['median']:.6f} | {r['any_fit_range_departure']} |")
lines+=['','243确实学会了一部分省查询的行为：243条不回本追加被取消，最大负向项经常来自剩余补查规模、下一批查询量和截距。保留这个有效区域，不能因整体下降抹去它。','','22条回本误停中有14条至少一个特征超出fit8输入范围；但243条有效取消中也有37条如此。33条回本保留中也有13条。因此单靠“有特征超出训练范围”不能决定是否停止。更窄的85条保护正例范围也无效：全部55条回本外层记录，都至少一个特征超出该范围。','','只有5条误停的前5个负向项含超出fit8范围的特征；没有一条的最大负向单项超出该特征训练范围。外推是具体案例线索，尚不能解释全部误停。','','## 主要损失组的具体计算','','种子2、1月5日call001的5个相关前缀，平均奖励标准化后约−7.83至−8.02，而自身fit8最低约−5.14；平均负载方向相反，也超出训练范围。两项分别贡献约−0.35，叠加约−0.70。负载限制项另约−0.395，其他项继续叠加，最终形成过强的取消方向。','','这5条记录的父先验仍在保护正例的一维范围内；失败来自多个公共项组合后的修正量，不能只解释成父先验低。它们来自同一调用和相关前缀，不是5个独立场景。','','已有公共41表示还有近似确定关系：3582条公开输入中，平均奖励＋平均负载≈2，最大偏差1.2e−7；奖励标准差与负载标准差最大偏差6.0e−8。相关信息在两列里重复出现，单位L2对这种表示的惩罚含义也会随之改变。不过单项贡献依赖表示，不能把两列都指认为独立原因，更不能据此宣布删列就会改善。','','## 下一步与保留结论','','下一步246先核对全部公共特征的语义关系、各池自身fit8的尺度/秩与风险权重，并核对父先验作为修正目标偏置时的量纲。只使用公开定义与自身训练支持确定唯一有限方案；不按外层误停挑删特征，不扫描裁剪阈值，不试多个策略选最好。','','230仍是主要内部候选；243种子1的小幅净效用改善、243条有效取消都保留。现在的目标是减少过强修正造成的收益损失，同时保住这些小改善，而非给某个调用加例外。','','逐项计算重现全部610个发布分数；2851条训练输入和85保护正例的贡献逐项核对，源数据与权重哈希不变。所有日期、种子、反例保留。这里是已暴露开发记录上的离线解释，不是因果消融或完整周/实际时延证据。最终8、物理仿真、电网、论文和FRAME比较均未新增。']
(O/'GRID245_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');C.verify();C.save('independent_audit.json',dict(passed=True,all610publishedscores_reproduced=True,all2936training_constraint_rows_contributions_reproduced=True,all43termsretained=True,sources_models_unchanged=True,public_semantics_source_sha256=before,zero_fit_forward_policy=True));C.archive('audited');dest=R/'论文协作资料/FRAME_修正方向核查_GRID245';dest.mkdir(exist_ok=False)
for n in ['GRID245_report.md','category_summary.json','interpretation_support.json','public_semantics_check.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/n,dest/n)
msg='GRID245固定43项分解完成，0拟合/推理/新策略；610分数、2851fit8和85保护贡献重现。22误停14有任一fit8范围外特征，243有效取消也37有，55回本全超至少一项正例保护范围，不作简单OOD门。Jan5种子2主要5相关前缀平均奖励/负载各约-.35叠加，平均奖励标准化-7.83至-8.02低于fit最低-5.14；但最大负向单项均在训练范围内，非完整解释。公共奖励均值+负载均值≈2、标准差近重复，单位L2表示含义需核查，不据外层删列。下一246仅全41语义/自身fit尺度秩/风险及偏置量纲资格，固定唯一有限方案后训练。保留230、243种子1及有效省查询区域。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID245修正方向与支持\n\n'+msg+'\n'
 if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
 else:t+=entry
 p.write_text(t,encoding='utf8',newline='\n')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID245后\n\n1. 242/243拟合评分审计交付、244/245只读分析完成，不重做。230仍主候选，保留243种子1及243有效取消区域。\n2. GRID246仅资格0拟合/新策略：核对全部41公共特征定义中的确定关系、各池自身fit8尺度/秩和原风险目标/父先验偏置量纲。不要按外层挑删特征或裁剪阈值，不把Jan5线索当所有误停共同原因。只依据公共定义与自身训练标签/支持固定唯一有限匹配方案，并明确容量、目标或经验约束改变后再实施。\n3. 245重现610原parentCONT决定；22误停14任一fit范围外，243有效取消37范围外，55回本均超至少一项正例范围，简单OOD门不成立。相关奖励/负载表达不是独立因果，不能直接删列试多个组合。\n4. 各学习阶段最多22模型/标量、单次600秒，不延长盲扫。原费用标签/全种子日期/控制器/排序别名/锁站/128/64/停滞2/.9保持，最终8/物理/电网/论文/FRAME不动。源码输入、模型步长、预测分阶段先发；Git严格串行等helper退出。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID245修正方向与支持核查\n\n[报告](protocols/'+A.name+'/audited/GRID245_report.md)；保留有效取消，先查表示与训练目标，再固定有限学习方案。\n\n').encode()+p.read_bytes());handoff='''# GRID245后持续交接

先读docs RESEARCH_STATE/NEXT_STEPS，以最新为准；已有运行器跟进，完整项不重做。242、243拟合评分审计交付及244/245只读完成。243为11线性+11步长(.052s)，回本保留33/55、漏22，整体不替换230；种子1小折中及243有效不回本取消保留。

244：22653同状态，22漏查7低于/4高于/11在85保护正例b范围内，旧239共同7仍全漏。cal4正例高裕度在父12/12和inner8 11/12池都存在，不单归为父内部乐观。

245：610父CONT决定、2851fit8和85保护贡献重现，全43项保留。22误停14有任一fit范围外特征，243有效取消也37有；全部55回本超至少一项保护正例范围，不作简单OOD门。Jan5种子2主要5相关前缀平均奖励/负载各约-.35叠加，均值标准化约±8超fit约±5.14，但所有误停最大负项都在训练单维范围内，不能单因解释。3582公共输入平均奖励+负载≈2(偏差1.2e-7)、标准差近重复，单位L2对表示的含义需核查，不按外层挑列。

下一步GRID246仅资格，0拟合/新预测/新策略：核对全部41公共特征公开定义中的确定关系、各池自身fit8尺度/秩、原风险权重与父先验作为修正偏置的量纲。仅据自身训练标签及公开支持固定唯一有限匹配学习方案，改变容量/目标/经验约束先明确记录；不试多组合外层择优，不扫描裁剪阈值/alpha，不给Jan5例外。不预设删冗余或裁剪有效，不立即重训。

每学习阶段<=22模型/标量、单次600秒，不延长/盲扫/重做完整项。源码输入先发，模型步长先发，全部预测先发后接结果。原费用标签/全种子日期/控制器/排序别名/锁站/128/64/停滞2/.9不变，最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。全为已暴露开发，局部代理不等整周/实际时延，不将多查询质量改善冒充等预算。按AGENTS更新docs/归档首页，实际源码A归档，git_push.cmd推送，严格等helper退出后下一helper/屏障。244交付W变量工程错已保留并仅补齐交付，非当前故障，无科研重跑。用户授权自主推进直至喊停，同步grl起点，不恢复旧自动化或归档聊天。
''';(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
archives=[]
for p in A.glob('*/archive_manifest.json'):
 for n,h in C.read(p).items():assert C.sha(p.parent/n)==h
 archives.append(dict(path=p.relative_to(REPO).as_posix(),sha256=C.sha(p)))
copies=[]
for p in dest.iterdir():assert C.sha(p)==C.sha(O/p.name);copies.append(dict(path=p.relative_to(R).as_posix(),sha256=C.sha(p)))
C.save('delivery_check.json',dict(passed=True,archives=archives,collaboration=copies,source_model_freezes_unchanged=True));d=A/'delivery';d.mkdir(exist_ok=False)
for p in [O/'CONTINUOUS_HANDOFF.md',O/'delivery_check.json',C.W/'finalize.py']:shutil.copy2(p,d/p.name)
(d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');print('GRID245_DELIVERED',json.dumps(extra))
