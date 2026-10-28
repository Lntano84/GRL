from pathlib import Path
import json,hashlib,shutil,sys,collections
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,REPO,A=C.R,C.O,C.REPO,C.A
C.verify();assert C.read(O/'complete.json')['passed'] and not (O/'delivery_check.json').exists();rows=C.read(O/'paired_rows.json');group=C.read(O/'group_summary.json');pool=C.read(O/'pool_summary.json');positive=C.read(O/'positive_support.json');protected=C.read(O/'protected_positive_rows.json');summary=C.read(O/'summary.json');checks=[]
for r in group:
 rr=[x for x in rows if x['role']==r['role'] and x['domain']==r['domain'] and x['positive']==r['positive']];assert len(rr)==r['n'];assert abs(sum(x['b12'] for x in rr)/len(rr)-r['b12']['mean'])<1e-12
 assert sum(x['b12']>0 for x in rr)==r['parent_continue'] and sum(x['b8']>0 for x in rr)==r['inner_continue'];checks.append(dict(role=r['role'],domain=r['domain'],positive=r['positive'],rows=len(rr)))
perpool=[]
for p in sorted({r['pool'] for r in pool}):
 d={r['role']:r for r in pool if r['pool']==p and r['domain']=='STOP' and r['positive']}
 perpool.append(dict(pool=p,parent_cal_minus_fit=d['CAL4']['b12']['mean']-d['FIT8']['b12']['mean'],inner_cal_minus_fit=d['CAL4']['b8']['mean']-d['FIT8']['b8']['mean']))
assert len(perpool)==12
miss=[r for r in positive if not r['continue_']];assert len(miss)==22 and len(positive)==55 and len(protected)==85
support=dict(parent_CAL4_positive_mean_higher_than_FIT8_pools=sum(r['parent_cal_minus_fit']>0 for r in perpool),inner_CAL4_positive_mean_higher_than_FIT8_pools=sum(r['inner_cal_minus_fit']>0 for r in perpool),perpool=perpool,selected_protected_records=85,selected_protected_unique_calls=len({r['call'] for r in protected}),selected_protected_unique_samples=len({r['sample_id'] for r in protected}),missed_unique_calls=len({r['call'] for r in miss}),missed_unique_samples=len({r['sample_id'] for r in miss}),protected_roles=dict(collections.Counter(r['role'] for r in protected)))
C.save('interpretation_support.json',support)
C.save('decision.json',dict(keep230main=True,retain243seed1tradeoff=True,no_evidence_for_single_margin_cutoff=True,no_causal_claim_parent_optimism_sufficient=True,next='GRID245read-onlysame-public41 frozen coefficient contribution/support geometry onfit8/cal4positive constraints andall existingparentCONT helddecisions. No newmodelorpolicy/alpha. Preregister allfeatures/records and matched saved models;no choosefeaturethreshold fromoutererrors.'))
lines=['# GRID244：先验分布与保护样本覆盖核查','','本轮只读核对已保存的22653条同状态记录、12个固定池、85条经验保护正例和55条有回本价值的原追加记录。没有拟合、校准、新模型推理或新策略评分。','', '## 主要发现','','先验统一没有解决修正器的泛化问题。243漏掉22条回本记录，其中7条低于自身保护正例的先验范围、4条高于范围，另外11条就在范围内。此前239共同漏掉的7条，本轮仍全部漏掉。单独按先验大小判定是否安全，缺乏支持。','','固定4天的正例更容易获得高先验，这在两套先验中都存在。原STOP正例的统计如下，所有记录在b>0筛选之前比较：','','| 数据角色 | 正例记录 | 父12天先验均值 | 固定8天先验均值 | 父先验>0记录 | 8天先验>0记录 |','|---|---:|---:|---:|---:|---:|']
for role in ['FIT8','CAL4']:
 r=next(x for x in group if x['role']==role and x['domain']=='STOP' and x['positive']);lines.append(f"| {role} | {r['n']} | {r['b12']['mean']:.6f} | {r['b8']['mean']:.6f} | {r['parent_continue']} | {r['inner_continue']} |")
lines+=['',f"逐池比较，父先验在{support['parent_CAL4_positive_mean_higher_than_FIT8_pools']}/12池呈现cal4正例均值高于fit8；固定8天先验也有{support['inner_CAL4_positive_mean_higher_than_FIT8_pools']}/12池如此。后者没有用cal4标签拟合这些收益头，因此不能将整个差异解释成父头在内部日期上过拟合。日期组成、原STOP样本分布、头的训练规模与标准化也不同，现有比较不能分离单一原因。",'', '对不回本的原STOP记录，父先验比8天先验更低：fit8平均差−0.080879，cal4平均差−0.135385；对回本记录，fit8平均差−0.102828，cal4平均差+0.040912。父头并非对所有内部记录都一味提高分数。','','## 保护范围与实际漏查','',f"85条保护记录由41条fit8和44条cal4组成，涉及{support['selected_protected_unique_calls']}个调用、{support['selected_protected_unique_samples']}个前缀样本；它们包含跨种子和相关前缀重复，不能当85个独立场景。22条漏查涉及{support['missed_unique_calls']}个调用和{support['missed_unique_samples']}个前缀样本。经验正例保护全部通过，只能说明这些已见向量不被取消。",'','主要损失所在的种子2/1月5日call001，其先验约0.380—0.586，位于该池保护正例的0.290—0.837范围内；但修正终点约−1.422至−1.841，最终分数约−0.822至−1.172。问题已经超出了先验一维取值：同样的先验区间，公共特征产生了更强的取消方向。','','因此下一步先分解固定修正器的公共特征贡献及其在训练/保护记录中的覆盖。仅用现有向量和已发布系数，检查哪些方向在漏查与无利取消上共同出现；不训练新模型，不试新阈值，也不给某个调用设例外。','','## 保留与边界','','230仍为主要内部候选。243种子1的小幅净效用改善与查询节省保留，但不挑种子替换；所有不利结果保留。','','本轮没有比较新的第三策略，也未扩大数据。父模型、基础NET/STOP和部分上游表示仍用outer12；内部cal4不是全链独立验证。原outer4标签祖先隔离保持，但这些开发记录已经暴露。局部代理效用和查询数不等于完整周成本或实际控制时延。']
(O/'GRID244_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');C.verify();C.save('independent_audit.json',dict(passed=True,group_counts_means_and_signs_recomputed=True,all85empiricalconstraints_and55publishedprofitabledecisions_checked=True,unchanged_sources_andmodels=True,no_new_fits_headforward_policy=True,group_checks=checks))
C.archive('audited');dest=R/'论文协作资料/FRAME_先验分布核查_GRID244';dest.mkdir(exist_ok=False)
for name in ['GRID244_report.md','summary.json','interpretation_support.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/name,dest/name)
msg='GRID244只读22653同状态/12池完成，0拟合/推理/新策略。243的22回本漏查7低于、4高于、11在85保护正例先验范围内；旧239共同7仍全漏。原STOP正例cal4先验均值父.472638/inner8 .431727，fit8父-.246068/inner8-.143240，父12/12池及inner8 11/12池cal均值更高，不能单因归为父先验乐观。种子2Jan5大损失先验在保护范围内而修正方向强负；下一步245只读公共41贡献/支持，不换alpha/设阈值/新策略。保留230与243种子1折中。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID244先验与覆盖核查\n\n'+msg+'\n'
 if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
 else:t+=entry
 p.write_text(t,encoding='utf8',newline='\n')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID244后\n\n1. 242/243所有拟合评分审计交付已完成，244同状态只读核查完成，不重做。230仍主候选，保留243种子1小折中。\n2. GRID245先资格固定只读分解：使用243已发布theta/alpha、242固定公共41+b、全部自身fit8/cal4保护记录和全部已有parentCONT外层决定；逐特征分解方向，比较漏查与有效无利取消的支持。0新拟合/预测/策略、0新阈值或alpha。不能按外层挑特征、删案例、设Jan5例外。\n3. 244发现一维先验覆盖不能区分11个范围内漏查；cal4正例高裕度在未用cal4标签训练的inner8头也存在，不将父内部乐观当唯一原因。下一有限学习方案只能据自身训练支持资格固定，不能直接外层择优。\n4. 每学习阶段<=22模型/标量、单次600秒，无延长盲扫。原费用标签/全种子日期/控制器/排序别名/锁站/128/64/停滞2/.9不变，最终8/物理/电网/论文/FRAME不动。源码输入、模型步长、预测分阶段先发布。Git严格串行等helper退出。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID244先验分布与保护覆盖核查\n\n[报告](protocols/'+A.name+'/audited/GRID244_report.md)；22漏查中11条先验位于保护范围内，下一步只读特征方向。\n\n').encode()+p.read_bytes())
(O/'CONTINUOUS_HANDOFF.md').write_text('''# GRID244后持续交接

先读docs RESEARCH_STATE/NEXT_STEPS，以最新状态为准。242、243拟合评分审计交付已完成；244只读22653同状态/12池完成，不重做。243回本保留33/55、漏22，整体不替换230；种子1小折中保留。244的22漏查7低于、4高于、11在自身85保护正例先验范围内；旧239共同7仍全漏。原STOP正例cal4父/inner8均值.472638/.431727，fit8为-.246068/-.143240；父12/12及inner8 11/12池cal更高，不单因解释为父内部乐观。Jan5种子2损失先验在保护范围内但公共特征修正方向强负。

下一步GRID245仅只读资格与贡献分解：先固定所有41公共特征+b项、截距、12已发布243模型和全部训练fit8/联合12保护/已有outerparentCONT记录；按已保存分数核对线性方向贡献与覆盖，漏查、回本保留、无利取消、无利保留四类全列。不能按外层挑特征或设新阈值/alpha，不训练选择器，不给Jan5例外，不评价第三策略。0新模型/标量/收益头推理/新策略分数/物理。各候选池用自身训练标签和公共支持才能提出下一有限学习方案。不能将后验特征解释当独立验证。

各后续学习阶段最多22模型/标量及一次600秒，不延长/盲扫/重做完整项。原费用、标签、全日期种子、控制器/排序/别名/锁站/128/64/停滞2/.9保持；最终8不生成或读标签，不新物理/电网/论文/FRAME，不安装依赖。旧开发记录已暴露，局部代理不等整周质量/时延，不将多查询质量提升冒充等预算。所有源码输入先发，模型步长先发，全部预测先发后连标签。按AGENTS更新docs/归档首页，使用实际A常量及git_push.cmd，等helper退出再下一helper。用户允许自主推进至喊停，更新grl跟进起点，不恢复旧自动化或归档聊天。
''',encoding='utf8')
archives=[]
for path in A.glob('*/archive_manifest.json'):
 for name,h in C.read(path).items():assert C.sha(path.parent/name)==h
 archives.append(dict(path=path.relative_to(REPO).as_posix(),sha256=C.sha(path)))
copies=[]
for p in dest.iterdir():assert C.sha(p)==C.sha(O/p.name);copies.append(dict(path=p.relative_to(R).as_posix(),sha256=C.sha(p)))
C.save('delivery_check.json',dict(passed=True,manifests=archives,collaboration=copies,previous243delivery_verified=C.read(C.C/'delivery_check.json')['passed']))
d=A/'delivery';d.mkdir(exist_ok=False)
for p in [O/'CONTINUOUS_HANDOFF.md',O/'delivery_check.json',W/'finalize.py']:shutil.copy2(p,d/p.name)
(d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');print('GRID244_DELIVERED',json.dumps(support,ensure_ascii=False))
