"""Close one saved-only diagnosis and publish its bounded research decision."""
from pathlib import Path
import collections,hashlib,json,shutil
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid292_activation_support';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID292_activation_support';C=R/'论文协作资料/FRAME_实际激活诊断_GRID292'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert read(O/'review.json')['passed'] and not (O/'decision.json').exists()
s=read(O/'activation_public_summary.json');rows=read(O/'public_actual_rows.json');own=read(O/'ownfit_public_support.json');labels=read(O/'ownfit_label_support.json');eligible=[r for r in rows if r['parent_CONT']];rejected=[r for r in rows if not r['parent_CONT']]
assert len(eligible)==57 and len(rejected)==272 and all(38 in r['public_support']['ALL'] for r in rejected)
detail=dict(parent_positive=57,unique_parent_positive_public_states=len({r['public_signature'] for r in eligible}),parent_positive_KEEP=sum(r['residual_KEEP'] for r in eligible),nonKEEP_eligible=sum(not r['residual_KEEP'] for r in eligible),unique_nonKEEP_eligible=len({r['public_signature'] for r in eligible if not r['residual_KEEP']}),parent_nonpositive=272,nonpositive_outside_b_coordinate=272,eligible_outside_any_ALLfit=sum(bool(r['public_support']['ALL']) for r in eligible),eligible_outside_coordinate_counts=dict(collections.Counter(k for r in eligible for k in r['public_support']['ALL'])),eligible_known_batch_lengths=dict(sorted(collections.Counter(r['next_queries'] for r in eligible).items())),eligible_known_query_sum=sum(r['next_queries'] for r in eligible),eligible_saved_correction_sign=dict(collections.Counter('positive' if r['saved_correction']>0 else 'negative' if r['saved_correction']<0 else 'zero' for r in eligible)),actual_fit_date_overlap=sorted({r['week'][:10] for r in rows}&{d for p in own.values() for d in p['dates']}),cancelled_rows=[{k:r[k] for k in ['pool','week','repeat','step','next_queries','parent_b','saved_logit','saved_correction','public_support']} for r in rows if r['extra_cancel']])
save(O/'activation_interpretation.json',detail)
decision=dict(status='COMPLETE_READ_ONLY_BRANCH_CLOSED',new_fits=0,new_predictions=0,new_physics=0,scientific='The208 parent already rejects272/329 nativeSTOP calls. The261 residual has57 eligible row uses(30 unique public states), of which2 areKEEP;32 eligible batches have length1. Only one unique state is additionally cancelled, twice, one query each. Broad outside-range count is dominated by already-rejected b<=0, not an identified actionable residual support failure.',forbidden_next='No automatic retraining, OOD gate, threshold/alpha/risk/feature/seed sweep or further forest/cancellation repairs.',next='Only static inventory of existing March shared-cost-failure evidence; if old reports already answer it, close without a new diagnostic. Any genuinely missing evidence must receive one finite saved-data protocol before execution. No new physical or model predictions.')
save(O/'decision.json',decision)
lines=['# GRID292：实际调用范围与自身训练支持，一次只读诊断','',
 '裁决：结束当前残差取消规则微调支线，不继续训练或扫描阈值。261不是没有被调用；主要限制是208已先拒绝绝大多数补查，而剩余批次通常很短。本轮没有发现支持新OOD门、缩放或换模型的可定位证据。','',
 '## 这次核对了什么','',
 '只消费GRID291已发布的全部93条BOUNDED261完整收据及匹配父决定，包含全部12池和KEEP；index186两次原NN停止决定单列为部分记录，不进入完整时域统计。1415个源码/引用输入先封存发布916b43ba，全部实际公共成员、坐标支持和保存裕度先发布770d3860，然后才连接260各池自身fit8原标签/权重。未读新teacher/outer目标，未运行theta、NN或收益头预测，也没有新策略分数/回报、物理或拟合。',
 '原261数值源码的Q常量解析为outputs/grid260_bounded_basis_plan，继承原训练行/标签/权重和既有祖先限制，不执行旧训练或审计。291 residual_input40是pre-tanh日志，对其前38坐标映射一次；260自身fit的前38已tanh，直接复用，不双重变换。b与nativeSTOP位保持。公共字段提取明确不读取NPZ的y/actual_NET/weight/source_mass，公共发布屏障核过具体Git字节后才连接。','',
 '## 实际作用范围','',
 '| 层次 | 保存行数 | 说明 |','|---|---:|---|',
 '| 261臂原NN推理决定 | 678 | 包含继续到下一批的原NN调用 |',
 '| 原NN停止，调用208/261 | 329 | 对应176个按pool/状态/批次/X精确描述的公共签名 |',
 '| 208已拒绝补查 | 272 | 父b≤0，261禁止把父STOP改CONT |',
 '| 父CONT，261才可能额外取消 | 57 | 30个公共签名；其中2行是f0_s1 KEEP |',
 '| 排除KEEP后的实际新增作用行 | 55 | 29个公共签名，仍非独立场景 |',
 '| 261保留父CONT | 55 | 包含上述2行KEEP |',
 '| 261额外取消 | 2 | 同一物理状态两次重复，各省1次查询 |','',
 '57个父CONT的已知批长：1查询32行，7查询4行，11查询1行，13查询2行，14查询12行，29查询4行，64查询2行；账面合计509个候选。该合计只是固定已观察调用的已知批长，不是可合法取消量、可实现整周节省上界或替代策略回报。取消有收益批次可能改变后续物理轨迹，不能据此推导自动全停。',
 '这些父CONT中，保存的261修正49行正、6行负、2行零（KEEP）；只有2行过零并取消。唯一取消状态为f2_s1 / March step641，父b=0.142502967741，保存logit=-0.000895585880，下一批仅1条；两次重复一致。该状态在自己ALL/STOP/CONT单坐标范围内。没有跨seed额外取消重复。','',
 '## 为什么不能把大范围“越界”当新失败机制','',
 '329行中276行超出自身fit8至少一个前39维坐标范围，表面比例很大；其中272行的父b≤0，而原fit输入已按b>0筛选，因此它们必然在b坐标38上越界。原208已经拒绝这272行，不能把这部分当261有资格继续决策的分布失效，也不能把它们算新增省查询机会。',
 '真正57个父CONT中，只有4行超出ALL-fit至少一维范围，分别为两个August状态各两次重复（canonical坐标23或18）；53行没有单维越界。按STOP/CONT域分别比较的越界17/10也全量保留，但既有referenceNN8域与部署原NN12STOP语义不同，不能择一个范围来建门。全部39维的min/max和每行完整越界索引已公开，没有挑特征或学习距离。',
 '单维范围内不等于联合支持或可靠预测，单维范围外也不证明错误或因果。本诊断没有把距离/范围变成置信度、阈值、安全规则或新策略。四个实际日期与各自身fit日期并无重合，只说明日期来源不同，未将其冒充独立盲测；所有场景已在历史开发中暴露。','',
 '## 自身训练标签只解释原支持，不用于挑新规则','',
 '12池原fit2859行，其中活跃11池2725、固定KEEP134；活跃STOP正例222、辅助CONT正例296保持原值。STOP正例在各活跃池占该域风险87.94%—98.11%，CONT为95.44%—99.49%。这不是正例未被看到或优化未收敛的证据。高正例风险/父偏置/函数类可能共同影响继续倾向，但本次没有反事实或重新拟合，不能将49个正向修正简单归因于某一个因素，更不能按外层去调风险比例。',
 'ownfit_label_support.json逐池保留STOP/CONT正负数、原风险、正例调用和日期数；KEEP单列，相关前缀/seed/用途不当独立样本。没有计算新的训练误差分数、极值步长或校准强度。既有own12父/内部cal4祖先与非全链OOF限制继续复用，不重新宽泛审计。','',
 '## 全部池保留','',
 '| pool | 原NN STOP调用 | 父CONT | 额外取消 | KEEP |','|---|---:|---:|---:|---|']
for p,r in s['per_pool'].items():lines.append(f"| {p} | {r['rows']} | {r['parent_positive']} | {r['extra_cancel']} | {r['KEEP']} |")
lines+=['','## 研究路线裁决','',
 '不再为这个小作用范围追加森林、阈值、先验、风险比例或特征微调。保留261与208实际轨迹/成本一致，以及一个状态两次省1查询的小线索；不把它当稳定提速，也不因效应小删除它。原291的March共同成本失败仍是更实质的质量问题，不能靠本次新增取消量来宣称已经修复。',
 '下一唯一工作单元仅静态盘点已有March质量失败证据：先查旧报告/冻结来源是否已经解释208/261与FULL成本差。如果已有解释，就直接给裁决，不重跑诊断；只有能明确指出缺少哪个已保存字段/比较，才另外固定一次有限保存数据方案。不是再次接口核对、不是新的物理对照，也不立即训练。原FULL仅原有质量参照，不新增FRAME/BATCH1/日期或最终8；不根据结果改原控制器/费用/排序/锁站/128/64/停滞2/.9。',
 '本支线190历史拟合、6续训8285updates和604.703秒/678.156授权单列不变，本次新增拟合0。全部种子、12池、KEEP、四日期和部分记录保留，不择成功子集，不设全赢/5%新门槛。','',
 '## 证据入口','',
 '- activation_public_summary.json / public_actual_rows.json / public_parent_rows.json / public_partial_rows.json：全部固定公开成员与原保存决定。',
 '- ownfit_public_support.json / public_fit：每池自身原训练公共坐标和全部39维范围；无收益标签。',
 '- ownfit_label_support.json / saved_margin_support.json / activation_interpretation.json：公共屏障后自身标签支持及固定解释。',
 '- source_freeze.json、public_freeze.json、review.json、decision.json和complete.json；D协作副本FRAME_实际激活诊断_GRID292。']
(O/'GRID292_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
nextplan=dict(id='GRID293_EXISTING_MARCH_EVIDENCE_INVENTORY_ONLY',status='FIXED_STATIC_QUALIFICATION_NOT_EXECUTED',question='Do existing published reports and originalFULL/saved291 evidence already explain the shared March cost failure? Is any genuinely new saved-data gap left?',sources='Only existing archived reports, their frozen source/manifests,291 already-published cost/quality summary and original FULL reference summary. Trace actual source constants; do not guess aliases. Start old184 first-divergence and59/61/other relevant source-linked quality diagnoses rather than redoing them.',allowed='Static evidence inventory: exact prior question/date/controller/model versions, claims and limits, source availability/hash/location. Give reuse-versus-gap judgment. No new numeric trajectory comparison until a separate finite protocol states precisely what old evidence cannot answer.',forbidden=['No old complete audits, raw archives or neural/model reruns','No new physical/FRAME/BATCH1/date/final8','No new policy returns or outcome-based thresholds','No automatic model or controller changes'],operation_caps=dict(fits=0,predictions=0,new_physics=0),stop='If prior evidence covers the question, close with explicit judgment. If a real gap exists, preregister one bounded saved-only diagnostic with all pools and cost/quality counterexamples; do not execute it as an uncounted qualification step.')
save(O/'NEXT_PROTOCOL_PLAN.json',nextplan)
msg='GRID292一次保存激活/自身fit支持诊断完成：678原NN决定/329STOP入口，208先拒272，仅57父CONT(30公共签名，其中2KEEP)；32个可作用批次仅1query。261额外取消同一状态两重复共2query。276单维范围外中272仅父b<=0落在原b>0训练域外；实际57个父CONT只有4超ALL-fit范围，不支持OOD门/重新调参。训练STOP正风险87.94%-98.11%不当正例未见或未收敛。本次0拟合/新评分回报/NN头/物理，公共源916b43ba/公共770d3860先发布后自身标签；全部12池/KEEP/部分保留。关闭残差取消微调支线，下一仅盘点已有March共同成本失败证据，若已有解释直接复用裁决，不为持续推进重做实验。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-11-01 — GRID292实际激活范围诊断\n\n'+msg+'\n'
 p.write_text('# 当前：GRID292完成，残差取消微调支线关闭\n\n'+msg+'\n\n以下保留历史。\n\n'+t if name=='RESEARCH_STATE.md' else t+entry,encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前 — 不再取消规则微调，先复用质量失败旧证据\n\n1. 292一次诊断完成，0拟合/预测/物理，报告outputs/grid292_activation_support/GRID292_report.md；不重做公共导出、标签连接、291审计或归档。\n2. 293仅按NEXT_PROTOCOL_PLAN静态盘点已有March共同成本失败的报告/源码版本/冻结来源；已知解释直接复用，不重跑。\n3. 只有明确旧证据缺少哪个已保存字段/比较，才独立预注册一次有限保存诊断，当前293不进行数值轨迹重算或新策略回报。\n4. 208/261冻结，265—284森林修补与残差阈值/风险/特征搜索关闭；无新训练/物理/日期/FRAME/最终8/依赖。291186/192预算到限不补跑。\n5. 完成本轮Git/D协作/清单/具体blob远端SHA核对，同步持续交接与自动跟进。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID292实际激活诊断完成\n\n[报告](protocols/GRID292_activation_support/results/GRID292_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
hand='# GRID292完成，当前取消微调支线关闭\n\n'+msg+'\n\n先读integration/AGENTS及docs状态/NEXT、本目录报告/review/decision/complete/NEXT_PROTOCOL_PLAN。下一293只是旧证据静态盘点，尚未执行；不重新导出292/连接标签/计算分数，不恢复任何291worker/manager或补跑。原291186/192及全部失败保留，0追加预算。已有原始数据D无损归档不重复。公共实际日志前38只tanh一次、260已tanh，当前比较已完成不重算。公共支持内外不作置信/OOD门，不按外层改alpha风险费用种子。若旧March证据已足够就裁决结束，不为自动推进无穷加阶段。新物理或新学习需要明确新信息缺口和独立有限方案，不在此盘点范围。\n'
(O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8')
save(O/'complete.json',dict(passed=True,scope='One finite saved activation/support diagnosis complete; no new method improvement claimed.',new_fits=0,new_predictions=0,new_physics=0,diagnostic_branch_closed=True))
S=A/'results';S.mkdir(exist_ok=False);C.mkdir(exist_ok=False)
names=['GRID292_report.md','activation_interpretation.json','ownfit_label_support.json','saved_margin_support.json','review.json','decision.json','complete.json','NEXT_PROTOCOL_PLAN.json','CONTINUOUS_HANDOFF.md','design.json','source_freeze.json','public_freeze.json']
for name in names:shutil.copy2(O/name,S/name);shutil.copy2(O/name,C/name);assert sha(O/name)==sha(S/name)==sha(C/name)
shutil.copy2(W/'finalize.py',S/'finalize.py');save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()});save(C/'delivery_hashes.json',{n:sha(C/n) for n in names});print('GRID292_RESULTS_WRITTEN')
