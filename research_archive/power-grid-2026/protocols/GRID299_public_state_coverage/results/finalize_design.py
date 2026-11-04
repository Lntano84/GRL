"""Design-only response to explicit new-training-date planning choice. No generator imports."""
from pathlib import Path
import collections,datetime as dt,hashlib,json,math,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';W=Path(__file__).parent;O=R/'outputs/grid299_public_state_coverage';P=R/'outputs/grid297_paired_collection_preparation';A=I/'research_archive/power-grid-2026/protocols/GRID299_public_state_coverage';C=R/'论文协作资料/FRAME_新增训练日期方案_GRID299'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def monday(year,month,ordinal):
 first=dt.date(year,month,1);first+=dt.timedelta(days=(-first.weekday())%7);return first+dt.timedelta(days=7*(ordinal-1))
assert not (O/'decision.json').exists() and read(O/'complete.json')['passed']
s=read(O/'summary.json');rr=read(O/'resource_estimates.json');oldsplit=read(R/'outputs/grid100_data_foundation/split_manifest.json');oldroles=read(P/'pool_roles.json')
oldtrain=sorted(d for d,role in oldsplit['calendar_role'].items() if role=='train');assert len(oldtrain)==16
mapping={};dateitems=[]
for original in oldtrain:
 d=dt.date.fromisoformat(original);ordinal=1 if d.day<=7 else 3;assert d==monday(d.year,d.month,ordinal)
 new=monday(2038,d.month,ordinal).isoformat();mapping[original]=new
 seeds=[int.from_bytes(hashlib.sha256(('GRL300_NEWDATE_PLAN_V1|'+new+'|'+key).encode()).digest()[:4],'little')%(2**32-1) for key in ['load','renewable','forecast']]
 dateitems.append({'date':new,'original_role_slot':original,'month':d.month,'monday_ordinal':ordinal,'global_role':'NEW_TRAINING_FAMILY','replicate':0,'scenario_id':100,'scenario':new+'_100','environment_seed':0,'generator_seeds':dict(zip(['load_seed','renew_seed','gen_p_forecast_seed'],seeds))})
newdates=sorted(mapping.values());assert len(set(newdates))==16
oldall=sorted(oldsplit['calendar_role']);combined=sorted(dt.date.fromisoformat(d) for d in oldall+newdates);assert all((b-a).days>=7 for a,b in zip(combined,combined[1:])), 'no overlapping seven-day calendar intervals'
finalitems=[i for i in oldsplit['scenarios'] if i['partition']=='test'];assert len(finalitems)==8 and not oldsplit['final_test_materialized']
assert all(not (R/'work/grid100_data_foundation/trainval_env/chronics'/i['scenario']).exists() for i in finalitems),'reserved scenario unexpectedly materialized; stop design'
roles=[];baseline=[]
for old in oldroles:
 role={'pool':old['pool'],'KEEP':old['KEEP'],'fit_dates':[mapping[d] for d in old['fit_dates']],'cal_dates':[mapping[d] for d in old['cal_dates']],'outer_dates_excluded':[mapping[d] for d in old['outer_dates_excluded']]}
 assert len(role['fit_dates'])==8 and len(role['cal_dates'])==len(role['outer_dates_excluded'])==4
 assert len(set(role['fit_dates']+role['cal_dates']+role['outer_dates_excluded']))==16;roles.append(role)
 for date in sorted(role['fit_dates']+role['cal_dates']):baseline.append({'pool':role['pool'],'date':date,'week':date+'_100','role':'fit8' if date in role['fit_dates'] else 'cal4','KEEP':role['KEEP'],'arm':'FROZEN261_REFERENCE','environment_seed':0,'scenario_id':100})
assert len(baseline)==144 and collections.Counter(i['role'] for i in baseline)=={'fit8':96,'cal4':48}
# The first future live replay, if approved, is a fixed fit8 member, never a cal-driven selection.
pilot=next(i for i in baseline if i['pool']=='f0_s0' and i['role']=='fit8');baseline=[pilot]+[i for i in baseline if i!=pilot]
for index,i in enumerate(baseline):i['index']=index
g=read(R/'outputs/grid100_data_foundation/generation_manifest.json');times=[x['wall_s'] for x in g['completed']];assert len(times)==40
oldmean=25309.4708868/288;genmean=sum(times)/len(times);central_opposite=s['selected4quarters']['records'];max_opposite=4*144
resources={'generation_families':16,'generation_simple_s':16*genmean,'generation_receipt_scenarios':len(times),'generation_receipt_mean_s':genmean,'baseline_episodes':144,'central_density_opposite_episodes':central_opposite,'central_total_episodes':144+central_opposite,'maximum_opposite_episodes':max_opposite,'maximum_total_episodes':144+max_opposite,'central_simple_hours':((144+central_opposite)*oldmean+16*genmean)/3600,'upper_count_simple_hours':((144+max_opposite)*oldmean+16*genmean)/3600,'proposed_single_total_window_s':86400,'proposed_generation_subcap_s':7200,'proposed_single_generation_s':1800,'proposed_episode_s':1200,'physical_steps_cap':2017*(144+max_opposite),'forecast_cap':int(9000000/288*(144+max_opposite)),'resource_passed':False,'new_year_generation_qualified':False,'not_authorization':True,'limits':'generation OPF/solver work is separate from replay physical-step/forecast counters and consumes the same24h; loading/startup/failed items included; original scope clocks not inherited; estimated averages do not guarantee completion.'}
save(O/'scope_choice.json',{'user_message_reference':'request_user_input_async/call_0a01c526d046411c9b9b25404544ee0a/0','user_instruction':'准备新增训练日期的数据方案（推荐；只设计，不启动模拟）','authorized_now':'design_only','new_environment_generation_approved':False,'new_physical_collection_approved':False})
save(O/'new_calendar_manifest.json',{'year':2038,'selection':'Same month and first/third Monday slot as original16 train dates, now2038. No calendar slots chosen by outcomes;16 distinct new synthetic date families, not16 independently sourced real-world weeks. One scenario100, no replicate sweep.','dates':dateitems,'old_to_new_slot_map':mapping,'reserved_original_test_scenario_identifiers':[x['scenario'] for x in finalitems],'reserved_test_files_read_or_generated':False,'disjoint_all_old24_calendar_intervals':True})
save(O/'new_pool_roles.json',roles);save(O/'proposed_new_baseline_manifest.json',baseline);save(O/'new_date_resource_estimates.json',resources)
refs=read(P/'model_references.json');assert len(refs)==12;save(O/'frozen_model_references.json',refs)
statics=[R/'outputs/grid100_data_foundation'/n for n in ['design.json','split_manifest.json','generation_manifest.json','generation_trainval_finished.json']]+[R/'work/grid100_data_foundation'/n for n in ['prepare.py','generate.py']]+[P/n for n in ['pool_roles.json','model_references.json','model_freeze.json','published_model_bindings.json']]
env=R/'work/grid100_data_foundation/trainval_env';statics += [p for p in env.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.py']]
save(O/'design_input_references.json',{p.relative_to(R).as_posix():sha(p) for p in statics})
proposal={'status':'DESIGN_READY_NOT_IMPLEMENTED_NOT_AUTHORIZED','new_calendar_families':16,'new_year':2038,'scope_authorized_now':'design_only','planning_question':'Can more distinct original-policy decision states on separate training weeks provide real matched suffix-cost support, without another feature/risk/threshold sweep?',
 'fixed_policy':'Original12 owner-fNN/208/261 and original normalizers/heads/model weights. Zero retraining in collection, f0_s1KEEP. No new grid, modified queryNET, reward, ordering, aliases, initial lock,128/64/stagnation2/.9.',
 'generation':'Separate proposed work/grid300_new_training_dates/train_env with byte-identical existing static grid and generation settings except allowed dates. Existing Chronix2Grid generator and16 fixed seed triples; only proposed2038 scenario100. No writes to old trainval_env, no old/test chronics copied as new. Static source/API compatibility for2038 not tested. One declared training generation pilot then remaining15, all within original new24h window; no substitution/reseed.',
 'baseline':'All144 frozen original261 references on new own12 dates (96fit,48cal,KEEP12). Keep all finished/failure/missing states. First approved replay is fixed f0_s0 fit8 pilot, not smoke outside budget. Publish baseline source identities and public candidate list before any opposite replay.',
 'selection':'For every baseline use all saved nativeSTOP events with pending1..64; fixed same4 quarter bins and earliest public marker per bin, no costs or future completion in selection. Up to576 opposite episodes. No bins or dates replaced for lack of events; state uniqueness and dates reported perpool before labels.',
 'opposite':'At each selected marker run exactly one episode with the opposite of original261 decision (original261CONT->STOP; original261STOP->onebatchCONT). Common original261 prefix from week start; then original261. Reuse original261 baseline as one source arm. Exact full joint action/observation prefix hashes and original model/context/source/norm aliases checked before intervention. Observation/action vectors only identity inputs, never treated as restorable environment/optimizer snapshot.',
 'publication':'Source/model/date/role/seed/hash and exact scope approval before generation or replay. Baseline raw receipts/identity manifest/public marker set first, then opposite source records; all raw source receipts published before distinct saved label derivation. Raw traces/models localD; Git only allowed code/references/receipts/derived reports.',
 'target':'Independent diagnostic C_CONT-C_STOP official raw_cost suffix to2017. Bothfull samehorizon for numeric cost target; safety, illegal/ambiguous/exceptions and completion separate, no penalty imputation. Baseline shared across multiple states means correlated targets; no count inflation. No new labels or fits by this design.',
 'training_boundary':'No model or exact performance criterion selected from cal4/new outcomes. Use each poolfit8 labels only; cal4 cannot repair fit8 support, outer4 excluded, KEEP retained. Even32 possible newfit states/pool are an upper bound, not a guarantee or mathematically sufficient sample count. Require coverage on more than the one previously shared beneficial state/date before a finite learning hypothesis is separately qualified; no prequalified41-dimensional head. At most11models+11scalars/one600s, no rescue retrain or source/seed/smoothing/threshold search.',
 'limits':'New2038 is a synthetic calendar/seeds cohort on the same grid/profile mechanisms, not independent physical provenance. Old2037 validation is exposed regression testing, not temporal prediction/blind evaluation. Original final8 remain sealed. Original own12/internalcal4 ancestry of frozen models remains. More query observations do not prove latency or whole-week quality.',
 'stopping':'Single total24h includes generation, loads, starts, failures and all replay; generation2h subcap, per-generation30min, per-episode20min, max720episodes/1452240steps/22500000forecasts. First exceeded cap or integrity failure preserves partials and stops, no rerun/resume/window extension/date reselection. One main coverage acquisition only, no sequence of topping-up until training wins.',
 'later_validation':'Any learned candidate later requires its separate explicit full192 development scope and budget, all original dates/pools/KEEP/reverse repeats; not included in24h. Original2914h exhausted, never resume. Final8 not part of proposal.',
 'next_implementation':'Only if user asks to implement: build reviewable new-date generator manager and counterfactual-with-baseline reuse adapter/exact authorization guard, finite stub checks; do not create another inventory ladder. Ask once for specific generation+max720/24h physical scope only after implementation is concrete. Current choice is designonly, not runtime approval.'}
save(O/'new_training_date_proposal.json',proposal)
decision={'status':'OLD_DATE_TOPUP_NOT_RECOMMENDED_NEWDATE_DESIGN_READY','old_date_topup_approved':False,'new_date_collection_approved':False,'new_training_qualified':False,'reason':'Existing fourquarter96opposite option adds only19 net distinct observed states and leaves9..13 candidatefit states/pool. User chose design of new training-date coverage instead. New plan exposes real generation/baseline/opposite/resource costs; no guaranteed successful labels or improvement.','user_scope':'new_training_date_design_only','next':'Review new16date,144baseline/up to576opposite,one24h design. Implementation not begun; no generator/live predictor/model fit started.'}
save(O/'decision.json',decision)
date_rows='\n'.join('| '+x['original_role_slot']+' | '+x['date']+' | '+x['scenario']+' |' for x in dateitems)
pool_rows='\n'.join('| '+pool+' | '+str(v['fit8']['records'])+' | '+str(v['cal4']['records'])+' |' for pool,v in s['by_pool_role'].items())
report=f'''# GRID299：覆盖盘点完成，新增训练日期方案

2026-11-04用户要求继续，并明确选择“准备新增训练日期的数据方案（推荐；只设计，不启动模拟）”。本次实际工作只有公开保存调用盘点和可审阅设计，0新环境生成/物理回放、0NN/收益头预测、0费用标签、0训练。旧297/298的完整采集、数值审计和归档均未重做。

原日期四段补采的源码/输入2331项引用先由2b455fc4发布，1012份原261保存调用在{s['wall_s']:.6f}秒/600秒内首次检查。每个原pool/week只选原297首次干预动作与原261一致的那条存档作为基线，选择只依赖干预前的普通261决定，未比较后缀费用/收益或完成状态。JSON容器整体解析，实际只访问协议白名单公开字段，非物理未读取字节/全链盲测声明。

| 公开状态覆盖 | 池记录数 | 不同日期/状态键 |
| --- | ---: | ---: |
| 全部合格nativeSTOP | 408 | 71 |
| 固定四段最早事件 | 229 | 39 |
| 已有首次事件可复用 | 133 | 20 |
| 需要新增相反动作回放 | 96 | 21 |

新增记录与旧记录还有2个状态重合，所以净增加19个不同状态。96条里41fit8/55cal4，其中KEEP3；347个空时间段保留不替换。每条新点只需一条相反动作轨迹，另一臂复用其原261基线（71个基线被共享），避免重复采集基线；共享基线和跨池日期使目标相关，不能把96当独立样本。粗估2.343小时仅按原297平均，不是完成保证。以下是旧日期四段方案的候选覆盖，尚未生成新费用目标：

| 池 | fit8候选 | cal4候选 |
| --- | ---: | ---: |
{pool_rows}

**裁决：不启动这96条旧日期补采。** 每活跃池仍仅9..13个fit候选，且候选不等有效完整周标签或有利样本；它对目前费用头的信息增益有限。样本少并非数学上不能拟合，当前裁决是研究投入与可验证性的判断，不设通用样本量定理。用户已选择准备新增日期方案。

新增方案选2038年16个训练日期家族，按原2037训练日期的月份及第一/第三个周一槽位映射，使用同一网架和生成机制。所有新七日区间彼此至少14天间隔，且与原全部24个日期区间分离；原最终8场景仍无生成目录，仅核其既有标识/角色。新年份/新seed是合成覆盖扩充，不代表新的真实物理来源或统计独立。旧2037开发验证不能升级为时间外推或盲测。

| 原角色槽位 | 新训练日期 | 新场景标识 |
| --- | --- | --- |
{date_rows}

每池新fit8/cal4/排除outer4角色按原槽位原样继承；所有3个模型seed和固定f0_s1KEEP保留，原NN12/208头/261残差及其scaler、归一化和模型文件不重训。新日期上并未验证运行兼容性，不会用参考NN8隐藏代替原父头输入。16份生成器load/renewable/forecast种子三元组通过固定scope/date/channel哈希形成，已落清单，不能按生成质量/成本再挑seed。

| 拟议工作 | 条数上限 | 口径 |
| --- | ---: | --- |
| 新场景生成 | 16 | 每日期scenario100一次，无复制场景扫种子 |
| 原261基线回放 | 144 | 全12池各新own12；96fit/48cal，KEEP12 |
| 相反动作回放 | 576 | 每基线最多4个固定时间段状态，每点只补一条 |
| 总物理回放 | 720 | 基线共享；没有事件不替补，不是720个独立场景 |
| 新费用模型训练 | 0 | 采集后仍需支持资格和唯一有限学习协议 |

新基线先完整保存并发布公开成员与哈希，才确定各段最早nativeSTOP标记；相反动作episode从原周起点重放原261共同前缀，在该处一次STOP/CONT反转，之后回原261。完整联合动作（含连续优化）和观测/模型/输入/排序前缀必须一致，不能拿观测向量当可恢复环境/controller快照。每个新fit日期最多4个目标，单池最多32个新fit目标；这是上限，事件缺失、失败、零费用差仍会降低信息量，**并不承诺采完就能训练或改进**。

资源用原297平均{oldmean:.3f}秒/episode以及40份原生成收据平均{genmean:.3f}秒/场景粗估。若新日期密度类似旧日期229个四段事件，约373回放加16生成，粗估{resources['central_simple_hours']:.2f}小时；全部时间段都有事件的720回放情形粗估{resources['upper_count_simple_hours']:.2f}小时。两者均非预算通过，新年份生成、后续动作和机器负载都可能改变时间。

**拟议单次总上限24小时**，包含加载/启动/生成/所有成功与失败回放；生成子上限2小时、每场景30分钟，每episode20分钟；回放最大1452240物理步/22500000所有forecast。生成器OPF/solver工作另记并计同一总窗，不藏在回放步数中。先到限即停并保存部分，无续窗、换日期、重复已付费项目或自动恢复。原291和297预算不可沿用。当前只设计，尚无该范围授权，也没有实际生成器或真实环境smoke。

成本目标继续为官方raw_cost的共同事件后缀C_CONT-C_STOP，只有两臂完整同2017步才有数值费用标签；失败完成、安全异常独立列出，不靠短周费用“胜出”。全部原始source先发布，再另定一次保存标签派生，不能借cal4正例补fit8，不按日期/调用/seed选例外。若新fit仍只一个有利状态/日期或质量目标不可识别，就结束该成本学习路线，不继续逐批加数据救结果。未来每学习组件至多11模型+11标量/单600秒，不扫模型/阈值/风险/正则或重做参考NN。

若今后要求实施，下一步只做本方案的可审阅生成器/基线复用adapter/授权与总截止守卫，再一次确认具体16生成+最多720回放/24h范围；不再增加盘点链。任何新候选的192条开发验证预算另列，未包括或授权；原最终8不读标签、不生成，新电网/论文/依赖不动。原NET费用奖励和控制器排序别名锁站128/64停滞2/.9不变，旧own12/internalcal4祖先和暴露相关性限制保留；局部查询或单批计时不能声称实际提速。
'''
(O/'GRID299_report.md').write_text(report,encoding='utf8')
msg='GRID299公开盘点1012调用/8.289秒：408候选/71状态，固定四段229池记录/39状态，旧日期补96条仅净增19状态、训练每池9..13候选，不推荐启动。用户明确仅设计新增训练日期：2038同月份周槽16新日期，原12池新own12角色/KEEP保持，144基线+至多576相反动作=720回放，粗估约'+f"{resources['central_simple_hours']:.1f}"+'至'+f"{resources['upper_count_simple_hours']:.1f}"+'小时、拟总24h上限；0生成/模拟预测/标签/训练，实施与新物理范围尚未授权。'
hand='# 当前：GRID299新增日期设计完成，未实施或启动\n\n'+msg+'\n\n先读GRID299_report/new_training_date_proposal/new_calendar_manifest/new_pool_roles/new_date_resource_estimates/scope_choice/decision/publication_review（交付后）。用户仅批准设计，不写physical_authorization、不启动生成器/manager/worker；不重复299盘点、297采集/归档或298目标。下一若用户要求实施，直接准备已确定方案的可审阅代码和精确守卫，再一次询问新物理范围，不额外盘点或扫方案。\n'
(O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8')
save(O/'NEXT_PROTOCOL_PLAN.json',{'status':'NEWDATE_DESIGN_DELIVERED_IMPLEMENTATION_NOT_STARTED','authorized_now':'design_only','new_physics_approved':False,'new_training_qualified':False,'next':'On explicit implementation request build this one generator/baseline-reuse/opposite adapter, exact scope guard and manager; only after reviewable code ask once for new16date/max720/24h physical range. No repeated inventories or old-date topup.','automation':'PAUSED_NO_UNATTENDED_SIMULATION'})
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n;t=p.read_text(encoding='utf8');p.write_text(('# 当前：GRID299新日期方案已完成，只设计未启动\n\n'+msg+'\n\n以下保留历史。\n\n'+t) if n=='RESEARCH_STATE.md' else t+'\n\n## 2026-11-04 — GRID299覆盖盘点/新增日期方案\n\n'+msg+'\n',encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前：GRID299新增日期方案已交付\n\n'+msg+'\n\n1. 完成报告与具体Gitblob/D协作核验，不重复公开盘点。\n2. 当前用户仅授权方案设计，无新生成、物理或训练；grl保持PAUSED。\n3. 用户若要求实施，直接准备固定方案代码/模型数据绑定/总时限与范围守卫，审阅完后才一次确认新物理预算。旧日期96条方案不启动。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID299新增训练日期方案（只设计）\n\n[报告](protocols/GRID299_public_state_coverage/results/GRID299_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
shutil.copy2(O/'CONTINUOUS_HANDOFF.md',R/'outputs/grid298_saved_suffix_targets/CONTINUOUS_HANDOFF.md')
S=A/'results';S.mkdir(exist_ok=False);C.mkdir(parents=True,exist_ok=False)
files=[p for p in O.glob('*.json') if p.name not in ['input_references.json']]+[O/'GRID299_report.md',O/'CONTINUOUS_HANDOFF.md']
for p in files:shutil.copy2(p,S/p.name);shutil.copy2(p,C/p.name)
shutil.copytree(O/'member_receipts',S/'member_receipts');shutil.copy2(Path(__file__),S/'finalize_design.py')
for n in ['inventory_stdout.log','inventory_stderr.log']:shutil.copy2(O/n,S/n.replace('.log','.txt'))
save(S/'archive_manifest.json',{p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file()})
save(C/'delivery_hashes.json',{p.name:sha(p) for p in files})
print('GRID299_DESIGN_DELIVERED',len(dateitems),len(baseline),resources['central_simple_hours'],resources['upper_count_simple_hours'],'ZERO_GENERATION_PHYSICS')
