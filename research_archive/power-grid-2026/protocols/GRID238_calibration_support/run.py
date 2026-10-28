"""Read-only positive support; separately preregister one future matched calibration pair."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid238_calibration_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID238_calibration_support'
S=R/'outputs/grid237_residual_holdgroup';Q=R/'outputs/grid236_shared_prior_inputs';PLAN=R/'outputs/grid235_calibration_ancestry';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def entry_decision(base_continue,b8,score):
    return bool(base_continue and (b8<=0 or score>=0))
def main():
    assert not O.exists() and read(S/'complete.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID238',scope='Onlyreadonlyfit8/cal4positive support,perpointcrossingsandancestry;zero actualalpha/model fitting,newforecasts orouterpolicy scores.',
        rows='All98positiveSTOPrecords retained inclconstantKEEP row. Individual b/(b-endpoint) descriptive only;doNOTselectperpool minimumorfitnewalpha here.',
        next='SeparatelypreregisterGRID239 matchedFIT8vsCAL4scalarcalibration on SAMEfixed237weights/sharedinner8prior. Thisnewstageexplicitlyextendscompleted237probe to oldexposedouter4offline development,neverretroactivelychanges237. Notphysicalcontrol/FRAME orfinal8.',
        selection='Onefixedpair,original221maxpositiveprotectionrule,nomatchingquerybudget/nothresholdorseedselection. Allactive11pools havepositive support inbothsets;f0_s1KEEP both. Necessarytotal22scalarfits,0newmodels.'))
    meta=read(META);rolepools={r['pool']:r for r in read(PLAN/'fixed_probe_pools.json')};calrows=read(S/'cal4_error_rows.json');support=[];positive_rows=[];publicchecks=[];paths=[W/'run.py',O/'design.json',META,S/'complete.json',S/'independent_audit.json',S/'trained_freeze.json',S/'scored_freeze.json',S/'cal4_error_rows.json',S/'cal4_error_counts.json',Q/'fit_qualification.json',Q/'generated_freeze.json',Q/'qualified_fit_freeze.json',PLAN/'fixed_probe_pools.json',PLAN/'qualification.json',R/'work/grid221_step_plan/run.py']
    for pool,roles in rolepools.items():
        z=np.load(Q/'fit_inputs'/f'{pool}.npz');h=np.load(Q/'cal_public'/f'{pool}.npz');p=np.load(S/'models'/f'{pool}.npz');t=p['theta'];fallback=bool(p['fallback_KEEP'][0]);fit_end=np.ones(len(z['ids'])) if fallback else z['b']+np.sum(z['X']*t[:-1],axis=1)+t[-1];cal=[r for r in calrows if r['pool']==pool];byid={r['index']:r for r in cal};assert np.array_equal(h['ids'],np.array([r['index'] for r in cal]));r=dict(pool=pool,constantKEEP=fallback,roles={})
        for role,ids,b,end,positive,stop,allowed in [('FIT8',z['ids'],z['b'],fit_end,z['y']>0,z['domain_STOP'],set(roles['fit_dates'])),('CAL4',h['ids'],h['b'],np.array([byid[int(i)]['logit'] for i in h['ids']]),np.array([byid[int(i)]['positive'] for i in h['ids']]),h['original_STOP'],set(roles['calibration_dates']))]:
            selected=positive&stop;assert {meta[int(i)]['date'] for i in ids}<=allowed;assert np.all(b>0);calls={meta[int(ids[k])]['physical_state_group'] for k in np.flatnonzero(selected)};r['roles'][role]=dict(STOP_rows=int(stop.sum()),positive=int(selected.sum()),positive_calls=len(calls),raw_positive_missed=int((selected&(end<0)).sum()),dates=sorted(allowed))
            for k in np.flatnonzero(selected):
                delta=float(end[k]-b[k]);cross=float(b[k]/(-delta)) if delta<0 and not fallback else None;positive_rows.append(dict(pool=pool,role=role,index=int(ids[k]),date=meta[int(ids[k])]['date'],call=meta[int(ids[k])]['physical_state_group'],prior=float(b[k]),endpoint=float(end[k]),raw_missed=bool(end[k]<0),descriptive_individual_crossing=cross,constantKEEP=fallback))
        if not fallback:assert r['roles']['FIT8']['positive']>0 and r['roles']['CAL4']['positive']>0
        support.append(r)
        cached=R/'outputs/grid190_extension_gate'/f'held_{pool}.npz';existing=R/'outputs/grid220_full_prior/public_held_inputs'/f'{pool}.npz';raw=np.load(cached);e=np.load(existing);assert np.array_equal(raw['ids'],e['ids']) and raw['features'].shape==(len(e['ids']),83);assert {meta[int(i)]['date'] for i in raw['ids']}<=set(roles['outer_dates']);publicchecks.append(dict(pool=pool,outer_public_rows=len(raw['ids']),ids_match_existing220=True,outer_dates=roles['outer_dates'],new_scores=0))
        paths += [Q/'fit_inputs'/f'{pool}.npz',Q/'cal_public'/f'{pool}.npz',S/'models'/f'{pool}.npz',cached,existing,R/roles['scaler_file']]+[R/hp for hp in roles['head_models']]
    assert len(positive_rows)==98 and sum(r['role']=='CAL4' and r['raw_missed'] for r in positive_rows)==9;active=[r for r in support if not r['constantKEEP']];assert len(active)==11
    rng=np.random.default_rng(238);tests=[]
    for trial in range(100):
        base=bool(rng.integers(0,2));b=float(rng.uniform(-1,1));score=float(rng.normal());action=entry_decision(base,b,score);assert not action or base
        if base and b<=0:assert action
        if base and b>0:assert entry_decision(base,b,b)
        tests.append(dict(test=trial,passed=True))
    future=dict(id='GRID239_PAIRED_CALIBRATION',status='prospective_not_executed',authorization='Usercontinuousautonomouslearningresearch;newboundedprotocol,separatefromcompleted237.',
        objective='CompareONLYpositive-calibration source on SAME237eight-date residualweights and SAMEsharedinner8prior/scalers. FIT8positiveconstraints vsCAL4positiveconstraints. Original221choose_alpha;tiesKEEP;noouterlabelcalibration.',
        operations=dict(new_scalar_fits=22,new_model_fits=0,new_NNfits=0,new_forest_fits=0,constantKEEPfiles=2,total_calibration_cap_s=600,extra_physics=0),
        parents='Reuse12frozen237theta,24published214inner0heads,213inner8scalers. No refit/rescale/newpriorsource.',
        arms='FIT8:eachownfit8originalSTOPnet>0;CAL4:eachowncal4originalSTOPnet>0. Exactlysamefullmaximumpositiveprotectionformula. f0_s1KEEPboth;allother11 havepositive support both. No parent222querybudget,union-positive thirdarm,alpha1testarm,threshold/penaltygrid orseedselection.',
        outer_development='New239protocolonly:oldexposed16dates originalouter4cachedstates,all3seeds. ThisisNOTadditional237probework andnotblindtest. No originalouter4labelbelongs toown237residual/head/upstreamancestors.',
        public_inputs='Regeneratepublicb8and42inputs on190heldraw83+214raw41usingfixedinner0heads/scalers;noNNforwards. Verifyoriginal208base_CONTandidsagainst220publicheld.',
        entry='Preserveoriginal208CONT entry atoriginalNETSTOP,atmostsameonebatch. Ifbase_CONTfalse noappend. Ifbase_CONTtruebutb8<=0,KEEPparentdecision(explicitout-of-supportpassthrough). Ifbase_CONTtrueandb8>0,useb8+alpha*(X42theta+c)>=0. Sameeligibilityandfallbackbotharms;norescueorhandexception.',
        controls='PrimaryFIT8vsCAL4matchedpair. Frozen208GATEonlycommonentrancecontext. No claimcross230or233differencesisolatecalibration;theytrainon12anddifferinprior/scalers/constraints.',
        barriers='239source/inputsfirstpublish;all24effectiveweights+stepspublishbeforeANYnewouter4score;all24outerpredictionspublishbeforeoutcomejoin. Allcounterexamples/seeds retained.',
        metrics='Offlinefixedteacher-prefix proxyNET,rawgain/query/actionmatch/allseed/date/callresults. Extraheadinferencecostnotmeasured;noactualcontroltime/fullweekquality claim. Nopredictivecal4positiveguaranteeasgeneralizationproof.',
        prohibited='No final8/newphysicalgrid/scenarios,paper/FRAME,NNfit,newheadfit,newfees/controller/128/64/stagnation2/.9/order/aliases/stationchanges. No hiddenlargesearchsplitintostages.')
    save('future239_design.json',future);paths.append(O/'future239_design.json');save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)})
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    summary=dict(active_pools=11,constantKEEPpools=1,positive_records=98,cal4_missed_positive_records=9,active_cal4_positive_counts=[r['roles']['CAL4']['positive'] for r in active],two_pools_only_one_calpositive=sum(r['roles']['CAL4']['positive']==1 for r in active),future_scalar_fits=22,new_fits_now=0,new_outer_scores_now=0)
    for n,v in [('positive_support.json',support),('positive_records.json',positive_rows),('existing_outer_public_qualification.json',publicchecks),('synthetic_entry_checks.json',tests),('summary.json',summary)]:save(n,v)
    save('qualification.json',dict(passed=True,only_readonly_and_synthetic_now=True,no_actual_extremal_step_selected=True,all11fit_and_calpositive_support_checked=True,all12existing_outer_public_shapes_ids_dates_checked_without_forecasts=True,synthetic_entry_subset_passthrough_checked=True,future239_design_independently_preregistered=True,actual_fits=0,actual_outer_scores=0));save('complete.json',dict(passed=True,qualification_only=True,new_model_fits=0,new_scalar_fits=0,new_forecasts=0,new_policy_evaluations=0,new_physics=0,final8_absent=True))
    lines=['# GRID238：成对校准的支持与预注册','','本轮只检查正例约束及数据来源，没有求新步长、训练模型、生成新预测或试新策略。','',
    '11个实际学习池在8天拟合集和4天检查集都存在原STOP回本正例，可以用完全相同的规则比较两种校准来源。不过4天每池仅有1—7条正例，其中两个池只有1条；还存在同调用和跨池重复。它们提供了可检查的约束，不是稳定泛化的保证。','',
    '已保留全部98条原STOP正例的先验、未校准终点及单样本过零比例。9条4天误停全部包含在内；本轮没有把这些比例取极值形成新alpha。','',
    '## 唯一的后续对照','',
    '另立GRID239，固定复用237的全部8天修正器、同源8天收益头和标准化参数，只改变强度校准的正例来源：一组用自身8天拟合正例，另一组用自身4天未参与修正器拟合的正例。两组都用原最大正例保护步长规则，不扫系数或挑种子。总计22个标量、0个新模型，600秒窗口。',
    '新协议将首次在旧的、已暴露的外层4日期缓存上做这组成对离线开发比较，明确超出已结束的237错误收集范围。不会回改237，也不会生成新物理仿真或读取最终8场景。',
    '两组共享原208继续入口和同一批候选。若8天先验不在修正器训练支持域，即b8不为正，统一保留原208的决定；不额外买候选或按调用设例外。这个支持域回退本身也是新协议的共同条件，会单独报告，不能省略。',
    '主比较只在这两个校准来源之间。原230和233训练集、先验和约束均不同，不能把跨版本差异解释成纯校准效果。','',
    '所有标量与权重先发布，再生成外层公共预测；全部预测发布后才接结果。离线查询、即时增益和代理净效用分别报告；额外收益头推理成本未计时，不作实际控制时延或整周质量结论。',
    '本轮仅完成资格与预注册，尚无这组对照的实验结果。']
    (O/'GRID238_plan.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');dest=R/'论文协作资料/FRAME_成对校准资格_GRID238';dest.mkdir(exist_ok=False)
    for n in ['GRID238_plan.md','future239_design.json','positive_support.json','summary.json','qualification.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg='GRID238只读98正例资格完成，0步长/模型拟合与新外层分数。11学习池FIT8/CAL4均有正例，CAL4每池1—7、两池仅1，不能声称稳定保证。独立预注册239：复用237端点/inner8先验与scalers，FIT8vsCAL4同221最大正例保护，22标量/0新模型/600秒；新阶段才做旧暴露outer4离线比较，共享208入口、b8<=0域外保持父决定，主比较只两校准来源。不回改237或据外层挑alpha，外层预测/标签屏障固定。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID238成对校准支持资格\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID238后\n\n1. 238资格/239独立预注册完成，无新步长/模型拟合/外层分数，不重做旧模型。\n2. 实施239单一成对对照：237权重固定，FIT8与CAL4各11个原221正例保护标量，共22/600秒；f0_s1双KEEP。禁止第三臂或调alpha。\n3. 239是新协议扩展到旧暴露outer4缓存离线开发，不回改237。用190held83/214raw41/213inner8scalers及固定24头生成b8+42输入；共享208entry，b8<=0支持域外保持父决定。全部模型步长先发、全部24外层预测先发后连结果。\n4. 主比较两校准来源，208仅入口上下文；不将230/233跨版本当纯校准消融。记录域外回退、全种子/日期/反例。无新物理/NN/头拟合/最终8/论文/FRAME；代理指标不等时延/整周质量。原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保持，Git串行等helper退出。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID238成对校准资格\n\n[方案](protocols/'+A.name+'/GRID238_plan.md)；239独立预注册，当前0新拟合/外层分数。\n\n').encode('utf8')+p.read_bytes());print('GRID238_QUALIFIED',json.dumps(summary))
if __name__=='__main__':main()
