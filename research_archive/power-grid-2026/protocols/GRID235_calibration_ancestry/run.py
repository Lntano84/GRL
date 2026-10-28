"""Qualification only: trace label ancestors and fix one conditional residual probe."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid235_calibration_ancestry';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID235_calibration_ancestry'
H=R/'outputs/grid214_inner_head_generation';G=R/'outputs/grid213_inner_head_plan';Q=R/'outputs/grid224_context_support_plan/revision02';P=R/'outputs/grid220_full_prior';META=R/'outputs/grid113_relational_learning/train_metadata.json';FULL=R/'outputs/grid192_gate_support'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    assert not O.exists() and read(R/'outputs/grid234_view_disagreement/complete.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID235',scope='Onlyancestryandfixedinner8/4support qualification;zero fits/newforwards/newpolicyscores. All36existingbundlesaudited;fixedinner0forall12pools selectedbyindexbeforeanynewscore.',
        questions='Can splitexistingcorrectiondatainto8/4 be independentcalibration? Trace oldb headlabels,baseNET/STOPselection,scalers,risknormalization,parentbudget,andcorrectionfit/calibration overlap. Distinguishinnercal overlap from actualouter4labelisolation.',
        fixed_next='One conditionalresidualprobe,not fullynested:reuse existinginner0 eight-dateevent+amount_depth4heads andtheirfit-onlyscalers toregenerate all12owntrainingdate priors/selection,withoutrefits/NNforwards. FreezeupstreambaseNET/STOPpopulation. Then fit public42residual onlyon8dates,and score4dates notusedbyresidual. Firstcollectuncalibrated layererrors;doNOTfitselectororcalibrationyet.',
        total_operations='Inputregeneration0fits/reuses24forestheads;subsequent atmost11new43parameterlinear models+1KEEP,0scalarfits,0NNfits,0physicalqueries. Single600s modelwindow,noextension/search. No hidden phases addingmodel/scalarops. Stopafterinternal4dateerrors;noouter4policyevaluation.',
        limits='BaseNET/STOPpopulation learnedouter12inclinner4 remainsconditioned,not independentOOF/blind/fullpipelinevalidation. Sameheadforallrows fixes new crossheadpriorcontamination only. Old222budgetnot inherited orcalibrated inthis diagnostic. Newprior/selection/scalingandfit8 explicitly differ from233;not pureout-of-fitcalibrationablation or claimed betterpolicy.',
        barriers='Publicexportsourcefirstpublished;existingheadweightpublicationverified;allregeneratedpublicpredictionspublishedbeforelabelsjoined. Allnewresidualweightsbefore4datepredictions,all4datepredictionsbefore4datelabeljoin. No final8/physics/paper/FRAME,newouterevaluation orheld-selectedparameters.'))
    meta=read(META);bundles=read(H/'bundles.json');graph={r['bundle']:r for r in read(G/'dependency_graph.json')};assert len(bundles)==36
    prior_sources={};allrecords=[];selected=[];old_overlap_rows=0;direct_overlap_rows=0;sourcepaths=[W/'run.py',O/'design.json',META,H/'bundles.json',H/'pre_score_publication.json',H/'trained_freeze.json',G/'dependency_graph.json',G/'qualified_freeze.json',P/'qualification.json',R/'outputs/grid234_view_disagreement/complete.json',R/'outputs/grid222_protected_step/calibration_receipts.json',R/'work/grid214_inner_head_generation/run.py',R/'work/grid213_inner_head_plan/run.py',R/'work/grid192_gate_support/run.py',R/'work/grid208_public_amount/run.py',R/'work/grid220_full_prior/run.py',R/'work/grid224_context_support_plan/run.py']
    for bundle in bundles:
        name=bundle['bundle'];f=bundle['outer_fold'];s=bundle['seed'];pool=f'f{f}_s{s}';g=graph[name];fit=set(bundle['fit_dates']);cal=set(bundle['score_dates']);outer=set(bundle['outer_evaluation_dates']);assert len(fit)==8 and len(cal)==4 and len(outer)==4 and not fit&cal and not (fit|cal)&outer;assert set(g['new_head_label_dates'])==fit and set(g['new_scaler_dates'])==fit and set(g['all_ancestor_label_dates'])==fit|cal and not outer&set(g['all_ancestor_label_dates'])
        path=R/g['baseNET_path'];assert sha(path)==g['baseNET_sha256'];z=np.load(H/'labelled_samples'/f'{name}.npz');margin=(z['event_p']*z['mu4']-z['public_charge'])/(z['event_p']*z['mu4']+z['public_charge']);sourcepaths += [path,H/'labelled_samples'/f'{name}.npz',G/'qualified_inputs'/f'{name}.npz']
        for k,i in enumerate(z['meta_ids']):
            i=int(i);assert meta[i]['date'] in cal;assert (pool,i) not in prior_sources;prior_sources[pool,i]=dict(bundle=name,head_dates=fit,b=float(margin[k]),actual_NET=float(z['actual_NET'][k]),original_STOP=bool(z['original_STOP_mask'][k]))
        allrecords.append(dict(bundle=name,fit_dates=sorted(fit),calibration_dates=sorted(cal),outer_dates=sorted(outer),baseNET_calibration_label_overlap=sorted(cal&set(g['baseNET_label_dates'])),head_calibration_label_overlap=sorted(cal&set(g['new_head_label_dates'])),outer_ancestor_label_overlap=[]))
    for b in bundles:
        if b['inner_fold']!=0:continue
        f,s=b['outer_fold'],b['seed'];pool=f'f{f}_s{s}';fit=set(b['fit_dates']);cal=set(b['score_dates']);full=np.load(FULL/'fit_features'/f'{pool}.npz');q=np.load(Q/'qualified_inputs'/f'{pool}.npz');old=np.load(P/'fit_inputs'/f'{pool}.npz');sourceids=full['ids'].astype(int);assert {meta[i]['date'] for i in sourceids}==fit|cal
        overlap=[];fit_rows=cal_rows=0
        for i,v in zip(q['ids'],q['b']):
            i=int(i);origin=prior_sources[pool,i];assert v==origin['b'];date=meta[i]['date']
            if date in fit:
                fit_rows+=1;shared=sorted(cal&origin['head_dates']);old_overlap_rows+=bool(shared);overlap.append(dict(index=i,source_bundle=origin['bundle'],calibration_label_ancestor_dates=shared))
            else:cal_rows+=1;assert not cal&origin['head_dates']
        assert all(len(r['calibration_label_ancestor_dates'])==4 for r in overlap)
        stop_support={};old_selected_support={}
        for role,dates in [('fit8',fit),('cal4',cal)]:
            ids=[i for i in sourceids if meta[i]['date'] in dates and prior_sources[pool,int(i)]['original_STOP']];oldids=[int(i) for i in old['ids'] if meta[int(i)]['date'] in dates];stop_support[role]=dict(all_originalSTOP_rows=len(ids),profitable_before_new_head_filter=sum(prior_sources[pool,int(i)]['actual_NET']>0 for i in ids));old_selected_support[role]=dict(rows=len(oldids),positive=sum(prior_sources[pool,i]['actual_NET']>0 for i in oldids));direct_overlap_rows+=len(oldids) if role=='cal4' else 0
        for head in ['event','amount_depth4']:sourcepaths.append(H/'models'/f"{b['bundle']}_{head}.pkl")
        sourcepaths += [FULL/'fit_features'/f'{pool}.npz',Q/'qualified_inputs'/f'{pool}.npz',P/'fit_inputs'/f'{pool}.npz',R/'outputs/grid208_public_amount/models'/f'{pool}.npz']
        selected.append(dict(pool=pool,bundle=b['bundle'],fixed_inner_fold=0,fit_dates=sorted(fit),calibration_dates=sorted(cal),outer_dates=b['outer_evaluation_dates'],source_originalSTOP_count=int(full['original_STOP_count']),source_rows=len(sourceids),old_mixed_fit8_rows=fit_rows,old_mixed_cal4_rows=cal_rows,old_fit8_prior_rows_with_cal4_head_label_ancestors=len(overlap),old_selected_STOP_support=old_selected_support,potential_STOP_support_before_new_head_filter=stop_support,warning='Theseoldselectedcountsarenotneweligibility;freshheadregenerationrequired.',head_models=[str((H/'models'/f"{b['bundle']}_{head}.pkl").relative_to(R)) for head in ['event','amount_depth4']],scaler_file=str((G/'qualified_inputs'/f"{b['bundle']}.npz").relative_to(R))))
    assert len(selected)==12;summary=dict(existing_bundles_audited=36,fixed_inner0_pools=12,old_mixed_fit8_rows_with_cal4_prior_head_ancestors=old_overlap_rows,old_STOP_cal4_rows_directly_used_in_residual_fit=direct_overlap_rows,all36_actual_outer4_ancestor_unions_disjoint=True,baseNET_cal4_overlap_in_all36=True,full_pipeline_independence_qualified=False,conditional_residual_probe_qualified=True,actual_new_fits=0,new_forwards=0,new_calibrations=0)
    ancestry=[dict(component='frozenbaseNET_hidden_STOPpopulation',direct_label_dates='ownouter12',overlaps_fixed_inner_cal4=True,action='Freezeanddeclareconditional,doNOTclaimfullpipelineOOF'),dict(component='original208scalers',direct_label_dates='none_direct_featurestatistics;samplingdependsNETouter12',overlaps_fixed_inner_cal4=True,action='Reuseexisting213inner8scalersforthenewheadandpublic42diagnostic'),dict(component='oldperrowmixedpriors',direct_label_dates='row-specific214inner8',overlaps_fixed_inner_cal4=True,action='Regeneratefit8andcal4withsamefixedinner0head'),dict(component='oldriskweights',direct_label_dates='absNETanddomainnormalizationallouter12',overlaps_fixed_inner_cal4=True,action='Recomputefit8-onlydomainnormalizationafterpublicpredictionssealed'),dict(component='old222querybudget',direct_label_dates='residualfitandquerydecisionsallouter12',overlaps_fixed_inner_cal4=True,action='DoNOTinheritbudgetinfirstconditionalerrorcollection;noalphafityet'),dict(component='oldresidualandpositivecalibration',direct_label_dates='sameouter12',overlaps_fixed_inner_cal4=True,action='Newresidualfit8only;holdcal4labelsasideuntilscorespublished')]
    for n,v in [('ancestry_components.json',ancestry),('existing_bundle_ancestry.json',allrecords),('fixed_probe_pools.json',selected),('summary.json',summary)]:save(n,v)
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(sourcepaths)})
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    save('qualification.json',dict(passed=True,scope='conditional_residual_layer_only',full_pipeline_independence=False,all36_outer4_ancestor_label_sets_disjoint=True,old_crosshead_and_parentbudget_leakage_relative_innercal_declared=True,fixed_inner0_not_chosen_bylabels=True,regenerate_priors_and_fit8_weights_before_newfit=True,final_eligibility_support_not_yet_generated=True,total_new_residual_models_cap=11,total_scalar_fits_cap=0,total_new_forest_NNfits=0,all_future_fit_cap_s=600,actual_fits=0,new_forwards=0));save('complete.json',dict(passed=True,qualification_only=True,new_fits=0,new_forwards=0,new_physics=0,final8_absent=True))
    lines=['# GRID235：校准数据的来源与隔离资格','','本轮没有训练或生成新预测，只核查36个既有分组的数据祖先，并固定所有池都使用inner0分组。','',
    '## 原外层结果与内部校准要分开看','',
    '36个分组的原外层4日期都没有进入既有模型的标签祖先。这次核查没有发现把原外层日期拿回训练的问题。',
    '但若直接把原12天修正器数据再切为8天拟合、4天校准，隔离并不成立：基模型和STOP样本筛选已经使用这12天；旧修正器本身也拟合了拟作校准的4天。',
    f"更具体地说，在固定inner0下，旧混合数据中拟作8天训练的{old_overlap_rows}条记录，其先验来自其他inner收益头，标签祖先都包含拟作校准的4天。旧STOP中另有{direct_overlap_rows}条拟作校准的记录已经被原修正器直接拟合。", 
    '原222的预算同样由12天拟合模型及其训练决策产生，不能继续继承它，再称校准4天完全未参与。','',
    '## 固定一个范围清楚的小诊断','',
    '下一步冻结基模型和原STOP样本总体，复用每池既有inner0的8天收益头与8天标准化参数，重新生成该池12天记录的先验。8天训练行和4天待核查行都使用同一对收益头，不混用其他inner头。根据新先验重新确定候选资格，不能直接沿用旧筛选后的记录。',
    '先保存并发布这些公共预测，再连接原有训练标签和统计8天支持。后续至多训练11个43参数公共修正器，4天不进入这个修正器的拟合；只收集修正层未拟合记录上的错误，暂不拟合步长或二级选择器，也不做新的外层策略比较。',
    '整个限定诊断复用24个已有森林头，新增森林、神经网络和物理查询均为0；新增修正器模型上限11、标量拟合0，单次模型总窗口600秒，不延长或换分组。若自身8天STOP正例不足，按预先规定保留常量继续并记录，不另挑分组。','',
    '## 能支持什么结论','',
    '这只能称为固定上游条件下的修正层留组诊断。基模型与STOP总体仍使用12天的信息，不能称全链嵌套、独立OOF标签或盲测。重新生成先验、标准化和训练分组也改变了样本分布，不能把它当成与233只差一个校准步骤的消融。',
    '先确认这种限定诊断能提供什么错误支持，再决定校准怎么学。不按已经看到的外层结果选分组、步长或组合策略。原费用、标签、基础控制器及最终8、物理仿真、论文、FRAME比较均不动。']
    (O/'GRID235_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8');A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');dest=R/'论文协作资料/FRAME_校准数据来源_GRID235';dest.mkdir(exist_ok=False)
    for n in ['GRID235_report.md','ancestry_components.json','fixed_probe_pools.json','summary.json','qualification.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg=f"GRID235仅资格、0拟合/0预测。36原outer4与全部标签祖先无交集；但固定inner0时旧混合fit8有{old_overlap_rows}条先验的收益头标签祖先含cal4，旧STOP cal4 {direct_overlap_rows}条已被修正器拟合，父222预算也含cal4。固定唯一条件诊断：复用inner0两头及8天scalers重生成12天先验/资格，先发布公共预测再连标签；后续最多11个43参修正器只fit8，score4只收集错误，0标量/新森林NN/物理，不做新outer策略评估。上游NET/STOP仍outer12，明确非全链独立。"
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID235校准祖先与限定诊断资格\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID235后\n\n1. 235资格完成，0拟合/0预测；不重做旧模型。36原outer4祖先隔离保持，内部cal4不能直接当独立样本。\n2. 236仅公共输入再生成：用235固定inner0两既有头及213inner8scalers，读取192缓存83/214raw41和公共charge，生成全部12天先验b及42输入/角色/STOP资格，0拟合/NNforward。核对旧214 cal4预测重现。源先发，全部24头公共预测先发后连标签；训练标签由214旧连续NET复用，不重新仿真。\n3. 236发布后统计新headCONT的fit8与cal4 STOP/CONT支持，并只在fit8分别归一化原结构质量×absNET风险半半。不能继承旧跨头b、outer12风险权重或父222预算。若fit8 STOP正例0，预定KEEP不选另fold；f0_s1 KEEP保留。\n4. 后续237最多11个43参公共修正器、600秒，标量拟合0，原L-BFGS-B与单位L2；权重先发、4天分数先发后连4天标签，仅收集错误，不做新outer4策略/alpha选择。上游NET/STOP仍outer12，条件层诊断不称独立OOF。\n5. 全限定诊断总操作上限已公开：0新森林/NN/物理、<=11残差模型、0标量；不悄悄拆大搜索。最终8/论文/FRAME不动，原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保留；Git严格等helper退出。自主推进至喊停。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID235校准来源核查\n\n[报告](protocols/'+A.name+'/GRID235_report.md)；原外层日期隔离保持，下一步限定修正层留组诊断。\n\n').encode('utf8')+p.read_bytes());print('GRID235_QUALIFIED',json.dumps(summary))
if __name__=='__main__':main()
