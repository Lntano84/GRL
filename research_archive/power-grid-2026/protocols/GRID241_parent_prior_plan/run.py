"""Qualification only for a controller-coherent frozen parent-prior candidate."""
from pathlib import Path
import hashlib,json,pickle,shutil,subprocess
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid241_parent_prior_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID241_parent_prior_plan'
B=R/'outputs/grid208_public_amount';E=R/'outputs/grid206_public_event';H=R/'outputs/grid214_inner_head_generation';ROLE=R/'outputs/grid235_calibration_ancestry/fixed_probe_pools.json';P=R/'outputs/grid220_full_prior';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    assert not O.exists() and read(R/'outputs/grid240_common_bottleneck/complete.json')['passed'];O.mkdir()
    design=dict(id='GRID241',scope='Structuralqualificationandoneprospectivecandidateonly;zero newforecasts/fits. DoNOTsubstituteb12intoold237theta orappend239thirdarm.',
        candidate='Frozen208controller-coherentprior for ALLfit8/cal4/deploymentrecords. Reuse208event+amountpairs andtheirmean83/scale83/rawmean41/rawscale41 toproduceparentgainp*mu andb12. Existing208entryb12>0 unchanged;doNOTcomputeinner8priorasadditionalentry oroffset.',
        residual='Samepublic41 standardizedbyfixed213inner8mean41/scale41 plus b12,42inputs43params,offsetb12. Public41scalerunchangedfrom237;priorandeligibilityintentionallychanged. Refit on8dates only,notreuseoldweights.',
        fitting='Rebuildfreshb12CONT originalSTOP/auxCONT rows from192fullown12source. Onlyfit8labels for0.5+0.5separatelynormalizedstructuralmass*absNETrisk. Meanlogistic+full43unitL2inclintercept,zero43init,originalL-BFGS-B. f0_s1orfit8STOPpositive0oremptydomain=>predeclaredKEEP,nofoldchange.',
        calibration='Oneoriginal221maxpositiveprotectionstep peractivepool usingALLavailableownouter12selectedoriginalSTOPpositive records(fit8+cal4). Explicit empiricalunionconstraint,notOOFindependentcalibration,not239FIT/CALsourceablation. No parentquerybudget/newalphagrid.',
        max_operations=dict(new_linear_fits=11,new_scalar_fits=11,total_fits_cap=22,predeclaredKEEP=1,new_forest_NNfits=0,new_physics=0,total_fit_calibration_cap_s=600),
        deployment='Usealreadycomputed208parentgain/charge margin;noadditionalinner8forest inferenceatdeployment. OnlycancelparentCONT atoriginalNETSTOP,atmostsameonebatch,b12<=0noappend. No domainfallbackthreshold/newrescue. Onlinefeature/inferenceoverheadnotmeasured.',
        comparison='Onecombinedconditionalcandidate vs shared208GATE,230onlyhistoricalinternalcontext. Notpureprior-sourceablation becauseeligibilityandpositiveconstraintpopulationchange. All3seeds/all16oldexposeddates;no calibration-sourcechoicefromouterresults.',
        ancestry='Parent208heads/NET/STOPpopulationouter12includeinternalcal4labels. Residualfit8onlybutunionstepuses12;not independentcal4errorvalidation/fullpipelineOOF/blindtest. Originalouter4ancestorlabelisolation remains.',
        barriers='Next242publicfit/calparentpriorregenerationfirstpublishedbeforetargetjoin;verifyoldouter208publicb12reproductionwithoutnewpolicy. Thennewcandidatecodeinputspublish,alltheta+stepspublishbeforeouterpolicyscores,allpredictionspublishbeforeoutcomejoin.',
        bounds='No actualfits/scoresin241. Noextraarms/NNscale/scenarios/physicalruns/final8/paper/FRAME,fee/label/controller/order/aliases/station/128/64/stagnation2/.9changes. Alladverse seeds retained.')
    save('design.json',design);meta=read(META);pub=read(B/'preheld_publication.json');assert pub['passed'] and pub['all12pairs_published'];trained=read(B/'trained_freeze.json');records=[];paths=[W/'run.py',O/'design.json',ROLE,META,B/'complete.json',B/'independent_audit.json',B/'preheld_publication.json',B/'trained_freeze.json',H/'public_raw41.npz',R/'work/grid208_public_amount/run.py',R/'work/grid206_public_event/run.py',R/'work/grid221_step_plan/run.py',R/'outputs/grid240_common_bottleneck/complete.json']
    for r in read(ROLE):
        pool=r['pool'];train_dates=set(r['fit_dates'])|set(r['calibration_dates']);outer=set(r['outer_dates']);cpfile=B/'models'/f'{pool}.npz';cp=np.load(cpfile);assert cp['mean'].shape==cp['scale'].shape==(83,) and cp['raw_mean'].shape==cp['raw_scale'].shape==(41,) and (cp['scale']>0).all() and (cp['raw_scale']>0).all();heads=[]
        for name in ['event','amount']:
            path=B/'models'/f'{pool}_{name}.pkl';assert sha(path)==trained[path.relative_to(B).as_posix()];blob=subprocess.check_output(['git','-C',str(REPO),'show',pub['git_sha']+':research_archive/power-grid-2026/protocols/GRID208_public_amount/trained/models/'+path.name]);assert hashlib.sha256(blob).hexdigest()==sha(path);model=pickle.loads(path.read_bytes());assert model.n_features_in_==124 and model.n_estimators==64 and model.max_depth==4
            if name=='event':assert np.array_equal(model.classes_,[0,1])
            heads.append(path.relative_to(R).as_posix());paths.append(path)
        efile=E/'fit_features'/f'{pool}.npz';efiledata=np.load(efile);amountfile=B/'fit_features'/f'{pool}.npz';amountdata=np.load(amountfile);event_dates={meta[int(i)]['date'] for i in efiledata['ids']};amount_dates={meta[int(i)]['date'] for i in amountdata['metadata_ids']};assert event_dates<=train_dates and amount_dates<=train_dates and not (event_dates|amount_dates)&outer
        existing=P/'public_held_inputs'/f'{pool}.npz';z=np.load(existing);assert np.array_equal(z['base_CONT'],z['b']>0);scaler=R/r['scaler_file'];q=np.load(scaler);assert q['mean41'].shape==q['scale41'].shape==(41,) and (q['scale41']>0).all();full=R/'outputs/grid192_gate_support/fit_features'/f'{pool}.npz';fullz=np.load(full);assert {meta[int(i)]['date'] for i in fullz['ids']}==train_dates
        records.append(dict(pool=pool,fit_dates=r['fit_dates'],cal4_dates=r['calibration_dates'],outer_dates=r['outer_dates'],parent_heads=heads,parent_scaler=cpfile.relative_to(R).as_posix(),residual_public41_scaler=scaler.relative_to(R).as_posix(),source_originalSTOP_count=int(fullz['original_STOP_count']),source_rows=len(fullz['ids']),event_label_dates=sorted(event_dates),amount_label_dates=sorted(amount_dates),old_outer_entry_equals_parent_margin_sign=True,new_parent_forecasts=0));paths += [cpfile,efile,amountfile,existing,scaler,full]
    save('qualified_pools.json',records);save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)})
    for path,h in read(O/'source_freeze.json').items():assert sha(R/path)==h,path
    save('qualification.json',dict(passed=True,all24parentheadweights_match_originalpublication=True,all12head_and_scaler_schemas_checked_without_forecasts=True,all12parent_label_ancestors_exclude_outer4=True,all12existing_entry_margin_signs_match=True,residual_public41scaler_preserved_from237=True,candidate_and_empirical_union_constraint_changes_disclosed=True,actual_fits=0,new_forecasts=0,max_future_linear_and_scalar_fits=22));save('complete.json',dict(passed=True,qualification_only=True,new_fits=0,new_forecasts=0,new_physics=0,final8_absent=True))
    text='''# GRID241：父先验一致候选的资格

下一候选只沿用原208已经计算的先验，不再让入口使用12天头、修正偏置却使用另一套8天头。训练、检查、外层部署都用同一对固定208收益头生成b12；原继续入口保持不变。

本轮核对了24个头的原始发布权重、12组标准化参数、标签日期祖先及既有入口与b12符号一致性，没有生成预测或拟合。公共41维额外修正器仍使用固定inner8标准化，以免同时悄悄改动这一部分。先验和筛选来源改变，必须重新构建输入及拟合权重，不能把旧237模型的b直接换掉。

固定一个联合候选：修正器只在8天拟合，步长用自身12天全部已知STOP回本正例作原规则的经验保护。总计最多11个43参数模型和11个标量、600秒，预定f0_s1继续。这个全正例约束与239的两种来源不同，是新候选的明确变更，不追加239第三组，也不声称只改变了先验一个因素。

下一步先公开重建的父先验，再连接标签核对支持；资格通过后才执行一次新拟合。部署可直接使用父方法已算好的裕度，额外修正器不需要再次推理8天收益头，但实际运行开销尚未计时。

父模型和原STOP总体已使用12天，步长也使用全部12天正例；不称内部4天独立验证或全链嵌套。原外层4天的标签祖先隔离保持，最终8和物理仿真不动。当前没有新候选效果结果。
'''
    (O/'GRID241_plan.md').write_text(text,encoding='utf8');A.mkdir(parents=True,exist_ok=False)
    for p in [W/'run.py',*O.iterdir()]:
        if p.is_file():shutil.copy2(p,A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');dest=R/'论文协作资料/FRAME_父先验一致资格_GRID241';dest.mkdir(exist_ok=False)
    for n in ['GRID241_plan.md','design.json','qualified_pools.json','qualification.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg='GRID241父先验一致候选资格完成，0拟合/预测；24原208头发布权重、12scalers/祖先与入口b12符号均核对。新候选训练/检查/部署沿用同一父b12，公共41 scaler保持inner8，重新筛选重拟合而不替换旧θ输入。修正器fit8、原规则保护自身outer12全部STOP正例，最多11线性+11标量/600秒；变更先验/资格及保护集合公开，不称单因素消融或cal4独立。242先公共再生成/发布/再接标签支持，当前无新效果结果。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID241父先验一致候选资格\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID241后\n\n1. 240只读与241父先验一致资格完成，0新模型/alpha/预测，不重做239。\n2. 242公共再生成：用241固定208头及parent scaler对192fullown12 raw83+214raw41生成同一b12；残差public41仍按固定inner8 scaler，加b12为42输入。以b12>0重建STOP/CONT资格，角色沿用固定8/4。先全部公共预测发布，再接旧214标签、只在fit8独立归一化两域风险半半。\n3. 新候选只fit8，单个原221步长保护全部own12已选STOP正例(显式联合经验约束)。<=11模型+11标量/600秒，f0_s1或fit8STOP正例0或任一域空则预定KEEP。资格后另阶段执行，不新设第三校准臂、不将旧237θ换b。\n4. 未来外层直接复用原208已发表b12和入口，无额外inner8头；模型/标量先发、预测先发后连接结果。候选改变先验/筛选/保护集合，不称纯先验消融，父/预算标签祖先outer12不称cal4独立。最终8/物理/电网/论文/FRAME不动，原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保持，Git严格等待helper。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID241父先验一致资格\n\n[方案](protocols/'+A.name+'/GRID241_plan.md)；尚未预测或拟合，下一步重建同源输入。\n\n').encode('utf8')+p.read_bytes());print('GRID241_QUALIFIED',len(records))
if __name__=='__main__':main()
