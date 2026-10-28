"""Independent saved-row reconstruction, document correction, and publication."""
from pathlib import Path
import json,hashlib,collections,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,R,S,A,REPO=C.O,C.R,C.S,C.A,C.REPO

def main():
    assert not (O/'independent_audit.json').exists();C.verify();meta=C.read(R/'outputs/grid113_relational_learning/train_metadata.json');bundles=C.read(S/'bundles.json');pools=collections.defaultdict(list);source_count=0
    for b in bundles:
        z=np.load(S/'labelled_samples'/f"{b['bundle']}.npz");ids=z['meta_ids'];assert np.allclose(z['actual_NET']+z['true_fee_normalized'],z['gain_normalized'],atol=1e-12,rtol=0)
        for k in range(len(ids)):
            if not z['original_STOP_mask'][k]:continue
            source_count+=1
            if not z['continue4'][k]:continue
            idx=int(ids[k]);assert meta[idx]['date'] not in b['outer_evaluation_dates'];pools[b['outer_fold'],b['seed']].append((idx,float(z['original_source_mass'][k]),float(z['actual_NET'][k]),meta[idx]['date'],meta[idx]['physical_state_group']))
    targets=C.read(O/'qualified_target_receipts.json');allposids=set();allposcalls=set();allposdates=set();n=positive=0;checks=0;fallback=[]
    for f,s in sorted(pools):
        key=f'f{f}_s{s}';rows=pools[f,s];z=np.load(O/'qualified_targets'/f'{key}.npz');ids=np.array([r[0] for r in rows]);rawmass=np.array([r[1] for r in rows]);net=np.array([r[2] for r in rows]);m=rawmass/rawmass.sum();cost=m*np.abs(net);y=net>0
        assert np.array_equal(ids,z['meta_ids']) and np.array_equal(net,z['actual_NET']) and np.array_equal(y,z['keep_label']);assert np.all(rawmass>0) and np.allclose(m,z['structural_mass'],atol=1e-15,rtol=1e-15);assert np.allclose(cost,z['cost_mass'],atol=1e-15,rtol=1e-15);assert np.allclose(z['normalized_fit_weight'],len(rows)*cost/cost.sum(),atol=1e-12,rtol=1e-12);assert bool(z['fallback_keep'][0])==not_any(y)
        if not y.any():fallback.append(key)
        for action in [False,True]:assert np.allclose(np.maximum(net,0)-action*net,np.abs(net)*(action!=y),atol=1e-15,rtol=1e-15)
        for row,pos in zip(rows,y):
            if pos:allposids.add(row[0]);allposcalls.add(row[4]);allposdates.add(row[3])
        st=C.read(O/'summary.json')['pools'][key]['base4_continue'];assert st['rows']==len(rows) and st['profitable_rows']==int(y.sum());assert st['profitable_calls']==len({r[4] for r in rows if r[2]>0});assert abs(st['positive_cost_mass']-float((m*np.maximum(net,0)).sum()))<1e-12;assert abs(st['negative_cost_mass']-float((m*np.maximum(-net,0)).sum()))<1e-12
        n+=len(rows);positive+=int(y.sum());checks+=1
    assert (source_count,n,positive,len(allposids),len(allposcalls),len(allposdates),checks)==(16565,3466,90,29,16,9,12);assert fallback==['f0_s1']
    perdate=C.read(O/'per_date_support.json');assert len(perdate)==144
    for q in perdate:
        f,s=[int(x[1:]) for x in q['pool'].split('_')];r=[r for r in pools[f,s] if r[3]!=q['excluded_date']];assert q['remaining_rows']==len(r) and q['remaining_profitable_rows']==sum(x[2]>0 for x in r);assert q['remaining_profitable_calls']==len({x[4] for x in r if x[2]>0})
    # Correct a historical stage-number citation in a separate, preserved revision.
    src=O/'GRID215_report.md';text=src.read_text(encoding='utf8');assert '旧GRID117已比较过相关加权损失' in text
    evidence=REPO/'docs/RESEARCH_STATE.md';assert 'GRID120 decision-objective screen complete' in evidence.read_text(encoding='utf8')
    corrected=text.replace('旧GRID117已比较过相关加权损失','旧GRID120已比较过相关加权损失');corrected=corrected.replace('当前保留GRID208。','GRID120中，按净收益加权的分类优于只学收益符号，但仍弱于当时的回归模型；该负证据保留。本次研究入口变化后的小森林取消模块，不将加权损失本身视为已证明的改进。\n\n当前保留GRID208。');corrected=corrected.replace('逐池去掉一个日期后，','各池最大单次调用占保护价值25.9%—64.5%，保护价值仍有集中。逐池去掉一个日期后，')
    revised=O/'GRID215_report_核对版.md';revised.write_text(corrected,encoding='utf8',newline='\n');C.save('document_revision_receipt.json',dict(reason='Source draft cited stage117; verified existing state shows utility-classification experiment isGRID120. Draft and frozen source preserved; corrected separate report adds adverse precedent and concentration interpretation.',original_file=src.name,original_sha256=C.sha(src),corrected_file=revised.name,corrected_sha256=C.sha(revised),scientific_arrays_unchanged=True,source_evidence=evidence.relative_to(R).as_posix()))
    C.verify()
    for p,h in C.read(O/'target_freeze.json').items():assert C.sha(O/p)==h
    C.save('independent_audit.json',dict(passed=True,source_rows_reconstructed=source_count,selected_rows_reconstructed=n,profitable_rows=positive,profitable_unique_states=len(allposids),profitable_calls=len(allposcalls),profitable_dates=len(allposdates),all12_target_files_and_cost_weights_reconstructed=True,all144_date_support_checks_reconstructed=True,one_batch_regret_identity_both_actions_verified=True,source_model_and_prediction_and_label_freezes_unchanged=True,document_historical_reference_corrected_separately=True,no_fits_or_new_model_forwards=True))
    C.save('delivery_receipt.json',dict(passed=True,report_file=revised.name,report_sha256=C.sha(revised),qualified_target_files=12,actual_fits=0,actual_model_forwards=0,actual_policy_evaluations=0,source_publication_sha=C.read(S/'delivery_receipt.json')['published_result_git_sha']))
    dest=R/'论文协作资料/FRAME_取消学习范围_GRID215';dest.mkdir(exist_ok=False)
    for name in [revised.name,'summary.json','decision.json','independent_audit.json','complete.json','delivery_receipt.json','document_revision_receipt.json']:shutil.copy2(O/name,dest/name)
    A.mkdir(parents=True,exist_ok=False)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,A/p.name)
    for p in O.iterdir():
        if p.is_file():shutil.copy2(p,A/p.name)
        else:shutil.copytree(p,A/p.name)
    for p in dest.iterdir():assert C.sha(p)==C.sha(O/p.name)==C.sha(A/p.name)
    (A/'archive_manifest.json').write_text(json.dumps({p.relative_to(A).as_posix():C.sha(p) for p in A.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
    msg='GRID215只读核查完成：取消学习入口定为原STOP且深4准备继续，3466模型状态行/90回本，对应29状态16调用9日期；11/12池有保护例，去掉任一日期后11池仍有正例，最大调用占保护价值25.9%—64.5%。只学取消、不混救回；标签为原下一批NET正负，原结构质量乘absNET作代价，保持原费，不均衡类别或调阈值。f0_s1无回本支持明确KEEP常量，不借跨折标签。12份目标核验，无训练/新前向/策略评分；原稿GRID117历史引用校正为GRID120，负结果保留。208仍主候选，下一轮先冻结11森林+1KEEP的输入/配置并通知训练。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID215取消训练域与目标\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID215后\n\n1. 取消入口：原STOP且深4准备继续；3466行90回本是相关模型记录，不独立实验。保留208主候选。\n2. 下一次固定11个小森林+f0_s1常量KEEP；训练前冻结124公共输入/缩放/原结构×absNET权重/模型配置和日期祖先，再通知训练。不改费或阈值，不同时做救回。\n3. 只在对应外层4日期评估限定取消策略；不使用受其他内层头祖先影响的内层切分宣称泛化，不混跨外层OOF。保留全部种子日期反例。\n4. 尚未训练取消分类器；不扩大网络/新物理/最终8/论文/新FRAME比较。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID215取消学习训练域\n\n[核对报告](protocols/'+A.name+'/'+revised.name+')；12份目标核验，零新训练或策略评分。\n\n').encode('utf8')+p.read_bytes());print('GRID215_AUDITED_AND_ARCHIVED',json.dumps(dict(rows=n,positive=positive,states=len(allposids),calls=len(allposcalls),fallback=fallback),ensure_ascii=False))
def not_any(y):return not bool(y.any())
if __name__=='__main__':main()
