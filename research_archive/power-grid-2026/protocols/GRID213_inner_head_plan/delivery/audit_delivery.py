"""Independent split/statistic checks and plan-only publication."""
from pathlib import Path
import ast,json,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def audit():
    C.verify();assert read(O/'complete.json')['actual_fits']==0 and not (O/'independent_audit.json').exists();meta,_,_,raw41=C.C.raw_inputs();records=read(O/'qualified_bundles.json');jobs=read(O/'future_jobs.json');deps=read(O/'dependency_graph.json');checked=0;job_keys=set()
    for r in records:
        f,k,s=r['outer_fold'],r['inner_fold'],r['seed'];name=f'f{f}_s{s}';bundle=r['bundle'];z=np.load(C.FIT/'fit_features'/f'{name}.npz');lab=np.load(C.LABEL/f'{name}.npz');q=np.load(O/'qualified_inputs'/f'{bundle}.npz');fy=np.load(O/'fit_targets'/f'{bundle}.npz');hy=np.load(O/'score_targets'/f'{bundle}.npz');fi,hi=q['fit_positions'],q['score_positions'];ids=z['ids'].astype(int);dates=np.array([meta[int(i)]['date'] for i in ids]);stop=fi<int(z['original_STOP_count']);weights=np.zeros(len(fi))
        for m in [stop,~stop]:weights[m]=int(stop.sum())*.5*z['mass'][fi[m]]/z['mass'][fi[m]].sum()
        assert np.allclose(weights,q['event_weights'],atol=1e-12,rtol=1e-12);assert set(dates[fi])==set(r['fit_dates']) and set(dates[hi])==set(r['score_dates']);assert not set(dates)&set(r['outer_evaluation_dates']);assert set(fi)|set(hi)==set(range(len(ids))) and not set(fi)&set(hi);assert np.array_equal(q['fit_meta_ids'],ids[fi]) and np.array_equal(q['score_meta_ids'],ids[hi])
        for arr,mask,mn,sn in [(z['x'],stop,'mean83','scale83'),(raw41[ids],np.ones(len(fi),bool),'mean41','scale41')]:
            avg=np.average(arr[fi[mask]],axis=0,weights=weights[mask]);std=np.sqrt(np.average((arr[fi[mask]]-avg)**2,axis=0,weights=weights[mask]));std[std<1e-8]=1.;assert np.allclose(avg,q[mn],atol=1e-10,rtol=1e-10) and np.allclose(std,q[sn],atol=1e-10,rtol=1e-10)
        out=C.make_inputs(z['x'],raw41[ids],z['mass'],int(z['original_STOP_count']),fi,hi);assert C.hashlib.sha256(out['fit_X'].tobytes()).hexdigest()==str(q['fit_X_sha256'][0]);assert C.hashlib.sha256(out['score_X'].tobytes()).hexdigest()==str(q['score_X_sha256'][0])
        altered83=z['x'].copy();altered41=raw41[ids].copy();altered83[hi]+=1234567.;altered41[hi]-=7654321.;altered=C.make_inputs(altered83,altered41,z['mass'],int(z['original_STOP_count']),fi,hi)
        for key in ['mean83','scale83','mean41','scale41','event_weights','fit_X']:assert np.array_equal(out[key],altered[key])
        assert not np.array_equal(out['score_X'],altered['score_X']);pos=lab['target'][fi]>0;assert np.array_equal(fy['event_y'],pos.astype(int));assert np.array_equal(fy['amount_positive_positions'],np.flatnonzero(pos));assert np.array_equal(fy['amount_y'],lab['target'][fi][pos]);assert np.allclose(fy['amount_weights'],weights[pos]*weights.sum()/weights[pos].sum(),atol=1e-10,rtol=1e-10);assert np.array_equal(hy['gain_normalized'],lab['target'][hi]);assert np.allclose(hy['actual_NET'],lab['target'][hi]-lab['known_charge'][hi],atol=1e-12,rtol=1e-12)
        dep=next(v for v in deps if v['bundle']==bundle);assert not set(dep['all_ancestor_label_dates'])&set(dep['outer_evaluation_dates']);assert set(dep['inner_score_overlap_baseNET_dates'])==set(r['score_dates']) and dep['head_only_out_of_fit'] and not dep['fully_nested_pipeline'];jj=[j for j in jobs if j['bundle']==bundle];assert len(jj)==3 and {j['head'] for j in jj}=={'event','amount_depth4','amount_depth5'}
        for j in jj:assert j['ordinal'] not in job_keys;job_keys.add(j['ordinal'])
        checked+=1
    assert checked==36 and job_keys==set(range(108));tree=ast.parse((C.W/'run.py').read_text(encoding='utf8'));assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['fit','predict','predict_proba','step'] for n in ast.walk(tree));C.verify();C.save('independent_audit.json',dict(passed=True,all36_bundles_dates_weights_scalers_labels_and_hashes_recomputed=True,all36_score_feature_poison_tests_leave_fit_statistics_unchanged=True,all108_jobs_unique_and_within_outer_training_dates=True,head_only_dependency_limitation_preserved=True,no_fit_prediction_or_optimizer_calls_in_plan_source=True,actual_fits=0,new_predictions=0,new_NET_forwards=0,new_physics=0))
    C.archive('audited');dest=R/'论文协作资料/FRAME_内层样本生成方案_GRID213';shutil.copy2(O/'independent_audit.json',dest/'independent_audit.json');msg='GRID213方案与零拟合资格核验完成：4外层×3内层×3种子=36组，每组8拟合/4生成日期；未来3头共108小森林。36组输入1054–1454行、正例145–211，标准化与权重只用内层拟合；外层日期祖先隔离、旧OOF复用冲突及基础NET内层拟合内效应均明示。评分特征毒化不改拟合统计，尚未训练或生成新模型。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-27 — GRID213内层样本生成方案资格\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID213方案核验后\n\n1. 208保持主候选。内层读出样本生成方案已核验：36组8/4日期，未来108个小森林，600秒拟合及内置核验上限，未启动。\n2. 下一步实现仅生成样本的固定运行器，严格消费资格文件；全部108权重封存后评分，评分封存后才连接标签，不训练路由或评分外层。\n3. 明确只对读出头交叉生成，基础NET/停止域仍有拟合内效应；新训练前通知范围，不新物理/最终8/论文/FRAME或调费用。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID213内层分歧样本生成方案\n\n[方案](protocols/'+C.A.name+'/audited/GRID213_plan.md)；36组资格通过，未来108小森林，本轮零拟合。\n\n').encode('utf8')+p.read_bytes());print('GRID213_INDEPENDENT_AUDIT',checked)
def delivery():
    C.verify();assert read(O/'independent_audit.json')['passed'];head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;count=0
    for mf in C.A.rglob('archive_manifest.json'):
        for p,h in read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
    dest=R/'论文协作资料/FRAME_内层样本生成方案_GRID213';files=list(dest.iterdir())
    for p in files:
        ap=C.A/'audited'/p.name;assert C.sha(p)==C.sha(O/p.name)==C.sha(ap);assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes()
    rec=dict(passed=True,verified_publication_head=head,remote_main=head,archive_files_checked=count,collaboration_files_checked=len(files),scope='plan_qualification_only',actual_fits=0,new_predictions=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'delivery_receipt.json',dest/'delivery_receipt.json');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
