"""Read-only lost-profit audit: amount support, cost algebra and leaf limits."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,hashlib,importlib.util,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid209_lost_gain';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID209_lost_gain';SRC=R/'outputs/grid208_public_amount';OLD=R/'outputs/grid206_public_event';LAB=R/'outputs/grid198_conditional_gain/fit_labels';CACHE=R/'outputs/grid190_extension_gate';FIT=R/'outputs/grid192_gate_support'
sp=importlib.util.spec_from_file_location('grid209_fixed_helpers',R/'work/grid208_public_amount/run.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    C.trained()
def prepare():
    assert not O.exists();O.mkdir(parents=True);C.trained();assert read(SRC/'independent_audit.json')['passed']
    save('design.json',dict(id='GRID209_LOST_GAIN',scope='All4681 fixed208vs206 predictions, all12old/newamounts;focus4removedprofitable starts,all retained/adverse groups.',questions='Do lostcases allunderestimate realizedpositiveamount? Fixed p cost/mu crossing descriptors;conditionmean isnotrealizedgain. Compareold/newpositivefit leaf support,depth4 and minleaf10 constraints,traininglabelspread.',algebra='For realizedpositivecases only describe mu/g andrequiredmu=knowncharge/frozenp;hindsightp*realizedgain>cost diagnostic only,no oraclepolicy evaluated. No targetmu0forzeros.',kernels='Reconstruct bothold83andnew124 amount predictions fromoriginal198positivefitweights and labels. New-minus-old support contributions sum to mu delta;top5trainingcall differences;STOP/CONTpositive amount support retained.',leaf_limits='Allnewheldpaths: actualleafdepth,positivefitrowcount,fitcalls/targetsd. Depth4leaf is potentiallysplittable onlyif>=20rows,variablepositive labels,and at leastone124feature has a threshold giving>=10rows eachchild. Existence isnotbestgreedysplit orbenefit guarantee. No deeper model built/scored.',limits='No fits/NNforward/newphysics/threshold/fee/features/rankchange,final8/paper/FRAME. All16exposed historicaltrain dates/allseeds/sharedprefixes;no causalorindependenttest claim. No automatic capacity increase/combination/replacement.'))
    files=[W/'run.py',O/'design.json',R/'work/grid208_public_amount/run.py',SRC/'complete.json',SRC/'independent_audit.json',SRC/'gate_predictions.json',SRC/'changed_choices.json',SRC/'family_summary.json',OLD/'gate_predictions.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']
    for d in [SRC,OLD]:files+=list((d/'models').iterdir())
    files+=list((SRC/'fit_features').glob('*.npz'))+list((FIT/'fit_features').glob('*.npz'))+list(LAB.glob('*.npz'))+list(CACHE.glob('held_f*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID209_PLANNED')
def evaluate():
    verify();assert not (O/'started.json').exists();save('started.json',dict(new_fits=0,new_NET_forwards=0,new_physics=0));meta,lookup,cases,raw=C.raw_inputs();pred=read(SRC/'gate_predictions.json');prev={(p['seed'],p['sample_id']):p for p in read(OLD/'gate_predictions.json')};ix={(p['fold'],p['seed'],lookup[p['sample_id'],p['position']]):p for p in pred};scale=read(R/'outputs/grid102_search_rl/design.json')['gain_scale'];rows=[];poolchecks=[]
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(SRC/'fit_features'/f'{name}.npz');orig=np.load(FIT/'fit_features'/f'{name}.npz');lab=np.load(LAB/f'{name}.npz');cp=np.load(SRC/'models'/f'{name}.npz');hz=np.load(CACHE/f'held_{name}.npz');pos=lab['indices'].astype(int);train=z['standard'];trainold=orig['standard'][pos];w,y=z['sample_weight'],z['target'];assert np.array_equal(w,lab['sample_weight']) and np.array_equal(y,lab['target']) and np.array_equal(train[:,:83],trainold);x=np.c_[(hz['features']-cp['mean'])/cp['scale'],(raw[hz['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']];fitmeta=[meta[int(i)] for i in z['metadata_ids']];assert not {m['date'] for m in fitmeta}&{meta[int(i)]['date'] for i in hz['ids']};kernels={};leafstats=[];newmodel=pickle.loads((SRC/'models'/f'{name}_amount.pkl').read_bytes());oldmodel=pickle.loads((OLD/'models'/f'{name}_amount.pkl').read_bytes());models=[('NEW',newmodel,train,x),('OLD',oldmodel,trainold,x[:,:83])]
            for label,model,tx,hx in models:
                kernel=np.zeros((len(x),len(y)))
                for tree in model.estimators_:
                    tl=tree.apply(tx);hl=tree.apply(hx);weight=np.bincount(tl,weights=w,minlength=tree.tree_.node_count);kernel+=(hl[:,None]==tl[None,:])*w[None,:]/weight[hl,None]/64
                    if label=='NEW':
                        t=tree.tree_;depth=np.zeros(t.node_count,int);stack=[0]
                        while stack:
                            node=stack.pop()
                            if t.children_left[node]!=-1:
                                for child in [t.children_left[node],t.children_right[node]]:depth[child]=depth[node]+1;stack.append(child)
                        nodeinfo={}
                        for leaf in np.unique(tl):
                            mask=tl==leaf;tt=tx[mask];yy=y[mask];n=len(yy);can=False
                            if n>=20 and np.unique(yy).size>1:
                                sortedx=np.sort(tt.astype(np.float32),axis=0);can=bool((np.diff(sortedx,axis=0)[9:n-10]>0).any())
                            nodeinfo[int(leaf)]=[int(depth[leaf]),n,int(can),float(np.std(yy)),len({fitmeta[k]['physical_state_group'] for k in np.flatnonzero(mask)})]
                        got=np.asarray([nodeinfo[int(leaf)] for leaf in hl]);assert np.array_equal(got[:,0],np.asarray(tree.decision_path(hx).sum(1)).ravel()-1);leafstats.append(got)
                mu=model.predict(hx);assert np.allclose(kernel.sum(1),1,atol=1e-10,rtol=1e-10) and np.allclose(kernel@y,mu,atol=1e-10,rtol=1e-10);kernels[label]=kernel
            stats=np.stack(leafstats);fitcalls=sorted({m['physical_state_group'] for m in fitmeta});gi={g:i for i,g in enumerate(fitcalls)};groups=np.array([gi[m['physical_state_group']] for m in fitmeta]);stop=pos<int(orig['original_STOP_count'])
            for j,i in enumerate(hz['ids']):
                p=ix[f,s,int(i)];old=prev[s,p['sample_id']];newmu=float(kernels['NEW'][j]@y);oldmu=float(kernels['OLD'][j]@y);assert abs(newmu-p['conditional_normalized_gain'])<1e-9 and abs(oldmu-old['conditional_normalized_gain'])<1e-9;assert p['event_probability']==old['event_probability'];a,b=p['gate_continue'],old['gate_continue'];change='BOTH' if a and b else 'NEW_ONLY' if a else 'OLD_ONLY' if b else 'NEITHER';profit=p['actual_next_NET']>0;category=('LOST_PROFIT' if profit else 'REMOVED_UNPROFIT') if change=='OLD_ONLY' else ('ADDED_PROFIT' if profit else 'ADDED_UNPROFIT') if change=='NEW_ONLY' else ('KEPT_PROFIT' if profit else 'KEPT_UNPROFIT') if change=='BOTH' else 'BOTH_STOP';diff=(kernels['NEW'][j]-kernels['OLD'][j])*y;dc=np.bincount(groups,weights=diff,minlength=len(fitcalls));gain=p['actual_next_gain']/scale;pp=p['event_probability'];limit=p['known_charge']/pp if pp>0 else None;maxids=np.argsort(-dc)[:5];minids=np.argsort(dc)[:5]
                rows.append(dict(fold=f,seed=s,sample_id=p['sample_id'],position=p['position'],call=p['physical_state_group'],date=p['date'],category=category,actual_gain_normalized=gain,actual_NET=p['actual_next_NET'],queries=p['actual_next_queries'],event_p=pp,known_charge=p['known_charge'],required_mu=limit,old_mu=oldmu,new_mu=newmu,new_mu_below_realized_positive=bool(newmu<gain) if gain>0 else None,realized_positive_mu_would_cross=bool(pp*gain>p['known_charge']) if gain>0 else None,old_margin=old['predicted_net'],new_margin=p['predicted_net'],depth4_tree_fraction=float(np.mean(stats[:,j,0]==4)),depth4_splittable_tree_fraction=float(np.mean((stats[:,j,0]==4)&(stats[:,j,2]>0))),minleaf_unsplittable_tree_fraction=float(np.mean(stats[:,j,1]<20)),mean_leaf_fit_rows=float(stats[:,j,1].mean()),mean_leaf_target_std=float(stats[:,j,3].mean()),mean_leaf_fit_calls=float(stats[:,j,4].mean()),NEW_STOP_amount=float(kernels['NEW'][j,stop]@y[stop]),NEW_CONT_amount=float(kernels['NEW'][j,~stop]@y[~stop]),top_increased_training_call_contributions=[dict(call=fitcalls[k],delta_mu=float(dc[k])) for k in maxids if dc[k]>0],top_decreased_training_call_contributions=[dict(call=fitcalls[k],delta_mu=float(dc[k])) for k in minids if dc[k]<0]))
            poolchecks.append(dict(fold=f,seed=s,held_rows=len(x),positive_fit_rows=len(y),all_old_new_native_mu_and_leaf_paths_checked=True))
    assert len(rows)==4681;categories={}
    for k in sorted({r['category'] for r in rows}):
        rr=[r for r in rows if r['category']==k];categories[k]=dict(pairs=len(rr),calls=len({r['call'] for r in rr}),folds=sorted({r['fold'] for r in rr}),mean_depth4_tree_fraction=float(np.mean([r['depth4_tree_fraction'] for r in rr])),mean_depth4_splittable_fraction=float(np.mean([r['depth4_splittable_tree_fraction'] for r in rr])),mean_minleaf_unsplittable_fraction=float(np.mean([r['minleaf_unsplittable_tree_fraction'] for r in rr])),mean_leaf_target_std=float(np.mean([r['mean_leaf_target_std'] for r in rr])))
    lost=[r for r in rows if r['category']=='LOST_PROFIT'];assert len(lost)==4 and len({r['call'] for r in lost})==3;result=dict(categories=categories,lost_cases=lost,all12pools=poolchecks,lost_amount_below_realized=sum(r['new_mu_below_realized_positive'] for r in lost),lost_realized_positive_mu_cross=sum(r['realized_positive_mu_would_cross'] for r in lost),new_fits=0,new_NET_forwards=0,new_physics=0);save('all_support_and_leaf_rows.json',rows);save('summary.json',result)
    lines=['# GRID209：被取消的四个回本配对','', '本轮固定所有模型，核查旧206与新208的全部4681个停止点。四个回本损失来自3个物理调用，集中在留出折0；不是四个独立实验。概率模型逐点相同，选择变化由条件收益预测变化引起，但不能把每个实际回本结果都解释为条件均值被低估。','', '|种子/调用|查询|旧幅度|新幅度|已实现正增量|按当前概率回本所需幅度|新预测净值|','|---|---:|---:|---:|---:|---:|---:|']
    for r in lost:lines.append(f"|{r['seed']}/{r['call']}|{r['queries']}|{r['old_mu']:.6f}|{r['new_mu']:.6f}|{r['actual_gain_normalized']:.6f}|{r['required_mu']:.6f}|{r['new_margin']:.6f}|")
    lines+=['',f"其中{result['lost_amount_below_realized']}个新幅度低于已实现正增量；另一个新幅度已经高于实际增量，却因乘以较低事件概率仍未过费。即使用这一个已实现增量作幅度，按原概率也只有{result['lost_realized_positive_mu_cross']}/4能过费。这里只是固定费用下的事后算式，没有运行使用未来答案的策略，也不将单次结果当真实条件均值。",'', '## 叶子是否受到容量上限约束','', '逐棵树核查叶子深度、正例数量及标签离散程度。潜在可继续划分要求：叶内至少20行、标签存在变化，且至少一个原有输入能在某阈值两侧各留下10行。深度4且满足这些条件，表示结构上仍有进一步区分的空间；它不保证额外一层能找到更好的泛化规则。没有生成或评分更深模型。','', '|组|配对/调用|折|深度4树比例均值|深度4且潜在可分比例|不足20行比例|叶内标签标准差均值|','|---|---:|---|---:|---:|---:|---:|']
    for k,z in categories.items():lines.append(f"|{k}|{z['pairs']}/{z['calls']}|{z['folds']}|{z['mean_depth4_tree_fraction']:.4f}|{z['mean_depth4_splittable_fraction']:.4f}|{z['mean_minleaf_unsplittable_fraction']:.4f}|{z['mean_leaf_target_std']:.4f}|")
    lines+=['','|损失配对|深度4比例|潜在可继续划分比例|不足20行比例|','|---|---:|---:|---:|']
    for r in lost:lines.append(f"|{r['seed']}/{r['sample_id'][:8]}|{r['depth4_tree_fraction']:.4f}|{r['depth4_splittable_tree_fraction']:.4f}|{r['minleaf_unsplittable_tree_fraction']:.4f}|")
    lines+=['','## 训练支持变化','', '新旧幅度均已精确还原为原198拟合正例标签的加权平均；两者样本、权重、标签相同，改变的是输入和由此产生的树叶。逐调用对幅度增减的贡献另存summary.json和全量记录，其和与新旧预测差一致。没有因某调用贡献大就删样本、加权、改费或写入触发规则。','', '下一步应依据全部保留与损失组共同判断容量、覆盖或概率—幅度组合的问题；不能专门放宽门槛把这四个结果救回来。当前所有结论仍来自已暴露教师路径，不是新完整周、跨电网或最终测试证据。']
    (O/'GRID209_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');verify();save('review.json',dict(passed=True,all4681_old_new_positive_fit_mu_reconstructed=True,all_new_leaf_depths_match_native_paths=True,all4_lost_and_retained_groups_kept=True,no_oracle_policy_scored=True,zero_mu_targets_not_added=True,new_fits=0,new_NET_forwards=0,new_physics=0));save('complete.json',dict(passed=True,new_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('complete');dest=R/'论文协作资料/FRAME_回本损失与叶子限制_GRID209';dest.mkdir(exist_ok=False)
    for n in ['GRID209_report.md','summary.json','review.json','complete.json','design.json']:shutil.copy2(O/n,dest/n)
    print('GRID209_COMPLETE',json.dumps(result,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
