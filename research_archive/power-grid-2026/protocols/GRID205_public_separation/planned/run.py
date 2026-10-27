"""Fixed public-feedback comparison, exact signatures and descriptive neighbors."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import ast,collections,hashlib,importlib.util,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid205_public_separation';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID205_public_separation';SRC=R/'outputs/grid203_tree_event';FIT=R/'outputs/grid192_gate_support';CACHE=R/'outputs/grid190_extension_gate';PUBLIC=R/'outputs/grid113_relational_learning'
sp=importlib.util.spec_from_file_location('grid205_fixed_helpers',R/'work/grid203_tree_event/run.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
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
def names():
    tree=ast.parse((R/'work/grid101_gain_learning/features.py').read_text(encoding='utf8'));base=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAMES' for t in n.targets));return base+['local_batches_done','local_budget_used_fraction','log_next_batch_queries','locked_remaining_fraction']
def descriptor(p,raw):
    fb=p['context_feedback'].astype(float);ca=p['context_action'].astype(float);na=p['next_action'].astype(float);gap=fb[:,0];nextdist=np.abs(na[:,17:34]).sum(1)/np.maximum(na[:,34:51].sum(1),1);contextdist=np.abs(ca[:,17:34]).sum(1)/np.maximum(ca[:,34:51].sum(1),1)
    # Known canonical connection patterns, excluding outcomes of the next batch.
    seen=ca[:,np.r_[0:17,34:51,51:68]];nxt=na[:,np.r_[0:17,34:51,51:68]];nearest=np.mean(nxt[:,None,:]!=seen[None,:,:],axis=2).min(1)
    extra=[np.log1p(len(fb)),*np.quantile(gap,[0,.25,.5,.75,1]),gap.mean(),gap.std(),*fb[:,1:].mean(0),nextdist.min(),nextdist.mean(),nextdist.max(),contextdist.min(),contextdist.mean(),contextdist.max(),nearest.min(),nearest.mean(),nearest.max(),len(na)]
    extra_names=['log_context_count',*[f'context_gap_q{x}' for x in ['0','25','50','75','100']],'context_gap_mean','context_gap_std',*[f'context_{n}_fraction' for n in ['finite','legal','strict_admissible','done','illegal','ambiguous','exception']],*[f'next_relative_distance_{x}' for x in ['min','mean','max']],*[f'context_relative_distance_{x}' for x in ['min','mean','max']],*[f'next_nearest_seen_distance_{x}' for x in ['min','mean','max']],'known_next_queries']
    out=np.r_[raw,np.asarray(extra)];assert len(out)==66 and len(extra_names)==25 and np.isfinite(out).all();return out,names()+extra_names
def sig(p,raw,factor=False):
    def rows(a):return sorted(np.ascontiguousarray(r,dtype='<f4').tobytes() for r in a)
    h=hashlib.sha256();parts=[np.asarray(raw,dtype='<f4').tobytes(),np.asarray(p['anchor_action'],dtype='<f4').tobytes(),np.asarray(p['known_query_cost'],dtype='<f4').tobytes()]
    if factor:
        for a in [p['context_action'],p['context_feedback']]:parts.append(b''.join(rows(a)))
    else:parts.append(b''.join(rows(np.c_[p['context_action'],p['context_feedback']])))
    parts.append(b''.join(rows(p['next_action'])))
    for a in parts:h.update(len(a).to_bytes(8,'little'));h.update(a)
    return h.hexdigest()
def prepare():
    assert not O.exists();O.mkdir(parents=True);C.trained();assert read(R/'outputs/grid204_event_paths/complete.json')['passed']
    save('design.json',dict(id='GRID205_PUBLIC_SEPARATION',scope='All4681 fixed eligible held STOPs, no model fits/NNforwards/physics. Compare821selected including53profitable,6positiveunprofitable,762zero cases;keepall unselected.',public='Only originaltrain_public.pkl train3582 and caseprefix raw41. Original41 names fixed;additional25 deterministic summaries of already-paid feedback and known next plans. Never next feedback/reward/IDs/dates as feature.',representations='RAW41, PUBLIC66 (raw41 plus fixed25 context and known-plan summaries), HIDDEN83 frozen. Nearest fit distances use perpool fit-only originalmixture sampleweight mean/std; existing83standardizer unchanged. Euclidean mean squared standardized distance;zero scales<1e-8 become1.',descriptive_neighbors='Allheld3 nearest distinct physical fitcalls perrepresentation, distance and trainonly IDs choose neighbors before attaching teacher gain/event. This is not a fitted/evaluated nearestneighbor gate. No vote threshold or policyperformance.',collisions='Exact raw fullpaired-public permutation-invariant signature and FACTOR unpaired signature; hidden83 exactbytes within foldseed. Count conflicting realized next labels, if any;none doesnot prove learnability. Raw global41 float32,paid feedback and known canonical actions only.',focused_comparison='Previous mainrestoredcall;pair each restored profitable heldstart to closest currentlyselected zero-gain heldstart fromothercall in samefoldseed, lexicographic querycount difference then PUBLIC66distance. Hindsight-conditioned diagnosis only, never deployed or used for fitting.',limits='No feature selection by scores, AUC scanning, fitweights/labels/fees/threshold changes, NNforward/final8/validation/newFRAME. All16exposed train dates/all3seeds;teacher-path sharedstarts not independent. Descriptors compare public information versus compressedreadout,not proof compressed information lost.'))
    files=[W/'run.py',O/'design.json',R/'work/grid203_tree_event/run.py',SRC/'gate_predictions.json',SRC/'interpretation_rows.json',SRC/'complete.json',SRC/'family_summary.json',R/'outputs/grid204_event_paths/summary.json',PUBLIC/'train_public.pkl',PUBLIC/'train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'work/grid101_gain_learning/features.py']+list((FIT/'fit_features').glob('*.npz'))+list(CACHE.glob('held_f*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID205_PLANNED')
def evaluate():
    verify();assert not (O/'started.json').exists();save('started.json',dict(new_fits=0,new_NET_forwards=0,new_physics=0));meta,lookup,cases=C.B.load();public=pickle.loads((PUBLIC/'train_public.pkl').read_bytes());assert len(public)==len(meta)==3582;bycase={c['sample_id']:c for c in cases};raw=np.array([bycase[m['sample_id']]['states'][m['position']] for m in meta],float);summaries=np.array([descriptor(p,x)[0] for p,x in zip(public,raw)]);feature_names=descriptor(public[0],raw[0])[1];assert summaries.shape==(3582,66)
    event=np.array([bycase[m['sample_id']]['gains'][m['position']]>0 for m in meta]);truegain=np.array([bycase[m['sample_id']]['gains'][m['position']] for m in meta]);signatures={'FULL_PAIRED':{},'FACTOR_UNPAIRED':{}};collisions={}
    for typ,factor in [('FULL_PAIRED',False),('FACTOR_UNPAIRED',True)]:
        buckets=collections.defaultdict(list)
        for i,(p,x) in enumerate(zip(public,raw)):buckets[sig(p,x,factor)].append(i)
        conflicts=[dict(indices=ids,events=event[ids].tolist(),calls=sorted({meta[i]['physical_state_group'] for i in ids})) for ids in buckets.values() if len(set(event[ids]))>1];collisions[typ]=dict(states=len(meta),unique_signatures=len(buckets),repeated_signature_groups=sum(len(v)>1 for v in buckets.values()),conflicting_event_groups=conflicts)
    pred=read(SRC/'gate_predictions.json');byp={(p['fold'],p['seed'],lookup[p['sample_id'],p['position']]):p for p in pred};ledger=[];focuspairs=[];poolchecks=[];hiddenconflicts=[];featurestats=[]
    for f in range(4):
        for s in range(3):
            z=np.load(FIT/'fit_features'/f'f{f}_s{s}.npz');hz=np.load(CACHE/f'held_f{f}_s{s}.npz');fi=z['ids'].astype(int);hi=hz['ids'].astype(int);w=z['sample_weight'];n=int(z['original_STOP_count']);fitdates={meta[i]['date'] for i in fi};assert not fitdates&{meta[i]['date'] for i in hi};rep={};scales={}
            for name,arr in [('RAW41',raw),('PUBLIC66',summaries)]:
                mean=np.average(arr[fi],axis=0,weights=w);std=np.sqrt(np.average((arr[fi]-mean)**2,axis=0,weights=w));std[std<1e-8]=1.;rep[name]=((arr[fi]-mean)/std,(arr[hi]-mean)/std);scales[name]=dict(mean=mean.tolist(),scale=std.tolist())
            rep['HIDDEN83']=(z['standard'],(hz['features']-z['mean'])/z['scale']);stats={}
            pp=[byp[f,s,int(i)] for i in hi];assert all(abs(float(public[i]['known_query_cost'][0])-p['known_charge'])<1e-10 for i,p in zip(hi,pp));fitgroups=[meta[i]['physical_state_group'] for i in fi];fitids_bycall=collections.defaultdict(list)
            for k,g in enumerate(fitgroups):fitids_bycall[g].append(k)
            pools=[]
            for j,(i,p) in enumerate(zip(hi,pp)):
                row=dict(fold=f,seed=s,index=int(i),sample_id=p['sample_id'],position=p['position'],call=p['physical_state_group'],date=p['date'],selected=p['gate_continue'],event=bool(event[i]),profitable=p['actual_next_NET']>0,next_queries=p['actual_next_queries'],actual_gain=p['actual_next_gain'],old_public66=summaries[i].tolist(),neighbors={})
                for name,(train,held) in rep.items():
                    dist=np.mean((train-held[j])**2,axis=1);order=sorted(((min((float(dist[k]),int(k)) for k in ids),g) for g,ids in fitids_bycall.items()),key=lambda v:(v[0][0],v[1]))[:3];nn=[]
                    for (d,k),g in order:
                        idx=int(fi[k]);nn.append(dict(index=idx,call=g,date=meta[idx]['date'],distance=d,fit_domain='STOP' if k<n else 'CONT',event=bool(event[idx]),gain=float(truegain[idx])))
                    assert len({v['call'] for v in nn})==3 and all(v['date'] in fitdates for v in nn);row['neighbors'][name]=nn
                pools.append(row)
            # Hidden exact collisions are checked within one model basis only.
            hh=collections.defaultdict(list)
            for j,i in enumerate(hi):hh[np.ascontiguousarray(rep['HIDDEN83'][1][j],dtype='<f8').tobytes()].append(int(i))
            hiddenconflicts+= [dict(fold=f,seed=s,indices=ids,events=event[ids].tolist()) for ids in hh.values() if len(set(event[ids]))>1]
            focus=[(j,r) for j,r in enumerate(pools) if r['call']=='2037-01-05_101|call001' and r['selected'] and r['profitable']];zero=[(j,r) for j,r in enumerate(pools) if r['selected'] and not r['event']]
            for j,r in focus:
                eligible=[(abs(q['next_queries']-r['next_queries']),float(np.mean((rep['PUBLIC66'][1][k]-rep['PUBLIC66'][1][j])**2)),q['sample_id'],k,q) for k,q in zero if q['call']!=r['call']];a=min(eligible,key=lambda t:t[:3]);other=a[-1];dif=summaries[other['index']]-summaries[r['index']];focuspairs.append(dict(fold=f,seed=s,restored_index=r['index'],restored_sample_id=r['sample_id'],restored_queries=r['next_queries'],zero_index=other['index'],zero_sample_id=other['sample_id'],zero_call=other['call'],zero_queries=other['next_queries'],query_count_difference=a[0],PUBLIC66_distance=a[1],all_raw_differences=dif.tolist(),largest_standardized_differences=[dict(feature=feature_names[k],restored=float(summaries[r['index'],k]),zero=float(summaries[other['index'],k]),standardized_difference=float(dif[k]/scales['PUBLIC66']['scale'][k])) for k in np.argsort(-abs(dif/np.array(scales['PUBLIC66']['scale'])))[:8]]))
            for label,mask in [('selected_profitable',np.array([p['gate_continue'] and p['actual_next_NET']>0 for p in pp])),('selected_zero',np.array([p['gate_continue'] and p['actual_next_gain']==0 for p in pp])),('selected_positive_unprofitable',np.array([p['gate_continue'] and p['actual_next_gain']>0 and p['actual_next_NET']<=0 for p in pp]))]:
                if mask.any():featurestats.append(dict(fold=f,seed=s,group=label,rows=int(mask.sum()),median=np.median(summaries[hi[mask]],axis=0).tolist(),q25=np.quantile(summaries[hi[mask]],.25,axis=0).tolist(),q75=np.quantile(summaries[hi[mask]],.75,axis=0).tolist()))
            poolchecks.append(dict(fold=f,seed=s,fit_rows=len(fi),held_rows=len(hi),fit_dates=sorted(fitdates),scaling_fit_only=True,scale_parameters=scales));ledger+=pools
    assert len(ledger)==4681;counts={k:sum(r['selected'] and (r['profitable'] if k=='profitable' else not r['event'] if k=='zero' else r['event'] and not r['profitable']) for r in ledger) for k in ['profitable','zero','positive_unprofitable']};assert counts==dict(profitable=53,zero=762,positive_unprofitable=6);neighbor_summary={}
    for label,rr in [('selected_profitable',[r for r in ledger if r['selected'] and r['profitable']]),('selected_zero',[r for r in ledger if r['selected'] and not r['event']]),('all_selected',[r for r in ledger if r['selected']])]:
        neighbor_summary[label]={name:dict(rows=len(rr),first_neighbor_positive=sum(r['neighbors'][name][0]['event'] for r in rr),three_distinct_call_positive_count=sum(sum(n['event'] for n in r['neighbors'][name]) for r in rr),median_first_distance=float(np.median([r['neighbors'][name][0]['distance'] for r in rr]))) for name in ['RAW41','PUBLIC66','HIDDEN83']}
    summary=dict(selected_counts=counts,exact_collisions=collisions,hidden_conflicting_event_groups=hiddenconflicts,neighbor_descriptions=neighbor_summary,focus_pairs=len(focuspairs),focus_query_matched_pairs=sum(r['query_count_difference']==0 for r in focuspairs),all_rows=4681,new_fits=0,new_NET_forwards=0,new_physics=0);save('public_descriptors.json',dict(names=feature_names,metadata_indices=list(range(len(meta))),values=summaries.tolist(),inputs_only=True));save('all_state_neighbors.json',ledger);save('feature_distributions.json',featurestats);save('focus_zero_pairs.json',focuspairs);save('fit_scaling_and_checks.json',poolchecks);save('summary.json',summary)
    lines=['# GRID205：已有反馈能否区分有利与无效补查','','本轮读取全部4681个固定停止点，保留当前821次追加中的53次局部回本、6次正奖励但未回本、762次零奖励。比较原41维公开摘要、加上25项固定已购反馈/已知接线结构描述后的66维摘要、以及原冻结隐藏83维。未拟合或部署任何新分类器。','','## 输入和分析边界','','新增描述只用已购买反馈的奖励差分布、合法性标记，以及已知待查候选与当前/已查接线的距离；没有使用待查结果、日期、编号或后验奖励作为特征。3582个训练路径状态全部保留，模型所属折的12个拟合日期用于标准化，留出日期不参与标准化。原83维仍用旧模型固定标准化。','','## 完全相同输入是否出现冲突','','|输入签名|状态|不同签名|重复组|事件标签冲突组|','|---|---:|---:|---:|---:|']
    for name,z in collisions.items():lines.append(f"|{name}|{z['states']}|{z['unique_signatures']}|{z['repeated_signature_groups']}|{len(z['conflicting_event_groups'])}|")
    lines+=['',f"同折、同种子的83维隐藏输入另有{len(hiddenconflicts)}组完全相同但事件结果不同的留出输入。无冲突只能排除这批记录中的精确冲突，不能证明可学习、可泛化或没有近似信息丢失。配对/非配对签名分别将接线与反馈一起排序/分别排序，模拟原FACTOR不保留对应关系这一限制。",'', '## 最近训练样本的描述','', '每个留出状态，在对应拟合折找三个不同物理调用的最近状态。检索只按固定距离和调用去重，不按标签选邻居；检索完成后附上已有教师标签。没有训练或评价最近邻策略，也未以这些统计决定触发门槛。距离依赖所选表示与标准化，三种表示的距离数值不能直接比较。','', '|查询组|表示|配对数|最近训练调用对应状态为正例|前三调用正例总数|最近距离中位数|','|---|---|---:|---:|---:|---:|']
    for label,rr in neighbor_summary.items():
        for name,z in rr.items():lines.append(f"|{label}|{name}|{z['rows']}|{z['first_neighbor_positive']}|{z['three_distinct_call_positive_count']}|{z['median_first_distance']:.6f}|")
    lines+=['','## 主要恢复案例与无效追加对照','', f"主要恢复调用的{len(focuspairs)}个种子—起点配对，分别与同折同种子、另一个调用的已选零奖励状态作事后对照。先匹配查询数，再按PUBLIC66距离配对；其中{sum(r['query_count_difference']==0 for r in focuspairs)}个查询数完全相同。这是用真实零奖励条件筛出的解释配对，绝不是在线检索或可部署规则。",'', '|种子/恢复起点前8位|查询数|零奖励对照调用|对照查询数|公开描述的前三项标准化差异|','|---|---:|---|---:|---|']
    for r in focuspairs:lines.append(f"|{r['seed']}/{r['restored_sample_id'][:8]}|{r['restored_queries']}|{r['zero_call']}|{r['zero_queries']}|"+'；'.join(f"{z['feature']} {z['standardized_difference']:+.3f}" for z in r['largest_standardized_differences'][:3])+'|')
    lines+=['','全66项中位数和四分位数按每个模型池、三类追加分别保存，不以差异大小筛选训练特征。配对中存在数值差异也不说明模型可可靠区分；同一个成功调用的多起点和跨种子相关。','', '下一步需结合最近训练支持和损失来源，判断更值得改事件识别还是幅度估计，不能仅因为某项对照差异大就新增一个触发条件。本轮所有描述都基于已暴露教师路径，未改模型、费用、标签、阈值或原FRAME比较。']
    (O/'GRID205_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');verify();save('review.json',dict(passed=True,all4681_public_and_hidden_descriptive_comparisons=True,all12_fit_scalings_held_dates_excluded=True,neighbors_chosen_without_teacher_labels=True,all53_profitable_762zero_6positive_unprofitable_retained=True,focus_pairs_explicitly_hindsight_only=True,new_fits=0,new_NET_forwards=0,new_physics=0));save('complete.json',dict(passed=True,new_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('complete');dest=R/'论文协作资料/FRAME_公开反馈区分_GRID205';dest.mkdir(exist_ok=False)
    for n in ['GRID205_report.md','summary.json','focus_zero_pairs.json','review.json','complete.json','design.json']:shutil.copy2(O/n,dest/n)
    print('GRID205_COMPLETE',json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':globals()[sys.argv[1]]()
