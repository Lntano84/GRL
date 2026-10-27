"""Training-support feasibility only; no router, combination or policy fitting."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid212_cancellation_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID212_cancellation_support';OLD=R/'outputs/grid208_public_amount';NEW=R/'outputs/grid210_amount_depth';FULL=R/'outputs/grid206_public_event/fit_features';FIT=R/'outputs/grid192_gate_support';LABEL=R/'outputs/grid196_gain_readout/fit_labels';CACHE=R/'outputs/grid190_extension_gate'
sp=importlib.util.spec_from_file_location('grid212_fixed_helpers',R/'work/grid210_amount_depth/run.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
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
    assert not O.exists();O.mkdir(parents=True);C.trained();assert read(R/'outputs/grid211_budget_sources/independent_audit.json')['passed']
    save('design.json',dict(id='GRID212_CANCELLATION_SUPPORT',scope='Frozen208and210 heads on original192 fullfitrows;existing held cancellations only for support comparison. No newmodel/router/combinedpolicy.',question='Doesfit supportcontain old208CONT/new210STOP states withboth profitable andunprofitablelabels in actualNETSTOPdomain? SeparateoriginalSTOP andpreSTOPCONT domains,all12pools.',scoring='36frozenforestscorecalls onfittingrows:existingevent andbothamounts. Input124from206fullfitarrays,knowncharge original192rawx82float32;actualNET=196target-196knownchargefloat64. Record any fee-rounding labeldifference,never adjust fees orlabels.',labels='Attach actualgain/feeNET only after candidatecancel mask oldp*oldmu>charge andnot oldp*newmu>charge fixed. This is descriptive selecteddomain,nottraininganswersacceptedforfit;in-sampleheads choosefittingdisagreements,optimism/exposure bias mustbe reported.',held='All157savedheldcancellationpairs. Nearestup to3distinctfitcalls amongactualSTOP-domain candidatecancellations in existing124standardizedspace,labels appendedafterdistance. Empty support explicitlyretained;no nearestneighborpolicy oraccuracytest.',coverage='Perpool/domain labels,dates,calls/unique state IDs and originalmixture weight. Fit12held4isolation. Folds/seeds repeat data,notindependent. No labelconditioned featureselection.',limits='Zero fits/NNforwards/newphysics/final8/paper/FRAME;no routerconstruction,policycombination/evaluation,fees/thresholdchanges orautomatictraining. Only frozenforestfittingrow inference allowed.'))
    files=[W/'run.py',O/'design.json',R/'work/grid210_amount_depth/run.py',R/'outputs/grid211_budget_sources/independent_audit.json',R/'outputs/grid211_budget_sources/summary.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']
    for d in [OLD,NEW]:files += [d/n for n in ['gate_predictions.json','complete.json','independent_audit.json']]+list((d/'models').iterdir())
    files+=list(FULL.glob('*.npz'))+list((FIT/'fit_features').glob('*.npz'))+list(LABEL.glob('*.npz'))+list(CACHE.glob('held_f*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID212_PLANNED')
def evaluate():
    verify();assert not (O/'started.json').exists();save('started.json',dict(new_fits=0,frozen_forest_fit_score_calls_planned=36));meta,lookup,cases,raw=C.raw_inputs();oldheld={(p['seed'],p['sample_id']):p for p in read(OLD/'gate_predictions.json')};held=read(NEW/'gate_predictions.json');fitledger=[];heldledger=[];pools=[];scorecalls=0
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';full=np.load(FULL/f'{name}.npz');orig=np.load(FIT/'fit_features'/f'{name}.npz');lab=np.load(LABEL/f'{name}.npz');x=full['standard'];ids=full['ids'].astype(int);n=int(orig['original_STOP_count']);assert np.array_equal(ids,orig['ids']);assert np.array_equal(x[:,:83],orig['standard']);assert np.array_equal(lab['target']>0,orig['y']>0);charge=orig['x'][:,82];event=pickle.loads((OLD/'models'/f'{name}_event.pkl').read_bytes());old=pickle.loads((OLD/'models'/f'{name}_amount.pkl').read_bytes());new=pickle.loads((NEW/'models'/f'{name}_amount.pkl').read_bytes());p=event.predict_proba(x)[:,1];muold=old.predict(x);munew=new.predict(x);scorecalls+=3;cancel=(p*muold>charge)&~(p*munew>charge);net=lab['target']-lab['known_charge'];roundnet=lab['target']-charge;records=[]
            fd={meta[int(i)]['date'] for i in ids};hz=np.load(CACHE/f'held_{name}.npz');hd={meta[int(i)]['date'] for i in hz['ids']};assert len(fd)==12 and len(hd)==4 and not fd&hd
            for domain,mask in [('actual_STOP',np.arange(len(x))<n),('preSTOP_CONT',np.arange(len(x))>=n)]:
                idx=np.flatnonzero(cancel&mask);profit=idx[net[idx]>0];unprofit=idx[net[idx]<=0];records.append(dict(domain=domain,total_rows=int(mask.sum()),candidate_cancel_rows=len(idx),profitable=len(profit),unprofitable=len(unprofit),zero_gain=int((lab['gain'][idx]==0).sum()),profitable_dates=len({meta[int(ids[k])]['date'] for k in profit}),profitable_calls=len({meta[int(ids[k])]['physical_state_group'] for k in profit}),unprofitable_dates=len({meta[int(ids[k])]['date'] for k in unprofit}),unprofitable_calls=len({meta[int(ids[k])]['physical_state_group'] for k in unprofit}),original_mixture_weight=float(orig['mass'][idx].sum()),profitable_original_weight=float(orig['mass'][profit].sum()),rounding_profit_sign_differences=int(np.sum((net[idx]>0)!=(roundnet[idx]>0)))))
                for k in idx:
                    m=meta[int(ids[k])];fitledger.append(dict(fold=f,seed=s,index=int(ids[k]),sample_id=m['sample_id'],position=m['position'],date=m['date'],call=m['physical_state_group'],domain=domain,event_p=float(p[k]),old_mu=float(muold[k]),new_mu=float(munew[k]),charge=float(charge[k]),actual_NET=float(net[k]),actual_gain=float(lab['gain'][k]),original_mass=float(orig['mass'][k])))
            stopidx=np.flatnonzero(cancel&(np.arange(len(x))<n));stopgroups={meta[int(ids[k])]['physical_state_group'] for k in stopidx};cp=np.load(OLD/'models'/f'{name}.npz');hx=np.c_[(hz['features']-cp['mean'])/cp['scale'],(raw[hz['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']];byglobal={int(i):j for j,i in enumerate(hz['ids'])};poolheld=[q for q in held if q['fold']==f and q['seed']==s and not q['gate_continue'] and oldheld[s,q['sample_id']]['gate_continue']]
            for q in poolheld:
                idx=lookup[q['sample_id'],q['position']];j=byglobal[idx];neighbors=[]
                if len(stopidx):
                    dist=np.mean((x[stopidx]-hx[j])**2,axis=1);order=np.argsort(dist,kind='stable');seen=set()
                    for pos in order:
                        k=int(stopidx[pos]);m=meta[int(ids[k])];g=m['physical_state_group']
                        if g in seen:continue
                        seen.add(g);neighbors.append(dict(index=int(ids[k]),call=g,date=m['date'],distance=float(dist[pos]),actual_NET=float(net[k]),profitable=bool(net[k]>0)))
                        if len(neighbors)==3:break
                heldledger.append(dict(fold=f,seed=s,sample_id=q['sample_id'],position=q['position'],date=q['date'],call=q['physical_state_group'],profitable=q['actual_next_NET']>0,actual_NET=q['actual_next_NET'],fit_STOP_cancel_rows=len(stopidx),fit_STOP_cancel_calls=len(stopgroups),neighbors=neighbors))
            pools.append(dict(fold=f,seed=s,domains=records,fit_dates=sorted(fd),held_dates=sorted(hd),held_removed_rows=len(poolheld)))
    assert scorecalls==36 and len(heldledger)==157 and sum(q['profitable'] for q in heldledger)==6;totals={}
    for domain in ['actual_STOP','preSTOP_CONT']:
        rr=[r for r in fitledger if r['domain']==domain];totals[domain]=dict(repeated_model_state_rows=len(rr),unique_state_indices=len({r['index'] for r in rr}),profitable_rows=sum(r['actual_NET']>0 for r in rr),unprofitable_rows=sum(r['actual_NET']<=0 for r in rr),pools_with_both_labels=sum(any(d['domain']==domain and d['profitable']>0 and d['unprofitable']>0 for d in p['domains']) for p in pools),pools_without_profitable_cancellations=sum(any(d['domain']==domain and d['profitable']==0 for d in p['domains']) for p in pools))
    htot={label:dict(rows=len(rr),empty_fit_support=sum(not r['neighbors'] for r in rr),nearest_profitable=sum(bool(r['neighbors']) and r['neighbors'][0]['profitable'] for r in rr),any_of_three_profitable=sum(any(n['profitable'] for n in r['neighbors']) for r in rr)) for label,rr in [('held_profitable_removed',[r for r in heldledger if r['profitable']]),('held_unprofitable_removed',[r for r in heldledger if not r['profitable']])]};summary=dict(fit_domains=totals,held_support=htot,pools=pools,new_fits=0,frozen_forest_fit_score_calls=scorecalls,new_NET_forwards=0,new_physics=0);save('fit_candidate_cancellations.json',fitledger);save('held_cancellation_neighbors.json',heldledger);save('summary.json',summary)
    lines=['# GRID212：取消补查的训练支持是否足够','','本轮没有训练或执行路由、组合策略。只用现有事件与两版收益森林在原拟合输入上各评分一次，共36次冻结森林批量调用。候选取消严格由固定分歧定义：208认为应追加、210认为应停止；之后才附上真实增量标签。实际NET停止域与搜索中原本继续域分开统计。','','## 每个模型的拟合支持','', '|折/种子|域|原行数|候选取消|实际回本/不回本|回本调用/日期|不回本调用/日期|','|---|---|---:|---:|---:|---:|---:|']
    for p in pools:
        for d in p['domains']:lines.append(f"|{p['fold']}/{p['seed']}|{d['domain']}|{d['total_rows']}|{d['candidate_cancel_rows']}|{d['profitable']}/{d['unprofitable']}|{d['profitable_calls']}/{d['profitable_dates']}|{d['unprofitable_calls']}/{d['unprofitable_dates']}|")
    lines+=['','## 汇总与留出损失支持','', '|域|重复模型—状态行|不同原状态|实际回本行|不回本行|有正反两类的模型池|无回本样本的模型池|','|---|---:|---:|---:|---:|---:|---:|']
    for d,z in totals.items():lines.append(f"|{d}|{z['repeated_model_state_rows']}|{z['unique_state_indices']}|{z['profitable_rows']}|{z['unprofitable_rows']}|{z['pools_with_both_labels']}|{z['pools_without_profitable_cancellations']}|")
    lines+=['','同一原状态可能在多个外折拟合池、种子重复出现，不能把模型—状态行数当独立训练场景。候选由拟合过这些数据的森林挑选，存在拟合内选择效应；标签真实不代表所选分歧能代表留出部署分布。','', '|留出类别|配对|无拟合STOP候选支持|最近调用状态回本|前三调用含回本|','|---|---:|---:|---:|---:|']
    for label,z in htot.items():lines.append(f"|{label}|{z['rows']}|{z['empty_fit_support']}|{z['nearest_profitable']}|{z['any_of_three_profitable']}|")
    lines+=['','最近邻只用对应折的已有124维标准化距离，并按物理调用去重，标签检索后附加；不是训练、测试最近邻分类器，也没有据此做触发规则。距离很近也不证明动作价值相同。','', '六个被取消的回本配对：','', '|种子/调用|拟合STOP候选行/调用|最近三个训练调用标签|','|---|---:|---|']
    for q in heldledger:
        if q['profitable']:lines.append(f"|{q['seed']}/{q['call']}|{q['fit_STOP_cancel_rows']}/{q['fit_STOP_cancel_calls']}|"+'；'.join(f"{n['call']}:{'回本' if n['profitable'] else '不回本'}" for n in q['neighbors'])+'|')
    lines+=['','## 后续边界','', '这些统计用于决定能否研究选择性取消，不能直接据拟合内分歧去训练并声称可保护留出收益。若样本稀少或缺少回本反例，应先解决分歧训练样本的来源与覆盖，而不是以留出六个损失补标签或调门槛。','', '208仍保留为当前候选。本轮不构造新策略、不扩模型、不调费用、标签或阈值；无新网络前向、物理、最终8、论文或原FRAME比较。']
    (O/'GRID212_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');verify();save('review.json',dict(passed=True,all12_fitSTOP_and_CONT_domains_separated=True,all157held_cancellations_support_retained=True,fit_held_dates_disjoint=True,labels_not_used_in_gate_masks_or_neighbor_retrieval=True,no_router_fit_or_policy_combination=True,new_fits=0,frozen_forest_fit_score_calls=36,new_NET_forwards=0,new_physics=0));save('complete.json',dict(passed=True,new_fits=0,frozen_forest_fit_score_calls=36,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('complete');dest=R/'论文协作资料/FRAME_取消补查训练支持_GRID212';dest.mkdir(exist_ok=False)
    for n in ['GRID212_report.md','summary.json','held_cancellation_neighbors.json','review.json','complete.json','design.json']:shutil.copy2(O/n,dest/n)
    print('GRID212_COMPLETE',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
