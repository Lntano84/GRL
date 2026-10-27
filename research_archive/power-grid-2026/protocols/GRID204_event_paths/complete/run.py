"""Read-only path and fitting-support explanations for fixed event heads."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,hashlib,importlib.util,json,pickle,shutil,subprocess,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid204_event_paths';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID204_event_paths';NEW=R/'outputs/grid203_tree_event';OLD=R/'outputs/grid201_tree_amount';FIT=R/'outputs/grid192_gate_support';CACHE=R/'outputs/grid190_extension_gate'
sp=importlib.util.spec_from_file_location('grid204_frozen_helpers',R/'work/grid203_tree_event/run.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
BLOCKS={'main_hidden64':(0,64),'relation_hidden16':(64,80),'NET_Q2':(80,82),'known_charge1':(82,83)}

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
    assert not O.exists();O.mkdir(parents=True);C.trained();assert read(NEW/'complete.json')['passed'] and read(NEW/'independent_audit.json')['passed']
    save('design.json',dict(id='GRID204_EVENT_PATHS',scope='All4681 saved held STOP choices, all12 fixed201/203 heads; no fit/NNforward/newphysics.',motivation='Keep203improvement;explain sameinput event differences in largest restored call and all losses before further learning.',linear='Exact intercept plus standard83*coef blocks main64/relation16/Q2/knowncharge1, telescopes to fixed logit. Contributions correlated and basis-dependent,not causal.',trees='Root class frequency plus child-minus-parent classfrequency along native decision paths, averaged64trees and grouped by split feature block. Exact telescoping probability,not SHAP/causal effect; no feature zeroing or counterfactual policy.',support='Forest native leaf probability equals sum over fitrows of average64same-leaf normalized sampleweights times eventlabel. Reconstruct allheld kernels; split STOP/CONT positive/negative support, effective row/call count, top5 contributing positive fitcalls. Fit dates exclude corresponding held dates; no nearest unseen teacher outcomes used.',focus='Previous frozen largest-call2037-01-05_101|call001 and five largest adverse calls from203interpretation; all4681 retained. Compare baseNET already-recovered starts across allthree seeds, not just eligible rows.',limits='All exposed teacher-path exploratory diagnosis; no parameter/threshold/fee/label changes or automatic training/FRAME/final8/paper. Kernel support is training influence algebra,not independent evidence or causal proof. New probabilities native mixture scores,not calibrated STOP risk.'))
    files=[W/'run.py',O/'design.json',R/'work/grid203_tree_event/run.py',NEW/'complete.json',NEW/'independent_audit.json',NEW/'interpretation.json',NEW/'interpretation_rows.json',NEW/'gate_predictions.json',NEW/'oof_trajectories.json',OLD/'gate_predictions.json',R/'outputs/grid113_relational_learning/train_metadata.json']+list((NEW/'models').iterdir())+list((OLD/'models').iterdir())+list((FIT/'fit_features').glob('*.npz'))+list(CACHE.glob('held_f*.npz'))
    save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID204_PLANNED')

def evaluate():
    verify();assert not (O/'started.json').exists();save('started.json',dict(new_fits=0,new_NET_forwards=0,new_physics=0));meta,lookup,cases=C.B.load();pred=read(NEW/'gate_predictions.json');prev={(p['seed'],p['sample_id']):p for p in read(OLD/'gate_predictions.json')};byidx={(p['fold'],p['seed'],lookup[p['sample_id'],p['position']]):p for p in pred};diag=read(NEW/'interpretation.json');focuscall=max(diag['per_call_contributions'],key=lambda r:r['macro_delta_utility'])['call'];adverse=[r['call'] for r in sorted(diag['per_call_contributions'],key=lambda r:r['macro_delta_utility'])[:5]];rows=[];poolchecks=[];eps=0.
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(FIT/'fit_features'/f'{name}.npz');hz=np.load(CACHE/f'held_{name}.npz');cp=np.load(OLD/'models'/f'{name}.npz');forest=pickle.loads((NEW/'models'/f'{name}_event.pkl').read_bytes());x=(hz['features']-cp['mean'])/cp['scale'];train=z['standard'];w=z['sample_weight'];labels=z['y'];n=int(z['original_STOP_count']);trainmeta=[meta[int(i)] for i in z['ids']];heldmeta=[meta[int(i)] for i in hz['ids']];fitdates={m['date'] for m in trainmeta};assert not fitdates&{m['date'] for m in heldmeta};assert len(fitdates)==12
            groupnames=sorted({m['physical_state_group'] for m in trainmeta});gi={g:i for i,g in enumerate(groupnames)};groupidx=np.array([gi[m['physical_state_group']] for m in trainmeta]);kernel=np.zeros((len(x),len(train)));path=np.zeros((len(x),4));roots=np.zeros(len(x));blocks=list(BLOCKS);manualp=np.zeros(len(x))
            for tree in forest.estimators_:
                t=tree.tree_;values=t.value[:,0,:];nodep=values[:,1]/values.sum(axis=1);hl=tree.apply(x);tl=tree.apply(train);leafsum=np.bincount(tl,weights=w,minlength=t.node_count);same=hl[:,None]==tl[None,:];kernel+=same*w[None,:]/leafsum[hl,None]/64;manualp+=nodep[hl]/64;roots+=nodep[0]/64
                for j in range(len(x)):
                    node=0;count=0
                    while t.children_left[node]!=-1:
                        feature=t.feature[node];child=t.children_left[node] if np.float32(x[j,feature])<=t.threshold[node] else t.children_right[node];block=next(b for b,(lo,hi) in BLOCKS.items() if lo<=feature<hi);path[j,blocks.index(block)]+=(nodep[child]-nodep[node])/64;node=child;count+=1
                    assert node==hl[j] and count<=4
            native=forest.predict_proba(x)[:,1];assert np.allclose(kernel.sum(1),1,atol=1e-10,rtol=1e-10);assert np.allclose(kernel@labels,native,atol=1e-10,rtol=1e-10);assert np.allclose(roots+path.sum(1),native,atol=1e-10,rtol=1e-10);assert np.allclose(manualp,native,atol=1e-10,rtol=1e-10);logits=x@cp['event_coef']+cp['event_intercept'][0];oldprob=np.exp(-np.logaddexp(0.,-logits));components={b:(x[:,lo:hi]*cp['event_coef'][lo:hi]).sum(1) for b,(lo,hi) in BLOCKS.items()};assert np.allclose(sum(components.values())+cp['event_intercept'][0],logits,atol=1e-10,rtol=1e-10)
            for j,idx in enumerate(hz['ids']):
                p=byidx[f,s,int(idx)];old=prev[s,p['sample_id']];assert abs(native[j]-p['event_probability'])<1e-10 and abs(oldprob[j]-old['event_probability'])<1e-10;a,b=p['gate_continue'],old['gate_continue'];category='BOTH' if a and b else 'NEW_ONLY' if a else 'OLD_ONLY' if b else 'NEITHER';kk=kernel[j];gc=np.bincount(groupidx,weights=kk,minlength=len(groupnames));gp=np.bincount(groupidx,weights=kk*labels,minlength=len(groupnames));support=dict(STOP_positive=float(kk[:n]@labels[:n]),CONT_positive=float(kk[n:]@labels[n:]),STOP_negative=float(kk[:n]@(1-labels[:n])),CONT_negative=float(kk[n:]@(1-labels[n:])),effective_rows=float(1/(kk@kk)),effective_calls=float(1/(gc@gc)),supported_calls=int((gc>0).sum()),supported_dates=len({m['date'] for i,m in enumerate(trainmeta) if kk[i]>0}),top_positive_calls=[dict(call=groupnames[k],probability_contribution=float(gp[k]),total_support_mass=float(gc[k])) for k in sorted(range(len(gp)),key=lambda k:-gp[k])[:5] if gp[k]>0]);assert abs(support['STOP_positive']+support['CONT_positive']-native[j])<1e-10;assert abs(sum(support[k] for k in ['STOP_positive','CONT_positive','STOP_negative','CONT_negative'])-1)<1e-10
                row=dict(fold=f,seed=s,sample_id=p['sample_id'],position=p['position'],call=p['physical_state_group'],date=p['date'],group=category,actual_gain=p['actual_next_gain'],actual_NET=p['actual_next_NET'],known_charge=p['known_charge'],mu=p['conditional_normalized_gain'],old_p=float(oldprob[j]),new_p=float(native[j]),old_logit=float(logits[j]),linear_intercept=float(cp['event_intercept'][0]),linear_blocks={b:float(v[j]) for b,v in components.items()},tree_root_p=float(roots[j]),tree_path_blocks={b:float(path[j,k]) for k,b in enumerate(blocks)},support=support);rows.append(row)
            error=float(np.max(abs(kernel@labels-native)));eps=max(eps,error);poolchecks.append(dict(fold=f,seed=s,held_rows=len(x),fit_rows=len(train),max_kernel_error=error,held_fit_dates_disjoint=True,root_p=float(roots[0])))
    assert len(rows)==4681
    def summary(rr):
        if not rr:return dict(rows=0)
        return dict(rows=len(rr),calls=len({r['call'] for r in rr}),positive=sum(r['actual_gain']>0 for r in rr),profitable=sum(r['actual_NET']>0 for r in rr),mean_old_p=float(np.mean([r['old_p'] for r in rr])),mean_new_p=float(np.mean([r['new_p'] for r in rr])),mean_linear_logit_blocks={b:float(np.mean([r['linear_blocks'][b] for r in rr])) for b in BLOCKS},mean_tree_path_blocks={b:float(np.mean([r['tree_path_blocks'][b] for r in rr])) for b in BLOCKS},mean_root_p=float(np.mean([r['tree_root_p'] for r in rr])),mean_STOP_positive_support=float(np.mean([r['support']['STOP_positive'] for r in rr])),mean_CONT_positive_support=float(np.mean([r['support']['CONT_positive'] for r in rr])),mean_effective_calls=float(np.mean([r['support']['effective_calls'] for r in rr])))
    groups={k:summary([r for r in rows if r['group']==k]) for k in ['NEW_ONLY','OLD_ONLY','BOTH','NEITHER']};focus=[r for r in rows if r['call']==focuscall];positivefocus=[r for r in focus if r['group']=='NEW_ONLY' and r['actual_gain']>0];focusseeds=[dict(seed=s,**summary([r for r in positivefocus if r['seed']==s])) for s in range(3)];tr=read(NEW/'oof_trajectories.json');baseline=[]
    for s in range(3):
        rr=[r for r in tr if r['method']==f'NET_seed{s}' and r['physical_state_group']==focuscall];baseline.append(dict(seed=s,all_starts=len(rr),full_action_matches=sum(r['action_match'] for r in rr),eligible_starts=sum(r['eligible'] for r in rr),full_reward_recovered=sum(abs(r['gain']-r['full_gain'])<1e-12 for r in rr),mean_gain=float(np.mean([r['gain'] for r in rr]))))
    losses={g:summary([r for r in rows if r['call']==g and r['group'] in ['NEW_ONLY','OLD_ONLY']]) for g in adverse};result=dict(groups=groups,focus_call=focuscall,focus_positive_by_seed=focusseeds,focus_original_NET=baseline,adverse_calls=losses,all12pools=poolchecks,max_reconstruction_error=eps,all4681_checked=True,new_fits=0,new_NET_forwards=0,new_physics=0);save('event_path_rows.json',rows);save('summary.json',result)
    lines=['# GRID204：同一输入下，概率模型为什么分歧','','本轮只读12对固定模型，核查全部4681个可追加停止点。旧线性概率的对数几率按主隐藏64维、关系隐藏16维、NET两动作值及已知查询费用分解；新森林按每条决策路径的父子节点正例比例变化分解。两者都可精确还原输出，但不在同一尺度，不能将数值横向相加或当作因果贡献。','','## 新概率的训练依据','','对每个留出输入，森林预测可以写成训练样本标签的加权平均：64棵树中，和它落在同一叶子的训练样本得到权重。逐样本重建这组权重，核对其和为1，并核对正例部分之和等于预测概率。进一步区分这些支持来自原停止状态（STOP）还是搜索中原本继续的状态（CONT）。这是固定模型的代数解释，并不是独立验证或真实概率校准。','','|选择变化组|配对数|正收益/回本|旧概率均值|新概率均值|STOP正例支持|CONT正例支持|有效支持调用数均值|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,z in groups.items():lines.append(f"|{k}|{z['rows']}|{z['positive']}/{z['profitable']}|{z['mean_old_p']:.6f}|{z['mean_new_p']:.6f}|{z['mean_STOP_positive_support']:.6f}|{z['mean_CONT_positive_support']:.6f}|{z['mean_effective_calls']:.2f}|")
    lines+=['','这些均值只描述组内模型行为，不是原日期宏平均绩效。正负例和全部状态均保留，不按结果重新选阈值或训练权重。','','## 上轮主要恢复调用','',focuscall,'','|种子|新增且正收益配对|旧概率均值|新概率均值|STOP正例支持|CONT正例支持|','|---|---:|---:|---:|---:|---:|']
    for z in focusseeds:
        if z['rows']:lines.append(f"|{z['seed']}|{z['rows']}|{z['mean_old_p']:.8f}|{z['mean_new_p']:.6f}|{z['mean_STOP_positive_support']:.6f}|{z['mean_CONT_positive_support']:.6f}|")
        else:lines.append(f"|{z['seed']}|0|—|—|—|—|")
    lines+=['','三个种子的原NET在该调用全部起点上：','', '|种子|起点|已恢复完整奖励|已选完整动作|仍可追加|','|---|---:|---:|---:|---:|']
    for z in baseline:lines.append(f"|{z['seed']}|{z['all_starts']}|{z['full_reward_recovered']}|{z['full_action_matches']}|{z['eligible_starts']}|")
    lines+=['','不能把缺少某个种子的新增恢复直接解释为该概率头失败；需先看原NET是否已经取到了奖励，以及是否还存在可追加机会。','','该调用各新增正收益配对的分解：','', '|种子/起点前8位|旧logit：主/关系/Q/费用/截距|新概率：根/主/关系/Q/费用|有效支持调用|','|---|---|---|---:|']
    for z in positivefocus:lines.append(f"|{z['seed']}/{z['sample_id'][:8]}|"+'/'.join(f"{z['linear_blocks'][b]:+.4f}" for b in BLOCKS)+f"/{z['linear_intercept']:+.4f}|{z['tree_root_p']:.4f}/"+'/'.join(f"{z['tree_path_blocks'][b]:+.4f}" for b in BLOCKS)+f"|{z['support']['effective_calls']:.2f}|")
    lines+=['','## 反向调用完整保留','']
    for g,z in losses.items():lines.append(f"- {g}：{z['rows']}个改变决策配对，其中正收益{z.get('positive',0)}、回本{z.get('profitable',0)}；新旧概率及支持来源见逐行记录。")
    lines+=['','## 解释边界和下一步','','隐藏特征之间及其与NET动作值、费用之间相关。线性系数分组与树路径增量依赖模型参数和所走路径，不能说某输入组是电网收益的原因。CONT正例支持多也不意味着应该删除或增加这类数据；停止条件下的概率校准仍需独立检查。','', '这轮先保留被找回收益的线索，结合其训练支持与相似反例判断下一项学习改动，不从单个成功调用直接设计触发规则。全部是已暴露教师路径研究；无新拟合、神经网络前向、仿真、最终8或原FRAME比较。']
    (O/'GRID204_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');verify();save('review.json',dict(passed=True,all4681_old_logits_native_tree_paths_and_training_kernels_reconstructed=True,all12_fit_held_dates_disjoint=True,all_focus_and_adverse_and_baseNET_seeds_retained=True,new_fits=0,new_NET_forwards=0,new_physics=0));save('complete.json',dict(passed=True,all4681=True,new_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('complete');dest=R/'论文协作资料/FRAME_概率分歧解释_GRID204';dest.mkdir(exist_ok=False)
    for n in ['GRID204_report.md','summary.json','review.json','complete.json','design.json']:shutil.copy2(O/n,dest/n)
    print('GRID204_COMPLETE',json.dumps(result,ensure_ascii=False))

if __name__=='__main__':globals()[sys.argv[1]]()
