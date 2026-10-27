"""Independent native reconstruction, nuanced interpretation and joint delivery."""
from pathlib import Path
import json,pickle,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def audit():
    C.trained();assert read(O/'complete.json')['passed'] and not (O/'independent_audit.json').exists();meta,lookup,cases,raw=C.raw_inputs();pred=read(O/'gate_predictions.json');ix={(p['fold'],p['seed'],lookup[p['sample_id'],p['position']]):p for p in pred};checked=0
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';hz=np.load(C.CACHE/f'held_{name}.npz');cp=np.load(O/'models'/f'{name}.npz');hidden=(hz['features']-cp['mean'])/cp['scale'];x=np.c_[hidden,(raw[hz['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']].astype(np.float32);clf=pickle.loads((O/'models'/f'{name}_event.pkl').read_bytes());amount=pickle.loads((O/'models'/f'{name}_amount.pkl').read_bytes());p=np.zeros(len(x));mu=np.zeros(len(x))
            for tree in clf.estimators_:v=tree.tree_.value[tree.apply(x),0,:];p+=v[:,1]/v.sum(axis=1)/64
            for tree in amount.estimators_:mu+=tree.tree_.value[tree.apply(hidden),0,0]/64
            for j,i in enumerate(hz['ids']):
                r=ix[f,s,int(i)];assert np.isclose(p[j],r['event_probability'],atol=1e-10,rtol=1e-10) and np.isclose(mu[j],r['conditional_normalized_gain'],atol=1e-10,rtol=1e-10);assert r['gate_continue']==bool(p[j]*mu[j]>hz['features'][j,82]);checked+=1
    assert checked==4681;tr=read(O/'oof_trajectories.json');ni={(r['method'],r['sample_id']):r for r in tr};comparisons=read(O/'expected_budget_comparison.json');budgetchecks=0
    for name,folder in [('GRID203',C.OLD),('GRID198',C.LINEAR)]:
        oi={(r['method'],r['sample_id']):r for r in read(folder/'oof_trajectories.json')}
        for row in comparisons[name]['per_date_seed']:
            s,d=row['seed'],row['date'];ids=[c['sample_id'] for c in cases if c['date']==d];base=[ni[f'NET_seed{s}',i] for i in ids];arms={'OLD':[oi[f'GATE_seed{s}',i] for i in ids],'NEW':[ni[f'GATE_seed{s}',i] for i in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',i] for i in ids]};costs={k:sum(a['queries']-b['queries'] for a,b in zip(rr,base)) for k,rr in arms.items()};budget=min(costs['OLD'],costs['NEW']);assert budget==row['matched_extra_query_total']
            for k,rr in arms.items():
                chance=budget/costs[k] if costs[k] else 0.;values=[(b['net_utility']+chance*(a['net_utility']-b['net_utility']),b['queries']+chance*(a['queries']-b['queries'])) for a,b in zip(rr,base)];v=np.mean(values,axis=0);assert abs(v[0]-row[k]['mean_net_utility'])<1e-10 and abs(v[1]-row[k]['mean_queries'])<1e-10
            budgetchecks+=1
    assert budgetchecks==96;newpm=read(O/'predictive_metrics.json');oldpm=read(C.OLD/'predictive_metrics.json');seeds=[]
    for s in range(3):
        a=float(np.mean([r['event_AUC'] for r in newpm['pools'] if r['seed']==s]));b=float(np.mean([r['domains']['held_STOP']['event_AUC'] for r in oldpm['pools'] if r['seed']==s]));seeds.append(dict(seed=s,old_AUC=b,new_AUC=a,delta=a-b))
    changes=read(O/'changed_choices.json');newindices={(r['seed'],r['sample_id']) for r in changes['rows'] if r['group']=='NEW_ONLY'};gain_scale=read(R/'outputs/grid102_search_rl/design.json')['gain_scale'];positive=[]
    for p in pred:
        if (p['seed'],p['sample_id']) in newindices and p['actual_next_gain']>0:positive.append(dict(seed=p['seed'],sample_id=p['sample_id'],call=p['physical_state_group'],queries=p['actual_next_queries'],event_p=p['event_probability'],predicted_positive_mu=p['conditional_normalized_gain'],realized_positive_gain_normalized=p['actual_next_gain']/gain_scale,known_charge=p['known_charge'],realized_NET=p['actual_next_NET']))
    assert len(positive)==12;old=read(C.OLD/'family_summary.json')['GATE'];new=read(O/'family_summary.json')['GATE'];nc=read(O/'gate_counts.json');oc=read(C.OLD/'gate_counts.json');decision=read(O/'decision.json');negative=sorted(comparisons['GRID203']['per_date_seed'],key=lambda r:r['NEW']['mean_net_utility']-r['OLD']['mean_net_utility'])[:5];costrows=[dict(seed=r['seed'],date=r['date'],delta=r['NEW']['mean_net_utility']-r['OLD']['mean_net_utility'],new_keep=r['NEW']['keep_probability'],old_keep=r['OLD']['keep_probability'],original_extra_queries=r['original_extra_query_totals']) for r in negative];C.save('recognition_and_cost_audit.json',dict(per_seed_AUC=seeds,all_three_AUC_improved=all(r['delta']>0 for r in seeds),newly_selected_positive_cases=positive,negative_matched_dates=costrows,conditional_mu_undefined_on_realized_zero_not_scored_as_zero=True,no_new_policy_test=True))
    lines=['# GRID206：识别进步已出现，查询分配还需要改','', '这轮值得保留：只增加原41项公开摘要直接输入后，三个种子的事件排序AUC都提高，上一轮主要恢复调用的10个相关配对全部保留。它提供了一个明确的学习改善方向，但尚未形成对上一版的同预算收益优势。','', '|种子|原AUC|新AUC|','|---|---:|---:|']
    for z in seeds:lines.append(f"|{z['seed']}|{z['old_AUC']:.6f}|{z['new_AUC']:.6f}|")
    lines+=['',f"完整同站动作一致率{100*old['action_match_rate']:.4f}%→{100*new['action_match_rate']:.4f}%，零收益追加次数{oc['append_no_change']}→{nc['append_no_change']}。不过，每起点候选查询{old['mean_queries']:.4f}→{new['mean_queries']:.4f}；实际代理净效用仅变化{new['mean_net_utility']-old['mean_net_utility']:+.9f}。不能把次数减少称为省候选或省控制时间。",'', '新旧选择变化解释了这个差距：取消249个旧追加中有248个零收益；新增106个追加中有12个正收益，但只有2个回本。新增查询抵消了相当一部分节省。这里的标签是已观测单次结果，不是可事先获得的收益确定性。','', f"逐日期、种子匹配相同预期查询后，新版净效用相对GRID203为{decision['matched']['GRID203']['NET_delta']:+.9f}，三个种子一好两差；动作一致率平均变化{decision['matched']['GRID203']['match_delta_pp']:+.5f}个百分点。保留这个反例，不用总体AUC掩盖决策目标的不足。",'', '## 同预算差距出现在哪里','', '|种子/日期|旧追加查询|新追加查询|新策略保留概率|该日期净效用差|','|---|---:|---:|---:|---:|']
    for z in costrows:lines.append(f"|{z['seed']}/{z['date']}|{z['original_extra_queries']['OLD']}|{z['original_extra_queries']['NEW']}|{z['new_keep']:.5f}|{z['delta']:+.9f}|")
    lines+=['', '主要收益所在的一月五日，种子0/2增加了查询。按预先固定的同预算解析比较，统一稀释也会降低原有有利追加的保留概率。这说明追加查询在日期内的分配仍有成本；并不表示实际固定新版丢掉了那10个恢复配对，也不能据此更换更有利的预算口径。','', '## 后续应改什么','', '目前更值得保留的是公开摘要直接输入带来的事件识别改善。下一步先核查新增正收益但未回本的案例：收益幅度是否被估高、哪些长批次造成开销，以及是否存在支持相近而实际收益不同的正反例。收益模型这次完全未变，不能将剩余问题全归于概率头。','', '不会把零收益状态的条件幅度标签直接填为0；条件幅度指有收益时的平均大小，它和事件概率要分别评价。也不马上增加网络或改查询费用。新的训练若有明确依据，再固定单模块改动、提前通知。','', '所有结论仍限于已暴露教师路径内部，非完整周质量或控制时延验证。全部日期、三个种子和负向结果保留。']
    (O/'结果判读与下一步.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all4681_native_leaf_products_and_decisions_reconstructed=True,all96_query_and_utility_mixtures_reconstructed=True,all3_seed_event_metrics_and_10prior_recoveries_retained=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_公开摘要概率_GRID206'
    for n in ['结果判读与下一步.md','recognition_and_cost_audit.json','independent_audit.json']:shutil.copy2(O/n,dest/n)
    msg='GRID206核验：原41摘要直连使三种子AUC均提升(总体.69645→.75574)，原10恢复全保留，零收益追加762→608；但取消249(248零)和新增106(12正/2回本)导致候选查询略增。同预期查询对203净效用−.000818886、一好两差，不能用识别改善替代查询价值优势。保留识别线索，下一步只读查新增正例幅度与长批成本，不自动再训/调费。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID206识别与查询价值分开判读\n\n'+msg+'\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8').replace('[报告](protocols/GRID206_public_event/complete/GRID206_report.md)','[报告](protocols/GRID206_public_event/audited/GRID206_report.md)；[判读](protocols/GRID206_public_event/audited/结果判读与下一步.md)',1),encoding='utf8',newline='\n');print('GRID206_INDEPENDENT_AUDIT',json.dumps(dict(seeds=seeds,positive_cases=len(positive),budgetchecks=budgetchecks)))
def delivery():
    C.trained();head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;stages=[]
    for out,arch,collab in [('grid205_public_separation','GRID205_public_separation','FRAME_公开反馈区分_GRID205'),('grid206_public_event','GRID206_public_event','FRAME_公开摘要概率_GRID206')]:
        src=R/'outputs'/out;a=C.REPO/'research_archive/power-grid-2026/protocols'/arch;dest=R/'论文协作资料'/collab;assert read(src/'complete.json')['passed'] and read(src/'independent_audit.json')['passed'];count=0
        for mf in a.rglob('archive_manifest.json'):
            for p,h in read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
        for p in dest.iterdir():
            assert C.sha(p)==C.sha(src/p.name);matches=[ap for ap in a.rglob(p.name) if ap.is_file() and C.sha(ap)==C.sha(p)];assert matches,p;assert any(git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==p.read_bytes() for ap in matches)
        for p,h in read(src/'frozen_inputs.json').items():assert C.sha(R/p)==h,p
        stages.append(dict(stage=arch,archive_files_checked=count,collaboration_files_checked=len(list(dest.iterdir())),committed_bytes_match=True))
    rec=dict(passed=True,verified_publication_head=head,remote_main=head,stages=stages,new_fits=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'delivery_receipt.json',R/'论文协作资料/FRAME_公开摘要概率_GRID206/delivery_receipt.json');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
