"""Independent fixed native and mixture checks; preserve useful and adverse signals."""
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
            name=f'f{f}_s{s}';cp=np.load(O/'models'/f'{name}.npz');hz=np.load(C.CACHE/f'held_{name}.npz');x=np.c_[(hz['features']-cp['mean'])/cp['scale'],(raw[hz['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']].astype(np.float32);event=pickle.loads((O/'models'/f'{name}_event.pkl').read_bytes());amount=pickle.loads((O/'models'/f'{name}_amount.pkl').read_bytes());p=np.zeros(len(x));mu=np.zeros(len(x))
            for tree in event.estimators_:v=tree.tree_.value[tree.apply(x),0,:];p+=v[:,1]/v.sum(1)/64
            for tree in amount.estimators_:mu+=tree.tree_.value[tree.apply(x),0,0]/64
            for j,i in enumerate(hz['ids']):
                r=ix[f,s,int(i)];assert np.isclose(p[j],r['event_probability'],atol=1e-10,rtol=1e-10) and np.isclose(mu[j],r['conditional_normalized_gain'],atol=1e-10,rtol=1e-10);assert r['gate_continue']==bool(p[j]*mu[j]>hz['features'][j,82]);checked+=1
    assert checked==4681;ni={(r['method'],r['sample_id']):r for r in read(O/'oof_trajectories.json')};comparisons=read(O/'expected_budget_comparison.json');budgetchecks=0
    for name,folder in [('GRID208',C.OLD),('GRID206',C.PRIOR)]:
        oi={(r['method'],r['sample_id']):r for r in read(folder/'oof_trajectories.json')}
        for row in comparisons[name]['per_date_seed']:
            s,d=row['seed'],row['date'];ids=[c['sample_id'] for c in cases if c['date']==d];base=[ni[f'NET_seed{s}',i] for i in ids];arms={'OLD':[oi[f'GATE_seed{s}',i] for i in ids],'NEW':[ni[f'GATE_seed{s}',i] for i in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',i] for i in ids]};costs={k:sum(a['queries']-b['queries'] for a,b in zip(rr,base)) for k,rr in arms.items()};budget=min(costs['OLD'],costs['NEW']);assert budget==row['matched_extra_query_total']
            for k,rr in arms.items():
                chance=budget/costs[k] if costs[k] else 0.;values=[(b['net_utility']+chance*(a['net_utility']-b['net_utility']),b['queries']+chance*(a['queries']-b['queries']),float(b['action_match'])+chance*(int(a['action_match'])-int(b['action_match']))) for a,b in zip(rr,base)];v=np.mean(values,axis=0);assert np.allclose(v,[row[k]['mean_net_utility'],row[k]['mean_queries'],row[k]['action_match_rate']],atol=1e-10,rtol=1e-10)
            budgetchecks+=1
    assert budgetchecks==96;decision=read(O/'decision.json');assert decision['main_recovery_retained']==decision['main_recovery_total']==10 and decision['previous_four_recovered']==0;pm=read(O/'predictive_metrics.json')['all12_mean'];opm=read(C.OLD/'predictive_metrics.json')['all12_mean'];family=read(O/'family_summary.json');oldfamily=read(C.OLD/'family_summary.json');new,old=family['GATE'],oldfamily['GATE'];matched=comparisons['GRID208'];s1=next(r for r in matched['per_seed'] if r['seed']==1);raws1=next(r for r in decision['actual_per_seed'] if r['seed']==1);checks=dict(conditional_positive_MSE_delta=pm['positive_conditional_MSE']-opm['positive_conditional_MSE'],mean_predicted_gain_delta=pm['mean_expected_gain']-opm['mean_expected_gain'],actual_queries_delta=new['mean_queries']-old['mean_queries'],actual_NET_delta=new['mean_net_utility']-old['mean_net_utility'],matched_NET_delta=decision['matched']['GRID208']['NET_delta'],seed1_matched_NET_delta=s1['NET_delta'],seed1_actual_NET_delta=raws1['NET_delta']);C.save('tradeoff_interpretation.json',checks)
    lines=['# GRID210判读：保留深4主候选，不继续加深','', '这轮只把条件收益树从深4加到深5。全部参数和正例数组核验表明，只有max_depth变化，概率模型完全固定。主要10个恢复配对保留，但旧4个损失一个也未找回。结构上有继续分割的空间，没有转化成这些案例的有效修复。','', f"平均候选查询从{old['mean_queries']:.5f}到{new['mean_queries']:.5f}，提供了更低查询的取舍；局部奖励保留从{100*old['gain_capture']:.4f}%到{100*new['gain_capture']:.4f}%，代理净效用从{old['mean_net_utility']:.9f}到{new['mean_net_utility']:.9f}。原始策略中一个种子净效用小幅提高、两个下降，种子2的查询还略增。",'', f"同预期查询对208平均净效用变化{decision['matched']['GRID208']['NET_delta']:+.9f}，动作变化{decision['matched']['GRID208']['match_delta_pp']:+.5f}个百分点。种子1保留了小正向：同预期查询净效用{s1['NET_delta']:+.9f}、动作{s1['match_delta_pp']:+.5f}个百分点；但其实际固定策略净效用为{raws1['NET_delta']:+.9f}。这两个参照不同，不能把前者写成原始策略实际获胜。",'', f"正例条件幅度MSE由{opm['positive_conditional_MSE']:.6f}到{pm['positive_conditional_MSE']:.6f}，总体预测奖励均值反而由{opm['mean_expected_gain']:.6f}到{pm['mean_expected_gain']:.6f}，实际均值仍为{pm['mean_actual_gain']:.6f}。误差略低与少查、漏掉部分奖励可以同时发生。平均误差不能替代查询选择的评价，这也不能单凭一轮就证明目标函数错误。",'', '当前保留GRID208作为较均衡的内部学习候选，同时保存GRID210较低查询端和种子1的同预算小改善。后续不再扫描更深模型；先解释种子1改善的查询分配和种子2主要损失，分清有效取消无益查询与误删有利追加。', '', '这轮新增12个选择中没有一个实际回本，取消157个旧追加时损失6个回本配对；全部案例和成本贡献保留。不能仅看少了135次零收益追加就称学习更好，也不因总体下降丢掉种子1线索。', '', '已复核全部4681原生树输出和96组同预期预算混合，所有种子、日期及反例保留。没有新增物理回放、最终测试、原FRAME比较或论文改动。']
    (O/'结果判读与下一步.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all4681_native_leaf_products_and_decisions_reconstructed=True,all96_expected_query_match_and_utility_mixtures_reconstructed=True,all_positive_and_adverse_seed_signals_retained=True,no_deeper_model_or_threshold_search=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_收益深度对照_GRID210'
    for n in ['结果判读与下一步.md','tradeoff_interpretation.json','independent_audit.json']:shutil.copy2(O/n,dest/n)
    msg='GRID210核验：深5正例MSE略降但总体预测均值更低，实际净效用一好两差，同预期预算对208平均负；种子1同预算小正向保留，不能与其实际负向混用。主10恢复保住、旧4损失未找回，取消另6回本配对。保留208主候选与210低查询端，不再加深，先只读解释种子1改善与种子2损失。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID210取舍与后续核验\n\n'+msg+'\n',encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID210核验后\n\n1. 保留208较均衡主候选和210低查询端/种子1同预算小改善，不再扫深度。\n2. 只读解释种子1改善、种子2主要损失及6个新取消回本配对；区分实际策略与解析同预期预算、平均MSE与查询价值。\n3. 有明确依据再通知固定新训练；不调费用阈值，不扩物理/最终8/论文/原FRAME。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8').replace('[报告](protocols/GRID210_amount_depth/complete/GRID210_report.md)','[报告](protocols/GRID210_amount_depth/audited/GRID210_report.md)；[判读](protocols/GRID210_amount_depth/audited/结果判读与下一步.md)',1),encoding='utf8',newline='\n');print('GRID210_INDEPENDENT_AUDIT',json.dumps(checks))
def delivery():
    C.trained();head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;stages=[]
    for out,arch,collab in [('grid209_lost_gain','GRID209_lost_gain','FRAME_回本损失与叶子限制_GRID209'),('grid210_amount_depth','GRID210_amount_depth','FRAME_收益深度对照_GRID210')]:
        src=R/'outputs'/out;a=C.REPO/'research_archive/power-grid-2026/protocols'/arch;dest=R/'论文协作资料'/collab;assert read(src/'complete.json')['passed'] and read(src/'independent_audit.json')['passed'];count=0
        for mf in a.rglob('archive_manifest.json'):
            for p,h in read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
        for p in dest.iterdir():
            assert C.sha(p)==C.sha(src/p.name);matches=[ap for ap in a.rglob(p.name) if ap.is_file() and C.sha(ap)==C.sha(p)];assert matches;assert any(git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==p.read_bytes() for ap in matches)
        for p,h in read(src/'frozen_inputs.json').items():assert C.sha(R/p)==h,p
        stages.append(dict(stage=arch,archive_files_checked=count,collaboration_files_checked=len(list(dest.iterdir())),committed_bytes_match=True))
    rec=dict(passed=True,verified_publication_head=head,remote_main=head,stages=stages,new_fits=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'delivery_receipt.json',R/'论文协作资料/FRAME_收益深度对照_GRID210/delivery_receipt.json');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
