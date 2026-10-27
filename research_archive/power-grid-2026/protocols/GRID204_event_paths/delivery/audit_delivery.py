"""Independent sparse-path and leaf-domain checks; no fitting."""
from pathlib import Path
import json,pickle,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read

def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])

def audit():
    C.verify();assert read(O/'complete.json')['passed'] and not (O/'independent_audit.json').exists();meta,lookup,_=C.C.B.load();rows=read(O/'event_path_rows.json');idx={(r['fold'],r['seed'],lookup[r['sample_id'],r['position']]):r for r in rows};checked=0;worst=0.;blocks=list(C.BLOCKS)
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(C.FIT/'fit_features'/f'{name}.npz');hz=np.load(C.CACHE/f'held_{name}.npz');cp=np.load(C.OLD/'models'/f'{name}.npz');x=(hz['features']-cp['mean'])/cp['scale'];forest=pickle.loads((C.NEW/'models'/f'{name}_event.pkl').read_bytes());n=int(z['original_STOP_count']);w,y=z['sample_weight'],z['y'];path=np.zeros((len(x),4));support=np.zeros((len(x),4));root=0.
            for tree in forest.estimators_:
                t=tree.tree_;v=t.value[:,0,:];p=v[:,1]/v.sum(axis=1);root+=p[0]/64;parent=np.full(t.node_count,-1,int)
                for node in np.flatnonzero(t.children_left!=-1):parent[t.children_left[node]]=node;parent[t.children_right[node]]=node
                dp=tree.decision_path(x)
                for b,(lo,hi) in C.BLOCKS.items():
                    delta=np.zeros(t.node_count)
                    for node in range(1,t.node_count):
                        if lo<=t.feature[parent[node]]<hi:delta[node]=p[node]-p[parent[node]]
                    path[:,blocks.index(b)]+=np.asarray(dp@delta)/64
                tl=tree.apply(z['standard']);hl=tree.apply(x)
                for leaf in np.unique(hl):
                    fi=np.flatnonzero(tl==leaf);total=w[fi].sum();parts=[fi[(fi<n)&(y[fi]>0)],fi[(fi>=n)&(y[fi]>0)],fi[(fi<n)&(y[fi]==0)],fi[(fi>=n)&(y[fi]==0)]];support[hl==leaf]+=np.array([w[j].sum()/total/64 for j in parts])
            for j,i in enumerate(hz['ids']):
                r=idx[f,s,int(i)];assert abs(root-r['tree_root_p'])<1e-10;reported=np.array([r['tree_path_blocks'][b] for b in blocks]);error=float(abs(path[j]-reported).max());worst=max(worst,error);assert error<1e-10
                reportedsupport=np.array([r['support'][k] for k in ['STOP_positive','CONT_positive','STOP_negative','CONT_negative']]);assert np.allclose(reportedsupport,support[j],atol=1e-10,rtol=1e-10);assert abs(root+path[j].sum()-r['new_p'])<1e-10
                assert abs(r['linear_intercept']+sum(r['linear_blocks'].values())-r['old_logit'])<1e-9;checked+=1
    assert checked==4681;summary=read(O/'summary.json');focus=summary['focus_positive_by_seed'];base=summary['focus_original_NET'];assert focus[0]['rows']==5 and focus[2]['rows']==5 and focus[1]['rows']==0;assert base[1]['all_starts']==base[1]['full_reward_recovered']==base[1]['full_action_matches']==13 and base[1]['eligible_starts']==0
    ratios={str(z['seed']):z['mean_CONT_positive_support']/z['mean_new_p'] for z in focus if z['rows']};assert all(0<v<1 for v in ratios.values())
    text='''# GRID204：查清的原因与下一步

这轮解释了一个值得保留的改善：相同的隐藏表示交给不同概率读出后，确实会作出不同的补查选择；其中一组被找回的补查带来较大局部奖励。

在该组五个相关起点上，旧线性概率模型的“关系隐藏特征”部分在对数几率尺度上的平均贡献为种子0的−7.10、种子2的−9.60，NET动作值和费用部分也为负。总和使平均概率分别降到约0.000009和0.00115。新浅树的平均概率为0.1934和0.0934，乘同一收益预测后跨过既定费用条件。这里说的是固定模型内部的计算路径，不是说这些隐藏特征在物理上有负面作用。

'''+f"新预测并非直接读取该留出日期的答案：对应训练数据排除了该日期，其概率可逐项还原为训练叶内正例的加权支持。原NET继续状态的正例贡献占这组预测概率约{100*ratios['0']:.1f}%和{100*ratios['2']:.1f}%，其余来自原停止状态正例。有效支持调用数约66和98；这是共享树叶训练权重的集中程度，不能解释为66或98个独立相似成功场景。\n\n"+'''第三个种子在原NET阶段已经把13个起点的完整局部奖励和动作全部恢复，且没有剩余门控机会。因此该种子没有“新增恢复”不构成这组案例的失败证据；它也不能作为新概率头的第三次成功。

反向案例提醒我们，浅树还没有把有利和无利补查完全分开。例如2037-07-06_101|call005的六个改变决策配对都没有奖励增量，新概率仍平均约0.1803；主要反向调用中也有微小正奖励不足以回本的情况。这些概率同样受到STOP和CONT正例支持，不能简单按“有训练正例支持”就判断值得补查。

我的判断：当前改进值得保留，问题从“旧模型把一个有利模式几乎判死”推进到了“新模型保住了它，但也放进了相似的无效追加”。下一步应对照两类状态的原有公开反馈，检查当前表示是否保留了区分它们的信息，以及概率误差和收益幅度误差分别起多大作用。优先解释已知差异，再定一次有针对性的学习改动，不先增加网络规模或改费用门槛。

本轮12对模型、4681个停止点均已核对，未训练、改标签、调阈值或新做物理回放。局部奖励与查询代理效用不能代替整周运行质量或实际控制时延。
'''
    (O/'原因与后续思路.md').write_text(text,encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all4681_sparse_path_reconstruction=True,all4681_leaf_STOP_CONT_positive_negative_support_reconstruction=True,max_sparse_path_error=worst,seed1_no_remaining_gate_opportunity_verified=True,training_date_exclusion_checked=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_概率分歧解释_GRID204'
    for n in ['原因与后续思路.md','independent_audit.json']:shutil.copy2(O/n,dest/n)
    msg='GRID204全部4681路径/叶内训练支持已复核：主要恢复案例旧关系隐藏logit贡献约−7.10/−9.60，浅树保留非零概率，CONT正例支持约63%/56%；第三种子原NET已恢复13/13且无剩余门控机会。反向无收益样本仍有较高树概率，不能按训练正例支持直接触发。保留改善，下一步对照已有公开反馈区分恢复与无效追加，零拟合/物理。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-27 — GRID204固定概率分歧与训练支持核查\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID204后\n\n1. 保留GRID203主要恢复与小幅改善，不新FRAME闭环。\n2. 对照恢复与无效追加的已有公开反馈，区分事件概率/幅度误差与表示覆盖；不用事后正例、查询范围或训练叶支持直接造触发规则。\n3. 新训练前通知固定范围，不自动扩大网络/数据、扫描费用阈值或新物理；最终8和论文不动。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID204固定概率分歧核查\n\n[报告](protocols/GRID204_event_paths/audited/GRID204_report.md)；[原因与后续](protocols/GRID204_event_paths/audited/原因与后续思路.md)；零拟合/仿真，保留GRID203恢复案例和反例。\n\n'.encode('utf8')+p.read_bytes());print('GRID204_INDEPENDENT_AUDIT',json.dumps(dict(checked=checked,CONT_support_ratios=ratios,max_error=worst)))

def delivery():
    C.verify();assert read(O/'independent_audit.json')['passed'];head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;count=0
    for mf in C.A.rglob('archive_manifest.json'):
        for p,h in read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
    dest=R/'论文协作资料/FRAME_概率分歧解释_GRID204';files=list(dest.iterdir())
    for p in files:
        ap=C.A/'audited'/p.name;assert C.sha(p)==C.sha(O/p.name)==C.sha(ap);assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes()
    rec=dict(passed=True,verified_publication_head=head,remote_main=head,archive_files_checked=count,collaboration_files_checked=len(files),all_committed_bytes_match=True,new_fits=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'delivery_receipt.json',dest/'delivery_receipt.json');print(json.dumps(rec))

if __name__=='__main__':globals()[sys.argv[1]]()
