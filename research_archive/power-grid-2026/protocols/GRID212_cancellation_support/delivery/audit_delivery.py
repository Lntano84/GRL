"""Verify fixed fit disagreements and publish the sample-generation boundary."""
from pathlib import Path
import json,pickle,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def audit():
    C.verify();assert not (O/'independent_audit.json').exists();meta,lookup,_,raw=C.C.raw_inputs();fitrows=read(O/'fit_candidate_cancellations.json');heldrows=read(O/'held_cancellation_neighbors.json');checked=neighbors=0
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(C.FULL/f'{name}.npz');orig=np.load(C.FIT/'fit_features'/f'{name}.npz');lab=np.load(C.LABEL/f'{name}.npz');x=z['standard'];ids=z['ids'].astype(int);n=int(orig['original_STOP_count']);models=[pickle.loads((folder/'models'/f'{name}_{kind}.pkl').read_bytes()) for folder,kind in [(C.OLD,'event'),(C.OLD,'amount'),(C.NEW,'amount')]];scores=[]
            for j,model in enumerate(models):
                value=np.zeros(len(x))
                for tree in model.estimators_:
                    leaf=tree.apply(x);v=tree.tree_.value[leaf,0,:];value+=(v[:,1]/v.sum(1) if j==0 else v[:,0])/64
                scores.append(value)
            p,old,new=scores;cancel=(p*old>orig['x'][:,82])&~(p*new>orig['x'][:,82]);sel=np.flatnonzero(cancel);reported=[r for r in fitrows if r['fold']==f and r['seed']==s];assert [int(ids[k]) for k in sel]==[r['index'] for r in reported];net=lab['target']-lab['known_charge']
            for k,r in zip(sel,reported):assert abs(r['actual_NET']-net[k])<1e-12 and r['domain']==('actual_STOP' if k<n else 'preSTOP_CONT');checked+=1
            stop=np.flatnonzero(cancel&(np.arange(len(x))<n));cp=np.load(C.OLD/'models'/f'{name}.npz');hz=np.load(C.CACHE/f'held_{name}.npz');hx=np.c_[(hz['features']-cp['mean'])/cp['scale'],(raw[hz['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']];byid={int(i):j for j,i in enumerate(hz['ids'])}
            for r in [q for q in heldrows if q['fold']==f and q['seed']==s]:
                j=byid[lookup[r['sample_id'],r['position']]];dist=np.mean((x[stop]-hx[j])**2,axis=1);seen=set();keep=[]
                for k in sorted(range(len(stop)),key=lambda k:(float(dist[k]),k)):
                    g=meta[int(ids[stop[k]])]['physical_state_group']
                    if g in seen:continue
                    seen.add(g);keep.append(int(ids[stop[k]]))
                    if len(keep)==3:break
                assert keep==[v['index'] for v in r['neighbors']];assert all(v['date']!=r['date'] for v in r['neighbors']);neighbors+=1
    assert checked==972 and neighbors==157
    text='''# GRID212判读：先改训练样本来源，再学取消开关

取消补查仍有可研究的小改善。当前问题是训练覆盖：12个拟合池累计634条实际停止域分歧仅4条回本，10个池没有应保护的回本例；去重后427个状态，不能将重复行当独立数据。六个留出回本损失的最近三个不同训练调用分歧均没有回本例。这是当前分歧子集的覆盖问题，不证明整个公开输入不可区分或任何路由器都学不会。

直接用拟合内分歧训练取消开关，容易学成几乎全取消。后续应保留连续局部净收益和费用，不只看好/坏分类正确率；先让模型在未参与自身拟合的日期产生训练分歧。可在每个外层12个训练日期内部固定分组，但外层4个评估日期始终隔离。

现有16日期折外预测不能直接拼给另一个外层训练路由，因为生成预测的其他模型可能学过那个外层的评估日期，造成间接泄漏。若只对读出作内层交叉生成而冻结基础NET，还要说明表示与停止域仍有拟合内效应，不能声称整条流水线完全交叉拟合。

下一步先固定模型依赖、日期分组与训练成本，再通知是否开始新一轮训练。本轮未训练或执行路由/组合策略。208仍为当前候选，取消机制和210节省线索保留；无新网络前向、物理、最终8、论文或原FRAME比较。
'''
    (O/'判读与样本生成边界.md').write_text(text,encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all972_fit_disagreements_reconstructed_from_native_leaves=True,all157_neighbors_retrieved_again=True,all12_STOP_CONT_domains_separated=True,new_fits=0,native_forest_fit_score_calls_main=36,manual_leaf_audit=True,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_取消补查训练支持_GRID212'
    for n in ['判读与样本生成边界.md','independent_audit.json']:shutil.copy2(O/n,dest/n)
    msg='GRID212：STOP域拟合分歧634行仅4回本、10/12池无回本例，去重427状态；留出6回本损失最近三训练调用均无回本支持。机制保留，先厘清内层样本生成和外层日期依赖，不直接拼已有OOF训练路由；零拟合/组合/物理。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-27 — GRID212取消训练支持\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID212后\n\n1. 保留208及有效取消线索，不用拟合内单边分歧直接训路由。\n2. 先固定内层样本生成、外层模型/日期依赖与成本，禁止直接拼已有OOF导致间接泄漏；说明冻结NET的拟合内效应。\n3. 新训练先通知固定方案；不调费用阈值，不扩物理/最终8/论文/原FRAME。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID212取消训练支持\n\n[报告](protocols/'+C.A.name+'/audited/GRID212_report.md)；[判读](protocols/'+C.A.name+'/audited/判读与样本生成边界.md)；零拟合，先厘清样本来源。\n\n').encode('utf8')+p.read_bytes());print('GRID212_AUDITED',checked,neighbors)
def delivery():
    C.verify();head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;stages=[]
    for out,arch,collab in [('grid211_budget_sources','GRID211_budget_sources','FRAME_预算改善来源_GRID211'),('grid212_cancellation_support',C.A.name,'FRAME_取消补查训练支持_GRID212')]:
        src=R/'outputs'/out;a=C.REPO/'research_archive/power-grid-2026/protocols'/arch;dest=R/'论文协作资料'/collab;assert read(src/'complete.json')['passed'] and read(src/'independent_audit.json')['passed'];count=0
        for mf in a.rglob('archive_manifest.json'):
            for p,h in read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
        for p in dest.iterdir():
            assert C.sha(p)==C.sha(src/p.name);matches=[ap for ap in a.rglob(p.name) if ap.is_file() and C.sha(ap)==C.sha(p)];assert matches;assert any(git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==p.read_bytes() for ap in matches)
        for p,h in read(src/'frozen_inputs.json').items():assert C.sha(R/p)==h,p
        stages.append(dict(stage=arch,archive_files_checked=count,collaboration_files_checked=len(list(dest.iterdir())),committed_bytes_match=True))
    rec=dict(passed=True,verified_publication_head=head,remote_main=head,stages=stages,new_fits=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'delivery_receipt.json',R/'论文协作资料/FRAME_取消补查训练支持_GRID212/delivery_receipt.json');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
