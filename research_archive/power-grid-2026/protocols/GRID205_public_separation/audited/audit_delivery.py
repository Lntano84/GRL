"""Check saved public descriptors and label-blind retrieval; publish diagnosis."""
from pathlib import Path
import json,pickle,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def audit():
    C.verify();assert read(O/'complete.json')['passed'] and not (O/'independent_audit.json').exists();meta,_,cases=C.C.B.load();bycase={c['sample_id']:c for c in cases};public=pickle.loads((C.PUBLIC/'train_public.pkl').read_bytes());desc=np.asarray(read(O/'public_descriptors.json')['values']);assert desc.shape==(3582,66)
    for i,(p,m) in enumerate(zip(public,meta)):
        assert np.array_equal(desc[i,:41],bycase[m['sample_id']]['states'][m['position']]);fb=p['context_feedback'];assert np.allclose(desc[i,41:49],[np.log1p(len(fb)),*np.quantile(fb[:,0],[0,.25,.5,.75,1]),fb[:,0].mean(),fb[:,0].std()],atol=1e-6,rtol=1e-6);assert desc[i,-1]==len(p['next_action']);assert np.allclose(desc[i,49:56],fb[:,1:].mean(0),atol=1e-7,rtol=1e-7)
        q=dict(p,context_action=p['context_action'][::-1],context_feedback=p['context_feedback'][::-1],next_action=p['next_action'][::-1]);assert C.sig(p,desc[i,:41])==C.sig(q,desc[i,:41]);q=dict(p,context_feedback=p['context_feedback'][::-1]);assert C.sig(p,desc[i,:41],True)==C.sig(q,desc[i,:41],True)
    rows=read(O/'all_state_neighbors.json');checks=read(O/'fit_scaling_and_checks.json');counts=0
    for c in checks:
        f,s=c['fold'],c['seed'];z=np.load(C.FIT/'fit_features'/f'f{f}_s{s}.npz');hz=np.load(C.CACHE/f'held_f{f}_s{s}.npz');fi=z['ids'].astype(int);hi=hz['ids'].astype(int);ix={int(i):j for j,i in enumerate(hi)};byglobal={int(i):j for j,i in enumerate(fi)};pools=[r for r in rows if r['fold']==f and r['seed']==s];dates={meta[i]['date'] for i in fi}
        for name in ['RAW41','PUBLIC66','HIDDEN83']:
            if name=='HIDDEN83':train=z['standard'];held=(hz['features']-z['mean'])/z['scale']
            else:
                arr=desc[:,:41] if name=='RAW41' else desc;mean=np.array(c['scale_parameters'][name]['mean']);scale=np.array(c['scale_parameters'][name]['scale']);assert np.allclose(mean,np.average(arr[fi],axis=0,weights=z['sample_weight']),atol=1e-10,rtol=1e-10);train=(arr[fi]-mean)/scale;held=(arr[hi]-mean)/scale
            for r in pools:
                distance=np.mean((train-held[ix[r['index']]])**2,axis=1);order=sorted(range(len(fi)),key=lambda k:(float(distance[k]),meta[int(fi[k])]['physical_state_group'],k));seen=set();selected=[]
                for k in order:
                    g=meta[int(fi[k])]['physical_state_group']
                    if g not in seen:seen.add(g);selected.append(k)
                    if len(selected)==3:break
                assert [int(fi[k]) for k in selected]==[n['index'] for n in r['neighbors'][name]]
                for n in r['neighbors'][name]:assert abs(n['distance']-distance[byglobal[n['index']]])<1e-9 and n['date'] in dates and n['date']!=r['date']
                counts+=1
    assert counts==4681*3;summary=read(O/'summary.json');v=summary['neighbor_descriptions'];txt='''# GRID205判读与一次固定训练方案

本轮没有证明新策略有效，但找到值得检验的输入路线。53次回本追加中，原41维公开摘要的最近拟合调用对应状态有18个正例，旧隐藏83维为12个；762次零奖励追加中分别为53和54个。加上25项固定已购反馈/接线描述后的66维为18和41。最近三个不同训练调用的结果、所有种子和反例均已保留。

这些是已暴露数据上的邻近关系描述，不是最近邻分类器的测试成绩；没有训练或执行最近邻策略，不能把18/53解释为可部署召回率。三种表示距离各自标准化，距离大小不可直接比较。没有发现精确输入冲突也不能证明泛化能力。

主要恢复案例和零奖励对照在查询数相近时，原有奖励差、站内奖励波动、已购候选的负载率等摘要仍存在差异。模型可能通过隐藏表示保留了它们，也可能压缩得不利于浅树利用；目前不能宣布信息已经丢失。

下一次训练只检验一个改动：给GRID203概率森林增加原有41项公开摘要的直接输入，原83项隐藏输入保留，合计124维。使用完整41项，不根据本轮差异挑字段；66维描述中的另外25项不进入这次训练。41项标准化只用每折拟合数据与原权重计算，收益森林、基础NET、标签、训练样本、权重和决策费用不变。

仍为12个ExtraTreesClassifier，每个64棵树、深度4、叶子至少10行，原三个种子，不搜索参数。全部权重先发布，再以GRID203为主要内部对照，并保留GRID198参照、全部日期与反向结果和相同预期查询比较。这个改动会改变随机划分可用特征和函数类，不能称严格同容量，也不保证保持此前每一个改善。

本轮零拟合、零神经网络前向、零物理回放；未动最终8、原FRAME比较或论文。
''';(O/'判读与固定训练方案.md').write_text(txt,encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all3582_descriptors_checked_and_signatures_permutation_invariant=True,all14043_label_blind_distinct_call_neighbor_retrievals_reconstructed=True,all_fit_dates_exclude_held=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_公开反馈区分_GRID205'
    for n in ['判读与固定训练方案.md','independent_audit.json']:shutil.copy2(O/n,dest/n)
    msg='GRID205公开反馈区分诊断：53回本追加中RAW41最近拟合调用正例18、隐藏83为12；762零奖励中为53/54。仅描述邻近关系，非最近邻策略成绩或信息丢失证明。下一次固定检验83隐藏+完整原41摘要的124维概率森林输入，保持收益/NET/费用/标签/权重及12个浅树配置；不选特征、零新物理。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-27 — GRID205公开反馈对照与输入路线\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID205后\n\n1. 已通知固定下一次训练：概率森林接入全部原41公开摘要，保留83隐藏与固定收益/NET；124维、12组、同64树/深4/叶10，不选择25个探索摘要。\n2. 先冻结与资格核验，再训练和封存全部权重，之后进行全种子留出与相同预期查询分析。\n3. 保留小改善和反例，不扩模型/日期/物理、调费用阈值或动最终8/论文/新FRAME比较。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID205公开反馈区分诊断\n\n[报告](protocols/GRID205_public_separation/audited/GRID205_report.md)；[判读与方案](protocols/GRID205_public_separation/audited/判读与固定训练方案.md)；零拟合，下一次只检验原41摘要直接输入。\n\n'.encode('utf8')+p.read_bytes());print('GRID205_AUDITED',counts)
def delivery():
    C.verify();head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;count=0
    for mf in C.A.rglob('archive_manifest.json'):
        for p,h in read(mf).items():assert C.sha(mf.parent/p)==h;count+=1
    dest=R/'论文协作资料/FRAME_公开反馈区分_GRID205';files=list(dest.iterdir())
    for p in files:
        ap=C.A/'audited'/p.name;assert C.sha(p)==C.sha(O/p.name)==C.sha(ap);assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes()
    rec=dict(passed=True,verified_publication_head=head,remote_main=head,archive_files_checked=count,collaboration_files_checked=len(files),new_fits=0,new_physics=0);C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(O/'delivery_receipt.json',dest/'delivery_receipt.json');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
