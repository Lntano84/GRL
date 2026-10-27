"""Independent leaf-depth and legal split checks; no deeper model is fitted."""
from pathlib import Path
import json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read
def main():
    C.verify();assert read(O/'complete.json')['passed'] and not (O/'independent_audit.json').exists();meta,lookup,_,raw=C.C.raw_inputs();rows=read(O/'all_support_and_leaf_rows.json');pred=read(C.SRC/'gate_predictions.json');ix={(p['fold'],p['seed'],lookup[p['sample_id'],p['position']]):p for p in pred};bykey={(r['fold'],r['seed'],r['sample_id']):r for r in rows};checked=0
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(C.SRC/'fit_features'/f'{name}.npz');cp=np.load(C.SRC/'models'/f'{name}.npz');hz=np.load(C.CACHE/f'held_{name}.npz');x=np.c_[(hz['features']-cp['mean'])/cp['scale'],(raw[hz['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']];model=pickle.loads((C.SRC/'models'/f'{name}_amount.pkl').read_bytes());depth=np.zeros(len(x));possible=np.zeros(len(x));minimum=np.zeros(len(x))
            for tree in model.estimators_:
                d=np.asarray(tree.decision_path(x).sum(1)).ravel()-1;depth+=(d==4)/64;tl=tree.apply(z['standard']);hl=tree.apply(x)
                for leaf in np.unique(hl):
                    mask=tl==leaf;n=int(mask.sum());tx=np.sort(z['standard'][mask].astype(np.float32),axis=0);valid=n>=20 and np.unique(z['target'][mask]).size>1 and bool((tx[9]<tx[n-10]).any());possible[(hl==leaf)&(d==4)]+=int(valid)/64;minimum[hl==leaf]+=int(n<20)/64
            for j,i in enumerate(hz['ids']):
                p=ix[f,s,int(i)];r=bykey[f,s,p['sample_id']];assert abs(depth[j]-r['depth4_tree_fraction'])<1e-12;assert abs(minimum[j]-r['minleaf_unsplittable_tree_fraction'])<1e-12
                assert abs(possible[j]-r['depth4_splittable_tree_fraction'])<1e-12;checked+=1
    assert checked==4681;summary=read(O/'summary.json');lost=summary['lost_cases'];all_fraction=float(np.mean([r['depth4_splittable_tree_fraction'] for r in rows]));lost_fraction=summary['categories']['LOST_PROFIT']['mean_depth4_splittable_fraction'];text=f'''# GRID209判读与下一次固定容量对照

四个被取消的回本配对来自3个调用、同一留出折。三例的当前收益幅度低于单次已实现增量；另一例当前幅度高于实际增量，却仍因事件概率较低而未过费。因此不能把四例一概当成幅度低估，也不应抬高所有预测把它们强行救回。

模型结构也提供了一条可检验线索：四例平均{100*lost_fraction:.2f}%的树路径在深度4达到上限，叶内仍有至少20行正例、不同收益标签和合法的下一层输入划分。全部4681点的该比例均值为{100*all_fraction:.2f}%，保留、取消、无效追加等组也普遍存在，说明这一限制并非四个失败案例独有。它只是结构上的可分空间，不保证加深后泛化改善。

下一次只做一个容量对照：条件收益森林深度4→5，其余仍为64树、最小叶10、相同124维输入、正例目标与权重，固定三个种子。概率森林和基础NET冻结。只增加一层，不扫描深度/叶子/树数，不改变费用、阈值或标签。全部12组权重先封存，再看原16日期全部种子的留出误差、收益、查询与同预期预算，以及旧主恢复、4个损失和新反例。

当前较低查询的GRID208保留；下一轮即使不能找回四例，也按全部状态判断，不要求追回每个已实现回本结果。没有训练前就承诺更好，不以损失案例选用未来答案或改触发规则。

本轮已独立核验4681条路径和合法分割条件，零拟合、零网络前向、零物理回放；最终8、论文和原FRAME比较不动。
''';(O/'判读与固定容量对照.md').write_text(text,encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all4681_native_depth_paths_and_orderstatistic_split_checks_reproduced=True,all4_fixed_cost_algebra_and_allgroups_retained=True,allstates_mean_depth4_splittable=all_fraction,new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_回本损失与叶子限制_GRID209'
    for n in ['判读与固定容量对照.md','independent_audit.json']:shutil.copy2(O/n,dest/n)
    msg=f'GRID209四回本损失：3例幅度低于已实现增量，1例虽高但概率乘积不过费；不全归幅度。全4681点深4且可合法续分比例均值{all_fraction:.4f}，非仅失败案例。下一次固定条件树深4→5、其余64树叶10/124维/正例权重不变，概率/NET冻结，不扫描或调费。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-27 — GRID209损失与树叶容量核查\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID209后\n\n1. 保留208改善和所有反例，下一次只固定检验条件树深度4→5，其他配置、正例目标和权重不变。\n2. 新训练先通知，再冻结资格与全部权重封存；不以追回四例为唯一目标，不扫描深度或叶子。\n3. 概率/NET冻结；不调费用阈值、不扩物理/最终8/论文/原FRAME。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID209回本损失与叶子容量核查\n\n[报告](protocols/GRID209_lost_gain/audited/GRID209_report.md)；[判读与方案](protocols/GRID209_lost_gain/audited/判读与固定容量对照.md)；零拟合，不强行追回四例。\n\n'.encode('utf8')+p.read_bytes());print('GRID209_AUDITED',json.dumps(dict(checked=checked,all_splittable_fraction=all_fraction,lost_splittable_fraction=lost_fraction)))
if __name__=='__main__':main()
