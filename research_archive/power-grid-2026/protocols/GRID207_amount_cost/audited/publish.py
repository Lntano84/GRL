"""Independent leaf-domain reconstruction and diagnostic publication."""
from pathlib import Path
import json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read
def main():
    C.verify();assert read(O/'complete.json')['passed'] and not (O/'independent_audit.json').exists();meta,lookup,_=C.C.B.load();rows=read(O/'amount_support_rows.json')
    # Explicit metadata join, no query to an environment or new prediction target.
    predictions={(p['fold'],p['seed'],lookup[p['sample_id'],p['position']]):p for p in read(C.SRC/'gate_predictions.json')};bykey={(r['fold'],r['seed'],r['sample_id']):r for r in rows};checked=0
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(C.FIT/'fit_features'/f'{name}.npz');lab=np.load(C.LAB/f'{name}.npz');cp=np.load(C.SRC/'models'/f'{name}.npz');hz=np.load(C.CACHE/f'held_{name}.npz');forest=pickle.loads((C.SRC/'models'/f'{name}_amount.pkl').read_bytes());pos=lab['indices'].astype(int);train=z['standard'][pos];x=(hz['features']-cp['mean'])/cp['scale'];w,y=lab['sample_weight'],lab['target'];isstop=pos<int(z['original_STOP_count']);contrib=np.zeros((len(x),2))
            for tree in forest.estimators_:
                tl=tree.apply(train);hl=tree.apply(x)
                for leaf in np.unique(hl):
                    mask=tl==leaf;den=w[mask].sum();contrib[hl==leaf]+=np.array([float(w[mask&isstop]@y[mask&isstop])/den,float(w[mask&~isstop]@y[mask&~isstop])/den])/64
            for j,i in enumerate(hz['ids']):
                p=predictions[f,s,int(i)];r=bykey[f,s,p['sample_id']];assert np.allclose(contrib[j],[r['STOP_amount_contribution'],r['CONT_amount_contribution']],atol=1e-10,rtol=1e-10);assert abs(contrib[j].sum()-p['conditional_normalized_gain'])<1e-10;checked+=1
    assert checked==4681;stats=read(O/'summary.json')['categories'];ratio=stats['new_positive_unprofitable']['median_mu_to_realized_positive'];text=f'''# GRID207判读：幅度估计与零收益支出都要看

新增10个正收益但未回本配对的幅度预测/已实现增量比值中位数为{ratio:.2f}，覆盖6个物理调用；平均批次32.8个候选。这是已有正收益结果的误差线索，不是对未知真实条件均值的因果证明。

另94个新增零收益配对的日期宏平均净效用贡献为{stats['new_zero']['macro_delta_utility']:+.9f}，大于上述10个小正收益配对的负贡献绝对值{abs(stats['new_positive_unprofitable']['macro_delta_utility']):.9f}。因此不能将问题只归于长批次，也不能把“识别到正收益”直接等同“值得查询”。

下一次只检验条件收益输入：固定GRID206概率森林，将原41项公开摘要接入条件收益森林，83隐藏保留。使用原GRID198正例目标和权重，不给零收益状态编造条件幅度0；原64树、深4、叶10、12组和费用规则不变。两模块不会同时重训，全部种子和反例保留。

对现有收益森林的4681个预测及其STOP/CONT正例支持进行了第二种逐叶核验；不按支持来源删训练数据，不按批次区间设计新触发规则。无新拟合、神经网络前向或物理回放。
''';(O/'判读与固定下一步.md').write_text(text,encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all4681_leaf_domain_contributions_independently_reconstructed=True,positiveonly_mu_error_scope_preserved=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_收益幅度与成本_GRID207'
    for n in ['判读与固定下一步.md','independent_audit.json']:shutil.copy2(O/n,dest/n)
    msg=f'GRID207幅度核查完成：新增10正收益未回本配对(6调用)幅度/已实现比值中位{ratio:.2f}，但94零收益追加负贡献更大，不只长批次问题。下一次固定206概率，只给正例条件收益森林接原41摘要；目标/权重/64树深4叶10不变，已通知，零新物理。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-27 — GRID207幅度与批次开销核查\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID207后\n\n1. 已通知下一次固定训练：仅条件收益森林加入原41摘要，124维，冻结206概率/NET，原正例目标与权重、12组64树深4叶10。\n2. 先冻结资格，再训练封存全部权重，之后全种子留出与同预期查询比较；不能把零样本设为条件幅度0。\n3. 不调整费用、阈值、批次上限或扩日期/物理，最终8和论文不动。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID207条件收益与支出核查\n\n[报告](protocols/GRID207_amount_cost/audited/GRID207_report.md)；[判读](protocols/GRID207_amount_cost/audited/判读与固定下一步.md)；零拟合/物理。\n\n'.encode('utf8')+p.read_bytes());print('GRID207_AUDITED',checked)
if __name__=='__main__':main()
