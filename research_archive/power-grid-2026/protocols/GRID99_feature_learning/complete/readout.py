"""Descriptive readout only; never selects, refits or deploys a model."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid99_feature_learning'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
NAMES={'LOGISTIC':'逻辑回归','FOREST64':'浅层森林','TABPFN_V2':'TabPFNv2'}
def main():
    assert read(OUT/'review.json')['passed'];rs=read(OUT/'results.json');pred=read(OUT/'predictions.json')
    def get(arm,name,scheme='calendar_loo'):return next(r for r in rs if r['arm']==arm and r['model']==name and r['scheme']==scheme)
    lines=['# GRID99学习结果解读','',
        '这次只回答：在同一批已暴露状态、相同模型和日期留出下，补充输入是否改变了离线预测表现。目标仍是识别取消后轨迹不变的补查，不是直接学习经济价值。','']
    summaries=[]
    for name in NAMES:
        a=get('ORIGINAL15',name)['primary_nonpilot'];b=get('ENRICHED35',name)['primary_nonpilot'];z=b['thresholds']['0.95'];t=get('ENRICHED35',name)['timing'];transfer=get('ENRICHED35',name,'old_to_generated')['primary_nonpilot']
        keep=[p for p in pred if p['arm']=='ENRICHED35' and p['model']==name and p['scheme']=='calendar_loo' and p['original_label']=='KEEP'];assert len(keep)==1
        summaries.append(dict(model=name,auc_delta=b['auc']-a['auc'],ba_delta=b['balanced_accuracy']-a['balanced_accuracy'],threshold95=z,known_KEEP_probability=keep[0]['p_same_path']))
        lines += [f"**{NAMES[name]}**：主要53例AUC由{a['auc']:.3f}变为{b['auc']:.3f}，平衡准确率由{a['balanced_accuracy']:.3f}变为{b['balanced_accuracy']:.3f}。Brier由{a['brier']:.3f}变为{b['brier']:.3f}（越低越好）。",
            f"在预定0.95门槛下，提出取消{z['proposed_omissions']}例，{z['correct_redundancy']}例确实轨迹不变、{z['changed_path_misclassified']}例轨迹改变。补充输入的单例平均推理{t['single_row_mean_ms']:.3f}ms，不含特征提取。",
            f"old13训练、生成场景22个非试点评价的AUC为{transfer['auc']:.3f}，平衡准确率{transfer['balanced_accuracy']:.3f}。唯一已知KEEP旧试点的冗余预测概率为{keep[0]['p_same_path']:.6f}；不能把这一个例子当保护能力证明。",'']
    lines += ['三个模型与全部不利结果都保留，没有按结果换参数、换门槛或选择输入子集。概率0.95不是校准保证；零取消意味着零覆盖，不能用零误判夸大保护能力。轨迹改变的误判也不能一律理解为经济损害，原经济标签仍单列。','',
        '原15维重复必须重现GRID97，排除换运行环境、日期或代码导致的结果漂移。主要样本只有38个同轨迹、15个改变轨迹，始终猜同轨迹的普通准确率已经是71.7%；需要同时看平衡准确率、概率排序和实际取消覆盖/误判。','',
        '新增20列有确定组合、常量与重复项，所以任何整组增量都不能单独归因于候选结构、同站反馈或某一个新特征。这次没有做按标签选择或分组消融，也不自动追加。输入是在看过GRID97后提出的开发修改，改善不能称独立确认或跨网泛化。','',
        '本轮到固定比较结束：不把单次删补查的离线表现直接变成在线门控，不增加电网回放，不相加重叠案例的收益。是否进一步分析，由用户看完结果后决定。','']
    (OUT/'训练结果解读_中文.md').write_text('\n'.join(lines),encoding='utf8');(OUT/'readout.json').write_text(json.dumps(dict(models=summaries,model_selected_for_deployment=False,extra_fits=0),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':main()
