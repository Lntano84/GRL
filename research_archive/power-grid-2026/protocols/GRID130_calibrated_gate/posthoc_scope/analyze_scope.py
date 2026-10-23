"""Read-only cost coverage and all residual disagreements;no altered gate."""
from pathlib import Path
import hashlib,json,math
import numpy as np

R=Path('D:/Codex/GRL研究');S=R/'outputs/grid130_calibrated_gate';O=S/'posthoc_scope'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not O.exists();O.mkdir()
    paths=[Path(__file__),S/'validation_trajectories.json',S/'validation_results.json',S/'fits.json',S/'model.joblib',S/'decision.json']
    hashes={p.relative_to(R).as_posix():sha(p) for p in paths}
    rows=json.loads(paths[1].read_text(encoding='utf8'));rr=[r for r in rows if r['method']=='LR_CALIBRATED'];assert len(rr)==590
    skipped=[r for r in rr if r['queries']==0];kept=[r for r in rr if r['queries']>0];bad=[r for r in rr if r['action_disagreement']]
    perdate={}
    for d in sorted({r['date'] for r in rr}):
        aa=[r for r in rr if r['date']==d];ss=[r for r in aa if r['queries']==0];kk=[r for r in aa if r['queries']>0]
        query_sum=sum(r['full_queries'] for r in aa);saving=sum(r['full_queries'] for r in ss)/query_sum
        perdate[d]=dict(prefixes=len(aa),skipped=len(ss),skip_query_sum=sum(r['full_queries'] for r in ss),all_query_sum=query_sum,
            skip_mean_remaining=float(np.mean([r['full_queries'] for r in ss])),keep_mean_remaining=float(np.mean([r['full_queries'] for r in kk])),query_saving=saving)
        assert query_sum-sum(r['queries'] for r in aa)==sum(r['full_queries'] for r in ss)
    summary=dict(skipped=len(skipped),skipped_call_groups=len({r['call'] for r in skipped}),
        skipped_mean_remaining=float(np.mean([r['full_queries'] for r in skipped])),kept_mean_remaining=float(np.mean([r['full_queries'] for r in kept])),
        date_macro_query_saving=math.fsum(v['query_saving'] for v in perdate.values())/len(perdate),
        bad_prefixes=len(bad),bad_calls=sorted({r['call'] for r in bad}),
        bad_gain_lost_sum=math.fsum(r['gain_lost'] for r in bad),
        bad_score_range=[min(r['score'] for r in bad),max(r['score'] for r in bad)],per_date=perdate)
    base=json.loads(paths[2].read_text(encoding='utf8'))['LR_CALIBRATED']['macro'];assert abs(summary['date_macro_query_saving']-base['query_saving'])<1e-12
    assert hashes=={p.relative_to(R).as_posix():sha(p) for p in paths}
    (O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (O/'all_disagreements.json').write_text(json.dumps(bad,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (O/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (O/'review.json').write_text(json.dumps(dict(passed=True,source_unchanged=True,all_disagreements_retained=True,query_accounting_conserved=True,model_or_threshold_changes=0,new_fits=0,new_predictions=0),indent=2)+'\n',encoding='utf8')
    text='''# 校准后的查询节省为什么仍有限\n\n这是既有验证结果的只读解释，未重训、调整门槛或追加评分。\n\n校准门控跳过223/590条相关前缀，涉及47次教师调用；这些被跳过的补查平均只剩4.61个候选，保留的补查平均还剩69.71个候选。按日期宏平均只省4.81%查询。因此，目前可信区间主要覆盖计算量很小的补查；计算大头多数回退。这里“可信”仅指本次开发门槛的覆盖，不是统计校准保证，也不能从相关性推断某个输入特征就是因果机制。\n\n7条动作差异都在2037-08-03，来自2037-08-03_100|call005与2037-08-03_101|call000两次调用。它们是5+2个相关前缀，不是7个独立场景。对应局部漏失增量分别每前缀6.80685e-5与1.085997e-4；加总0.00055754也不是独立实验的整周经济损失。\n\n模型的无改进排序分数在这些误停上为0.979964至0.998520，超过校准门槛0.975700；该高分不能解释为经过验证的98%—99%成功概率。校准组零损失没有完全迁移。8月日期补查增量保留99.94345%，四日期宏平均99.98586%；严格动作一致开发筛查仍失败。\n\n本轮保护收益、减少误停的方向明确，但付出了多查的代价。相对固定一批，校准门控保留的局部增量更多、动作差异更少，但查询比例95.19%高于56.13%，净效用也较低；不能概括成超过固定一批。相对同权重固定门槛的保护提升，才是匹配的校准机制证据。\n\n下一问题应聚焦：开销较大但实际无增量的补查，能否从决策前信息中被可靠识别。首先使用既有训练/校准记录检查覆盖与表示，不根据这7条验证误停改阈值或加入特定触发规则，也不立即扩大数据或追加模型。\n'''
    (O/'节省为何有限_中文.md').write_text(text,encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_date'},ensure_ascii=False))

if __name__=='__main__':main()
