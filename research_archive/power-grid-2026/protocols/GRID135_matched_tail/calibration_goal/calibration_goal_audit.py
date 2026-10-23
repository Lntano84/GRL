"""Describe existing extreme calibration witnesses; no alternative cutoff evaluation."""
from common135 import *
import collections,shutil
def main():
    verify();assert read(O/'publication_recovery_01.json')['passed'];assert not (O/'calibration_goal_audit.json').exists()
    names=['interpretation.json','calibration_metadata.json','thresholds.json','train_calibration_logits.npz','validation_trajectories.json','decision.json','families.json']
    frozen={n:sha(O/n) for n in names};analysis=read(O/'interpretation.json');meta={r['sample_id']:r for r in read(O/'calibration_metadata.json')};anchors=collections.defaultdict(list)
    for name,a in analysis['calibration_anchors'].items():
        for sid in a['sample_ids']:anchors[meta[sid]['call']].append((name,sid))
    facts={}
    for call,aa in anchors.items():
        ids=sorted({sid for _,sid in aa});rr=[meta[sid] for sid in ids]
        facts[call]=dict(models=sorted({n for n,_ in aa}),sample_ids=ids,
            gain_range=[min(r['full_baseline_gain'] for r in rr),max(r['full_baseline_gain'] for r in rr)],
            pending_query_range=[min(r['known_saved_queries_if_skip'] for r in rr),max(r['known_saved_queries_if_skip'] for r in rr)])
    assert len(facts)==2 and len({n for f in facts.values() for n in f['models']})==12
    evidence=dict(passed=True,source_hashes=frozen,calibration_anchor_calls=facts,
        reason='All gains>0 receive the same calibration protection despite different magnitudes;state this target explicitly instead of equating it with original whole-week quality budget.',
        not_established=['tiny increments are numerical noise','discarding these cases is safe','relaxing cutoff solves larger misses','source structural information is unlearnable','new objective or bigger dataset will certainly work'],
        unchanged=dict(models=True,labels=True,cutoffs=True,decisions=True,original_quality_criterion=True),
        new_fits=0,new_scores=0,alternative_thresholds_evaluated=0,new_physical_calls=0)
    save('calibration_goal_audit.json',evidence)
    lines=['# GRID135补充：保护目标与校准见证','',
        '这次仅检查已经保存的校准见证，不计算替代门槛，也不重做任何模型决策。','',
        '## 固定门槛由哪些记录决定','']
    for call,f in facts.items():lines.append(f"- {call}：约束{len(f['models'])}份模型的门槛；补查奖励增量{f['gain_range']}，剩余查询{f['pending_query_range']}。这些是同一调用的少量相关前缀，不是独立场景。")
    lines+=['',
        '种子0和1的八份模型都由2037-01-05_100|call006的一条记录约束：补查214个候选，增量3.5762786865234375e−7。种子2的四份模型由2037-07-06_101|call001约束：增量1.4662742614746094e−5，剩余215或237个候选。',
        '保护规则是“全部正增量或动作变化校准见证都不得被跳过”，因此这些记录理应受到保护。这是方案本身的要求，不能事后把小增量删作噪声。','',
        '## 它解释什么，又解释不了什么','',
        '这个标准与原论文的整周成本不超过FULL的100.5%不是同一质量定义。只预测“有无任何增量”且要求保住所有正例，会把接近奖励数值分辨率的小差值与较大差值放在同一个类别；它适合作为严格开发诊断，但未必直接对应最终控制目标。',
        '不过种子0在另一调用误停的增量为0.0013315677642822266，明显高于这些校准见证。因此失败不能只归为门槛过严；它同时说明模型对不同日期上有用补查的分数还不可靠。单纯放宽门槛或忽略小差值，不能据现有证据推断会保护较大收益。',
        '当前阈值、标签、旧0.5%整周质量标准和所有结果完全不变，不以这份补充说明把未通过的模型改判通过。','',
        '## 对后续研究的约束','',
        '下一步先用现有训练/校准记录明确收益尺度、调用重复和跨日期覆盖，分开回答“能否识别有益补查”与“愿意承担多少增量损失来省计算”。若要改学习目标或质量预算，必须事前说明对应实际控制目标的依据并重新封存，不能靠验证错例挑损失容忍度。',
        'GRID125–127对收益回归、非负回归和代价相关目标的失败仍保留；不是简单加个收益权重就能解决。本轮不新增拟合、数据或物理回放。']
    (O/'保护目标与校准见证.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for n,h in frozen.items():assert sha(O/n)==h
    post=A/'calibration_goal';post.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'calibration_goal_audit.json',O/'保护目标与校准见证.md']:shutil.copy2(p,post/p.name)
    (post/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in post.iterdir()},indent=2)+'\n',encoding='utf8')
    collab=R/'论文协作资料/FRAME_完整尾部匹配学习_GRID135'
    for n in ['calibration_goal_audit.json','保护目标与校准见证.md']:
        shutil.copy2(O/n,collab/n);assert sha(O/n)==sha(collab/n)
    entry='''

## 2026-10-23 — GRID135保护目标与校准见证只读核查

十二份门槛均由两物理调用约束：种子0/1的8份来自2037-01-05_100|call006一前缀，214查询/G=3.5762786865e-7；种子2的4份来自2037-07-06_101|call001相关前缀，215–237查询/G=1.4662742615e-5。全部正增量同等保护与旧整周FULL+0.5%质量标准不等价；不据此删除微小增量作噪声。新验证误停G=.001331567764明显更大，不能把失败只归为阈值严或据此放宽阈值。模型/标签/门槛/动作/旧质量标准均未改，零替代门槛评估或新拟合。下一步先厘清业务保护目标与原数据调用/日期覆盖；旧125–127回归/代价目标失败仍保留，不简单回到加权损失。补充：research_archive/power-grid-2026/protocols/GRID135_matched_tail/calibration_goal/保护目标与校准见证.md
'''
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+entry,encoding='utf8')
    p=REPO/'docs/NEXT_STEPS.md';p.write_text(p.read_text(encoding='utf8')+'\n5. GRID135的12份门槛受两调用的微小正增量见证约束，但新误停收益更大；先澄清业务质量目标/收益尺度与模型跨日期可靠性，不能据验证例子调整容忍度。保留125–127代价/收益学习失败，不自动换加权损失或重新训练。\n',encoding='utf8')
    verify();print(json.dumps(evidence,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
