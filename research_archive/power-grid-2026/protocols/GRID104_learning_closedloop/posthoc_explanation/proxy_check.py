"""Explain first actual divergence under the already frozen immediate query price."""
from pathlib import Path
import json,sys,shutil
import numpy as np
root=Path('D:/Codex/GRL研究');sys.path.insert(0,str(root/'work/grid104_learning_closedloop'))
import registry as R
E=R.read(R.OUT/'readonly_explanation.json');cost=R.read(root/'outputs/grid102_search_rl/design.json')['cost_per_query']
diagnoses=[]
reward=lambda r:float(np.float32(r['reward']))
for e in E['first_divergences']:
    if not e.get('same_observation_and_base') or not e.get('same_global_feedback'):continue
    files=sorted((R.OUT/'runs'/f"r0__{e['week']}__FRAME_GUARD").glob('call[0-9][0-9][0-9].json'))
    call=next(R.read(p) for p in files if R.read(p)['step']==e['first_delivered_action_divergence'])
    global_rows=[r for r in call['outcomes'] if r['phase']!='local_completion'];local=[r for r in call['outcomes'] if r['phase']=='local_completion']
    best=max((reward(r) for r in global_rows if r['strict_admissible']));initial=best;points=[0.];batch_gain=[]
    for j in range(0,len(local),64):
        for r in local[j:j+64]:
            if r['strict_admissible']:best=max(best,reward(r))
        batch_gain.append(best-initial);points.append(best-initial-cost*min(j+64,len(local)))
    diagnoses.append(dict(week=e['week'],method=e['method'],step=e['first_delivered_action_divergence'],
        known_local_queries=len(local),actual_first_batch_gain=batch_gain[0] if batch_gain else 0.,
        actual_first_batch_cost=cost*min(64,len(local)),best_immediate_net_over_all_stopping_points=max(points),
        oracle_first_batch_choice=int(len(points)>1 and points[1]>0),
        note='Hindsight diagnostic on paid FRAME feedback, not a new simulation or a deployable input.'))
R.save(R.OUT/'proxy_value_diagnosis.json',dict(posthoc=True,cost_per_query=cost,diagnoses=diagnoses,new_forecasts=0,new_fits=0))
a=next(d for d in diagnoses if d['method']=='LOCAL_RF' and d['week']=='2037-03-02_100')
b=next(d for d in diagnoses if d['method']=='LOCAL_DQN0_0' and d['week']=='2037-03-02_100')
text='\n\n## 进一步定位：即时目标与闭环目标不能混为一谈\n\n'
text+=f"3月RF首次动作分歧在第{a['step']}步。利用FRAME在同一状态、同一基础动作下实际查询过的候选，下一批真实奖励增益为{a['actual_first_batch_gain']:.8f}，原固定费用为{a['actual_first_batch_cost']:.8f}；所有合法分批停止点中的最佳即时净收益为{a['best_immediate_net_over_all_stopping_points']:.8f}（立即停止为0）。因此，这次RF不查的决定符合当前即时收益减费用目标，不能简单归因于模型预测不准；但其最终整周成本仍比FRAME高1.112%。\n\n"
text+=f"3月DQN种子0/1在第{b['step']}步漏掉的下一批真实奖励增益为{b['actual_first_batch_gain']:.8f}，查询费用只有{b['actual_first_batch_cost']:.8f}，下一批在原目标下也值得查。这与RF案例不同，暴露了停止策略的误判。两类问题需要分开处理：当前目标下决策错误，以及当前目标下合理却未改善闭环质量。\n\n"
text+='这些是共同前缀内已揭示反馈的定位，不能把单次分歧当作整周成本涨幅的完整因果解释。当前费用是启发式代理，奖励是一步预测奖励，不是整周货币成本。扩大网络只能尝试改善前一类预测误差，不能自动修复后一类目标差异；本轮不修改费用或标签救结果。\n'
p=R.OUT/'闭环失败与开销解释_中文.md';raw=p.read_text(encoding='utf8')
raw=raw.replace('种子2保持了FRAME轨迹，但不能据此挑选最好种子。','3月的种子2保持了FRAME轨迹，但不能据此挑选最好种子。')
p.write_text(raw+text,encoding='utf8')
collab=root/'论文协作资料/FRAME_学习补查闭环_GRID104_20261022';dest=R.ARCHIVE/'posthoc_explanation'
for name in ['闭环失败与开销解释_中文.md','proxy_value_diagnosis.json']:
    shutil.copy2(R.OUT/name,collab/name);shutil.copy2(R.OUT/name,dest/name)
shutil.copy2(Path(__file__),dest/'proxy_check.py')
R.save(dest/'archive_manifest.json',{p.name:R.sha(p) for p in dest.iterdir() if p.is_file() and p.name!='archive_manifest.json'})
print(json.dumps(dict(march_rf=a,march_dqn=b),ensure_ascii=False,indent=2))
