"""Pure pre-completion feature arithmetic. No outcomes, labels or simulator API."""
import math
import numpy as np

FEATURES={
 'station_remaining_log1p':('候选结构','log(1+剩余同站代表数)','已知候选及别名'),
 'station_queried_fraction':('候选结构','已查同站代表数/全部同站代表数','已知候选及已查ID'),
 'remaining_best_partition_distance_min':('候选结构','剩余意向分区到当前最佳分区的最小成对连接差异','动作命令与当前拓扑'),
 'remaining_best_partition_distance_mean':('候选结构','上述成对连接差异的均值','动作命令与当前拓扑'),
 'remaining_best_partition_distance_max':('候选结构','上述成对连接差异的最大值','动作命令与当前拓扑'),
 'remaining_seen_nearest_distance_mean':('候选结构','每个剩余分区到最近已查同站分区的差异，再取均值','动作命令及已查ID'),
 'remaining_seen_nearest_distance_max':('候选结构','上述最近已查差异的最大值','动作命令及已查ID'),
 'remaining_novel_partition_fraction':('候选结构','没有相同成对连接签名的剩余分区占比','动作命令及已查ID'),
 'remaining_current_partition_distance_mean':('候选结构','剩余分区到当前实际分区的平均成对连接差异','动作命令与当前拓扑'),
 'station_legal_reward_std':('同站反馈','同站已查合法、未结束、有限正rho反馈的奖励总体标准差','仅已查同站反馈'),
 'station_best_runnerup_reward_gap':('同站反馈','同站已查可接纳奖励第一名与第二名之差；不足两例填0','仅已查同站反馈'),
 'station_runnerup_missing':('同站反馈','同站已查可接纳反馈不足两例的指示量','仅已查同站反馈'),
 'station_legal_feedback_fraction':('同站反馈','同站已查反馈合法、未结束、有限正rho的比例','仅已查同站反馈'),
 'station_other_best_minus_outside_best':('同站反馈','排除当前最佳后，同站可接纳最好奖励减站外可接纳最好奖励；任一缺失填0，沿用原station_support_missing','仅已查反馈'),
 'incumbent_clearance_to_limit':('当前余量','rho_limit-incumbent_rho；原特征的确定组合，不是新观测','原15维'),
 'incumbent_clearance_to_safe':('当前余量','0.9-incumbent_rho；原特征的确定组合，不是新观测','原15维及原阈值'),
 'current_max_rho':('当前状态','当前在线线路最大rho','当前观测'),
 'current_overload_excess_mean':('当前状态','当前在线线路max(rho-1,0)之和/在线线路数','当前观测'),
 'station_incident_max_rho':('当前状态','当前最佳站所接在线线路最大rho；无在线接线填0','当前观测与端点归属'),
 'station_incident_overload_share':('当前状态','该站所接过载在线线路数/全网过载在线线路数；无过载填0','当前观测与端点归属'),
}

def reward(r):return float(np.float32(r['reward']))
def rank(r):return reward(r),-int(r['pool_id'])
def partition_signature(topology):
    """Same-bus connectivity of each unordered device pair; bus-label invariant."""
    a=np.asarray(topology);u,v=np.triu_indices(len(a),1)
    return ((a[u]>0)&(a[v]>0)&(a[u]==a[v])).astype(np.uint8)

def compute(c):
    # Deliberately narrow API: only redacted prefix records and public structure.
    assert set(c)=={'order','aliases','subids','visible','best_id','target_ids','target_plans','current_station_topology','current_rho','current_line_status','incident_lines','rho_limit'}
    order=c['order'];visible=c['visible'];seen={r['pool_id'] for r in visible}
    assert len(seen)==len(visible) and [r['pool_id'] for r in visible]==order[:len(visible)]
    assert all(set(r)=={'pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible'} for r in visible)
    best=max((r for r in visible if r['strict_admissible']),key=rank);assert best['pool_id']==c['best_id']
    subs=c['subids'];station=subs[c['best_id']]
    targets={int(v) for k,v in c['aliases'].items() if subs[int(k)]==station}
    assert targets==set(c['target_ids']) and targets<=set(order)
    local=[r for r in visible if r['pool_id'] in targets];remaining=[i for i in order if i in targets and i not in seen]
    assert remaining and local
    plans={int(i):np.asarray(v) for i,v in c['target_plans'].items()};assert set(plans)==targets
    signatures={i:partition_signature(v) for i,v in plans.items()}
    assert len(next(iter(signatures.values())))>0
    def dist(a,b):return float(np.mean(a!=b))
    d_best=[dist(signatures[i],signatures[c['best_id']]) for i in remaining]
    d_near=[min(dist(signatures[i],signatures[r['pool_id']]) for r in local) for i in remaining]
    current=partition_signature(c['current_station_topology'])
    legal=[r for r in local if not (r['done'] or r['illegal'] or r['ambiguous'] or r['exceptions']) and np.isfinite(r['rho']) and r['rho']>0 and np.isfinite(r['reward'])]
    eligible=sorted([reward(r) for r in local if r['strict_admissible']],reverse=True)
    other=[reward(r) for r in local if r['strict_admissible'] and r['pool_id']!=c['best_id']]
    outside=[reward(r) for r in visible if r['strict_admissible'] and r['pool_id'] not in targets]
    rho=np.asarray(c['current_rho'],float);live=np.asarray(c['current_line_status'],bool);incident=np.asarray(c['incident_lines'],bool)&live;overloaded=(rho>1)&live
    x=dict(station_remaining_log1p=math.log1p(len(remaining)),station_queried_fraction=len(local)/len(targets),
        remaining_best_partition_distance_min=min(d_best),remaining_best_partition_distance_mean=float(np.mean(d_best)),remaining_best_partition_distance_max=max(d_best),
        remaining_seen_nearest_distance_mean=float(np.mean(d_near)),remaining_seen_nearest_distance_max=max(d_near),remaining_novel_partition_fraction=sum(v>0 for v in d_near)/len(d_near),
        remaining_current_partition_distance_mean=float(np.mean([dist(signatures[i],current) for i in remaining])),
        station_legal_reward_std=float(np.std([reward(r) for r in legal])) if legal else 0.,station_best_runnerup_reward_gap=eligible[0]-eligible[1] if len(eligible)>1 else 0.,station_runnerup_missing=float(len(eligible)<2),
        station_legal_feedback_fraction=len(legal)/len(local),station_other_best_minus_outside_best=max(other)-max(outside) if other and outside else 0.,
        incumbent_clearance_to_limit=float(c['rho_limit'])-float(best['rho']),incumbent_clearance_to_safe=.9-float(best['rho']),
        current_max_rho=float(np.max(rho[live])) if live.any() else 0.,current_overload_excess_mean=float(np.maximum(rho[live]-1,0).sum()/live.sum()) if live.any() else 0.,
        station_incident_max_rho=float(np.max(rho[incident])) if incident.any() else 0.,station_incident_overload_share=float((overloaded&incident).sum()/overloaded.sum()) if overloaded.any() else 0.)
    assert list(x)==list(FEATURES) and all(np.isfinite(v) for v in x.values())
    return x
