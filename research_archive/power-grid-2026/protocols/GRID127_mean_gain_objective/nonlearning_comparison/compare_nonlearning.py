"""Read-only comparison on saved local-search cases; no fitting or simulator."""
from pathlib import Path
import collections
import hashlib
import json
import math
import pickle
import sys
import numpy as np

R = Path('D:/Codex/GRL研究')
O = R/'outputs/grid127_mean_gain_objective/nonlearning_comparison'
S = O.parent
sys.path.insert(0, str(R/'work/grid107_direct_value'))
from common107 import summarize, sealed

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')

def independent_summary(rows):
    dates = sorted({r['date'] for r in rows})
    bydate = {}
    for d in dates:
        rs = [r for r in rows if r['date'] == d]
        bydate[d] = {
            'mean_net_utility': math.fsum(r['net_utility'] for r in rs)/len(rs),
            'query_fraction': math.fsum(r['queries'] for r in rs)/math.fsum(r['full_queries'] for r in rs),
            'gain_capture': math.fsum(r['gain'] for r in rs)/math.fsum(r['full_gain'] for r in rs)
                if math.fsum(r['full_gain'] for r in rs) else None,
        }
    return {k: math.fsum(v[k] for v in bydate.values() if v[k] is not None)/sum(v[k] is not None for v in bydate.values()) for k in bydate[dates[0]]}

def main():
    sealed()
    inputs = [R/'outputs/grid102_search_rl/cases.pkl', R/'outputs/grid102_search_rl/design.json',
              R/'outputs/grid102_search_rl/validation_results.json',
              S/'validation_results.json', S/'oof_results.json',
              S/'full_validation_trajectories.json', S/'oof_trajectories.json',
              S/'posthoc_transfer/risk_parts.json', Path(__file__), R/'work/grid107_direct_value/common107.py',
              R/'outputs/grid107_direct_value/validation_results.json']
    frozen = {p.relative_to(R).as_posix(): sha(p) for p in inputs}
    cases = pickle.loads(inputs[0].read_bytes())
    design = read(inputs[1]); fee = design['cost_per_query']; scale = design['gain_scale']
    out = {}; differences = {}; checks = 0
    for domain, partition, fn in [('validation','validation','full_validation_trajectories.json'), ('oof','train','oof_trajectories.json')]:
        cs = [c for c in cases if c['partition'] == partition]
        lookup = {c['sample_id']: c for c in cs}
        assert len(lookup) == len(cs)
        assert len(cs) == (590 if domain == 'validation' else 2378)
        models = read(S/fn); old = read(S/('validation_results.json' if domain == 'validation' else 'oof_results.json'))
        baseline = {}
        for name in ['FULL_LOCAL','BATCH1']:
            rows = []
            for c in cs:
                p = len(c['gains']) if name == 'FULL_LOCAL' else 1
                assert len(c['gains']) == len(c['states']) == len(c['queries'])
                gain = float(c['gains'][:p].sum()); queries = int(c['queries'][:p].sum())
                rows.append(dict(sample_id=c['sample_id'], date=c['date'], physical_state_group=c['physical_state_group'],
                    gain=gain, queries=queries, full_gain=float(c['gains'].sum()), full_queries=int(c['queries'].sum()),
                    net_utility=(gain-fee*queries)/scale))
            result = summarize(rows)
            for k,v in independent_summary(rows).items():
                assert abs(v-result['macro'][k]) < 1e-12; checks += 1
            baseline[name] = {r['sample_id']: r for r in rows}
            out.setdefault(domain,{})[name] = result
        for arm in ['NET_ZERO','POS_KL']:
            allseeds = []
            for seed in range(3):
                name=f'{arm}_seed{seed}'; rs=[r for r in models if r['method']==name]
                assert len(rs) == len(cs) and {r['sample_id'] for r in rs} == set(lookup)
                for r in rs:
                    c=lookup[r['sample_id']]; p=r['batches']
                    assert r['date']==c['date'] and r['physical_state_group']==c['physical_state_group']
                    assert r['chosen_pool_id']==c['chosen'][p]
                    gain=float(c['gains'][:p].sum()); queries=int(c['queries'][:p].sum())
                    assert abs(gain-r['gain'])<1e-12 and queries==r['queries']
                    assert abs((gain-fee*queries)/scale-r['net_utility'])<1e-12
                    checks+=1
                result=summarize(rs)
                for k,v in independent_summary(rs).items():
                    assert abs(v-result['macro'][k])<1e-12 and abs(v-old[name]['all']['macro'][k])<1e-12;checks+=1
                allseeds.append(result)
            out[domain][arm]={'macro':{k:float(np.mean([v['macro'][k] for v in allseeds])) for k in allseeds[0]['macro']},
                'per_seed':allseeds,
                'per_date':{d:{k:float(np.mean([v['per_date'][d][k] for v in allseeds])) for k in ['mean_net_utility','query_fraction','gain_capture']} for d in allseeds[0]['per_date']}}
        # Exact all-prefix, all-seed contribution accounting against fixed one batch.
        counts=collections.Counter(c['date'] for c in cs); nd=len(counts)
        parts=collections.defaultdict(lambda:dict(records=0,calls=set(),utility_contribution=0.))
        calls=collections.defaultdict(float)
        for r in models:
            if not r['method'].startswith('POS_KL_seed'):continue
            c=lookup[r['sample_id']]; base=baseline['BATCH1'][r['sample_id']]
            if r['batches']==0:
                label='skip_profitable_first_batch' if base['net_utility']>0 else 'skip_nonpositive_first_batch'
            elif r['batches']==1:label='same_one_batch'
            else:label='buy_more_than_one_batch'
            w=1/(3*nd*counts[c['date']]); delta=(r['net_utility']-base['net_utility'])*w
            parts[label]['records']+=1;parts[label]['calls'].add(c['physical_state_group']);parts[label]['utility_contribution']+=delta
            calls[c['physical_state_group']]+=delta
        for v in parts.values():v['calls']=len(v['calls'])
        delta=out[domain]['POS_KL']['macro']['mean_net_utility']-out[domain]['BATCH1']['macro']['mean_net_utility']
        assert abs(sum(v['utility_contribution'] for v in parts.values())-delta)<1e-12;checks+=1
        differences[domain]={'vs_batch1':dict(parts),'total_utility_difference':delta,
            'all_call_contributions':sorted(calls.items(),key=lambda item:item[1]),
            'vs_full_local_utility_relative':out[domain]['POS_KL']['macro']['mean_net_utility']/out[domain]['FULL_LOCAL']['macro']['mean_net_utility']-1,
            'vs_batch1_utility_relative':out[domain]['POS_KL']['macro']['mean_net_utility']/out[domain]['BATCH1']['macro']['mean_net_utility']-1,
            'vs_net_zero_utility_relative':out[domain]['POS_KL']['macro']['mean_net_utility']/out[domain]['NET_ZERO']['macro']['mean_net_utility']-1}
    legacy=read(inputs[-1])
    for name in ['FULL_LOCAL','BATCH1']:
        for k in ['mean_net_utility','query_fraction','gain_capture']:
            assert abs(out['validation'][name]['macro'][k]-legacy[name]['all']['macro'][k])<1e-12;checks+=1
    assert frozen == {p.relative_to(R).as_posix():sha(p) for p in inputs}
    sealed();O.mkdir(parents=True,exist_ok=True)
    write(O/'source_hashes.json',frozen);write(O/'comparison.json',out);write(O/'differences.json',differences)
    write(O/'review.json',dict(passed=True,arithmetic_checks=checks,source_unchanged=True,final_tests_absent=True,
        new_fits=0,new_model_forwards=0,physical_rollouts=0,
        note='FULL_LOCAL is original FRAME local-completion behavior evaluated on identical FULL-teacher cases, not a new closed-loop FRAME trajectory. OOF baseline uses each training date once, paired with its held-out learned model.'))
    lines=['# 当前学习模型与不学习补查的对齐比较','','本次仅复算既有反馈，无训练、预测调用或物理回放。',
        'FULL_LOCAL对应原FRAME完成锁定站内剩余候选的行为；这里只替换保存状态上的局部补查，不能解释为原FRAME与新模型的整周运行对比。',
        '局部收益是相对当前最佳动作的预测奖励增量；不是控制总奖励，也不是整周成本或存活率。100%代表把该站剩余候选查完可得到的增量。',
        '净效用=(局部奖励增量−固定查询费×次数)/原收益尺度，费用是历史代理价格。查询比例、收益保留率先按日期计算，再宏平均。学习模型再平均三个种子。','']
    names={'FULL_LOCAL':'不学习：完整同站补查','BATCH1':'不学习：固定补一批','NET_ZERO':'原监督净收益回归','POS_KL':'新监督KL模型'}
    for domain,title in [('validation','4个原验证日期：590个起始前缀'),('oof','16日期分组折外：2378个起始前缀')]:
        lines += [f'## {title}','','| 方法 | 补查收益保留率 | 同站查询比例 | 扣费净效用 |','|---|---:|---:|---:|']
        for arm in names:
            m=out[domain][arm]['macro'];lines.append(f"| {names[arm]} | {100*m['gain_capture']:.2f}% | {100*m['query_fraction']:.2f}% | {m['mean_net_utility']:.8f} |")
        df=differences[domain];lines += ['',f"新KL净效用相对完整同站补查{100*df['vs_full_local_utility_relative']:+.2f}%，相对固定一批{100*df['vs_batch1_utility_relative']:+.2f}%，相对原监督回归{100*df['vs_net_zero_utility_relative']:+.2f}%。",'']
        lines += ['对固定一批的全部差值记账（相同调用/前缀相关，记录数不是独立场景）：']
        for k,v in df['vs_batch1'].items():lines.append(f"- {k}：{v['records']}条前缀—种子记录，{v['calls']}次调用，净效用贡献{v['utility_contribution']:+.8f}。")
        lines.append('')
    lines += ['## 判断','','新KL减少了无利查询，也漏掉更多有价值的补查，尚不能作为保持原质量的替代方案。固定补一批在两个评价域的收益保留率和净效用均高于新KL；它是必须保留的简单对照。',
        '原监督回归也未在两个评价域稳定超过固定一批，不能因它优于最新模型就称学习模块已成功。',
        '新KL没有整周物理验证；目前不能报告其相对原FRAME的整周成本差、完成率或实际提速。GRID104是更早RF/DQN权重的物理实验，不对应本轮模型。',
        '原验证和折外均为已暴露同网开发数据；原16训练日期的历史费用与尺度沿用，折外属于条件开发诊断，不是最终独立测试。前缀、同日副本、种子均不能当独立场景。',
        '下一步优先核查固定一批保住、新模型却放弃的正收益机会，以及决策前可见证据能否区分这些机会和无利查询。暂不重复更换损失或加大网络；当前不启动新训练、扩大场景或使用最终测试。','']
    (O/'与不学习方法比较_中文.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'review':read(O/'review.json'),'macros':{d:{a:v['macro'] for a,v in arms.items()} for d,arms in out.items()},'differences':{d:{k:v for k,v in x.items() if k!='all_call_contributions'} for d,x in differences.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
