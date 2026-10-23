"""Conserved OOF error aggregation; no fitted models or optimization."""
from pathlib import Path
import collections,json,pickle,hashlib
import numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid121_date_transfer'
read=lambda p:json.loads(p.read_text(encoding='utf8'))

def main():
    assert read(O/'review.json')['passed']
    tr=read(O/'oof_trajectories.json');cases=pickle.loads((O/'train_cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};by={(r['method'],r['sample_id']):r for r in tr};datecount=collections.Counter(c['date'] for c in cases)
    groups=collections.defaultdict(lambda:dict(net_contribution=0.,prefix_seed_pairs=0,physical_group='',date=''))
    divergences=collections.Counter();divcalls=collections.defaultdict(set)
    for s in range(3):
        for c in cases:
            h=by[f'HUBER_seed{s}',c['sample_id']];u=by[f'UTILITY_BCE_seed{s}',c['sample_id']]
            weight=1/3/16/datecount[c['date']];g=groups[c['physical_state_group']];g['net_contribution']+=weight*(u['net_utility']-h['net_utility']);g['prefix_seed_pairs']+=1;g['physical_group']=c['physical_state_group'];g['date']=c['date']
            for j,(aa,bb) in enumerate(zip(u['actions'],h['actions'])):
                if aa==bb:continue
                y=c['gains'][j]-read(O/'design.json')['cost_per_query']*c['queries'][j]
                key=f'utility{aa}_huber{bb}_next'+('positive' if y>0 else 'nonpositive')
                divergences[key]+=1;divcalls[key].add(c['physical_state_group']);break
    total=sum(g['net_contribution'] for g in groups.values());d=read(O/'decision.json');expected=d['family_oof_utility']['UTILITY_BCE']-d['family_oof_utility']['HUBER'];assert abs(total-expected)<1e-12
    neg=sorted((g for g in groups.values() if g['net_contribution']<0),key=lambda x:x['net_contribution']);pos=sorted((g for g in groups.values() if g['net_contribution']>0),key=lambda x:-x['net_contribution'])
    result=dict(passed=True,scope='Read-only aggregation of already audited OOF trajectories. Prefixes correlated; call contributions not independent causal effects or wholeweek costs.',
        total_difference=total,negative_contribution=sum(x['net_contribution'] for x in neg),positive_contribution=sum(x['net_contribution'] for x in pos),
        top_negative=neg[:8],top_positive=pos[:8],top3_negative_fraction_of_all_negative=sum(x['net_contribution'] for x in neg[:3])/sum(x['net_contribution'] for x in neg),
        first_common_prefix_divergences={k:dict(prefix_seed_pairs=v,distinct_physical_calls=len(divcalls[k])) for k,v in divergences.items()},new_fits=0,new_physical_calls=0)
    path=O/'transfer_error_concentration.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
