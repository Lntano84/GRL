"""Posthoc combination coverage; no fitted transformations or model."""
from common111 import O,read,save
import gzip,json,numpy as np
def main():
    results={(r['sample_id'],r['position']):r for r in read(O/'state_results.json')}
    counts={p:dict(positive=0,novel_best_combination=0,positive_without_unknown_changed_relations=0) for p in ['train','validation']};varies=0
    with gzip.open(O/'public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            row=json.loads(line);c=row['public_input'];r=results[row['sample_id'],row['position']]
            plans=np.array(c['seen_plans']+c['next_plans']);varies+=int(np.any((plans>0)!=(plans[0]>0)))
            if r['actual_positive']:
                co=counts[row['partition']];co['positive']+=1;i=c['next_ids'].index(r['best_next_id_for_posthoc_audit_only']);t=np.array(c['next_plans'][i]);u,v=np.triu_indices(len(t),1)
                b=(t[u]>0)&(t[v]>0)&(t[u]==t[v]);s=np.array(c['seen_plans']);s=(s[:,u]>0)&(s[:,v]>0)&(s[:,u]==s[:,v])
                co['novel_best_combination']+=int(not np.any(np.all(s==b,axis=1)))
                co['positive_without_unknown_changed_relations']+=int(r['best_next_changed_pairs']==r['best_next_supported_changed_pairs'])
    answer=dict(active_device_pattern_varies_states=varies,partitions=counts,posthoc=True)
    if (O/'combination_profile.json').exists():assert read(O/'combination_profile.json')==answer
    else:save('combination_profile.json',answer)
if __name__=='__main__':main()
