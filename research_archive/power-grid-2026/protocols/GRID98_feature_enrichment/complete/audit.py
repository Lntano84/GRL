"""Independent arithmetic on redacted inputs, no imported feature extractor."""
from pathlib import Path
import json,math,hashlib,statistics,struct
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid98_feature_enrichment'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def recompute(c):
    # Original policy compares float32 rewards; independently implement rounding.
    rew=lambda r:struct.unpack('f',struct.pack('f',r['reward']))[0]
    seen={r['pool_id'] for r in c['visible']};targets=set(c['target_ids']);remaining=[i for i in c['order'] if i in targets and i not in seen];local=[r for r in c['visible'] if r['pool_id'] in targets]
    best=next(r for r in local if r['pool_id']==c['best_id']);plans={int(k):v for k,v in c['target_plans'].items()}
    def signature(v):return [int(v[i]>0 and v[j]>0 and v[i]==v[j]) for i in range(len(v)) for j in range(i+1,len(v))]
    sigs={k:signature(v) for k,v in plans.items()}
    def distance(a,b):return sum(x!=y for x,y in zip(a,b))/len(a)
    db=[distance(sigs[i],sigs[c['best_id']]) for i in remaining]
    dn=[min(distance(sigs[i],sigs[r['pool_id']]) for r in local) for i in remaining]
    legal=[r for r in local if not r['done'] and not r['illegal'] and not r['ambiguous'] and not r['exceptions'] and math.isfinite(r['rho']) and r['rho']>0 and math.isfinite(r['reward'])]
    eligible=sorted((rew(r) for r in local if r['strict_admissible']),reverse=True)
    others=[rew(r) for r in local if r['strict_admissible'] and r['pool_id']!=c['best_id']]
    outside=[rew(r) for r in c['visible'] if r['strict_admissible'] and r['pool_id'] not in targets]
    live=[i for i,v in enumerate(c['current_line_status']) if v];over=[i for i in live if c['current_rho'][i]>1];incident=[i for i in live if c['incident_lines'][i]]
    return dict(station_remaining_log1p=math.log(1+len(remaining)),station_queried_fraction=len(local)/len(targets),
        remaining_best_partition_distance_min=min(db),remaining_best_partition_distance_mean=statistics.mean(db),remaining_best_partition_distance_max=max(db),
        remaining_seen_nearest_distance_mean=statistics.mean(dn),remaining_seen_nearest_distance_max=max(dn),remaining_novel_partition_fraction=sum(d>0 for d in dn)/len(dn),remaining_current_partition_distance_mean=statistics.mean([distance(sigs[i],signature(c['current_station_topology'])) for i in remaining]),
        station_legal_reward_std=statistics.pstdev([rew(r) for r in legal]) if legal else 0.,station_best_runnerup_reward_gap=eligible[0]-eligible[1] if len(eligible)>1 else 0.,station_runnerup_missing=float(len(eligible)<2),station_legal_feedback_fraction=len(legal)/len(local),station_other_best_minus_outside_best=max(others)-max(outside) if others and outside else 0.,
        incumbent_clearance_to_limit=c['rho_limit']-best['rho'],incumbent_clearance_to_safe=.9-best['rho'],current_max_rho=max((c['current_rho'][i] for i in live),default=0.),current_overload_excess_mean=sum(max(c['current_rho'][i]-1,0) for i in live)/len(live) if live else 0.,station_incident_max_rho=max((c['current_rho'][i] for i in incident),default=0.),station_incident_overload_share=sum(i in incident for i in over)/len(over) if over else 0.)
def main():
    d=read(OUT/'design.json');rows=read(OUT/'features.json');old=read(ROOT/'outputs/grid97_effect_learning/dataset.json');oldmap={r['case_id']:r for r in old};assert len(rows)==len(old)==56
    assert read(OUT/'original_folds_unchanged.json')==read(ROOT/'outputs/grid97_effect_learning/design.json')['folds']
    labs=read(OUT/'labels_and_groups_unchanged.json');assert labs==[{k:v for k,v in r.items() if k!='features'} for r in old]
    provenance={r['case_id']:r for r in read(OUT/'provenance.json')};checks=0
    for row in rows:
        i=row['case_id'];c=read(OUT/'contexts'/(i+'.json'));assert sha(OUT/'contexts'/(i+'.json'))==provenance[i]['context_sha256']
        expected=recompute(c);assert set(expected)==set(row['added20'])==set(d['new_features'])
        for k,v in expected.items():assert math.isclose(v,row['added20'][k],rel_tol=1e-11,abs_tol=1e-12),(i,k,v,row['added20'][k]);checks+=1
        assert row['original15']==oldmap[i]['features'];assert row['all35']==dict(row['original15'],**row['added20']);assert len(row['all35'])==35
        visible={r['pool_id'] for r in c['visible']};assert not visible&set(provenance[i]['remaining_ids']);assert set(c['visible'][0])=={'pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible'}
    for n,h in read(OUT/'freeze.json')['sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in read(ROOT/'outputs/grid97_effect_learning/freeze.json')['sha256'].items():assert sha(ROOT/n)==h,n
    b=read(OUT/'build_checks.json');assert b['passed'] and all(v>=56 for k,v in b['checks'].items() if k not in ['visible_hash_checks'])
    template=read(OUT/'template_metadata.json');assert template['template_forecast_counter_before']==template['template_forecast_counter_after']==0
    receipt=dict(passed=True,states=56,old_features_unchanged=15,added_features=20,independent_scalar_recomputations=checks,old_labels_unchanged=True,old_folds_unchanged=True,
        new_forecasts=0,new_replays=0,new_fits=0,new_predictions=0,build_checks=b['checks'],template_forecast_counter=0,
        limits='All physical outcomes inherited and untouched. Intended partition describes public action commands, not solved future state. No predictive improvement asserted.')
    (OUT/'review.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
