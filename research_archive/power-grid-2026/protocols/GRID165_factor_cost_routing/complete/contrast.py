"""Describe all first differences against existing experts; no composite policy."""
import collections,pickle,math
import numpy as np
import run as D
def main():
 D.verify_trained();assert D.read(D.O/'diagnosis_receipt.json')['passed'];assert not (D.O/'contrast_receipt.json').exists();source=D.R/'outputs/grid164_explicit_cost/state_scores.npz';hs=D.sha(source);extra=np.load(source);q=np.load(D.O/'state_scores.npz');meta=D.read(D.R/'outputs/grid113_relational_learning/validation_metadata.json');idx={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};cases={c['sample_id']:c for c in pickle.loads((D.R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};rows=[]
 fs=__import__('importlib.util').util.spec_from_file_location('g165_names',D.R/'work/grid101_gain_learning/features.py');f=__import__('importlib.util').util.module_from_spec(fs);fs.loader.exec_module(f);names=f.NAMES+['local_batches_done','local_budget_used_fraction','log_next_batch_queries','locked_remaining_fraction']
 for r in D.read(D.O/'first_difference_records.json'):
  seed=r['seed'];i=idx[r['sample_id'],r['position']];a=extra[f'COST42_seed{seed}__validation'][i];b=extra[f'ZERO42_seed{seed}__validation'][i];features={k:float(x) for k,x in zip(names,cases[r['sample_id']]['states'][r['position']])};rows.append(dict(r,cost42_continue=bool(a[1]>a[0]),base41_continue=bool(b[1]>b[0]),cost42_margin=float(a[1]-a[0]),public_features=features))
 summary=[]
 for cat in sorted({r['category'] for r in rows}):
  rr=[r for r in rows if r['category']==cat];summary.append(dict(category=cat,events=len(rr),full64=sum(r['known_next_queries']==64 for r in rr),partial=sum(r['known_next_queries']<64 for r in rr),next_queries_min=min(r['known_next_queries'] for r in rr),next_queries_median=float(np.median([r['known_next_queries'] for r in rr])),next_gain_zero=sum(r['audit_next_gain']==0 for r in rr),next_net_positive=sum(r['audit_next_net']>0 for r in rr),cost42_continue=sum(r['cost42_continue'] for r in rr),base41_continue=sum(r['base41_continue'] for r in rr)))
 assert hs==D.sha(source);D.verify_trained();D.save('existing_expert_contrasts.json',rows);D.save('existing_expert_summary.json',summary);D.save('contrast_receipt.json',dict(passed=True,post_hoc_descriptive=True,no_composite_policy_selected=True,no_new_scores=True,new_fits=0,cost42_score_source_sha256=hs,full64_is_original_batch_size_not_tuned=True));print(summary)
if __name__=='__main__':main()
