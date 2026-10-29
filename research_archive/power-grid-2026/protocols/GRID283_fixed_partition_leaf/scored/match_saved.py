def match(folder,arm,name):
 oldpred={(r['fold'],r['seed'],r['sample_id'],r['position']):r for r in C.read(folder/'gate_predictions.json')};prevmetrics=C.read(folder/'search_metrics.json');oldfam=C.read(folder/'family_summary.json')[arm];changes=[];paired=[]
 for r in new:
  z=oldpred[(r['fold'],r['seed'],r['sample_id'],r['position'])]
  for k in ['base_continue','actual_next_NET','actual_next_gain','actual_next_queries']:assert r[k]==z[k]
  if r['continue_']!=z['continue_']:
   sign=int(r['continue_'])-int(z['continue_']);changes.append(dict(**r,old_continue=z['continue_'],new_continue=r['continue_'],NET_delta=sign*r['actual_next_NET'],query_delta=sign*r['actual_next_queries'],gain_delta=sign*r['actual_next_gain']))
 for seed in range(3):
  n=metrics[f'FIXED_LEAF_seed{seed}'];z=prevmetrics[f'{arm}_seed{seed}'];d={k:n['macro'][k]-z['macro'][k] for k in n['macro']};nd={v['date']:v['cases'] for v in n['per_date']};ss=[v for v in changes if v['seed']==seed]
  for k,f in [('mean_net_utility','NET_delta'),('mean_queries','query_delta'),('mean_gain','gain_delta')]:assert abs(sum(v[f]/(16*nd[v['date']]) for v in ss)-d[k])<1e-12
  paired.append(dict(seed=seed,macro_delta=d))
 categories=[]
 for label,sign,positive in [('profitable_restorations',1,True),('nonprofitable_restorations',1,False),('profitable_cancellations',-1,True),('nonprofitable_cancellations',-1,False)]:
  ss=[v for v in changes if (int(v['new_continue'])-int(v['old_continue']))==sign and (v['actual_next_NET']>0)==positive];v=dict(category=label,records=len(ss),raw_query_delta=sum(t['query_delta'] for t in ss),calls=len({t['call'] for t in ss}),dates=len({t['date'] for t in ss}),macro_NET_delta=0.,macro_query_delta=0.)
  for t in ss:
   nd={r['date']:r['cases'] for r in metrics[f"FIXED_LEAF_seed{t['seed']}"]['per_date']};den=3*16*nd[t['date']];v['macro_NET_delta']+=t['NET_delta']/den;v['macro_query_delta']+=t['query_delta']/den
  categories.append(v)
 assert abs(sum(v['macro_NET_delta'] for v in categories)-(nf['mean_net_utility']-oldfam['mean_net_utility']))<1e-12 and abs(sum(v['macro_query_delta'] for v in categories)-(nf['mean_queries']-oldfam['mean_queries']))<1e-12
 result=dict(old_family=oldfam,new_family=nf,family_delta={k:nf[k]-oldfam[k] for k in nf},per_seed=paired,changes=len(changes),categories=categories,raw_query_delta=sum(v['query_delta'] for v in changes))
 C.save('matched'+name+'_changes.json',changes);C.save('matched'+name+'_summary.json',result);return result,changes
