"""GRID262 fixed descriptive support diagnosis; no fitting or new policy evaluation."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,collections
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid262_cancellation_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID262_cancellation_support'
N=R/'outputs/grid261_bounded_basis';Q=R/'outputs/grid260_bounded_basis_plan';P=R/'outputs/grid249_context_bit_plan';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def verify():
 for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8')

def published(phase):
 head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 for rel,h in rd(A/phase/'archive_manifest.json').items():assert hashlib.sha256(git('show',head+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==h
 return head

def prepare():
 assert not O.exists() and rd(N/'complete.json')['passed'];O.mkdir()
 head=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 for rel in ['complete.json','independent_audit.json','gate_preservation_summary.json']:
  prev=REPO/'research_archive/power-grid-2026/protocols/GRID261_bounded_basis/audited'/rel
  assert git('show',head+':'+prev.relative_to(REPO).as_posix())==(N/rel).read_bytes()
 design=dict(id='GRID262',scope='ONEfixedreadonlydescription77cancelledparentCONTrecords;nooptimizer/alpha/newpolicy/newouterreturn/physics/full261reaudit.',focus='All77cancellations,allseeds/pools/adverse/supportgaps. Savedscore margins and additive public/b/bit/intercept contributions only;nohybridstrategy.',neighbors='OneEuclideannearestownfit8record PER nativeSTOP/CONT domain in existingbounded40inputspace;exclude samephysical_state_group and self;stable firstrow tie. No learnedmetric/k/thresholdsearch. Publicpairs/distances/indices publishedbefore owntraining y/NET/weight joins.',public_fields='ownfit arrays X,b,ids,domain_STOP only;249 ids/X/b;261 publishedtheta/alpha/logit. metadata sample/date/physicalgroup only. Focusselectionusesalreadyexposed261decisions;notblind.',label_fields='Only corresponding ownfit8 y/actual_NET/weight/source_mass AFTERpublicpairpublication. Existing261 cancellationgain/query reusedforcostdescription;no new outcome generation.',contribution='Score=parentb+alpha*(sum38publicterms+theta38*b+theta39*STOP+intercept). Allactivealpha1. Verify only77existingpublishedscores,notfullpolicyrescore.',fixed='38tanh/parentb/bit/offset,originalfeeslabelsdatesseeds/controller/order/aliases/station/128/64/stagnation2/.9 unchanged. No final8/newgrid/paper/FRAME/install.',interpretation='DescriptiveNNsupport/callconcentration notcausal/independent sample/fullchainOOF. Priorown12/internalcal4 ancestry andoldexposeddevelopmentremain. Queriesnotactualtime/fullweek.',accounting='0newfits or NN/headforward;branch remains62 plus previously disclosed6continuations8285updates and604.703s/678.156cap. No parameter/transform sweep.')
 save('design.json',design);paths=[W/'run.py',O/'design.json',N/'gate_cancellations.json',N/'gate_preservation_summary.json',N/'search_metrics.json',N/'fit_receipts.json',N/'complete.json',N/'independent_audit.json',META]
 for folder in [N/'models',N/'outer_predictions',Q/'fit_inputs',P/'outer_parent_public']:paths+=sorted(folder.glob('*.npz'))
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID262_SOURCE_FROZEN')

def public():
 verify();head=published('source');assert not (O/'public_started.json').exists();save('public_started.json',dict(source_published=head))
 meta=rd(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};focus=rd(N/'gate_cancellations.json');assert len(focus)==77
 cache={};records=[]
 for r in focus:
  pool=f"f{r['fold']}_s{r['seed']}"
  if pool not in cache:cache[pool]=(np.load(Q/'fit_inputs'/f'{pool}.npz'),np.load(P/'outer_parent_public'/f'{pool}.npz'),np.load(N/'models'/f'{pool}.npz'),np.load(N/'outer_predictions'/f'{pool}.npz'))
  fit,h,m,pred=cache[pool];mid=lookup[r['sample_id'],r['position']];j=int(np.flatnonzero(h['ids']==mid)[0]);z=h['X'][j].copy();z[:38]=np.tanh(z[:38]);b=float(h['b'][j]);theta=m['theta'];alpha=float(m['alpha'][0]);terms=z*theta[:-1];parts=dict(parent_b=b,public38=float(terms[:38].sum()),parent_b_feature=float(terms[38]),STOP_feature=float(terms[39]),intercept=float(theta[-1]),alpha=alpha)
  endpoint=b+np.sum(terms)+theta[-1];score=b+alpha*(endpoint-b);assert score==pred['logit'][j] and score<0 and h['base_CONT'][j]
  neighbors=[]
  for stop in [True,False]:
   allowed=[i for i,idx in enumerate(fit['ids']) if bool(fit['domain_STOP'][i])==stop and int(idx)!=mid and meta[int(idx)]['physical_state_group']!=r['call']]
   if allowed:
    dist2=np.sum((fit['X'][allowed]-z)**2,axis=1);jfit=allowed[int(np.argmin(dist2))];idx=int(fit['ids'][jfit]);neighbors.append(dict(domain='STOP' if stop else 'CONT',fit_index=jfit,metadata_id=idx,sample_id=meta[idx]['sample_id'],position=meta[idx]['position'],date=meta[idx]['date'],call=meta[idx]['physical_state_group'],distance=float(np.sqrt(dist2.min())),eligible_candidates=len(allowed)))
   else:neighbors.append(dict(domain='STOP' if stop else 'CONT',missing=True,eligible_candidates=0))
  fitX=fit['X'];records.append(dict(pool=pool,seed=r['seed'],fold=r['fold'],sample_id=r['sample_id'],position=r['position'],metadata_id=mid,date=r['date'],call=r['call'],published_score=float(score),stop_margin=float(-score),correction=float(score-b),parts=parts,public38_min=float(z[:38].min()),public38_max=float(z[:38].max()),outside_ownfit_coordinate_range=int(np.sum((z<fitX.min(axis=0))|(z>fitX.max(axis=0)))),public_neighbors=neighbors))
 save('public_focus_pairs.json',records);save('public_access.json',dict(newpolicy_scores=0,reproduced_existing_focused_scores=77,fit_fields=['X','b','ids','domain_STOP'],outcome_fields_read_in_public=[],no_neighborlabel_search=True,source_publication=head));save('public_freeze.json',{name:sha(O/name) for name in ['public_focus_pairs.json','public_access.json']});archive('public');print('GRID262_PUBLIC_PAIRS',len(records))

def analyze():
 verify();head=published('public');assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(all_public_pairs_published=head));rows=rd(O/'public_focus_pairs.json');focus=rd(N/'gate_cancellations.json');lookup={(r['seed'],r['sample_id'],r['position']):r for r in focus};metrics=rd(N/'search_metrics.json');details=[];pools={};date=collections.defaultdict(lambda:dict(cancellations=0,rawquery=0,macro_query_saved=0.,macro_NET_saved=0.));call=collections.defaultdict(lambda:dict(cancellations=0,rawquery=0,macro_query_saved=0.,macro_NET_saved=0.))
 for r in rows:
  pool=r['pool'];fit=np.load(Q/'fit_inputs'/f'{pool}.npz')
  if pool not in pools:
   pp=dict(pool=pool,rows=len(fit['ids']),STOP_rows=int(fit['domain_STOP'].sum()),CONT_rows=int((~fit['domain_STOP']).sum()),domains={})
   for stop in [True,False]:
    mask=fit['domain_STOP']==stop;pos=fit['y']>0;name='STOP' if stop else 'CONT';total=float(fit['weight'][mask].sum());pp['domains'][name]=dict(positive=int((mask&pos).sum()),negative=int((mask&~pos).sum()),positive_risk_share=float(fit['weight'][mask&pos].sum()/total),b_min=float(fit['b'][mask].min()),b_max=float(fit['b'][mask].max()),positive_calls=len({rd(META)[int(i)]['physical_state_group'] for i in fit['ids'][mask&pos]}))
   pools[pool]=pp
  q=lookup[r['seed'],r['sample_id'],r['position']];r=dict(r);joined=[]
  for n in r['public_neighbors']:
   nn=dict(n)
   if not nn.get('missing'):
    i=nn['fit_index'];assert int(fit['ids'][i])==nn['metadata_id'];nn.update(y=int(fit['y'][i]),actual_NET=float(fit['actual_NET'][i]),risk_weight=float(fit['weight'][i]),source_mass=float(fit['source_mass'][i]))
   joined.append(nn)
  r['own_training_neighbors']=joined;r.pop('public_neighbors');r.update(actual_next_gain=q['actual_next_gain'],actual_next_NET=q['actual_next_NET'],actual_next_queries=q['actual_next_queries']);assert r['actual_next_gain']==0
  nd={x['date']:x['cases'] for x in metrics[f"BOUNDED_seed{r['seed']}"]['per_date']};denom=3*16*nd[r['date']];r['family_macro_query_saved']=q['actual_next_queries']/denom;r['family_macro_NET_saved']=-q['actual_next_NET']/denom
  for target in [date[r['date']],call[r['call']]]:
   target['cancellations']+=1;target['rawquery']+=q['actual_next_queries'];target['macro_query_saved']+=r['family_macro_query_saved'];target['macro_NET_saved']+=r['family_macro_NET_saved']
  details.append(r)
 shares=sorted(call.values(),key=lambda x:x['macro_query_saved'],reverse=True);tot=sum(r['family_macro_query_saved'] for r in details);net=sum(r['family_macro_NET_saved'] for r in details);g=rd(N/'gate_preservation_summary.json');assert abs(tot+g['family_delta']['mean_queries'])<1e-12 and abs(net-g['family_delta']['mean_net_utility'])<1e-12
 states=collections.defaultdict(list)
 for r in details:states[r['sample_id'],r['position']].append(r)
 repeated=[dict(sample_id=k[0],position=k[1],call=v[0]['call'],date=v[0]['date'],seeds=[r['seed'] for r in v],rawquery=sum(r['actual_next_queries'] for r in v)) for k,v in states.items() if len(v)>1]
 stats=lambda v:dict(min=float(np.min(v)),median=float(np.median(v)),max=float(np.max(v)))
 summary=dict(passed=True,focus77=len(details),query_saved_raw=sum(r['actual_next_queries'] for r in details),macro_query_saved=tot,macro_NET_saved=net,distinct_calls=len(call),distinct_dates=len(date),unique_state_records=len(states),two_seed_repeat_states=len(repeated),three_seed_repeat_states=sum(len(v)==3 for v in states.values()),top_call_macro_share=shares[0]['macro_query_saved']/tot,top5_call_macro_share=sum(v['macro_query_saved'] for v in shares[:5])/tot,top_call_raw_share=max(v['rawquery'] for v in call.values())/122,stop_margin=stats([r['stop_margin'] for r in details]),parent_b=stats([r['parts']['parent_b'] for r in details]),correction=stats([r['correction'] for r in details]),public38_sum=stats([r['parts']['public38'] for r in details]),outside_ownfit_any=sum(r['outside_ownfit_coordinate_range']>0 for r in details),neighbor_support={d:dict(missing=sum(any(n['domain']==d and n.get('missing') for n in r['own_training_neighbors']) for r in details),positive=sum(any(n['domain']==d and n.get('y')==1 for n in r['own_training_neighbors']) for r in details),negative=sum(any(n['domain']==d and n.get('y')==0 for n in r['own_training_neighbors']) for r in details),distance=stats([n['distance'] for r in details for n in r['own_training_neighbors'] if n['domain']==d and not n.get('missing')])) for d in ['STOP','CONT']},pool_count=len(pools),new_models=0,new_scalars=0,new_policy_scores=0,new_NN_head_forward=0,query_not_actualtime=True,correlated_exposed_development=True)
 for name,val in [('focus_support_details.json',details),('cost_by_date.json',date),('cost_by_call.json',call),('repeated_states.json',repeated),('own_fit_pool_support.json',list(pools.values())),('summary.json',summary)]:save(name,val)
 archive('analyzed');print('GRID262_ANALYZED',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
