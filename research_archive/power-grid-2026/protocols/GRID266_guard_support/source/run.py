"""Fixed readonly GRID266 explanation of existing191 cancellations; no new policy."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,collections
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid266_guard_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID266_guard_support';N=R/'outputs/grid265_guarded_bias';Q=R/'outputs/grid260_bounded_basis_plan';CAL=R/'outputs/grid264_guarded_bias_plan/qualified_inputs';RAW=R/'outputs/grid249_context_bit_plan/outer_parent_public';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8')
def published(phase):
 h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for rel,v in rd(A/phase/'archive_manifest.json').items():assert hashlib.sha256(git('show',h+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==v
 return h

def prepare():
 assert not O.exists() and rd(N/'complete.json')['passed'];h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for name in ['matched261_changes.json','matched261_summary.json','complete.json','NEXT_PROTOCOL_PLAN.json']:
  ar=REPO/'research_archive/power-grid-2026/protocols/GRID265_guarded_bias/audited'/name;assert git('show',h+':'+ar.relative_to(REPO).as_posix())==(N/name).read_bytes()
 O.mkdir();save('design.json',dict(id='GRID266',scope='ONEfixedreadonly191cancellations;6profitable/185nonprofitable;0newfit/scalar/threshold/policy/outerreturn/NNforward.',description='Use published261oldscore and265delta/score,raw249parentb andfixedbounded40input. Decompose existing public/b/bit/intercept terms;no alteredpolicy.',guard='Original500own12positive scores onfixed261,role/scoredistribution andlimitingrecord equal to ALREADYfitteddelta. Descriptive minima/quantiles not candidate thresholds, no policydecisions at alternatives.',support='For each191record,nearest1ownfit8 perSTOP/CONT in bounded40 Euclidean space,exclude samephysicalgroup/self,firsttie. Publish all382 publicpairs/distances BEFORE ownfit y/NET/weight join;no distancelearning or k/thresholdsearch.',cost='Only265published191rowchanges,metrics cases;date/call/seed macrocontributions andtwo categories. Correlatedprefix/seeduses notindependent.',ancestry='Reuse235/253known own12parent/internalcal4 caveat;no broadancestryrerun/newOOF claim.',limits='No final8/physics/grid/paper/FRAME/install;originalfee/label/controller/order/alias/station/128/64/stagnation2/.9 unchanged. Branchremains73plus6historicalcontinuations8285updates and604.703s/678.156cap. Next hypotheses onlyowntraining/publicsupport,no outerselectedstrength or Jan5exception.'))
 paths=[W/'run.py',O/'design.json',N/'matched261_changes.json',N/'matched261_summary.json',N/'search_metrics.json',N/'NEXT_PROTOCOL_PLAN.json',META,*list((N/'models').glob('*.npz')),*list((N/'outer_predictions').glob('*.npz')),*list((Q/'fit_inputs').glob('*.npz')),*list(CAL.glob('*.npz')),*list(RAW.glob('*.npz'))]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID266_SOURCE_FROZEN')

def public():
 verify();h=published('source');assert not (O/'public_started.json').exists();save('public_started.json',dict(source_published=h));meta=rd(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};changes=rd(N/'matched261_changes.json');assert len(changes)==191;records=[];guard=[];cache={}
 for file in sorted((N/'models').glob('*.npz')):
  pool=file.stem;m=np.load(file);cal=np.load(CAL/f'{pool}_cal.npz');oldscore=cal['b']+float(m['alpha'][0])*((cal['b']+np.sum(cal['X']*m['theta'][:-1],axis=1)+m['theta'][-1])-cal['b']);pos=cal['y']>0;delta=float(m['delta'][0]);lim=np.flatnonzero(pos&(oldscore==delta));stats=lambda v:dict(min=float(v.min()),median=float(np.median(v)),max=float(v.max())) if len(v) else None
  guard.append(dict(pool=pool,KEEP=bool(m['scalar_KEEP'][0]),delta=delta,positive_count=int(pos.sum()),positive_score=stats(oldscore[pos]),fit8_positive_score=stats(oldscore[pos&cal['fit_role']]),cal4_positive_score=stats(oldscore[pos&~cal['fit_role']]),limiting_records=[dict(metadata_id=int(cal['ids'][i]),date=meta[int(cal['ids'][i])]['date'],call=meta[int(cal['ids'][i])]['physical_state_group'],fit8_role=bool(cal['fit_role'][i]),published_style_oldscore=float(oldscore[i])) for i in lim]))
 for r in changes:
  pool=f"f{r['fold']}_s{r['seed']}"
  if pool not in cache:cache[pool]=(np.load(Q/'fit_inputs'/f'{pool}.npz'),np.load(RAW/f'{pool}.npz'),np.load(N/'models'/f'{pool}.npz'),np.load(N/'outer_predictions'/f'{pool}.npz'))
  fit,raw,m,pred=cache[pool];mid=lookup[r['sample_id'],r['position']];j=int(np.flatnonzero(raw['ids']==mid)[0]);jj=int(np.flatnonzero(pred['ids']==mid)[0]);x=raw['X'][j].copy();x[:38]=np.tanh(x[:38]);b=float(raw['b'][j]);theta=m['theta'];alpha=float(m['alpha'][0]);delta=float(m['delta'][0]);oldscore=float(pred['source261_logit'][jj]);newscore=float(pred['logit'][jj]);terms=x*theta[:-1];assert newscore==oldscore-delta and newscore<0 and oldscore>=0 and delta>0
  assert oldscore==b+alpha*((b+np.sum(terms)+theta[-1])-b)
  neighbors=[]
  for stop in [True,False]:
   allowed=[i for i,idx in enumerate(fit['ids']) if bool(fit['domain_STOP'][i])==stop and int(idx)!=mid and meta[int(idx)]['physical_state_group']!=r['call']]
   if allowed:
    dist2=np.sum((fit['X'][allowed]-x)**2,axis=1);i=allowed[int(np.argmin(dist2))];idx=int(fit['ids'][i]);neighbors.append(dict(domain='STOP' if stop else 'CONT',fit_index=i,metadata_id=idx,date=meta[idx]['date'],call=meta[idx]['physical_state_group'],distance=float(np.sqrt(dist2.min())),candidate_count=len(allowed)))
   else:neighbors.append(dict(domain='STOP' if stop else 'CONT',missing=True,candidate_count=0))
  records.append(dict(pool=pool,fold=r['fold'],seed=r['seed'],sample_id=r['sample_id'],position=r['position'],metadata_id=mid,date=r['date'],call=r['call'],oldscore=oldscore,newscore=newscore,delta=delta,oldscore_to_delta=oldscore/delta,parent_b=b,bounded_correction=oldscore-b,public38_sum=float(terms[:38].sum()),parent_b_feature=float(terms[38]),STOP_feature=float(terms[39]),intercept=float(theta[-1]),alpha=alpha,neighbors=neighbors))
 save('public_focus_pairs.json',records);save('existing_guard_distribution.json',guard);save('public_access.json',dict(existing_focus_records191=True,public_neighborpairs382=True,ownfit_fields=['X','b','ids','domain_STOP'],ownfit_outcomefields_read=[],oldcal_y_used_only_to_describe_existing500guard=True,newfit=0,new_outerstrategy=0));save('public_freeze.json',{n:sha(O/n) for n in ['public_focus_pairs.json','existing_guard_distribution.json','public_access.json']});archive('public');print('GRID266_PUBLIC_PAIRS',len(records))

def analyze():
 verify();h=published('public');assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(publicpairs_published=h));public=rd(O/'public_focus_pairs.json');change={(r['seed'],r['sample_id'],r['position']):r for r in rd(N/'matched261_changes.json')};metrics=rd(N/'search_metrics.json');details=[];groups=collections.defaultdict(list)
 for r in public:
  r=dict(r);c=change[r['seed'],r['sample_id'],r['position']];fit=np.load(Q/'fit_inputs'/f"{r['pool']}.npz");neighbors=[]
  for n in r['neighbors']:
   n=dict(n)
   if not n.get('missing'):
    i=n['fit_index'];assert int(fit['ids'][i])==n['metadata_id'];n.update(y=int(fit['y'][i]),actual_NET=float(fit['actual_NET'][i]),risk_weight=float(fit['weight'][i]))
   neighbors.append(n)
  r['neighbors']=neighbors;r.update(actual_next_NET=c['actual_next_NET'],actual_next_gain=c['actual_next_gain'],actual_next_queries=c['actual_next_queries'],profitable=c['actual_next_NET']>0);nd={x['date']:x['cases'] for x in metrics[f"SHIFT_seed{r['seed']}"]['per_date']};den=3*16*nd[r['date']];r['family_macro_NET_delta']=c['NET_delta']/den;r['family_macro_query_delta']=c['query_delta']/den;details.append(r);groups['profitable' if r['profitable'] else 'nonprofitable'].append(r)
 stats=lambda v:dict(min=float(min(v)),median=float(np.median(v)),max=float(max(v)))
 summaries={};callrows=[]
 for name,rr in groups.items():
  bycall=collections.defaultdict(lambda:dict(rows=0,raw_queries_saved=0,macro_NET_delta=0.,macro_query_saved=0.))
  for r in rr:
   t=bycall[r['call']];t['rows']+=1;t['raw_queries_saved']+=r['actual_next_queries'];t['macro_NET_delta']+=r['family_macro_NET_delta'];t['macro_query_saved']-=r['family_macro_query_delta']
  denom=sum(v['macro_query_saved'] for v in bycall.values());order=sorted(bycall.values(),key=lambda x:x['macro_query_saved'],reverse=True)
  summaries[name]=dict(records=len(rr),calls=len(bycall),dates=len({r['date'] for r in rr}),raw_queries_saved=sum(r['actual_next_queries'] for r in rr),macro_NET_delta=sum(r['family_macro_NET_delta'] for r in rr),macro_query_delta=sum(r['family_macro_query_delta'] for r in rr),maxcall_query_share=order[0]['macro_query_saved']/denom,top5call_query_share=sum(x['macro_query_saved'] for x in order[:5])/denom,oldscore=stats([r['oldscore'] for r in rr]),score_to_delta=stats([r['oldscore_to_delta'] for r in rr]),parent_b=stats([r['parent_b'] for r in rr]),bounded_correction=stats([r['bounded_correction'] for r in rr]),nearest={d:dict(positive=sum(any(n['domain']==d and n.get('y')==1 for n in r['neighbors']) for r in rr),negative=sum(any(n['domain']==d and n.get('y')==0 for n in r['neighbors']) for r in rr),missing=sum(any(n['domain']==d and n.get('missing') for n in r['neighbors']) for r in rr)) for d in ['STOP','CONT']})
  callrows += [dict(category=name,call=k,**v) for k,v in bycall.items()]
 prior=rd(N/'matched261_summary.json');assert sum(len(v) for v in groups.values())==191
 assert abs(sum(v['macro_NET_delta'] for v in summaries.values())-prior['family_delta']['mean_net_utility'])<1e-12
 assert abs(sum(v['macro_query_delta'] for v in summaries.values())-prior['family_delta']['mean_queries'])<1e-12
 guards=rd(O/'existing_guard_distribution.json');lim=[dict(pool=g['pool'],**r) for g in guards for r in g['limiting_records'] if not g['KEEP']]
 summary=dict(passed=True,categories=summaries,guardlimiting_records=len(lim),guardlimiter_fit8=sum(r['fit8_role'] for r in lim),guardlimiter_cal4=sum(not r['fit8_role'] for r in lim),all500guardretained_is_empirical_not_independent=True,newfits=0,newscalars=0,newouterstrategy=0,newNNforward=0,branch73=True)
 for n,v in [('focus_support_details.json',details),('cost_by_call.json',callrows),('guard_limiting_records.json',lim),('summary.json',summary)]:save(n,v)
 archive('analyzed');print('GRID266_ANALYZED',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
