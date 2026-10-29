"""GRID263 fixed own-fit8 risk, logistic-loss and query-cost description. No optimizer."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,collections
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid263_training_residual';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID263_training_residual';Q=R/'outputs/grid260_bounded_basis_plan';M=R/'outputs/grid261_bounded_basis/models';META=R/'outputs/grid113_relational_learning/train_metadata.json';PLAN=R/'outputs/grid262_cancellation_support/NEXT_PROTOCOL_PLAN.json'
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

def prepare():
 assert not O.exists() and rd(PLAN)['id']=='GRID263'
 head=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 for folder,apath in [(Q/'fit_inputs',REPO/'research_archive/power-grid-2026/protocols/GRID260_bounded_basis_plan/qualified/fit_inputs'),(M,REPO/'research_archive/power-grid-2026/protocols/GRID261_bounded_basis/trained/models')]:
  assert len(list(folder.glob('*.npz')))==12
  for file in folder.glob('*.npz'):assert git('show',head+':'+(apath/file.name).relative_to(REPO).as_posix())==file.read_bytes()
 O.mkdir();save('design.json',dict(id='GRID263',scope='ONEfixedownfit8descriptivequalification;all12poolsincludingKEEP,11active summarizedseparately;nooptimizer/newalpha/newouterpolicyscores.',inputs='260fit_inputs only X,b,ids,domain_STOP,y,actual_NET,weight,source_mass;261fixedmodels theta/alpha/fallback_KEEP;113metadata next_queries/date/physical_state_group;262fixednextplan. No outer public/prediction/outcome files.',score='Existing deployedzero threshold:score=b+alpha*((b+sum(X*theta)+intercept)-b),b>0selectedalready;no thresholdscan. Separate objective_z= X@theta+intercept+b for original219meanNLL+full41L2.',groups='Eachpool nativeSTOP/CONT x positive/negative x fixedcontinue/stop,allrows retained;counts,knownnextqueries,source_mass*queries,rawrisk/totalrisk,weightedNLL. WrongnegativeCONT =nonprofitableyetcontinue;wrongpositiveSTOP separate.',concentration='AllfalseCONTnegative rows groupedphysicalcall/date. Rawrowcostandwithinpool risk/loss separate;ownpoolmacromeansunweightedacross11active;duplicate prefixes/seeds/uses notindependent, no new efficacy result.',gradient='Existingtheta only. Eachdomain/label weightedlogisticgradient normalized totaloriginalweight,fullunitL2 theta;sum-gradient check descriptive,not newfit or candidateupdate.',constraints='Originalfee/labels/risks/features/scales/dates/seeds/controllers/order/aliases/station/128/64/stagnation2/.9 unchanged. No final8/newphysics/grid/paper/FRAME/install.',following='Evidencefor nextuniqueboundedhypothesis qualification only;do not immediatelyreweight,addnetwork,scanpriorstrength/threshold/alpha or choosebyexposedouter.',accounting='0fits/alpha/NNheadforward,branch remains62plus6historicalcontinuations8285updates and604.703s/678.156cap.'))
 paths=[W/'run.py',O/'design.json',PLAN,META,*sorted((Q/'fit_inputs').glob('*.npz')),*sorted(M.glob('*.npz'))];save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('input_publication.json',dict(passed=True,git_sha=head,all12ownfitfiles_andfixedmodels_specificblobs_verified=True,noouterinputsread=True));archive('source');print('GRID263_SOURCE_FROZEN')

def analyze():
 verify();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes()
 assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(source_published=head,ownfitonly=True))
 meta=rd(META);allrows=[];pools=[]
 for file in sorted((Q/'fit_inputs').glob('*.npz')):
  z=np.load(file);m=np.load(M/file.name);x=z['X'];b=z['b'];theta=m['theta'];alpha=float(m['alpha'][0]);keep=bool(m['fallback_KEEP'][0]);y=z['y'];w=z['weight'];den=float(w.sum());assert x.shape[1]==40 and theta.shape==(41,) and (b>0).all() and den>0
  end=b+np.sum(x*theta[:-1],axis=1)+theta[-1];score=b+alpha*(end-b);cont=score>=0
  objective_z=x@theta[:-1]+theta[-1]+b;nll=np.logaddexp(0,objective_z)-y*objective_z;loss=w*nll/den;prob=np.exp(-np.logaddexp(0,-objective_z));aug=np.c_[x,np.ones(len(x))];l2=float(.5*(theta@theta));gradient=aug.T@(w*(prob-y)/den)+theta
  groups=[];gradients=[]
  for stop in [True,False]:
   for positive in [True,False]:
    base=(z['domain_STOP']==stop)&(y==int(positive));gg=aug[base].T@(w[base]*(prob[base]-y[base])/den)
    gradients.append(dict(domain='STOP' if stop else 'CONT',label=int(positive),vector=gg.tolist(),norm=float(np.linalg.norm(gg))))
    for decision in [True,False]:
     ids=np.flatnonzero(base&(cont==decision));qs=[int(meta[int(z['ids'][i])]['next_queries']) for i in ids];mass=z['source_mass'][ids];callset={meta[int(z['ids'][i])]['physical_state_group'] for i in ids};ds={meta[int(z['ids'][i])]['date'] for i in ids}
     groups.append(dict(domain='STOP' if stop else 'CONT',label=int(positive),continue_=decision,rows=len(ids),raw_queries=sum(qs),source_mass_queries=float(mass@np.array(qs)),risk_sum=float(w[ids].sum()),risk_fraction_total=float(w[ids].sum()/den),weighted_NLL=float(loss[ids].sum()),unweighted_NLL=float(nll[ids].sum()),calls=len(callset),dates=len(ds)))
  assert abs(sum(g['weighted_NLL'] for g in groups)-float(loss.sum()))<1e-12
  assert np.max(abs(np.sum([g['vector'] for g in gradients],axis=0)+theta-gradient))<1e-12
  for i,mid in enumerate(z['ids']):
   mt=meta[int(mid)];allrows.append(dict(pool=file.stem,KEEP=keep,metadata_id=int(mid),sample_id=mt['sample_id'],position=mt['position'],date=mt['date'],call=mt['physical_state_group'],domain='STOP' if z['domain_STOP'][i] else 'CONT',label=int(y[i]),actual_NET=float(z['actual_NET'][i]),continue_=bool(cont[i]),score=float(score[i]),objective_z=float(objective_z[i]),next_queries=int(mt['next_queries']),source_mass=float(z['source_mass'][i]),risk_weight=float(w[i]),risk_fraction_total=float(w[i]/den),weighted_NLL=float(loss[i]),unweighted_NLL=float(nll[i])))
  pools.append(dict(pool=file.stem,KEEP=keep,rows=len(y),groups=groups,original_mean_NLL=float(loss.sum()),full_unit_L2=l2,objective=float(loss.sum()+l2),gradient_inf=float(abs(gradient).max()),gradients=gradients,L2_gradient=theta.tolist(),max_score_objectivez_difference=float(abs(score-objective_z).max()) if not keep else None))
 active=[r for r in allrows if not r['KEEP']];wrongneg=[r for r in active if r['label']==0 and r['continue_']];wrongpos=[r for r in active if r['label']==1 and not r['continue_']]
 def groupcost(records,key):
  out=collections.defaultdict(lambda:dict(rows=0,raw_queries=0,risk_fraction_pool_sum=0.,weighted_NLL_pool_sum=0.,source_mass_query_pool_sum=0.))
  for r in records:
   t=out[r[key]];t['rows']+=1;t['raw_queries']+=r['next_queries'];t['risk_fraction_pool_sum']+=r['risk_fraction_total'];t['weighted_NLL_pool_sum']+=r['weighted_NLL'];t['source_mass_query_pool_sum']+=r['source_mass']*r['next_queries']
  return out
 stats=lambda v:dict(min=float(min(v)),mean=float(np.mean(v)),max=float(max(v)))
 domain={}
 for d in ['STOP','CONT']:
  rr=[r for r in active if r['domain']==d];neg=[r for r in rr if r['label']==0];bad=[r for r in neg if r['continue_']];pos=[r for r in rr if r['label']==1];fn=[r for r in pos if not r['continue_']]
  domain[d]=dict(rows=len(rr),positive=len(pos),negative=len(neg),falseCONT_negative=len(bad),falseSTOP_positive=len(fn),negative_raw_queries=sum(r['next_queries'] for r in neg),falseCONT_raw_queries=sum(r['next_queries'] for r in bad),falseCONT_query_fraction_of_negative=sum(r['next_queries'] for r in bad)/sum(r['next_queries'] for r in neg),falseCONT_negative_risk_fraction_of_negative=sum(r['risk_fraction_total'] for r in bad)/sum(r['risk_fraction_total'] for r in neg),falseCONT_meanpool_risk=sum(r['risk_fraction_total'] for r in bad)/11,falseCONT_meanpool_weightedNLL=sum(r['weighted_NLL'] for r in bad)/11,all_meanpool_weightedNLL=sum(r['weighted_NLL'] for r in rr)/11,wrongneg_calls=len({r['call'] for r in bad}),wrongneg_dates=len({r['date'] for r in bad}))
 calls=groupcost(wrongneg,'call');dates=groupcost(wrongneg,'date');callq=sorted((v['raw_queries'] for v in calls.values()),reverse=True);allq=sum(callq)
 summary=dict(passed=True,all12_rows=len(allrows),active11_rows=len(active),KEEP_rows=len(allrows)-len(active),active_positive=sum(r['label'] for r in active),falseCONT_negative=len(wrongneg),falseSTOP_positive=len(wrongpos),domain=domain,wrongneg_raw_queries=allq,wrongneg_calls=len(calls),wrongneg_dates=len(dates),topcall_rawqueryshare=callq[0]/allq,top5calls_rawqueryshare=sum(callq[:5])/allq,active_mean_NLL=float(np.mean([p['original_mean_NLL'] for p in pools if not p['KEEP']])),active_mean_L2=float(np.mean([p['full_unit_L2'] for p in pools if not p['KEEP']])),active_gradientinf=stats([p['gradient_inf'] for p in pools if not p['KEEP']]),new_models=0,new_scalars=0,new_outerpolicy_scores=0,NNheadforward=0,onlyownfit8_diagnostic=True,not_independent_records_or_savings_forecast=True)
 for name,val in [('training_row_diagnostics.json',allrows),('pool_risk_loss_gradient.json',pools),('negative_falseCONT.json',wrongneg),('positive_falseSTOP.json',wrongpos),('negative_falseCONT_by_call.json',calls),('negative_falseCONT_by_date.json',dates),('summary.json',summary)]:save(name,val)
 archive('analyzed');print('GRID263_ANALYZED',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
