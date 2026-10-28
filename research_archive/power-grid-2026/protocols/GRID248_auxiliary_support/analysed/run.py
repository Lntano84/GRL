"""Fixed own-fit8 cross-domain public support, then separate label diagnosis."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid248_auxiliary_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID248_auxiliary_support';Q=R/'outputs/grid246_canonical_plan';S=R/'outputs/grid247_canonical_residual';PLAN=R/'outputs/grid241_parent_prior_plan';META=R/'outputs/grid113_relational_learning/train_metadata.json';ACCESS=[]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def fields(p,names):
 z=np.load(p);out={n:z[n].copy() for n in names};ACCESS.append(dict(path=p.relative_to(R).as_posix(),fields=names));return out

def verify():
 for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
def prepare():
 assert not O.exists() and read(S/'complete.json')['passed'];O.mkdir()
 save('design.json',dict(id='GRID248',fixed_priorplan='247NEXT_PROTOCOL_PLAN',scope='Only ownfit8 canonical39 publicgeometry/STOPCONTtraininglabels andoriginalrisk; zerooptimizer/model/scalar/heldprediction/newpolicy/physical. No calibrationlabels.',
 metric='Euclidean on SAME24739standardizedinputs,no fittedmetric/ksearch. For EVERYrow nearest1same-domain and1cross-domain in SAMEpool, exclude self ANDsamephysical_state_group. Tiesfirstoriginalrowindex. Missingcandidate staysmissing.',
 barriers='Codeinputfirstpublish. Exportfullpublicvectors/ids/domain/groups/rolesandallneighborpairs/distanceswithoutlabels. Publishallpublicpairsbefore originalownfit8labels/riskjoin.',
 summaries='All12pools inclfixedKEEP, all3seeds, STOPCONT andpositive/nonpositive conditional; rawrecordrates andoriginalriskweightedrates separate. Active11macro separatefromKEEP. Withinzero distances andtiesexplicit. Pairgeometry notlearnedselector/predictiveperformance/causaltransferproof.',
 unchanged='Allrows/labels/weights/scalers/fees/seeds/dates/controllers/order/aliases/station128/64/stagnation2/.9. No feature/riskratio/threshold/alpha changes, no call exceptions. No final8/newdata/physics/paper/FRAME/install.'))
 paths=[W/'run.py',O/'design.json',META,S/'NEXT_PROTOCOL_PLAN.md',S/'complete.json',Q/'qualified_freeze.json',PLAN/'qualified_pools.json']+list((Q/'fit_inputs').glob('*.npz'));save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID248_SOURCE_FROZEN')
def generate():
 verify();assert not (O/'generation_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();save('generation_started.json',dict(sourcepublished=head,labels_read=False,newfits=0));meta=read(META);rows=[];vectors=[];poolcounts=[]
 for r in read(PLAN/'qualified_pools.json'):
  pool=r['pool'];z=fields(Q/'fit_inputs'/f'{pool}.npz',['X','ids','domain_STOP','fallback_KEEP']);x,ids,stop=z['X'],z['ids'],z['domain_STOP'];assert x.shape==(len(ids),39);dates={meta[int(i)]['date'] for i in ids};assert dates<=set(r['fit_dates']) and not dates&set(r['outer_dates']);groups=np.array([meta[int(i)]['physical_state_group'] for i in ids]);rr=[]
  for j,i in enumerate(ids):
   dist=np.sum((x-x[j])**2,axis=1);diffgroup=groups!=groups[j];neighbors={}
   for name,mask in [('within',diffgroup&(stop==stop[j])),('cross',diffgroup&(stop!=stop[j]))]:
    choices=np.flatnonzero(mask)
    if len(choices):
     k=int(choices[np.argmin(dist[choices])]);neighbors[name]=dict(index=k,id=int(ids[k]),call=str(groups[k]),squared_distance=float(dist[k]),distance=float(np.sqrt(dist[k])),tie_count=int((dist[choices]==dist[k]).sum()))
    else:neighbors[name]=None
   m=meta[int(i)];row=dict(pool=pool,index=j,id=int(i),sample_id=m['sample_id'],date=m['date'],call=str(groups[j]),domain='STOP' if stop[j] else 'CONT',fallback_KEEP=bool(z['fallback_KEEP'][0]),neighbors=neighbors);rr.append(row);rows.append(row);vectors.append(dict(pool=pool,index=j,id=int(i),X=x[j].tolist()))
  poolcounts.append(dict(pool=pool,rows=len(rr),STOP=int(stop.sum()),CONT=int((~stop).sum()),unique_calls=len(set(groups)),missing_within=sum(r['neighbors']['within'] is None for r in rr),missing_cross=sum(r['neighbors']['cross'] is None for r in rr)))
 assert len(rows)==2851 and len(poolcounts)==12 and not any({'y','weight','actual_NET'}&set(a['fields']) for a in ACCESS);save('public_rows.json',rows);save('public_vectors.json',vectors);save('public_pool_counts.json',poolcounts);save('public_access.json',ACCESS);save('public_freeze.json',{p.name:sha(p) for p in [O/'public_rows.json',O/'public_vectors.json',O/'public_pool_counts.json',O/'public_access.json']});verify();archive('public');print('GRID248_PUBLIC_PAIRS',len(rows),sum(r['missing_cross'] for r in poolcounts),sum(r['missing_within'] for r in poolcounts))
def public():
 verify()
 for p,h in read(O/'public_freeze.json').items():assert sha(O/p)==h,p

def publication():
 public();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'public/public_freeze.json').relative_to(REPO).as_posix())==(O/'public_freeze.json').read_bytes();save('pre_label_publication.json',dict(passed=True,git_sha=head,all2851publicpairs_vectorspublished=True));print('GRID248_PUBLIC_PUBLISHED')
def stat(v):
 v=np.array(v,dtype=float);return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),q10=float(np.quantile(v,.1)),q90=float(np.quantile(v,.9))) if len(v) else dict(n=0)
def describe(rr):
 result=dict(rows=len(rr),positive=sum(r['y'] for r in rr),unique_calls=len({r['call'] for r in rr}),risk_mass=float(sum(r['weight'] for r in rr)))
 for kind in ['within','cross']:
  avail=[r for r in rr if r[kind] is not None];mass=sum(r['weight'] for r in avail);bad=[r for r in avail if r[kind]['y']!=r['y']];result[kind]=dict(available=len(avail),missing=len(rr)-len(avail),label_disagreements=len(bad),raw_disagreement_rate=len(bad)/len(avail) if avail else None,riskweighted_disagreement=sum(r['weight'] for r in bad)/mass if mass else None,neighbor_positive=sum(r[kind]['y'] for r in avail),distance=stat([r[kind]['distance'] for r in avail]),zero_distance=sum(r[kind]['distance']==0 for r in avail))
 both=[r for r in rr if r['within'] is not None and r['cross'] is not None];result['both_available']=len(both);result['cross_strictly_closer']=sum(r['cross']['distance']<r['within']['distance'] for r in both);ratios=[r['cross']['distance']/r['within']['distance'] for r in both if r['within']['distance']>0];result['distance_ratio_cross_over_within']=stat(ratios);return result

def join():
 public();assert read(O/'pre_label_publication.json')['passed'] and not (O/'label_join_started.json').exists();save('label_join_started.json',dict(publicpairsfirst=True));rows=read(O/'public_rows.json');lookup={}
 for file in sorted((Q/'fit_inputs').glob('*.npz')):
  z=fields(file,['ids','y','weight','actual_NET','domain_STOP']);pool=file.stem
  for j,i in enumerate(z['ids']):lookup[pool,int(i)]=dict(y=bool(z['y'][j]),weight=float(z['weight'][j]),NET=float(z['actual_NET'][j]),domain_STOP=bool(z['domain_STOP'][j]))
 joined=[]
 for r in rows:
  y=lookup[r['pool'],r['id']];assert y['domain_STOP']==(r['domain']=='STOP');out={k:v for k,v in r.items() if k!='neighbors'};out.update(y)
  for kind in ['within','cross']:
   n=r['neighbors'][kind]
   if n is None:out[kind]=None
   else:
    q=lookup[r['pool'],n['id']];assert n['call']!=r['call'];assert (q['domain_STOP']==y['domain_STOP'])==(kind=='within');out[kind]=dict(n,y=q['y'],NET=q['NET'])
  joined.append(out)
 pooled=[];perpool=[]
 for domain in ['STOP','CONT']:
  for label in ['ALL','POSITIVE','NONPOSITIVE']:
   z=[r for r in joined if r['domain']==domain and (label=='ALL' or r['y']==(label=='POSITIVE'))];pooled.append(dict(domain=domain,label=label,**describe(z)))
   for pool in sorted({r['pool'] for r in joined}):perpool.append(dict(pool=pool,domain=domain,label=label,**describe([r for r in z if r['pool']==pool])))
 macro=[]
 for domain in ['STOP','CONT']:
  z=[r for r in perpool if r['domain']==domain and r['label']=='ALL' and r['pool']!='f0_s1'];macro.append(dict(domain=domain,active_pools=len(z),within_poolmean_risk_disagreement=float(np.mean([r['within']['riskweighted_disagreement'] for r in z])),cross_poolmean_risk_disagreement=float(np.mean([r['cross']['riskweighted_disagreement'] for r in z]))))
 save('labelled_pair_diagnosis.json',joined);save('pooled_summary.json',pooled);save('pool_summary.json',perpool);save('active11_macro.json',macro);save('label_access.json',ACCESS);public();save('review.json',dict(passed=True,all2851fixed_publicpairs_joined_toownfit8_only=True,all12pools_seeds_missing_counterexamples_retained=True,no_newfit_model_policy=True,original_sources_unchanged=True));save('complete.json',dict(passed=True,read_only=True,newfits=0,newforecasts=0,newheldscores=0,source_rows=2851,pools=12));archive('analysed');print('GRID248_DIAGNOSED',json.dumps(dict(pooled=pooled,active11macro=macro),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
