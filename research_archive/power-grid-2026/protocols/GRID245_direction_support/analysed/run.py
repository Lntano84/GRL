"""Describe all frozen linear contributions, never construct a new policy."""
from pathlib import Path
import ast,hashlib,json,shutil,subprocess,sys,collections
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid245_direction_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID245_direction_support';Q=R/'outputs/grid242_parent_prior_inputs';P=R/'outputs/grid243_parent_coherent'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
 for folder,name in [(Q,'qualified_fit_freeze.json'),(Q,'generated_freeze.json'),(P,'trained_freeze.json'),(P,'scored_freeze.json')]:
  for p,h in read(folder/name).items():assert sha(folder/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
def prepare():
 assert not O.exists() and read(R/'outputs/grid244_prior_distribution/complete.json')['passed'];O.mkdir()
 tree=ast.parse((R/'work/grid101_gain_learning/features.py').read_text(encoding='utf8'));names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='NAMES' for x in n.targets));names+=['local_batches_done','local_budget_used_fraction','log_next_batch_queries','locked_remaining_fraction','parent_b12_residual','intercept'];assert len(names)==43;save('feature_names.json',names)
 save('design.json',dict(id='GRID245',scope='Read-only all43 frozen terms for all610 parentCONT decisions, allfit8 inputs and85 selectedempiricalpositiveconstraints; no fitting/scalar/headforward/newpolicy scores.',
 decomposition='Nativeprior b12 + alpha*(sum X42*theta42 + intercept), reproduce savedpublished score with original floating grouping. All41 publicterms,parentmargin residual and intercept included, nofeatureselection. Contributions describe the existinglinearcorrector only.',
 support='Perpool training min/max perfeature and protectedpositive min/max; flag strict range departures fordescription only, nothreshold/rule fit. All4outcome categories retained. No newalphamin/maxor thirdpolicy.',
 limits='Heldlabels oldexposed development; notcausalfeatureablation norproofNNfailure. No callspecificexception, featuredrop fromheld, seedselection, newdata/final8/physics/paper/FRAME/dependencies. Fixedcontrols/fees/labels/candidates unchanged;230main and243seed1tradeoff retained.'))
 paths=[W/'run.py',O/'design.json',O/'feature_names.json',R/'work/grid101_gain_learning/features.py',R/'outputs/grid113_relational_learning/train_metadata.json',P/'gate_predictions.json',P/'complete.json',P/'fit_receipts.json',R/'outputs/grid244_prior_distribution/protected_positive_rows.json']
 for folder in ['models','outer_predictions']:paths+=list((P/folder).glob('*.npz'))
 for folder in ['fit_inputs','calibration_inputs','outer_parent_public']:paths+=list((Q/folder).glob('*.npz'))
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID245_QUALIFIED')
def stats(x):
 x=np.asarray(x)
 return dict(n=len(x),min=float(x.min()),median=float(np.median(x)),max=float(x.max()),mean=float(x.mean())) if len(x) else dict(n=0)
def run():
 verify();assert not (O/'analysis_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'qualified/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();save('analysis_started.json',dict(git_sha=head,newfits=0,newpolicy_scores=0));names=read(O/'feature_names.json');meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');idx={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};gp=read(P/'gate_predictions.json');allrows=[];training=[];constraints=[];matrices=[];extreme=[];pool_summaries=[]
 for file in sorted((P/'models').glob('*.npz')):
  pool=file.stem;f,s=[int(x[1:]) for x in pool.split('_')];p=np.load(file);theta=p['theta'];alpha=float(p['alpha'][0]);fit=np.load(Q/'fit_inputs'/file.name);cal=np.load(Q/'calibration_inputs'/file.name);h=np.load(Q/'outer_parent_public'/file.name);pub=np.load(P/'outer_predictions'/file.name);hlookup={int(i):j for j,i in enumerate(h['ids'])};positive=cal['y']>0
  lo,hi=fit['X'].min(0),fit['X'].max(0);plo=cal['X'][positive].min(0) if positive.any() else None;phi=cal['X'][positive].max(0) if positive.any() else None
  for z,container,label in [(fit,training,'fit'),(cal,constraints,'constraints')]:
   mask=np.ones(len(z['ids']),dtype=bool) if label=='fit' else z['y']>0;X=z['X'][mask];contrib=np.c_[X*theta[:-1],np.full(len(X),theta[-1])];direction=np.sum(X*theta[:-1],axis=1)+theta[-1]
   for k,i in enumerate(z['ids'][mask]):container.append(dict(pool=pool,id=int(i),positive=bool(z['y'][mask][k]),direction=float(direction[k]),b=float(z['b'][mask][k]),weighted_loss_mass=float(z['weight'][mask][k]) if label=='fit' else None,largest_negative_feature=names[int(np.argmin(contrib[k]))],contributions=contrib[k].tolist()))
  rr=[]
  for r in gp:
   if r['fold']!=f or r['seed']!=s or not r['base_continue']:continue
   i=idx[r['sample_id'],r['position']];j=hlookup[i];x=h['X'][j];terms=np.r_[x*theta[:-1],theta[-1]];end=float(h['b'][j])+float(np.sum(x*theta[:-1]))+float(theta[-1]);score=float(h['b'][j])+alpha*(end-float(h['b'][j]));assert abs(score-float(pub['logit'][j]))<1e-12 and bool(pub['continue_'][j])==r['continue_'];kind=('profitable' if r['actual_next_NET']>0 else 'nonprofitable')+('_keep' if r['continue_'] else '_cancel');out=(x<lo)|(x>hi);pout=((x<plo)|(x>phi)) if plo is not None else np.ones(42,bool);neg=sorted(range(43),key=lambda k:float(terms[k]));row=dict(r,pool=pool,id=i,kind=kind,b12=float(h['b'][j]),endpoint=end,direction=float(end-h['b'][j]),score=score,alpha=alpha,contributions=terms.tolist(),outside_fit_features=[names[k] for k in np.flatnonzero(out)],outside_protected_features=[names[k] for k in np.flatnonzero(pout)],top_negative=[dict(feature=names[k],contribution=float(terms[k]),value=float(x[k]) if k<42 else 1.,fit_min=float(lo[k]) if k<42 else 1.,fit_max=float(hi[k]) if k<42 else 1.,outside_fit=bool(out[k]) if k<42 else False) for k in neg[:5]],largest_negative_feature=names[neg[0]]);rr.append(row);allrows.append(row);matrices.append(terms)
  pool_summaries.append(dict(pool=pool,alpha=alpha,training_rows=len(fit['ids']),protected_positive=int(positive.sum()),held_parentCONT=len(rr),missed=sum(r['kind']=='profitable_cancel' for r in rr),miss_any_fit_range_departure=sum(r['kind']=='profitable_cancel' and bool(r['outside_fit_features']) for r in rr)))
 assert len(allrows)==610 and len(training)==2851 and len(constraints)==85;categories=[];feature_tables=[]
 for kind in ['profitable_keep','profitable_cancel','nonprofitable_cancel','nonprofitable_keep']:
  rr=[r for r in allrows if r['kind']==kind];ct=np.array([r['contributions'] for r in rr]);categories.append(dict(kind=kind,rows=len(rr),direction=stats([r['direction'] for r in rr]),any_fit_range_departure=sum(bool(r['outside_fit_features']) for r in rr),any_positive_range_departure=sum(bool(r['outside_protected_features']) for r in rr),largest_negative_features=dict(collections.Counter(r['largest_negative_feature'] for r in rr))))
  for j,name in enumerate(names):feature_tables.append(dict(kind=kind,index=j,feature=name,contribution=stats(ct[:,j]),negative_count=int((ct[:,j]<0).sum()),outside_fit_count=sum(name in r['outside_fit_features'] for r in rr),outside_protected_count=sum(name in r['outside_protected_features'] for r in rr)))
 # Every term retained; rank below only explains existing decisions, never chooses a feature subset.
 save('held_contributions.json',allrows);save('fit_contributions.json',training);save('protected_contributions.json',constraints);save('all_feature_summary.json',feature_tables);save('category_summary.json',categories);save('pool_summary.json',pool_summaries)
 miss=[r for r in allrows if r['kind']=='profitable_cancel'];jan=[r for r in miss if r['seed']==2 and r['call']=='2037-01-05_101|call001'];assert len(miss)==22 and len(jan)==5
 save('summary.json',dict(parentCONT610=len(allrows),all43terms_included=True,fit8rows=len(training),protected85=len(constraints),missed22=len(miss),jan5_related_misses=len(jan),newfits=0,newpolicyscores=0,categories=categories));verify();save('review.json',dict(passed=True,all610savedpublishedscoresreproduced=True,all43termsretained=True,all2851fit_inputs_and85positiveconstraints=True,sources_andmodels_unchanged=True,allseeds_counterexamples_retained=True,no_newfit_forecast_policy=True));save('complete.json',dict(passed=True,analysis_only=True));archive('analysed');print('GRID245_ANALYSED',json.dumps(dict(categories=categories,jan5=[dict(pool=r['pool'],call=r['call'],terms=r['top_negative'],outside=r['outside_fit_features']) for r in jan]),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
