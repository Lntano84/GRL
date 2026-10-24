"""Fixed one-forest global-compression study; no environment imports or queries."""
from pathlib import Path
import collections,datetime,functools,gzip,hashlib,importlib.util,json,math,os,pickle,shutil,statistics,sys,time
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid140_global_compression';SOURCE=R/'outputs/grid139_global_feasibility'
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID140_global_compression'
spec=importlib.util.spec_from_file_location('grid140_saved_reference',R/'work/grid139_global_feasibility/run.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def gz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]
@functools.lru_cache(maxsize=None)
def call_at(source):return read(R/source)
@functools.lru_cache(maxsize=1)
def subids():return read(R/'outputs/grid28/offline.json')['public_subids']
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    P.verify()
def verify_trained():
    verify()
    for n,h in read(O/'trained_freeze.json').items():assert sha(O/n)==h,n
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
def eligible_data(part):
    # Filtering uses partition metadata before selecting model inputs/labels.
    task=[r for r in gz(SOURCE/'compression_task_targets.jsonl.gz') if r['partition']==part]
    ids={r['sample_id'] for r in task};inputs={r['sample_id']:r for r in gz(SOURCE/'public_inputs.jsonl.gz') if r['sample_id'] in ids}
    meta={r['sample_id']:r for r in gz(SOURCE/'metadata.jsonl.gz') if r['sample_id'] in ids}
    return [dict(meta=meta[r['sample_id']],features=inputs[r['sample_id']]['features'],target=r) for r in task]
def transform(rows,names,medians):
    raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in names] for r in rows],float)
    missing=np.isnan(raw);out=np.where(missing,np.asarray(medians)[None,:],raw)
    return np.concatenate([out,missing.astype(float)],axis=1)
def export_model(est):
    return dict(kind='RandomForestClassifier',classes=est.classes_.tolist(),trees=[dict(left=t.tree_.children_left.tolist(),right=t.tree_.children_right.tolist(),feature=t.tree_.feature.tolist(),threshold=t.tree_.threshold.tolist(),value=t.tree_.value[:,0,:].tolist()) for t in est.estimators_])
def exported_scores(model,xx):
    xx=np.asarray(xx,np.float32);p=np.zeros((len(xx),len(model['classes'])))
    for t in model['trees']:
        for j,x in enumerate(xx):
            n=0
            while t['left'][n]!=-1:n=t['left'][n] if x[t['feature'][n]]<=t['threshold'][n] else t['right'][n]
            v=np.asarray(t['value'][n],float);p[j]+=v/v.sum()
    p/=len(model['trees']);return p[:,model['classes'].index(1)]
def probability_metrics(rows,scores,threshold):
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,balanced_accuracy_score
    y=np.asarray([r['target']['stop_label'] for r in rows],int);p=np.asarray(scores,float);pred=p>threshold
    return dict(n=len(y),positive=int(y.sum()),negative=int((1-y).sum()),threshold=float(threshold),accuracy=float(np.mean(pred==y)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),
        stop_proposals=int(pred.sum()),wrong_stop_labels=int(np.sum(pred&(y==0))),missed_stop_labels=int(np.sum((~pred)&(y==1))),
        majority_stop_accuracy=float(y.mean()),auxiliary_only=True)
def replay(ref,mode,threshold=None,scores=None):
    c=call_at(ref['source']);order=c['order'];m=len(order);subs=subids();truth={r['pool_id']:r for r in c['outcomes']}
    n=min(128,m);initial=P.best(c['outcomes'][:n]);prefix_safe=initial is not None and initial['rho']<.9;b=initial;stale=0;visited=[];stop_kind='exhaustion';stopped_score=None
    while n<m:
        force=(not prefix_safe and b is not None and b['rho']<.9) or (b is not None and n>=256 and stale>=2)
        if force:stop_kind='original_forced_stop';break
        if b is not None and mode.startswith('CAP'):
            if n>=int(mode[3:]):stop_kind='simple_early_stop';break
        if b is not None and mode=='MODEL':
            assert scores is not None and threshold is not None;k=(ref['call_id'],n);assert k in scores,k;score=scores[k]
            visited.append(dict(visible_count=n,score=score,stop=score>threshold))
            if score>threshold:stop_kind='learned_early_stop';stopped_score=score;break
        before=b;n=min(n+64,m);b=P.best(c['outcomes'][:n]);stale=0 if P.improves(before,b) else stale+1
    pre=b;extra=[]
    if b is not None and n<m and (not prefix_safe or P.bid(b)!=P.bid(initial)):
        station=subs[b['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if subs[int(alias)]==station};extra=[i for i in order[n:] if i in ids];b=P.best(c['outcomes'][:n]+[truth[i] for i in extra])
    paid=order[:n]+extra;assert len(paid)==len(set(paid));original_q=ref['global_n']+ref['local_queries'];q=len(paid)
    result=dict(call_id=ref['call_id'],partition=ref['partition'],date=ref['date'],scenario=ref['scenario'],global_queries=n,local_queries=len(extra),queries=q,original_queries=original_q,
        selected=P.bid(b),original_selected=ref['frame_choice'],action_match=P.bid(b)==ref['frame_choice'],stop_kind=stop_kind,stopped_score=stopped_score,model_decisions=visited,model_decision_count=len(visited),
        bad_learned_stop=stop_kind=='learned_early_stop' and (P.bid(b)!=ref['frame_choice'] or q>=original_q))
    if mode=='FRAME':assert result['action_match'] and q==original_q
    return result
def refs(partition=None,dates=None):
    return [r for r in read(SOURCE/'call_ledger.json') if (partition is None or r['partition']==partition) and (dates is None or r['date'] in dates)]
def summarize(rows):
    assert rows;sc=[]
    for scenario in sorted({r['scenario'] for r in rows}):
        rr=[r for r in rows if r['scenario']==scenario];sc.append(dict(scenario=scenario,date=rr[0]['date'],calls=len(rr),queries=sum(r['queries'] for r in rr),
            original_queries=sum(r['original_queries'] for r in rr),query_reduction=1-sum(r['queries'] for r in rr)/sum(r['original_queries'] for r in rr),action_differences=sum(not r['action_match'] for r in rr)))
    dd=[]
    for date in sorted({r['date'] for r in sc}):
        rr=[r for r in sc if r['date']==date];dd.append(dict(date=date,query_reduction=statistics.mean(r['query_reduction'] for r in rr),action_differences=sum(r['action_differences'] for r in rr),queries=sum(r['queries'] for r in rr),original_queries=sum(r['original_queries'] for r in rr)))
    return dict(calls=len(rows),scenarios=len(sc),dates=len(dd),queries=sum(r['queries'] for r in rows),original_queries=sum(r['original_queries'] for r in rows),
        action_differences=sum(not r['action_match'] for r in rows),action_matches=sum(r['action_match'] for r in rows),model_decisions=sum(r['model_decision_count'] for r in rows),
        model_stops=sum(r['stop_kind']=='learned_early_stop' for r in rows),query_increase_calls=sum(r['queries']>r['original_queries'] for r in rows),
        date_macro_query_reduction=statistics.mean(r['query_reduction'] for r in dd),by_date=dd,by_scenario=sc)
