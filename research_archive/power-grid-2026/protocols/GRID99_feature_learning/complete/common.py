"""GRID99: fixed offline trajectory-effect prediction, never a deployed gate."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
W=Path(__file__).parent
OUT=ROOT/'outputs/grid99_feature_learning'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
for k in ['HF_HOME','HUGGINGFACE_HUB_CACHE','TABPFN_MODEL_CACHE_DIR','XDG_CACHE_HOME']:
    os.environ[k]=str(ROOT/'tmp/grid99'/k)
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_OFFLINE']='1'
import json,hashlib,time
from datetime import datetime,timezone
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,brier_score_loss,confusion_matrix
from tabpfn import TabPFNClassifier
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
def utc():return datetime.now(timezone.utc).isoformat()
def verify_freeze():
    f=read(OUT/'freeze.json')
    for p,h in f['sha256'].items():assert sha(ROOT/p)==h,p
    return len(f['sha256'])
def estimator(name,d):
    c=d['models'][name]
    if name=='LOGISTIC':return Pipeline([('scale',StandardScaler()),('model',LogisticRegression(**c))])
    if name=='FOREST64':return RandomForestClassifier(**c)
    return TabPFNClassifier(model_path=str(ROOT/d['checkpoint_path']),**c)
def arrays(rows,ids,features):
    rr=[rows[i] for i in ids]
    return np.asarray([[r['features'][f] for f in features] for r in rr],float),np.asarray([r['y_same_path'] for r in rr],int)
def score(rows):
    if not rows:return {'n':0}
    y=np.array([r['y'] for r in rows]);p=np.array([r['p_same_path'] for r in rows]);pred=p>=.5
    result=dict(n=len(rows),same_path=int(y.sum()),changed_path=int((1-y).sum()),auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,p)) if y.sum() else None,balanced_accuracy=float(balanced_accuracy_score(y,pred)) if len(set(y))==2 else None,
        accuracy=float(np.mean(y==pred)),brier=float(brier_score_loss(y,p)),confusion_changed_then_same=confusion_matrix(y,pred,labels=[0,1]).tolist())
    result['thresholds']={}
    for t in [.5,.95]:
        selected=[r for r in rows if r['p_same_path']>=t]
        errors=[r for r in selected if r['y']==0]
        result['thresholds'][str(t)]=dict(proposed_omissions=len(selected),correct_redundancy=len(selected)-len(errors),changed_path_misclassified=len(errors),coverage=len(selected)/len(rows),precision=(len(selected)-len(errors))/len(selected) if selected else None,
            case_ids=[r['case_id'] for r in selected],false_omission_ids=[r['case_id'] for r in errors],false_omission_original_labels=[r['original_label'] for r in errors])
    return result
