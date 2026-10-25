"""Exact accounting of existing RF scores along their tree paths, not causal SHAP."""
import numpy as np
import run as D
def account(export,xx):
 names=D.read(D.B.O/'preprocessing.json')['features'];contrib={k:0. for k in names};root=0.;leaf=0.;positive=export['classes'].index(1)
 def p(tree,i):v=tree['value'][i];return v[positive]/sum(v)
 x=np.asarray(xx,np.float32)
 for t in export['trees']:
  n=0;root+=p(t,n)
  while t['left'][n]!=-1:
   j=t['feature'][n];child=t['left'][n] if x[j]<=t['threshold'][n] else t['right'][n];contrib[names[j%18]]+=p(t,child)-p(t,n);n=child
  leaf+=p(t,n)
 count=len(export['trees']);root/=count;leaf/=count;contrib={k:v/count for k,v in contrib.items()};assert abs(root+sum(contrib.values())-leaf)<1e-12
 return dict(root_probability=root,contribution=contrib,score=leaf)
def main():
 D.verify();assert not (D.O/'tree_path_accounting.json').exists();oldpre=D.read(D.B.O/'preprocessing.json');newpre=D.read(D.M.O/'preprocessing.json');oldex=D.read(D.B.O/'model_export.json');newex=D.read(D.M.O/'model_export.json');cases=[]
 for r in D.read(D.O/'changed_calls.json'):cases.append(dict(call_id=r['call_id'],group=r['category'],features=r['features'],old_score=r['old_score192'],new_score=r['new_score192']))
 for r in D.read(D.O/'calibration_wrong_exits_suppressed.json'):cases.append(dict(call_id=r['call_id'],group='CALIBRATION_WRONG_EXIT_SUPPRESSED',features=r['features'],old_score=r['old_score'],new_score=r['new_score']))
 output=[]
 for r in cases:
  row=[dict(features=r['features'])];a=account(oldex,D.B.transform(row,oldpre['features'],oldpre['medians'])[0]);b=account(newex,D.B.transform(row,newpre['features'],newpre['medians'])[0]);assert abs(a['score']-r['old_score'])<1e-12 and abs(b['score']-r['new_score'])<1e-12;delta={k:b['contribution'][k]-a['contribution'][k] for k in a['contribution']};root=b['root_probability']-a['root_probability'];assert abs(root+sum(delta.values())-(r['new_score']-r['old_score']))<1e-12
  ordered=sorted(delta,key=delta.get);output.append(dict(call_id=r['call_id'],group=r['group'],old_score=r['old_score'],new_score=r['new_score'],old_root=a['root_probability'],new_root=b['root_probability'],root_change=root,path_feature_changes=delta,top_decreases=[dict(feature=k,delta=delta[k]) for k in ordered[:3]],top_increases=[dict(feature=k,delta=delta[k]) for k in ordered[-3:][::-1]],old_account=a,new_account=b))
 D.save('tree_path_accounting.json',output);D.save('tree_path_receipt.json',dict(passed=True,cases=len(output),model_score_reconstructions=2*len(output),root_plus_path_sums_exact=True,not_causal_explanation=True,not_conditional_expectation_shap=True,no_feature_intervention=True,new_fits=0,new_data=0));D.verify();print([{k:r[k] for k in ['call_id','group','root_change','top_decreases']} for r in output])
if __name__=='__main__':main()
