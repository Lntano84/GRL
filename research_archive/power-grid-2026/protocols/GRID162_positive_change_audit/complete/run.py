"""Exploratory paired accounting of existing improvements, not a new acceptance test."""
from pathlib import Path
import collections,hashlib,importlib.util,json,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid162_positive_change_audit';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID162_positive_change_audit'
sp=importlib.util.spec_from_file_location('g162_saved',R/'work/grid158_single_192/run.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M);B=M.B
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
 M.verify_trained()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
 assert not O.exists();O.mkdir(parents=True);M.verify_trained()
 save('design.json',dict(id='GRID162_POSITIVE_CHANGE_AUDIT',authorization='User: explore improved points and why they changed before requiring stable overall improvement.',
  status='Post-hoc exploratory decomposition. No new acceptance gate and no independent confirmation.',
  current='All62 original development calls pairedRF140 vsRF158;retain all added correct exits,lost correct exits,wrong exits and unchanged calls. Cross two existing192 score arrays with two existing frozen cutoffs only as arithmetic diagnostic,never choose a deployment threshold. Check all82 calibration calls too to expose cost of the crossed readings.',
  historical='All3 paired seeds/all4 old local-task dates ofGRID116 FACTOR vsBASE. Separate realized reward contribution from proxy query-charge contribution,retain negative seeds/dates. Different target from current action identity;never pool the two tasks.',
  priorities='An informative local improvement remains worth investigating even if overall adoption gate fails. Keep original success criteria for later confirmation;do not relabel historical failure as success.',
  prohibited=['new fitting/predictions','threshold search or selection','drop adverse rows','change labels','new physical/data/test8','paper','automatic scale']))
 paths=[M.O/n for n in ['replays.json','cutoff.json','train_scores.npz','validation_rows.json.gz','validation_scores.npz','review.json']]+[B.O/n for n in ['fit_rows.json.gz','calibration_rows.json.gz','train_scores.npz','validation_scores.npz','cutoff.json']]+[R/'outputs/grid116_matched_module'/n for n in ['design.json','decision.json','validation_results.json','payoff_accounting.json','review.json']]+[O/'design.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen_exploratory'));archive('frozen');print('GRID162_FROZEN')
def analyse():
 verify();assert not (O/'review.json').exists();records=read(M.O/'replays.json');va=B.eligible_data('validation');oldrows=M.rowfile(B.O/'fit_rows.json.gz')+M.rowfile(B.O/'calibration_rows.json.gz')+va;oz=np.load(B.O/'train_scores.npz');old=M.scoremap(oldrows,np.concatenate([oz['fit'],oz['calibration'],np.load(B.O/'validation_scores.npz')['scores']]));newrows=M.rowfile(M.O/'fit_rows.json.gz')+M.rowfile(M.O/'calibration_rows.json.gz')+M.rowfile(M.O/'validation_rows.json.gz');nz=np.load(M.O/'train_scores.npz');new=M.scoremap(newrows,np.concatenate([nz['fit'],nz['calibration'],np.load(M.O/'validation_scores.npz')['scores']]));cut=read(M.O/'cutoff.json');oldtau=cut['old_original'];newtau=cut['primary'];features={(r['meta']['call_id'],r['meta']['visible_count']):r['features'] for r in oldrows};vr=[r for r in records if r['evaluation_partition']=='validation'];older={r['call_id']:r for r in vr if r['method']=='RF140_GLOBAL'};newer={r['call_id']:r for r in vr if r['method']=='RF158_192_CALIBRATED'};paired=[]
 for cid in older:
  a,b=older[cid],newer[cid];delta=a['queries']-b['queries'];category='UNCHANGED'
  if not b['action_match']:category='ADDED_WRONG_ACTION'
  elif delta>0:category='ADDED_CORRECT_SAVING'
  elif delta<0:category='LOST_CORRECT_SAVING'
  f=features.get((cid,192));paired.append(dict(call_id=cid,date=b['date'],scenario=b['scenario'],category=category,old_queries=a['queries'],new_queries=b['queries'],saved_relative_to_old=delta,old_selected=a['selected'],new_selected=b['selected'],original_selected=b['original_selected'],new_action_match=b['action_match'],old_score192=old.get((cid,192)),new_score192=new.get((cid,192)),old_cutoff=oldtau,new_cutoff=newtau,old_model_at_new_cutoff_stops=old.get((cid,192),-1)>newtau,features=f))
 changes=[r for r in paired if r['category']!='UNCHANGED'];summary={k:dict(calls=sum(r['category']==k for r in paired),query_delta=sum(r['saved_relative_to_old'] for r in paired if r['category']==k),dates=sorted({r['date'] for r in paired if r['category']==k})) for k in ['ADDED_CORRECT_SAVING','LOST_CORRECT_SAVING','ADDED_WRONG_ACTION','UNCHANGED']};assert sum(x['calls'] for x in summary.values())==62;assert sum(r['saved_relative_to_old'] for r in paired)==344
 crossed=[]
 for part,refs in [('calibration',B.refs('train',dates=read(B.O/'design.json')['calibration_dates'])),('validation',B.refs('validation'))]:
  for model,sm in [('OLD140',old),('NEW158',new)]:
   for threshold_name,tau in [('OLD_CUTOFF',oldtau),('NEW_CUTOFF',newtau)]:
    rr=[M.shadow(ref,tau,sm) for ref in refs];crossed.append(dict(partition=part,model=model,cutoff_name=threshold_name,threshold=tau,**B.summarize(rr),joint_failures=sum(r['bad_learned_stop'] for r in rr)))
 # This tests sufficiency for the four observed decision changes, not safe interchangeability of calibrated scores.
 wins=[r for r in paired if r['category']=='ADDED_CORRECT_SAVING'];assert len(wins)==4 and all(r['old_model_at_new_cutoff_stops'] for r in wins)
 save('all62_paired_calls.json',paired);save('changed_calls.json',changes);save('current_change_summary.json',summary);save('fixed_score_cutoff_crossing.json',crossed)
 v=read(R/'outputs/grid116_matched_module/validation_results.json');d=read(R/'outputs/grid116_matched_module/design.json');fee=d['unchanged']['cost_per_query'];scale=d['unchanged']['gain_scale'];historical=[];perdate=[]
 for seed in range(3):
  f=v[f'BASE_FACTOR_seed{seed}']['all'];b=v[f'DIRECT{seed}']['all'];ff,bb=f['macro'],b['macro'];g=(ff['mean_gain']-bb['mean_gain'])/scale;q=-fee*(ff['mean_queries']-bb['mean_queries'])/scale;net=ff['mean_net_utility']-bb['mean_net_utility'];assert abs(net-g-q)<1e-12
  historical.append(dict(seed=seed,factor_utility=ff['mean_net_utility'],base_utility=bb['mean_net_utility'],gain_contribution=g,query_charge_contribution=q,net_delta=net,relative_utility_change=ff['mean_net_utility']/bb['mean_net_utility']-1))
  for date in f['per_date']:
   x,y=f['per_date'][date],b['per_date'][date];gg=(x['mean_gain']-y['mean_gain'])/scale;qq=-fee*(x['mean_queries']-y['mean_queries'])/scale;nn=x['mean_net_utility']-y['mean_net_utility'];assert abs(nn-gg-qq)<1e-12;perdate.append(dict(seed=seed,date=date,gain_contribution=gg,query_charge_contribution=qq,net_delta=nn))
 family={k:float(np.mean([r[k] for r in historical])) for k in ['factor_utility','base_utility','gain_contribution','query_charge_contribution','net_delta']};family['relative_utility_change']=family['factor_utility']/family['base_utility']-1;family['positive_seeds']=sum(x['net_delta']>0 for x in historical);family['dates']=[dict(date=date,**{k:float(np.mean([x[k] for x in perdate if x['date']==date])) for k in ['gain_contribution','query_charge_contribution','net_delta']}) for date in sorted({x['date'] for x in perdate})]
 save('factor_vs_base_seeds.json',historical);save('factor_vs_base_seed_dates.json',perdate);save('factor_vs_base_summary.json',family);verify();save('review.json',dict(passed=True,paired_current_calls=62,fixed_cross_policy_calls=4*(82+62),historical_paired_seeds=3,historical_paired_seed_dates=12,accounting_equalities_passed=True,new_fits=0,new_model_predictions=0,new_physics=0,new_labels=0,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(current=summary,factor=family,cross=[{k:x[k] for k in ['partition','model','cutoff_name','action_differences','queries','joint_failures']} for x in crossed]),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
