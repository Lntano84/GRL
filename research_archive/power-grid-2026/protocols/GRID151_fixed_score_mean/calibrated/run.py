"""One fixed equal-score combination; constituent models never refit."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid151_fixed_score_mean'
sp=importlib.util.spec_from_file_location('grid151_linear',R/'work/grid150_linear_control/run.py');G=importlib.util.module_from_spec(sp);sp.loader.exec_module(G)
B=G.B;REPO=G.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID151_fixed_score_mean'
read=G.read;sha=G.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
 G.verify_trained();assert read(G.O/'complete.json')['passed_audit']
 for name in ['source_hashes.json','code_freeze.json']:
  for p,h in read(O/name).items():assert sha(R/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def note(stage,text):
 entry=f'\n\n## 2026-10-25 — GRID151 {stage}\n\n{text}\n'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
  if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);old=title+'\n'+entry+'\n'+rest
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID151 '+stage+'\n\n1. '+text+'\n2. 仅一次固定0.5/0.5原分数均值，不试logit均值/权重/阈值网格，不重训成分模型。\n3. 主标签/搜索/日期/强对照保持；开发已暴露，不作独立确认，不自动扩数据/物理/最终8。\n',encoding='utf8',newline='\n')

def prepare():
 assert not O.exists();O.mkdir();G.verify_trained();assert read(G.O/'review.json')['passed']
 save('design.json',dict(id='GRID151_FIXED_EQUAL_SCORE_MEAN',authorization='User authorizes continuous justified investigation. Progress-only notice,no refitting.',
  task='Same originalFRAME action-preserving global search compression,unchanged18features/labels/date split/control/aliases/force stops/completion.',
  motivation='RF140 and LR150 are different inductive biases on identical inputs/weights;inspect complementarity rather than retune either model. Existing constituent development outputs are exposed;this is further development,not a fresh test.',
  constituent_models=['FrozenRF140','FrozenLR150'],score='Exactly0.5*pSTOP_RF140 +0.5*pSTOP_LR150 at same public checkpoint,no weights learned or alternative combination.',
  parameters='No new fitted parameters. Different inference cost than one model;offline candidate query counts do not establish real-time improvement.',
  calibration='Same empirical procedure from0.5,highest bad actually reached stop score,strictgreater and reexecute completecalcalls. No formal risk guarantee.',
  primary='MEAN151_CALIBRATED',secondary='MEAN151_FIXED_0.5 retained',
  evidence_order='Prepare all scripts,seal score formula/source/models;compute/calibrate on386fit/198cal only,push cutoff and derivedscore arrays before loading constituent validation arrays for combination.',
  complementarity='All original122continue prefixes;within same depth/entry-safe stratum,report positive-negative pairs bothcorrect/RFonly/LRonly/bothwrong/tied;counts not independent tests. No thresholds selected from these tables.',
  screen='Original0action/0jointfailure,macroquery reduction>CAP256,eachdatequeries<=CAP256 andone strict. No automatic adoption/physics/scale.',
  prohibited=['fit','newmodel/parameter/combinationweight/grid','validationderivedthreshold','risk/labelchange','newphysics/data/test8','paper/scale']))
 paths=[O/'design.json',R/'work/grid150_linear_control/run.py',R/'work/grid140_global_compression/common.py']
 for root in [G.O,B.O]:
  paths += [root/n for n in ['model.pkl','model_export.json','train_scores.npz','validation_scores.npz','cutoff.json','trained_freeze.json']]
 paths += [G.O/n for n in ['all_search_replays.json','summary.json','review.json','publication_verification.json']]
 paths += [B.O/n for n in ['fit_rows.json.gz','calibration_rows.json.gz','design.json']]
 save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')})
 verify();save('state.json',dict(stage='frozen'));archive('frozen');note('frozen','两冻结模型同18项、同标签，固定各半算术均值一次检查，不扫描组合，零拟合；所有开发已暴露，不声称新独立验证。')
 print('GRID151_FROZEN')

def calibrate():
 verify();assert not (O/'started.json').exists();save('started.json',dict(fits=0))
 rz=np.load(B.O/'train_scores.npz');lz=np.load(G.O/'train_scores.npz')
 pf=.5*rz['fit']+.5*lz['fit'];pc=.5*rz['calibration']+.5*lz['calibration']
 np.savez_compressed(O/'train_scores.npz',fit=pf,calibration=pc)
 cal=G.S.rowfile(B.O/'calibration_rows.json.gz');dates=read(B.O/'design.json')['calibration_dates']
 tau,h,initial,final=G.calibrate(B.refs('train',set(dates)),G.smap(cal,pc))
 save('calibration_history.json',h);save('calibration_fixed_calls.json',initial);save('calibration_final_calls.json',final)
 save('cutoff.json',dict(primary_threshold=tau,strict_greater=True,empirical_only=True,validation_combined=False))
 save('derived_freeze.json',{n:sha(O/n) for n in ['design.json','train_scores.npz','calibration_history.json','calibration_fixed_calls.json','calibration_final_calls.json','cutoff.json']})
 archive('calibrated');note('calibrated',f'固定等权分数校准完成，阈值{tau:.17g}，零拟合；发布公式/数组/阈值后再合并开发分数。')
 print('GRID151_CALIBRATED',tau,h[-1]['summary']['model_stops'])

def publication():
 verify();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote
 p=(A/'calibrated/derived_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{p}'],cwd=REPO)).hexdigest()==sha(O/'derived_freeze.json')
 save('prevalidation_publication.json',dict(passed=True,local_sha=local,remote_sha=remote,derived_freeze_sha256=sha(O/'derived_freeze.json')));print('GRID151_PUBLISHED')

def evaluate():
 verify();pub=read(O/'prevalidation_publication.json');assert pub['passed'] and pub['derived_freeze_sha256']==sha(O/'derived_freeze.json');assert not (O/'validation_started.json').exists()
 save('validation_started.json',dict(published_sha=pub['local_sha'],fits=0));rz=np.load(B.O/'train_scores.npz');lz=np.load(G.O/'train_scores.npz')
 rv=np.load(B.O/'validation_scores.npz')['scores'];lv=np.load(G.O/'validation_scores.npz')['scores'];pv=.5*rv+.5*lv;np.savez_compressed(O/'validation_scores.npz',scores=pv)
 fit=G.S.rowfile(B.O/'fit_rows.json.gz');cal=G.S.rowfile(B.O/'calibration_rows.json.gz');val=B.eligible_data('validation');rows=fit+cal+val
 rf=np.concatenate([rz['fit'],rz['calibration'],rv]);lr=np.concatenate([lz['fit'],lz['calibration'],lv]);pp=.5*rf+.5*lr;sm=G.smap(rows,pp)
 cut=read(O/'cutoff.json')['primary_threshold'];prior=read(G.O/'all_search_replays.json');old={(r['evaluation_partition'],r['method'],r['call_id']):r for r in prior};methods=list(dict.fromkeys(r['method'] for r in prior));assert len(methods)==17
 dates=read(B.O/'design.json');records=[];sums=[]
 for part,refs in [('fit',B.refs('train',set(dates['fit_dates']))),('calibration',B.refs('train',set(dates['calibration_dates']))),('validation',B.refs('validation'))]:
  for name in methods+['MEAN151_FIXED_0.5','MEAN151_CALIBRATED']:
   rr=[]
   for ref in refs:
    r=B.replay(ref,'MODEL',.5 if name.endswith('0.5') else cut,sm) if name.startswith('MEAN151') else dict(old[part,name,ref['call_id']])
    rr.append(dict(r,method=name,evaluation_partition=part))
   records+=rr;sums.append(dict(method=name,evaluation_partition=part,**B.summarize(rr),joint_failures=sum(r['bad_learned_stop'] for r in rr)))
 save('all_search_replays.json',records);save('summary.json',sums)
 metrics=[];pairwise=[]
 parts=['fit']*386+['calibration']*198+['validation']*142
 for part in ['fit','calibration','validation']:
  ii=[i for i,p in enumerate(parts) if p==part];rr=[rows[i] for i in ii]
  for name,p in [('RF140',rf),('LR150',lr),('MEAN151',pp)]:metrics.append(dict(model=name,partition=part,**G.ranks(rr,p[ii],cut if name=='MEAN151' else read((B.O if name=='RF140' else G.O)/'cutoff.json')['primary_threshold'])))
  groups={}
  for i in ii:groups.setdefault((rows[i]['meta']['visible_count'],rows[i]['features']['prefix_safe']),[]).append(i)
  for (depth,safe),jj in sorted(groups.items()):
   pos=[i for i in jj if rows[i]['target']['stop_label']];neg=[i for i in jj if not rows[i]['target']['stop_label']];counts=dict(both_correct=0,rf_only_correct=0,lr_only_correct=0,both_wrong=0,tied=0)
   for i in pos:
    for j in neg:
     a=rf[i]-rf[j];b=lr[i]-lr[j]
     k='tied' if a==0 or b==0 else 'both_correct' if a>0 and b>0 else 'rf_only_correct' if a>0 else 'lr_only_correct' if b>0 else 'both_wrong';counts[k]+=1
   pairwise.append(dict(partition=part,depth=depth,prefix_safe=safe,pairs=len(pos)*len(neg),**counts))
 save('ranking_metrics.json',metrics);save('same_depth_complementarity.json',pairwise)
 get=lambda n:next(s for s in sums if s['evaluation_partition']=='validation' and s['method']==n)
 a=get('MEAN151_CALIBRATED');b=get('CAP256');bd={r['date']:r for r in b['by_date']};dd=[dict(date=r['date'],action_differences=r['action_differences'],queries=r['queries'],cap256_queries=bd[r['date']]['queries']) for r in a['by_date']]
 passed=a['action_differences']==0 and a['joint_failures']==0 and a['date_macro_query_reduction']>b['date_macro_query_reduction'] and all(r['queries']<=r['cap256_queries'] for r in dd) and any(r['queries']<r['cap256_queries'] for r in dd)
 save('decision.json',dict(primary='MEAN151_CALIBRATED',passes_development_screen=passed,ready_for_frame_coverage_proposal=passed,ready_to_scale=False,adopted=False,
  action_differences=a['action_differences'],joint_failures=a['joint_failures'],query_reduction_macro=a['date_macro_query_reduction'],cap256_macro=b['date_macro_query_reduction'],per_date=dd,new_fits=0,new_physical=0,final8_absent=True))
 verify();save('state.json',dict(stage='evaluated_pending_audit'));print(json.dumps(read(O/'decision.json'),ensure_ascii=False))

def audit():
 verify();assert read(O/'state.json')['stage']=='evaluated_pending_audit'
 for n,h in read(O/'derived_freeze.json').items():assert sha(O/n)==h
 for part in ['fit','calibration']:
  a=np.load(O/'train_scores.npz')[part];b=np.load(B.O/'train_scores.npz')[part];c=np.load(G.O/'train_scores.npz')[part];assert np.array_equal(a,(b+c)/2)
 v=np.load(O/'validation_scores.npz')['scores'];assert np.array_equal(v,(np.load(B.O/'validation_scores.npz')['scores']+np.load(G.O/'validation_scores.npz')['scores'])/2)
 fit=G.S.rowfile(B.O/'fit_rows.json.gz');cal=G.S.rowfile(B.O/'calibration_rows.json.gz');val=B.eligible_data('validation');z=np.load(O/'train_scores.npz');sm=G.smap(fit+cal+val,np.concatenate([z['fit'],z['calibration'],v]))
 tau,h,_,_=G.calibrate(B.refs('train',set(read(B.O/'design.json')['calibration_dates'])),sm);assert tau==read(O/'cutoff.json')['primary_threshold'] and h==read(O/'calibration_history.json')
 old={(r['evaluation_partition'],r['method'],r['call_id']):r for r in read(G.O/'all_search_replays.json')};records=read(O/'all_search_replays.json');assert len(records)==19*305
 refs={r['call_id']:r for r in B.refs()};checks=dict(old_rows=0,new_rows=0,complete_searches=0);bad=[]
 for r in records:
  if r['method'].startswith('MEAN151'):
   actual=B.replay(refs[r['call_id']],'MODEL',.5 if r['method'].endswith('0.5') else tau,sm);assert all(r[k]==v for k,v in actual.items());checks['new_rows']+=1
   if r['bad_learned_stop']:bad.append(r)
  else:assert all(r[k]==v for k,v in old[r['evaluation_partition'],r['method'],r['call_id']].items());checks['old_rows']+=1
  c=B.call_at(refs[r['call_id']]['source']);n=r['global_queries'];a=B.P.best(c['outcomes'][:n]);first=B.P.best(c['outcomes'][:128]);extra=[]
  if a is not None and n<len(c['order']) and (first is None or first['rho']>=.9 or B.P.bid(a)!=B.P.bid(first)):
   station=B.subids()[a['pool_id']];ids={int(v) for k,v in c['aliases'].items() if B.subids()[int(k)]==station};extra=[v for v in c['outcomes'][n:] if v['pool_id'] in ids]
  assert B.P.bid(B.P.best(c['outcomes'][:n]+extra))==r['selected'] and n+len(extra)==r['queries'];checks['complete_searches']+=1
 save('all_new_policy_failures.json',bad);save('review.json',dict(passed=True,new_fits=0,score_means_checked=726,mean_formula_fixed=True,original_calibration_reexecuted=True,checks=checks,
  source_models_weights_labels_cutoffs_unchanged=True,combination_formula_cutoff_published_before_dev=True,new_physical=0,final8_absent=True));save('state.json',dict(stage='audited'));print('GRID151_AUDITED',checks)

if __name__=='__main__':{'prepare':prepare,'calibrate':calibrate,'publication':publication,'evaluate':evaluate,'audit':audit}[sys.argv[1]]()
