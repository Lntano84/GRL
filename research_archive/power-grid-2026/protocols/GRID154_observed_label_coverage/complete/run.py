"""Identify unchanged global-compression targets using only existing FRAME paid feedback."""
from pathlib import Path
import collections,hashlib,importlib.util,json,shutil,sys
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid154_observed_label_coverage';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID154_observed_label_coverage'
spec=importlib.util.spec_from_file_location('g154_reference',R/'work/grid140_global_compression/common.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
 B.verify_trained()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
 B.P.sealed()
def cases():return [r for r in read(R/'outputs/grid153_path_coverage/call_coverage.json') if r['repeat']==0]
def prepare():
 assert not O.exists();O.mkdir(parents=True);B.verify_trained();assert read(R/'outputs/grid153_path_coverage/review.json')['passed']
 save('design.json',dict(id='GRID154_OBSERVED_LABEL_COVERAGE',population='All30 existing actual FRAME calls, four exposed development dates, repeat0. No new cohort or physical samples.',
  target='Unchanged: STOP iff stopping at an eligible pre-forced-stop checkpoint and doing original same-station completion preserves exact original FRAME selected pool_id AND strictly reduces total candidate queries.',
  identification='Compute required completion IDs from public order/aliases and current observed incumbent. If total query count is non-saving label CONTINUE even without all feedback. If saving and required candidates all paid in existing call, label from exact selection. Otherwise UNKNOWN, never relabel unknown as continue.',
  eligible='Original global checkpoints with an incumbent, before original forced stop; no extending past forced stop, no inventing incumbent, no threshold tuning.',
  audit='Reexecute official FRAME and independent manual reference on all actual paid calls; compare identifiable labels against existing exact-context teacher only as separate audit. Teacher suffix cannot fill unknown in observed-only dataset.',
  limits='No model inference/fit/physics/new generation/test8. Partial observed labels can be selected/missing nonrandomly; not a new training set or independent test. No subset/rule chosen based on labels.'))
 paths=[R/'outputs/grid153_path_coverage'/n for n in ['call_coverage.json','review.json','complete.json','source_hashes.json']]
 for r in cases():paths.append(R/r['source'])
 paths +=[R/'work/grid86/policies.py',R/'work/grid139_global_feasibility/run.py',R/'outputs/grid139_global_feasibility/compression_task_targets.jsonl.gz',O/'design.json']
 save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(set(paths))});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen'));archive('frozen');print('GRID154_FROZEN')
def identify(c,n,truth):
 order=c['order'];prefix=order[:n];b0=B.P.best([truth[i] for i in order[:min(128,len(order))]]);b=B.P.best([truth[i] for i in prefix]);assert b is not None
 safe=b0 is not None and b0['rho']<.9;extra=[]
 if n<len(order) and (not safe or B.P.bid(b)!=B.P.bid(b0)):
  subs=B.subids();station=subs[B.P.bid(b)];ids={int(rep) for a,rep in c['aliases'].items() if subs[int(a)]==station};extra=[i for i in order[n:] if i in ids]
 missing=[i for i in extra if i not in truth];q=n+len(extra);original=len(c['outcomes']);selected=None
 if q>=original:label=False;reason='NON_SAVING'
 elif missing:label=None;reason='UNOBSERVED_COMPLETION'
 else:
  selected=B.P.bid(B.P.best([truth[i] for i in prefix+extra]));label=selected==c['chosen_pool_id'];reason='OBSERVED_SAME_ACTION' if label else 'OBSERVED_DIFFERENT_ACTION'
 return dict(visible_count=n,total_queries_if_stopped=q,original_queries=original,required_completion_ids=extra,unobserved_completion_ids=missing,selected_if_identifiable=selected,stop_label=label,reason=reason)
def analyse():
 verify();assert not (O/'review.json').exists();checks=collections.Counter();spec=importlib.util.spec_from_file_location('g154_official',R/'work/grid86/policies.py');G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
 teacher_targets={(x['call_id'],x['visible_count']):x for x in B.gz(B.SOURCE/'compression_task_targets.jsonl.gz')};rows=[];callrows=[];audit=[]
 for ref in cases():
  c=read(R/ref['source']);truth={r['pool_id']:r for r in c['outcomes']};assert len(truth)==len(c['outcomes']);seen=[]
  def reveal(i,phase):
   assert i not in seen and i in truth;seen.append(i);return dict(truth[i],phase=phase)
  result=G.run(c['order'],reveal,'FRAME_GUARD',B.subids(),c['aliases'],limit=c['rho_limit']);manual=B.P.manual_guard(c,B.subids());ng=manual['global_n'];origids=c['order'][:ng]+manual['local']
  assert seen==origids==[r['pool_id'] for r in c['outcomes']];assert result['chosen_pool_id']==B.P.bid(manual['after'])==c['chosen_pool_id'];assert result['pre_extra_choice']==c['pre_extra_choice'];checks['official_manual_original_calls']+=1
  for o in c['outcomes']:
   assert o['strict_admissible']==bool(0<o['rho']<c['rho_limit'] and not o['done'] and not o['illegal'] and not o['ambiguous'] and not o['exceptions']);checks['legal_feedback']+=1
  current=[]
  for n in manual['visited']:
   if n>=ng or B.P.best(c['outcomes'][:n]) is None:continue
   x=identify(c,n,truth);x.update(source=ref['source'],call_id=ref['scenario']+'|'+Path(ref['source']).stem,date=ref['date'],scenario=ref['scenario'],step=c['step'],teacher_context_matched=bool(ref['teacher_exact_hits']));rows.append(x);current.append(x)
   # Independent set/selection recomputation, no missing candidate lookup.
   required=c['order'][:n]+x['required_completion_ids'];assert len(required)==len(set(required));assert len(required)==x['total_queries_if_stopped']
   if x['stop_label'] is not None and x['reason']!='NON_SAVING':
    direct=sorted((truth[i] for i in required if truth[i]['strict_admissible']),key=lambda o:(-B.P.key(o)[0],o['pool_id']))
    chosen=direct[0]['pool_id'] if direct else None;assert x['stop_label']==(chosen==c['chosen_pool_id']);checks['independent_known_selection']+=1
   # Withhold a required local result: no positive/negative selection may be invented.
   if x['reason']!='NON_SAVING' and not x['unobserved_completion_ids'] and x['required_completion_ids']:
    masked=dict(truth);masked.pop(x['required_completion_ids'][0]);assert identify(c,n,masked)['stop_label'] is None;checks['masked_completion_unknown']+=1
   for tid in ref['teacher_exact_hits']:
    target=teacher_targets[(tid,n)];ta=target['stop_label'];audit.append(dict(actual_call=x['call_id'],teacher_call=tid,visible_count=n,observed_label=x['stop_label'],teacher_label=ta,unknown=x['stop_label'] is None))
    if x['stop_label'] is not None:assert x['stop_label']==bool(ta);checks['exact_teacher_label_matches']+=1
   checks['eligible_checkpoints']+=1
  positives=[x for x in current if x['stop_label'] is True];best=min(positives,key=lambda x:x['total_queries_if_stopped'],default=None)
  callrows.append(dict(call_id=ref['scenario']+'|'+Path(ref['source']).stem,date=ref['date'],scenario=ref['scenario'],eligible=len(current),known=sum(x['stop_label'] is not None for x in current),positive=len(positives),unknown=sum(x['stop_label'] is None for x in current),original_queries=len(c['outcomes']),global_queries=ng,local_queries=len(manual['local']),earliest_observed_stop=positives[0]['visible_count'] if positives else None,minimum_observed_queries=best['total_queries_if_stopped'] if best else len(c['outcomes']),has_unresolved=any(x['stop_label'] is None for x in current)))
 def counts(rr):return dict(prefixes=len(rr),stop=sum(x['stop_label'] is True for x in rr),continue_=sum(x['stop_label'] is False for x in rr),unknown=sum(x['stop_label'] is None for x in rr))
 perdate=[]
 for date in sorted({x['date'] for x in callrows}):
  cr=[x for x in callrows if x['date']==date];rr=[x for x in rows if x['date']==date];total=sum(x['original_queries'] for x in cr);best=sum(x['minimum_observed_queries'] for x in cr)
  perdate.append(dict(date=date,calls=len(cr),**counts(rr),original_queries=total,observed_hindsight_queries=best,observed_hindsight_reduction=1-best/total,calls_with_known_stop=sum(bool(x['positive']) for x in cr)))
 verify();save('observed_targets.json',rows);save('call_summary.json',callrows);save('teacher_label_audit_only.json',audit);save('per_date.json',perdate)
 save('summary.json',dict(calls=len(callrows),dates=4,**counts(rows),date_macro_observed_hindsight_reduction=sum(x['observed_hindsight_reduction'] for x in perdate)/4,reason_counts=dict(collections.Counter(x['reason'] for x in rows)),unknown_teacher_labels=dict(collections.Counter(str(x['teacher_label']) for x in audit if x['unknown'])),hindsight_not_policy=True,unknown_unfilled=True))
 save('review.json',dict(passed=True,checks=dict(checks),new_fits=0,new_model_inferences=0,new_physics=0,source_hashes_unchanged=True,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(read(O/'summary.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
