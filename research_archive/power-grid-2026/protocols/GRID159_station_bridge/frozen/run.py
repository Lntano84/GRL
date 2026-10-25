"""Existing-fit-only global-to-local path attribution and public station coverage inventory."""
from pathlib import Path
import collections,gzip,hashlib,importlib.util,json,shutil,sys
import numpy as np
spec=importlib.util.spec_from_file_location('g159_reference',Path('D:/Codex/GRL研究/work/grid140_global_compression/common.py'));B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
R=B.R;W=Path(__file__).parent;O=R/'outputs/grid159_station_bridge';REPO=B.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID159_station_bridge'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def rowfile(p):
 with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def gzsave(n,x):
 with gzip.open(O/n,'wt',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False)
def verify():
 B.verify_trained()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
 B.P.sealed()
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
 assert not O.exists();O.mkdir(parents=True);B.verify_trained();assert read(R/'outputs/grid158_single_192/complete.json')['passed_audit']
 save('design.json',dict(id='GRID159_STATION_BRIDGE',population='Original12 fit dates only,386 prefixes and161 teacher-state calls. All55 CONTINUE rows retained;repeated prefixes do not count as independent calls.',
  mechanism='Descriptive bridge-local case: original final action selected in original local completion, final original global incumbent station differs from incumbent station at current prefix, and original final action absent from stop-now queried set. No causal wholeweek claim.',
  labels='Original same-action-and-saving STOP unchanged. Allpositive331 andnegative55 counted. Future global/local results used only separate audit,never online fields.',
  public_profile='For each publicly represented station: all representative IDs linked through aliases, seen/remaining counts and best observed admissible reward/rho. Query may count toward multiple alias stations;station counts are not mutually exclusive.',
  qualification='All386 public profiles computed from explicit redacted prefix contract and compared after hidden suffix poisoning. Raw profile inventory only,no predictive screen/fit/selection/threshold/rule.',
  prohibited=['new fits/predictions','new labels','new cal/dev extraction','new physical/data/test8','paper','adoption or scale']))
 paths=[R/'outputs/grid150_linear_control/original_fit_rows.json.gz',R/'outputs/grid158_single_192/complete.json',B.SOURCE/'call_ledger.json',R/'outputs/grid28/offline.json',O/'design.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen'));archive('frozen');print('GRID159_FROZEN')
def profile(public):
 assert set(public)=={'order','aliases','prefix'};order=public['order'];prefix=public['prefix'];assert [x['pool_id'] for x in prefix]==order[:len(prefix)];seen={x['pool_id']:x for x in prefix};by=collections.defaultdict(set);subs=B.subids()
 for a,rep in public['aliases'].items():by[subs[int(a)]].add(int(rep))
 out=[]
 for station,ids in sorted(by.items()):
  ids=ids&set(order)
  if not ids:continue
  observed=ids&seen.keys();best=B.P.best([seen[i] for i in observed]);out.append(dict(station=station,representatives=len(ids),seen=len(observed),remaining=len(ids-observed),best_observed_pool=B.P.bid(best),best_observed_reward=B.P.key(best)[0] if best is not None else None,best_observed_rho=best['rho'] if best is not None else None))
 return out
def analyse():
 verify();assert not (O/'review.json').exists();rows=rowfile(R/'outputs/grid150_linear_control/original_fit_rows.json.gz');assert len(rows)==386;refs={r['call_id']:r for r in B.refs()};audits=[];profiles=[];checks=collections.Counter()
 for row in rows:
  cid=row['meta']['call_id'];ref=refs[cid];c=B.call_at(ref['source']);n=row['meta']['visible_count'];truth={x['pool_id']:x for x in c['outcomes']};b=B.P.best(c['outcomes'][:n]);initial=B.P.best(c['outcomes'][:128]);safe=initial is not None and initial['rho']<.9;station=B.subids()[b['pool_id']];extra=[]
  if not safe or B.P.bid(initial)!=B.P.bid(b):
   ids={int(rep) for a,rep in c['aliases'].items() if B.subids()[int(a)]==station};extra=[i for i in c['order'][n:] if i in ids]
  queried=set(c['order'][:n]+extra);stopped=B.P.best([truth[i] for i in queried]);pre=truth[ref['global_choice']];final=truth[ref['frame_choice']];manual=B.P.manual_guard(c,B.subids());fromlocal=ref['frame_choice'] in manual['local'];stationchanged=B.subids()[pre['pool_id']]!=station;absent=final['pool_id'] not in queried;bridge=bool(fromlocal and stationchanged and absent);label=row['target']['stop_label'];q=len(queried)
  assert label==(B.P.bid(stopped)==ref['frame_choice'] and q<ref['global_n']+ref['local_queries']);assert not (label and bridge)
  audits.append(dict(sample_id=row['meta']['sample_id'],call_id=cid,date=ref['date'],scenario=ref['scenario'],visible_count=n,stop_label=label,stop_queries=q,original_queries=ref['global_n']+ref['local_queries'],stop_choice=B.P.bid(stopped),original_global_choice=pre['pool_id'],original_choice=final['pool_id'],original_final_from_local=fromlocal,global_station_changed=stationchanged,original_final_absent_at_stop=absent,bridge_local=bridge,global_reward_gain=B.P.key(pre)[0]-B.P.key(b)[0],local_reward_gain=B.P.key(final)[0]-B.P.key(pre)[0],future_fields_audit_only=True))
  public=dict(order=c['order'],aliases=c['aliases'],prefix=[{k:x[k] for k in ['pool_id','reward','rho','strict_admissible']} for x in c['outcomes'][:n]]);pp=profile(public);poison=dict(c);poison['outcomes']=c['outcomes'][:n]+[dict(x,reward=-999,rho=999,strict_admissible=False) for x in c['outcomes'][n:]];redacted=dict(order=poison['order'],aliases=poison['aliases'],prefix=[{k:x[k] for k in ['pool_id','reward','rho','strict_admissible']} for x in poison['outcomes'][:n]]);assert profile(redacted)==pp
  for st in pp:assert st['seen']+st['remaining']==st['representatives'];checks['station_count_partitions']+=1
  profiles.append(dict(sample_id=row['meta']['sample_id'],call_id=cid,date=ref['date'],visible_count=n,station_profile=pp));checks['public_suffix_invariance']+=1
 negative=[x for x in audits if not x['stop_label']];bridge=[x for x in negative if x['bridge_local']];assert len(negative)==55;per=[]
 for date in sorted({x['date'] for x in audits}):
  rr=[x for x in negative if x['date']==date];bb=[x for x in bridge if x['date']==date];per.append(dict(date=date,continue_prefixes=len(rr),continue_calls=len({x['call_id'] for x in rr}),bridge_prefixes=len(bb),bridge_calls=len({x['call_id'] for x in bb})))
 def desc(xs):return dict(min=float(min(xs)),median=float(np.median(xs)),max=float(max(xs))) if xs else None
 summary=dict(prefixes=386,continue_prefixes=55,continue_calls=len({x['call_id'] for x in negative}),continue_dates=len({x['date'] for x in negative}),bridge_continue_prefixes=len(bridge),bridge_continue_calls=len({x['call_id'] for x in bridge}),bridge_continue_dates=len({x['date'] for x in bridge}),bridge_global_reward_gain=desc([x['global_reward_gain'] for x in bridge]),bridge_local_reward_gain=desc([x['local_reward_gain'] for x in bridge]),public_profiles=386,new_fits=0,new_scores=0,new_labels=0,profile_is_not_trained_or_validated=True)
 verify();save('mechanism_audit_only.json',audits);gzsave('public_station_profiles.json.gz',profiles);save('per_date.json',per);save('summary.json',summary);save('review.json',dict(passed=True,checks=dict(checks),new_fits=0,new_scores=0,new_physics=0,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
