"""Reuse serialized spaces; diagnose public physical/feedback context for the frozen global target."""
from pathlib import Path
import ast,collections,copy,gzip,hashlib,importlib.util,json,shutil,sys,warnings
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid148_state_context'
spec=importlib.util.spec_from_file_location('grid148_previous',R/'work/grid147_generalization_diagnosis/run.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
G=P.G;B=G.B;REPO=G.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID148_state_context'
read=G.read;sha=G.sha;SCHEMA=R/'outputs/grid03/scoring_smoke/agent_episode';META=R/'outputs/grid98_feature_enrichment/template_metadata.json'
FIELDS=['current_overload_count','current_overload_excess_sum','incumbent_incident_max_rho','incumbent_overload_endpoint_share',
 'incumbent_hops_to_maxrho_line','incumbent_min_hops_to_overload','visible_admissible_overload_endpoint_fraction',
 'visible_best_overload_endpoint_reward_gap','next_batch_overload_endpoint_fraction','next_batch_hops_to_maxrho_mean',
 'incumbent_station_changed_since_prefix','incumbent_station_changed_since_previous_batch']
VISIBLE={'pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible'}
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def verify():
 G.verify_trained()
 for n in ['source_hashes.json','code_freeze.json','state_sources.json']:
  for p,h in read(O/n).items():assert sha(R/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
 assert not O.exists();O.mkdir();G.verify_trained();assert read(P.O/'complete.json')['passed']
 save('design.json',dict(id='GRID148_PUBLIC_STATE_GLOBAL_LABEL_DIAGNOSIS',
  population='All726 already exposed global-compression prefixes:386fit/198cal/142dev,305 saved physical calls. No new data.',
  history='GRID98/99 already tried local structure/pressure for trajectory equivalence;GRID110-123 recovered/learned local action-feedback context for net gain with adverse results. Reuse reconstruction;do not relabel those outcomes as current action-preserving compression results.',
  schema='Reuse GRID03 serialized observation/action spaces and GRID110 schema;native deserialization and source AST field order,4460/1605 full roundtrips. Forbid environment construction/reset/step/simulate/forecast.',
  descriptors=FIELDS,semantics='Current bus graph:two nodes/substation,live lines connect actual topo_vect endpoint buses,no internal edge between distinct busbars. Distances to maxrho line endpoints and all currently overloaded endpoints. Station distance=min over its currently present buses;None if unreachable. Current hop distance is not electrical sensitivity or future solved impact.',
  cutoff='rho>1 defines current overloaded live lines;maxrho ties use first line index. No tuned threshold/hop radius.',
  alias='Candidate memberships include every public original-action station aliased to its representative. Next/visible endpoint coverage uses any member. Incumbent station follows original canonical public_subids lookup.',
  feedback='Only firstn paid outcomes and public next64 IDs. Incumbent station changes compare128 andn-64 prefixes. Future outcomes/labels cannot enter descriptor API.',
  comparison='Three fixed views:original18+missing,state12+missing,all30+missing. Fit386-only median/IQR with range fallback. For all726queries nearest5 distinct fit calls at same depth/initial-safe branch,exclude same date,save every neighbor and label. Geometric diagnosis only,no classifier/threshold or model selection.',
  measures='For all continue and stop rows,report support/k<5,zero-continue neighborhoods and date counts for each view;do not show only a favorable case. Current target and all negatives remain unchanged.',
  cases='Reuse all four GRID147 prespecified cases and their original two fit-neighbors;inspect saved original128/192 trace for actual error. No new favorable case selection.',
  checks='All305 vector/hash/schema checks,909 public commands,726 suffix-redaction/rejected-future-key/bus-label swap checks;independent scalar BFS/geometry checks.',
  prohibited=['fit','predict','new policy or threshold replay','environment initialization/solve','new labels or dates','final8tests','paper or figure','automatic training/scale']))
 paths=[O/'design.json',SCHEMA/'dict_observation_space.json',SCHEMA/'dict_action_space.json',META,
  R/'outputs/grid110_structure_audit/decoded_schema.json',R/'outputs/grid110_structure_audit/review.json',
  R/'outputs/grid110_structure_audit/GRID110_report.md',R/'outputs/grid113_relational_learning/decision.json',
  P.O/'saved_state_inventory.json',P.O/'case_comparison_summary.json',P.O/'feedback_context.json',
  B.O/'fit_rows.json.gz',B.O/'calibration_rows.json.gz',B.O/'preprocessing.json',
  R/'outputs/grid139_global_feasibility/call_ledger.json',R/'outputs/grid28/offline.json',
  R/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz',R/'work/grid110_structure_audit/extract.py',
  R/'work/grid00/.venv/Lib/site-packages/grid2op/l2rpn_utils/idf_2023.py',
  R/'work/grid00/.venv/Lib/site-packages/grid2op/Space/SerializableSpace.py']
 # Some historical stage directory names differ; archive the exact located report instead.
 paths=[p for p in paths if p.exists()]
 save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
 save('state_sources.json',read(P.O/'saved_state_source_hashes.json'))
 save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();save('state.json',dict(stage='frozen'));archive('frozen');print('GRID148_FROZEN')

def graph(physical,meta):
 rho=np.asarray(physical['rho']);live=np.asarray(physical['line_status'],bool);topo=np.asarray(physical['topo_vect'],int)
 assert set(topo.tolist())<={-1,1,2}
 starts=np.r_[0,np.cumsum(meta['sub_info'])];buses=[{2*s+int(v)-1 for v in topo[starts[s]:starts[s+1]] if v>0} for s in range(118)]
 adj=[set() for _ in range(236)];edges=[]
 for i in np.flatnonzero(live):
  a=int(topo[meta['line_or_pos_topo_vect'][i]]);b=int(topo[meta['line_ex_pos_topo_vect'][i]]);assert a>0 and b>0
  u=2*meta['line_or_to_subid'][i]+a-1;v=2*meta['line_ex_to_subid'][i]+b-1;adj[u].add(v);adj[v].add(u);edges.append((u,v,int(i)))
 assert live.any()
 peak=int(np.argmax(np.where(live,rho,-np.inf)));over=set(int(i) for i in np.flatnonzero(live&(rho>1)))
 core_peak={v for a,b,i in edges if i==peak for v in [a,b]};core_over={v for a,b,i in edges if i in over for v in [a,b]}
 def distance(core):
  d={v:0 for v in core};q=collections.deque(core)
  while q:
   u=q.popleft()
   for v in adj[u]:
    if v not in d:d[v]=d[u]+1;q.append(v)
  return [min((d[v] for v in buses[s] if v in d),default=None) for s in range(118)]
 return dict(peak=peak,over=over,peak_distance=distance(core_peak),over_distance=distance(core_over),edges=edges,
  rho=rho,live=live,over_stations={meta[k][i] for k in ['line_or_to_subid','line_ex_to_subid'] for i in over})

def describe(c,meta):
 assert set(c)=={'order','aliases','visible','subids','physical'}
 assert set(c['physical'])=={'rho','line_status','topo_vect'}
 visible=c['visible'];assert visible and all(set(r)==VISIBLE for r in visible)
 n=len(visible);order=c['order'];assert [r['pool_id'] for r in visible]==order[:n]
 b=B.P.best(visible);assert b is not None
 sub=c['subids'][b['pool_id']];gg=graph(c['physical'],meta)
 members={i:set() for i in order}
 for k,v in c['aliases'].items():members[int(v)].add(c['subids'][int(k)])
 assert all(members.values())
 incident=[i for i in np.flatnonzero(gg['live']) if meta['line_or_to_subid'][i]==sub or meta['line_ex_to_subid'][i]==sub]
 eligible=[r for r in visible if r['strict_admissible']]
 endpoint=[r for r in eligible if members[r['pool_id']]&gg['over_stations']]
 first=B.P.best(visible[:128]);prev=B.P.best(visible[:-64]) if n>128 else None
 nxt=order[n:min(n+64,len(order))];assert nxt
 nh=[min((gg['peak_distance'][s] for s in members[i] if gg['peak_distance'][s] is not None),default=None) for i in nxt]
 f=dict(current_overload_count=len(gg['over']),current_overload_excess_sum=float(np.maximum(gg['rho'][gg['live']]-1,0).sum()),
  incumbent_incident_max_rho=float(max(gg['rho'][incident])) if incident else 0.,
  incumbent_overload_endpoint_share=len(set(incident)&gg['over'])/len(gg['over']) if gg['over'] else 0.,
  incumbent_hops_to_maxrho_line=gg['peak_distance'][sub],incumbent_min_hops_to_overload=gg['over_distance'][sub],
  visible_admissible_overload_endpoint_fraction=len(endpoint)/len(eligible),
  visible_best_overload_endpoint_reward_gap=float(B.P.key(b)[0]-B.P.key(B.P.best(endpoint))[0]) if endpoint else None,
  next_batch_overload_endpoint_fraction=sum(bool(members[i]&gg['over_stations']) for i in nxt)/len(nxt),
  next_batch_hops_to_maxrho_mean=float(np.mean(nh)) if all(v is not None for v in nh) else None,
  incumbent_station_changed_since_prefix=None if first is None else c['subids'][first['pool_id']]!=sub,
  incumbent_station_changed_since_previous_batch=None if prev is None else c['subids'][prev['pool_id']]!=sub)
 assert list(f)==FIELDS and all(v is None or np.isfinite(v) for v in f.values());return f

def run():
 verify();assert not (O/'started.json').exists();save('started.json',dict(fits=0,physical=0))
 warnings.filterwarnings('ignore',message='Numba cannot be loaded.*')
 import grid2op
 from grid2op.Environment import BaseEnv
 from grid2op.Observation import ObservationSpace,BaseObservation
 from grid2op.Action import ActionSpace
 calls=collections.Counter()
 def forbidden(*a,**kw):calls['forbidden']+=1;raise RuntimeError('GRID148 forbids physical/environment methods')
 grid2op.make=BaseEnv.__init__=BaseEnv.step=BaseEnv.reset=BaseObservation.simulate=BaseObservation.get_forecast_env=forbidden
 os=ObservationSpace.from_dict(read(SCHEMA/'dict_observation_space.json'));space=ActionSpace.from_dict(read(SCHEMA/'dict_action_space.json'))
 oldschema=read(R/'outputs/grid110_structure_audit/decoded_schema.json');meta=read(META)
 assert os.size()==4460 and space.size()==1605 and list(os.attr_list_vect)==oldschema['observation_fields'] and list(os.shape)==oldschema['observation_shapes']
 tree=ast.parse((R/'work/grid00/.venv/Lib/site-packages/grid2op/l2rpn_utils/idf_2023.py').read_text(encoding='utf8'))
 klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ObservationIDF2023')
 declared=next(ast.literal_eval(n.value) for n in klass.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='attr_list_vect' for t in n.targets));assert declared==list(os.attr_list_vect)
 for k in ['sub_info','line_or_to_subid','line_ex_to_subid','line_or_pos_topo_vect','line_ex_pos_topo_vect']:assert np.array_equal(getattr(space,k),meta[k])
 subids=B.subids();library=np.load(R/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz')['action_space'];av=lambda a:np.concatenate([a._get_array_from_attr_name(n).astype(np.float32) for n in type(a).attr_list_vect])
 for i,v in enumerate(library):
  a=space.from_vect(v);assert digest(av(a))==meta['library_hashes'][i]
  assert int(a.as_dict()['set_bus_vect']['modif_subs_id'][0])==subids[i]
 save('decoded_schema.json',dict(**oldschema,library_candidates=909,grid2op_version=grid2op.__version__,native_spaces_restored=True,source_ast_order_matches=True))
 physical={};state_context=[]
 inventory=read(P.O/'saved_state_inventory.json')['records'];iv={r['call_id']:r for r in inventory}
 for ref in B.refs():
  p=R/iv[ref['call_id']]['observation_file']
  with np.load(p) as z:
   obs=os.from_vect(z['observation']);act=space.from_vect(z['base_action']);assert np.array_equal(obs.to_vect(),z['observation']) and np.array_equal(av(act),z['base_action'])
   assert digest(obs.to_vect())==B.call_at(ref['source'])['state_hash']
   fields={k:np.array(getattr(obs,k),copy=True) for k in ['rho','line_status','topo_vect']}
   # Independent named field slices from the recorded vector, not guessed offsets.
   start=0
   for name,size in zip(os.attr_list_vect,os.shape):
    if name in fields:assert np.array_equal(fields[name],z['observation'][start:start+size])
    start+=size
  assert start==4460
  physical[ref['call_id']]=fields;gg=graph(fields,meta)
  assert float(max(gg['rho'][gg['live']]))==B.call_at(ref['source'])['rho_limit']
  # Independent iterative edge-relaxation distances for both source sets.
  for label,core in [('peak',{v for a,b,i in gg['edges'] if i==gg['peak'] for v in [a,b]}),('over',{v for a,b,i in gg['edges'] if i in gg['over'] for v in [a,b]})]:
   dist=np.full(236,np.inf);dist[list(core)]=0
   for _ in range(236):
    before=dist.copy()
    for u,v,i in gg['edges']:dist[u]=min(dist[u],before[v]+1);dist[v]=min(dist[v],before[u]+1)
    if np.array_equal(before,dist):break
   starts=np.r_[0,np.cumsum(meta['sub_info'])]
   for s in range(118):
    buses={2*s+int(b)-1 for b in fields['topo_vect'][starts[s]:starts[s+1]] if b>0};d=min((dist[b] for b in buses),default=np.inf)
    assert (None if not np.isfinite(d) else int(d))==gg[label+'_distance'][s]
  state_context.append(dict(call_id=ref['call_id'],date=ref['date'],partition=ref['partition'],peak_line=gg['peak'],overloaded_lines=sorted(gg['over']),
   overloaded_end_stations=sorted(gg['over_stations']),max_rho=float(max(gg['rho'][gg['live']])),current_step=int(obs.current_step),recorded_control_step=ref['step'],vector_roundtrip=True))
 rows=G.rowfile(B.O/'fit_rows.json.gz')+G.rowfile(B.O/'calibration_rows.json.gz')+B.eligible_data('validation');parts=['fit']*386+['calibration']*198+['validation']*142
 assert len(rows)==726;descriptors=[];public=[]
 for r in rows:
  c=B.call_at(r['meta']['source']);n=r['meta']['visible_count'];visible=[{k:v[k] for k in VISIBLE} for v in c['outcomes'][:n]]
  contract=dict(order=c['order'],aliases=c['aliases'],visible=visible,subids=subids,physical=physical[r['meta']['call_id']])
  f=describe(contract,meta)
  # Hidden data cannot enter the strict contract. Rebuild from a suffix-poisoned full record.
  poison=dict(c,outcomes=c['outcomes'][:n]+[dict(v,reward=-999.,rho=999.,pool_id=-1) for v in c['outcomes'][n:]],chosen_pool_id=-1)
  same=dict(contract,visible=[{k:v[k] for k in VISIBLE} for v in poison['outcomes'][:n]])
  assert describe(same,meta)==f
  swapped=copy.deepcopy(contract);t=swapped['physical']['topo_vect'];swapped['physical']['topo_vect']=np.where(t>0,3-t,t);assert describe(swapped,meta)==f
  try:describe(dict(contract,future_label=not r['target']['stop_label']),meta)
  except AssertionError:pass
  else:raise AssertionError('Future target accepted')
  descriptors.append(f);public.append(dict(sample_id=r['meta']['sample_id'],features=f))
 with gzip.open(O/'public_state_descriptors.jsonl.gz','wt',encoding='utf8') as z:
  for r in public:z.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 save('state_context.json',state_context)
 save('row_index.json',[dict(sample_id=r['meta']['sample_id'],call_id=r['meta']['call_id'],date=r['meta']['date'],partition=p,visible_count=r['meta']['visible_count']) for r,p in zip(rows,parts)])
 assert not calls['forbidden']
 save('extraction_review.json',dict(passed=True,serialized_vector_roundtrips=610,library_commands_checked=909,native_named_slice_checks=915,
  graph_station_distance_checks=305*118*2,public_rows=726,suffix_isolation=726,bus_label_swap=726,future_key_rejections=726,environment_or_physical_calls=dict(calls)))
 verify();save('state.json',dict(stage='extracted_pending_comparison'));print('GRID148_EXTRACTED',len(rows),len(physical))

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
