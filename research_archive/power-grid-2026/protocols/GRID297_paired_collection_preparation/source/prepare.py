"""Freeze a reviewable implementation; no model loads or environment imports."""
from pathlib import Path
import json,hashlib,shutil,subprocess,collections
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid297_paired_collection_preparation';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID297_paired_collection_preparation'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not O.exists();assert read(R/'outputs/grid296_data_route_proposal/complete.json')['passed']
O.mkdir();prior=read(R/'outputs/grid291_control_preparation/design.json');roles=read(R/'outputs/grid296_data_route_proposal/pool_role_counts.json');refs=read(R/'outputs/grid291_control_preparation/model_references.json');assert len(roles)==len(refs)==12
items=[];pairs=[]
for role in roles:
 pool=role['pool'];ref=next(x for x in refs if x['pool']==pool)
 for date in sorted(role['fit_dates']+role['cal_dates']):
  assert date not in role['outer_dates_excluded'];j=len(pairs);arms=['STOP_NEXT','CONT_NEXT'] if j%2==0 else ['CONT_NEXT','STOP_NEXT']
  pair=dict(pair_index=j,pool=pool,date=date,week=date+'_100',role='fit8' if date in role['fit_dates'] else 'cal4',KEEP=role['KEEP'],arms=arms);pairs.append(pair)
  for k,arm in enumerate(arms):
   item=dict(prior['execution_order'][0]);item.update(index=len(items),pool=pool,date=date,week=date+'_100',model_seed=ref['seed'],method=arm+'__'+pool,arm=arm,completion_mode='BOUNDED261_SINGLE_INTERVENTION',partition='own_training_'+pair['role'],pair_index=j,within_pair=k,repeat=0,KEEP=role['KEEP']);items.append(item)
assert len(items)==288 and len(pairs)==144 and len({(i['week'],i['method']) for i in items})==288
assert collections.Counter(p['role'] for p in pairs)=={'fit8':96,'cal4':48}
design=dict(id='GRID297_PAIRED_SUFFIX_COLLECTION',status='PREPARATION_ONLY_NO_PHYSICAL_AUTHORIZATION',environment=prior['environment'],environment_seed=0,execution_order=items,pairs=pairs,caps=dict(episode_s=1200,wall_s=28800,physical_steps=580896,public_forecasts=9000000),timing=prior['timing'],original_method_contract=prior['method_contract'],intervention='Exactly first episode nativeSTOP with pending>0; force STOP or one original batch<=64 only then resume original261. Public rule, no outcome selection.',label_join='All source receipts first published, only then separately authorized saved target derivation; no targets or policy returns computed by manager.',no_event='Missing trigger is recorded, no replacement or resampling; matching absent-event episode signatures required.',partial='Stop and preserve at first failed resource/engineering guard; no automatic retries/resume/renewed clocks.',reference_limits='Not independent OOF, honest forest, new final test or latency confirmation. Original own12/internalcal4 and exposed development ancestry retained.',preparation_allowed='Static sources/hash binding, generated-source compilation, synthetic stub flows, missing-auth process rejection. No torch/NN/head loads or predictions, simulate, solver or physical smoke test.')
save(O/'design.json',design);save(O/'model_references.json',refs);save(O/'pool_roles.json',roles);save(O/'data_collection_proposal.json',read(R/'outputs/grid296_data_route_proposal/data_collection_proposal.json'));save(O/'resource_estimates.json',read(R/'outputs/grid296_data_route_proposal/resource_estimates.json'))
models={p:h for ref in refs for p,h in ref['hashes'].items()}
for p,h in models.items():assert sha(R/p)==h,p
save(O/'model_freeze.json',models)
old=read(R/'outputs/grid291_control_preparation/source_freeze.json')
code={p:h for p,h in old.items() if p.startswith('work/') and p.endswith('.py')}
for p,h in code.items():assert sha(R/p)==h,p
data={};environment=R/prior['environment']
for p in environment.iterdir():
 if p.is_file():data[p.relative_to(R).as_posix()]=sha(p)
for date in sorted({p['date'] for p in pairs}):
 p=R/f'outputs/grid100_data_foundation/generation_receipts/{date}_100.json';receipt=read(p);assert receipt['item']['partition']=='train' and receipt['item']['scenario']==date+'_100'
 data[p.relative_to(R).as_posix()]=sha(p)
 for rel,h in receipt['files'].items():assert (R/rel).is_file();data[rel]=h
save(O/'data_freeze.json',data)
paths=list(W.glob('*.py'))+[O/n for n in ['design.json','model_references.json','model_freeze.json','pool_roles.json','data_freeze.json','data_collection_proposal.json','resource_estimates.json']]
code.update({p.relative_to(R).as_posix():sha(p) for p in paths});save(O/'source_freeze.json',code);save(O/'code_freeze.json',code)
# Pin exact already-published model bytes by matching their immutable hash; no
# checkpoint/array deserialization. Copies are referenced, not uploaded again.
tracked=subprocess.check_output(['git','-C',str(I),'ls-files','research_archive/power-grid-2026/protocols']).decode().splitlines();by_name={}
for rel in tracked:by_name.setdefault(Path(rel).name,[]).append(rel)
published={}
for local,h in models.items():
 candidates=[p for p in by_name.get(Path(local).name,[]) if (I/p).is_file() and (I/p).stat().st_size==(R/local).stat().st_size]
 match=next((p for p in candidates if sha(I/p)==h),None)
 if match is None:raise ValueError('no exact published payload binding: '+local)
 published[local]=dict(archive_path=match,sha256=h)
save(O/'published_model_bindings.json',published)
S=A/'source';S.mkdir(parents=True,exist_ok=False)
for p in W.glob('*.py'):shutil.copy2(p,S/p.name)
for p in O.iterdir():
 if p.is_file():shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
print('GRID297_SOURCE_FROZEN',len(items),len(models),len(code),len(data))
