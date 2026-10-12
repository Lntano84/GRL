"""One matched old/new comparison, with equivalent prior cache for all users."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid55'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not OUT.exists();Q=read(ROOT/'outputs/grid54/qualification.json');assert Q['passed'] and Q['cases']==130
base=read(ROOT/'outputs/grid52/design.json');assert read(ROOT/'outputs/grid52/review.json')['passed_saved_data']
methods=[dict(m) for m in base['methods']]
methods += [dict(m) for m in read(ROOT/'outputs/grid51/design.json')['methods'] if m['method'] in ['LOCAL_GNN0','LOCAL_GNN1']]
assert len(methods)==8 and len({m['method'] for m in methods})==8
weeks=base['weeks'];assert len(weeks)==8
order=[]
for repeat in [0,1]:
    for n,week in enumerate(weeks):
        shift=(n+4*repeat)%len(methods);sequence=methods[shift:]+methods[:shift]
        if repeat:sequence=list(reversed(sequence))
        order += [{**m,'week':week,'repeat':repeat,'stage_phase':'pilot'} for m in sequence]
D={**base,'stage':'GRID55_MATCHED_OLD_NEW_EQUIVALENT_CACHE_V1','methods':methods,'execution_order':order,
   'scope':'128fresh serial runs on ALL8already exposed GRID52dates. Old2/new2seeds are configuration comparison;three main baseline families remain FULL/PPO/SoftGNN2seeds. No new dates,training,teachers,models,thresholds or outcome-based filtering.',
   'timing':'Rotate/reverse8implementations over8datefamilies/two timing repeats. One fresh single-thread worker each. All original DC/safeQP caches/controller/legal masks/GRID28faststop unchanged. Equivalent static-prior cache applied to ALL original-prior users,including old/new/PPO. Lazy first build/fresh-field validation paid in online control;setup/physical/export separate.',
   'choice_before_access':'All original/soft-compressed weights frozen and preserved. No selecting one seed or weaker SoftGNN. No new baseline families. This is matched version selection and engineering ablation,not an independent confirmation.',
   'sealed_use':'GRID52dates already exposed;not called sealed/unseen again. They remain excluded from training/tuning. Historical GRID44/51/52results are immutable.',
   'version_selection':'Apply same original quality/survival/time investment screen to EACH version/BOTH seeds,relativeFULL. If only one version meets it,prefer that version as the engineering candidate. If both meet it,require>=2%seed-average paired control improvement,>=6/8positive families,and BOTHmatched seed mean directions positive to prefer faster version. Symmetric rule for old. Otherwise retain both/no clear preferred version;report quality-time frontier,do not tune new parameters or pick a seed. Screening is not significance,equivalence or paper acceptance.',
   'caps':{'episode_s':1500,'wall_s':18000,'physical_steps':270000,'public_forecasts':2500000,'new_fits':0,'reserved_evaluations':0}}

for name in ['worker.py','run.py','audit.py','preflight.py']:
    source=(ROOT/'work/grid52'/name).read_text(encoding='utf-8')
    source=source.replace('grid52','grid55').replace('GRID52','GRID55')
    if name=='run.py':
        key='    install_free_adapter(engine.Search)';assert source.count(key)==1
        source=source.replace(key,key+"\n    sys.path.insert(0,str(ROOT/'work/grid54'))\n    from cache_prior import install_prior_cache\n    prior_cache_modules=install_prior_cache()")
        key="'safe_cache_counts':[dict(getattr(m,'_grid30_safe_counts',{})) for m in optimizers],";assert source.count(key)==1
        source=source.replace(key,key+"\n            'prior_cache_modules':prior_cache_modules,\n            'prior_cache_stats':None if not hasattr(worker,'unused') and not hasattr(optimizers[0], '_grid54_unused') else None,")
        # Capture the control instance without changing decisions or measured work.
        source=source.replace('optimizers=[];components=[];solvers=[];', 'optimizers=[];controls=[];components=[];solvers=[];')
        source=source.replace('        old_init(self,*a,**kw)\n', '        old_init(self,*a,**kw)\n        controls.append(self)\n')
        source=source.replace("'prior_cache_stats':None if not hasattr(worker,'unused') and not hasattr(optimizers[0], '_grid54_unused') else None,",
            "'prior_cache_stats':[None if not hasattr(c,'_grid54_prior_metadata') else c._grid54_prior_metadata.stats() for c in controls],")
        source=source.replace("'reserved_evaluations':len(summaries)","'reserved_evaluations':0")
        source=source.replace('all6arms shareDC+safe cache;first reserved confirmation','all8arms shareDC+safe+equivalent prior cache;already exposed matched comparison')
    if name=='audit.py':
        source=source.replace('==96','==128').replace("'runs':96","'runs':128").replace('96条','128条').replace('96reserved-family','128matched-version')
        source=source.replace("'reserved_evaluations':96","'reserved_evaluations':0")
        key="pairs_to_compare += [(g,o) for g in ['COMPRESSED_GNN0','COMPRESSED_GNN1'] for o in ['LOCAL_PPO','SOFTGNN0','SOFTGNN1']]"
        assert source.count(key)==1
        source=source.replace(key,"pairs_to_compare += [(g,o) for g in ['LOCAL_GNN0','LOCAL_GNN1','COMPRESSED_GNN0','COMPRESSED_GNN1'] for o in ['LOCAL_PPO','SOFTGNN0','SOFTGNN1']]\n    pairs_to_compare += [('COMPRESSED_GNN0','LOCAL_GNN0'),('COMPRESSED_GNN1','LOCAL_GNN1')]")
        key="        profile=read(folder/'components.json');active=[c for c in profile['calls'] if c['in_control']]"
        assert source.count(key)==1
        source=source.replace(key,key+"\n        expected_prior=s['method'] not in ['FULL','SOFTGNN0','SOFTGNN1'] and bool(calls)\n        stats=profile['prior_cache_stats'];assert len(stats)==1\n        assert bool(stats[0] is not None)==expected_prior\n        if expected_prior:assert stats[0]['builds']==1 and stats[0]['validations']==len(calls)-1\n        assert all(m in profile['prior_cache_modules'] for m in ['prior','guard_adapter','online_adapter'])")
        key="    R={'passed_saved_data':True,"
        assert source.count(key)==1
        extra='''    old_comparisons=[c for c in comparisons if c['method'] in ['LOCAL_GNN0','LOCAL_GNN1'] and c['opponent']=='FULL']
    version_family=[]
    for week in D['weeks']:
        vals=[r['mean_time_reduction'] for c in comparisons if (c['method'],c['opponent']) in [('COMPRESSED_GNN0','LOCAL_GNN0'),('COMPRESSED_GNN1','LOCAL_GNN1')] for r in c['families'] if r['week']==week]
        assert len(vals)==2
        if all(v is not None for v in vals):version_family.append({'week':week,'new_over_old_mean_time_reduction':mean(vals)})
    paired_seed=[next(c['macro_control_reduction'] for c in comparisons if c['method']==f'COMPRESSED_GNN{s}' and c['opponent']==f'LOCAL_GNN{s}') for s in [0,1]]
    old_pass=all(r['investment_screen_passed'] for r in summaries if r['method'] in ['LOCAL_GNN0','LOCAL_GNN1'])
    new_pass=ownsummary['both_model_seeds_screen_passed']
    version_mean=mean([r['new_over_old_mean_time_reduction'] for r in version_family])
    positive=sum(r['new_over_old_mean_time_reduction']>0 for r in version_family)
    negative=sum(r['new_over_old_mean_time_reduction']<0 for r in version_family)
    selected='NO_CLEAR_PREFERRED_VERSION'
    if new_pass and not old_pass:selected='COMPRESSED_CANDIDATE_QUALITY_SCREEN'
    elif old_pass and not new_pass:selected='ORIGINAL_CANDIDATE_QUALITY_SCREEN'
    elif old_pass and new_pass and len(version_family)==8:
        if version_mean>=.02 and positive>=6 and all(v>0 for v in paired_seed):selected='COMPRESSED_CANDIDATE_PAIRED_SPEED'
        elif version_mean<=-.02 and negative>=6 and all(v<0 for v in paired_seed):selected='ORIGINAL_CANDIDATE_PAIRED_SPEED'
    version={'old_both_seeds_screen_passed':old_pass,'new_both_seeds_screen_passed':new_pass,
             'new_over_old_seed_average_time_reduction':version_mean,'family_bootstrap95':interval([r['new_over_old_mean_time_reduction'] for r in version_family]),
             'matched_seed_mean_time_reductions':paired_seed,'positive_families':positive,'negative_families':negative,
             'families':version_family,'engineering_preference':selected,'no_default_code_changed':True}
'''
        source=source.replace(key,extra+key)
        source=source.replace("'own_seed_average':ownsummary,","'own_seed_average':ownsummary,'version_selection':version,")
        source=source.replace('Fresh second date-family confirmation of soft-compressed candidate of frozen controller/model;', 'Matched old/new frozen-model comparison on already exposed dates;shared equivalent prior cache;')
        source=source.replace('八个预留日期家族确认','八个已暴露日期的旧新配对与共有缓存比较')
        source=source.replace("save('review.json',R);", "save('review.json',R);save('version_selection.json',version);")
        source=source.replace("lines+=['',f\"两软压缩模型种子均通过投资筛查", "lines+=['',f\"工程选型：{selected}；新对旧平均控制时间减少：{version_mean}。\",'',f\"两软压缩模型种子均通过投资筛查")
    if name=='preflight.py':
        source=source.replace('==96','==128').replace("'runs':96","'runs':128")
        source=source.replace(" and not (OUT/'extraction_complete.json').exists()",'')
        source=source.replace('no reserved-data observation or model inference','metadata/formula checks before this exposed-date comparison')
    (HERE/name).write_text(source,encoding='utf-8')
OUT.mkdir();(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=list(HERE.glob('*.py'))+list((ROOT/'work/grid54').glob('*.py'))+[ROOT/'outputs/grid54/qualification.json',OUT/'design.json']
files += [ROOT/p for p in read(ROOT/'outputs/grid52/code_freeze.json')]
files += [ROOT/m['checkpoint'] for m in methods if m['checkpoint']]
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in dict.fromkeys(files)},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:128serial matched old/new runs,ALL8exposed dates,2timingrepeats,three baseline families,no training.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'runs':128,'weeks':weeks,'implementations':len(methods),'new_training':0,'new_dates':0}))
