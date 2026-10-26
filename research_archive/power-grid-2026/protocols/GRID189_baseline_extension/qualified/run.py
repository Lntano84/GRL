"""Design and fit-side label support for at-most-one-batch NET extension."""
from pathlib import Path
import collections,hashlib,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid189_baseline_extension';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID189_baseline_extension'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def sources():return [R/'outputs/grid168_one_sided_transfer',*[R/f'outputs/grid170_remaining_date_folds/fold{f}' for f in [1,2,3]]]
def sealed():
    d=read(R/'outputs/grid100_data_foundation/design.json')
    for x in d['scenario_plan']:
        if x['partition']=='test':
            assert not (R/d['environment']/'chronics'/x['scenario']).exists()
            assert not (R/'outputs/grid100_data_foundation/datasets'/x['scenario']).exists()
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    sealed()
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed();assert read(R/'outputs/grid188_repeated_cases/history_reconciliation.json')['passed']
    save('design.json',dict(id='GRID189_BASELINE_EXTENSION_DESIGN',authorization='User:好的继续吧; concretize minimal change and qualify existing label support,not a newfit/physical authorization.',
        behavior='Freeze original NET parameters and execute unchanged to its first STOP. If candidates remain, invoke a separate gate once:STOP returns original best;EXTEND buys at most the next64locked-station representatives and returns best observed,without a second gate or resuming NET. Exhausted baselines bypass gate.',
        purpose='Recover some skipped local action improvements without cancelling any original NET continuation. This is optional extension,not faster-than-NET search.',
        boundaries='For same fixed input/controller/initial station/candidate order/aliases and admissibility+float32reward+ID ranking: Q_NET<=Q_EXT<=Q_FULL_LOCAL; extra<=64; rank(best_EXT)>=rank(best_NET). If NET already chose full-local optimum,extension cannot change it; hence full-local action-match indicator cannot decrease on that fixed candidate problem. Does not guarantee physical full-week survival,cost or real latency.',
        cost='Gate inference once only on eligible NETSTOP; original NET calls unchanged. Extra candidatequery/forecast/feature/inference cost must be counted. No guaranteed speedup vs NET; assess quality/compute tradeoff and FRAME reference separately.',
        difference='Unlike GRID123 shared auxiliary,original value weights and representation are frozen and new gate controls only an optional batch. Unlike GRID184 bilateral consensus,never turns a NETCONTINUE into STOP. Unlike GRID167 one-sided earlystop,the direction is only optional laterstop. Unlike GRID126/127,target choice cannot replace baseline value supervision.',
        label='At first NETSTOP with remaining candidates: chosen_after_next_batch != current_chosen. Preserve continuous next_batch_gain and oldNET as separate private evaluation fields. No final-full-suffix label copied onto next batch. Zero-gain tie-ID changes separate.',
        inputs='Candidate gate may reuse original current global41,paid local feedback,next<=64plans and frozenNET representation/currentmargin. No new future feedback/date/ID input. Architecture and loss to be fixed after fit-only eligibility counts; no gate fitted here.',
        qualification='All4existing folds x3seeds,only each fold12fitdates. One firstSTOP per original startingprefix; no classfiltering. Report date/physicalgroup/label support,positive gain andprofit separately. Fit-side endpoint-selection is in-sample with respect to frozen baseline; acknowledge gate deployment-distribution shift.',
        future_evaluation='When trained:all16OOF dates/all3seeds,paired BASE_F,ALWAYS_ADD1_AT_FSTOP and selective extension. Old auxiliary and fixedBATCH1 separately. Report rawgain/full-localactionmatch/query/NET and inference overhead; fixedADD1 essential to distinguish learning from extraquery. No allwin or5percent gate.',
        prohibited=['optimizer/modelforward/newfeature extraction here','held extension strategy scoring or oracle selection','new physical/data/final8/paper','threshold/rule search','automatic scale']))
    files=[W/'run.py',O/'design.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid113_relational_learning/train_public.pkl',R/'outputs/grid188_repeated_cases/analysis.json',R/'outputs/grid188_repeated_cases/history_reconciliation.json',R/'outputs/grid187_action_invariance_target/preflight.json']
    for p in sources():files += [p/n for n in ['fold_arrays.npz','state_scores.npz','fits.json','design.json']]+list((p/'models').glob('FACTOR_seed*.pt'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('frozen');print('GRID189_DESIGN_FROZEN')
def qualify():
    verify();assert not (O/'fit_support.json').exists();allcases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'];meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};price=read(R/'outputs/grid102_search_rl/design.json')['cost_per_query'];scale=read(R/'outputs/grid102_search_rl/design.json')['gain_scale'];records=[];pools=[];checks=collections.Counter()
    for f,path in enumerate(sources()):
        z=np.load(path/'fold_arrays.npz');fi=set(z['fit_ids'].tolist());fitdates={meta[i]['date'] for i in fi};helddates={meta[i]['date'] for i in z['held_ids']};assert not fitdates & helddates and len(fitdates)==12;cases=[c for c in allcases if c['date'] in fitdates];scores=np.load(path/'state_scores.npz')
        for seed in range(3):
            rows=[];exhausted=0
            for c in cases:
                ix=[lookup[c['sample_id'],j] for j in range(len(c['states']))];assert set(ix)<=fi
                q=scores[f'FACTOR_seed{seed}'][ix];go=q[:,1]>q[:,0];stop=next((j for j,b in enumerate(go) if not b),len(go));base_q=int(c['queries'][:stop].sum());base_gain=float(c['gains'][:stop].sum());full_q=int(c['queries'].sum());full_gain=float(c['gains'].sum())
                checks['fit_starting_problems']+=1
                if stop==len(go):exhausted+=1;assert c['chosen'][stop]==c['chosen'][-1];continue
                m=int(c['queries'][stop]);g=float(c['gains'][stop]);rank_changed=int(c['chosen'][stop+1])!=int(c['chosen'][stop]);tie=rank_changed and g==0
                for gate in [0,1]:
                    end=stop+gate;eq=int(c['queries'][:end].sum());eg=float(c['gains'][:end].sum())
                    assert base_q<=eq<=full_q and 0<=eq-base_q<=64 and base_gain<=eg<=full_gain
                    if c['chosen'][stop]==c['chosen'][-1]:assert c['chosen'][end]==c['chosen'][-1]
                    checks['both_gate_branch_bounds']+=1
                row=dict(fold=f,seed=seed,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=stop,metadata_index=ix[stop],fold_fit_only=True,
                    private_label=dict(next_action_changes=rank_changed,next_gain=g,next_NET=(g-price*m)/scale,tie_ID_only=tie),public_bookkeeping=dict(next_queries=m,original_NET_margin=float(q[stop,1]-q[stop,0])))
                rows.append(row);records.append(row)
            dates=sorted({r['date'] for r in rows});groups={d:sorted({r['physical_state_group'] for r in rows if r['date']==d}) for d in dates};sizes=collections.Counter(r['physical_state_group'] for r in rows)
            for r in rows:r['proposed_sampling_mass']=1/(len(dates)*len(groups[r['date']])*sizes[r['physical_state_group']])
            assert abs(sum(r['proposed_sampling_mass'] for r in rows)-1)<1e-10
            per=[]
            for d in dates:
                sub=[r for r in rows if r['date']==d];per.append(dict(date=d,eligible=len(sub),groups=len(groups[d]),action_change_positive=sum(r['private_label']['next_action_changes'] for r in sub),positive_NET=sum(r['private_label']['next_NET']>0 for r in sub)))
            pool=dict(fold=f,seed=seed,fit_dates=sorted(fitdates),held_dates=sorted(helddates),fit_starts=len(cases),eligible=len(rows),exhausted=exhausted,physical_groups=len({r['physical_state_group'] for r in rows}),positive_action_changes=sum(r['private_label']['next_action_changes'] for r in rows),positive_NET=sum(r['private_label']['next_NET']>0 for r in rows),positive_groups=len({r['physical_state_group'] for r in rows if r['private_label']['next_action_changes']}),positive_NET_groups=len({r['physical_state_group'] for r in rows if r['private_label']['next_NET']>0}),zero_gain_tie_changes=sum(r['private_label']['tie_ID_only'] for r in rows),positive_mass=sum(r['proposed_sampling_mass'] for r in rows if r['private_label']['next_action_changes']),per_date=per)
            pool['both_label_classes']=0<pool['positive_action_changes']<len(rows);pools.append(pool)
    assert len(pools)==12 and all(v['both_label_classes'] for v in pools)
    assert checks['fit_starting_problems']==3*3*2378
    save('fit_stop_rows.json',records);save('fit_support.json',dict(pools=pools,all12_both_label_classes=True,proposed_sampling='Equaldate then physicalgroup then eligibleprefix; no positiveclass upweight. Different conditional fitting population,not pure loss-only ablation.',counts_repeated_across_folds_andseeds=True,held_strategy_evaluations=0))
    verify();save('review.json',dict(passed=True,checks=dict(checks),all_rows_from_respective_fit_dates=True,one_STOP_per_fit_start=True,private_outcome_not_a_model_input=True,policy_bounds_checked_for_both_choices=True,new_fits=0,new_model_forwards=0,new_physics=0,held_extension_strategy_scores=0,final8_absent=True));archive('qualified');print('GRID189_QUALIFIED',json.dumps([dict(fold=p['fold'],seed=p['seed'],eligible=p['eligible'],positive_changes=p['positive_action_changes'],positive_groups=p['positive_groups'],positive_NET=p['positive_NET'],positive_mass=p['positive_mass']) for p in pools]))
if __name__=='__main__':globals()[sys.argv[1]]()
