"""Existing-input, exposure and value-range diagnosis. No model or optimizer."""
from pathlib import Path
import sys,collections,datetime,shutil,gzip
R=Path('D:/Codex/GRL研究');sys.path.insert(0,str(R/'work/grid122_false_stop_audit'))
import common122 as nn
sys.path.insert(0,str(R/'work/grid123_event_auxiliary'))
import common123 as src
import numpy as np
W=Path(__file__).parent;O=R/'outputs/grid125_value_underestimate';P=R/'outputs/grid124_shared_gradient_audit';S=src.O
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID125_value_underestimate'
read=src.read;save=src.save;sha=src.sha
ARMS=['NET_ZERO','AUX_NETSIGN','AUX_GAIN'];GROUPS=['baseCONT_auxSTOP__nextPositive','baseCONT_auxSTOP__nextNonpositive','baseSTOP_auxCONT__nextPositive','baseSTOP_auxCONT__nextNonpositive']

def verify():
    for fn in ['code_freeze.json','data_freeze.json']:
        for n,h in read(O/fn).items():assert sha(R/n)==h,n
    src.verify();nn.verify();src.sealed()

def archive(phase):
    a=A/phase;a.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,a/p.name)
    save(a/'archive_manifest.json',{p.name:sha(p) for p in a.iterdir() if p.is_file()})

def docs(message,steps):
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+message+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(steps,encoding='utf8')

def arrays():
    cases=src.pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};meta={part:read(S/f'{part}_metadata.json') for part in ['train','validation']};raw={};gain={};charge={};D=read(S/'design.json')
    for part in meta:
        raw[part]=np.stack([cm[m['sample_id']]['states'][m['position']] for m in meta[part]]).astype(float)
        gain[part]=np.array([cm[m['sample_id']]['gains'][m['position']]/D['gain_scale'] for m in meta[part]])
        charge[part]=np.array([D['cost_per_query']*cm[m['sample_id']]['queries'][m['position']]/D['gain_scale'] for m in meta[part]])
        assert (gain[part]>=0).all() and (charge[part]>0).all()
        assert all(m['next_queries']==cm[m['sample_id']]['queries'][m['position']] for m in meta[part])
        assert np.array_equal((gain[part]-charge[part]).astype('f'),np.load(S/f'{part}_data.npz')['net'])
    return cases,meta,raw,gain,charge

def divergence_rows():
    with gzip.open(P/'first_divergences.jsonl.gz','rt',encoding='utf8') as f:return [r for ln in f if (r:=__import__('json').loads(ln))['arm']=='AUX_GAIN' and r['group']!='same_actions']

def scope(split,meta,raw):
    if split=='full':
        ids=np.arange(len(meta['train']));held=np.arange(len(meta['validation']));part='validation';z=np.load(R/'outputs/grid102_search_rl/normalization.npz')
    else:
        dates=read(S/'design.json')['folds'][int(split[1:])]['held_dates'];ids=np.array([i for i,m in enumerate(meta['train']) if m['date'] not in dates]);held=np.array([i for i,m in enumerate(meta['train']) if m['date'] in dates]);part='train';z=np.load(src.CV/f'fold{split[1:]}_normalization.npz')
    fit_pre=(raw['train'][ids]-z['mean'])/z['scale'];held_pre=(raw[part][held]-z['mean'])/z['scale'];x=np.clip(fit_pre,-20,20).astype('f').astype(float);h=np.clip(held_pre,-20,20).astype('f').astype(float)
    return ids,held,part,x,h,fit_pre,held_pre

def prepare():
    assert not O.exists();O.mkdir(parents=True);src.verify();nn.verify();src.sealed();assert read(P/'complete.json')['passed'];arrays()
    save(O/'plan.json',dict(id='GRID125_VALUE_UNDERESTIMATE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User: 那开始查询吧, approving visible-input/support/underestimate diagnosis before new training.',
      scope='All frozen GRID124 AUX_GAIN first changed trajectories in original validation and conditional OOF,all profitable misses and all opposite outcomes;all45 saved NET_ZERO/AUX_NETSIGN/AUX_GAIN score arrays.',
      nearest='Original41 normalized/clipped Euclidean distances,fit dates only,all exact nearest ties;closest profitable/nonprofitable references;fit leave-whole-call-out distance95 as descriptive coverage. Reproduce old122fold rows,add full-training/validation coverage. No neighbor rule,k/metric search or fitted model.',
      input='All41 public summary fields plus fixed8 paid-feedback/action-plan descriptors fromGRID122,clipping counts and exact raw/input collisions. Same metadata/order;no extra features added to models.',
      support='Keep77 profitable-miss prefix-seed records visible and include every other divergence category. Aggregate within physicalcall before comparing groups;calls/prefixes not independent,labels onlyposthoc.',
      training='Existing exact expected probabilities and sampled exposure counts. Inspect positive net magnitude>1 because originalHuberbeta=1,not a chosen trigger. Report mean net and mean clipped net at fixed zero prediction perfitpool/perdate. These are label functionals,not fitted constant policies.',
      range='Best observed admissible reward never decreases,so normalized net target>=-known_charge. Count saved prediction violations with1e-6 numerical tolerance;check projectionmax(q1,-charge) leaves every q1>0 action unchanged. It is NOT a new search policy or proposed efficacy result.',
      known_limits='Original head1net andhead0event do not include an independent conditional-gain-size forecast;cannot uniquely assign each mistake to probability versus magnitude. Neighbor absence is not proof unavailable information;neighbor presence is not prooflearnability. ExistingGRID108MSE/longtrainingandGRID120weightedclassificationfailuresretained.',
      excluded=['new model forwards','gradient/backward','optimizer','weight updates','new fit','new simulation','new data','threshold sweep','final8test','paper/figures']))
    files=[O/'plan.json',P/'first_divergences.jsonl.gz',P/'head_consistency.json',P/'review_addenda.json',P/'complete.json',S/'predictions.npz',S/'fits.json',S/'design.json',S/'review.json',S/'code_freeze.json',S/'data_freeze.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid102_search_rl/normalization.npz',src.CV/'train_arrays.npz',src.CV/'metadata.json',src.CV/'design.json',nn.O/'support_rows.json',nn.O/'fit_leave_call_out_neighbors.json',nn.O/'fold_coverage.json',R/'outputs/grid108_loss_budget/GRID108_report.md',R/'outputs/grid120_decision_objective/GRID120_report.md']
    for part in ['train','validation']:files.extend([S/f'{part}_data.npz',S/f'{part}_metadata.json',src.S/f'{part}_public.pkl'])
    files+=list(src.CV.glob('fold*_normalization.npz'))+list(S.glob('exposure_*.npz'))+[src.OLD/f'training_exposure_seed{s}.npz' for s in range(3)]+[src.CV/f'exposure_fold{f}__HUBER_seed{s}.npz' for f in range(4) for s in range(3)]
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),nn.W/'common122.py',src.W/'common123.py']})
    save(O/'preflight.json',dict(passed=True,no_physical_or_model_calls=True,new_fits=0));archive('before')
    docs('## 2026-10-23 — GRID125 high-value false-stop support/range diagnosis frozen\n\nUserapprovesvisible-input/support/underestimateinvestigation,noretraining. AllGRID124AUX_GAINfirstdivergences,77profitablemissprefix-seedrecords/allcontrols. Original41nearesttiesandfitcall-out95,all8fixedpublicdescriptors,exactsavedexposure,Huberzero-gradientlabelstatistics,knownchargelowerbound/projection-signidentity. ReproduceGRID122foldsupport. No model/backward/fit/physical/test/newrule;oldMSEandweightedclassificationfailuresretained.',
      '# Next steps — GRID125高收益误停核查\n\n1. 核对已有输入、训练覆盖和实际抽样，逐项保留正反两类分歧。\n2. 检查已知查询费对应的价值下界，以及稀疏大收益在原Huber目标中的作用。\n3. 区分数据证据与解释假设；不自动重新训练或改触发条件。\n')
    print('GRID125_FROZEN',flush=True)

if __name__=='__main__':prepare()
