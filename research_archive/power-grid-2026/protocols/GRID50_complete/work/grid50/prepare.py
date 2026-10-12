"""Last activation-only followup;do not begin an activation/constant sweep."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid50'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
A=read(ROOT/'outputs/grid49/pilot_audit.json');assert not A['both_new_seeds_qualified']
assert next(r for r in A['results'] if r['method']=='FREE_GNN0')['passed_quality']
assert not next(r for r in A['results'] if r['method']=='FREE_GNN1')['passed_quality']
assert read(ROOT/'outputs/grid48/audit_fit.json')['passed']
for name in ['train.py','offline.py','audit_fit.py','online_adapter.py']:
    s=(ROOT/'work/grid48'/name).read_text(encoding='utf-8')
    s=s.replace('free_residual','compressed_residual').replace('FreeResidualScorer','CompressedResidualScorer')
    s=s.replace("'outputs/grid48'","'outputs/grid50'").replace('FREEGNN','COMPRESSEDGNN')
    s=s.replace('free residual','soft-compressed residual').replace('Free residual','Soft-compressed residual')
    s=s.replace('free-residual','soft-compressed-residual').replace('GRID48','GRID50')
    assert not (HERE/name).exists();(HERE/name).write_text(s,encoding='utf-8')
D={**read(ROOT/'outputs/grid48/design.json'),'stage':'GRID50_ONE_SOFT_COMPRESSION_FOLLOWUP',
   'change':'Only correction activation:original2*tanh(raw/2)becomes2*asinh(raw/2). Same originalscale2/local slope1,graph parameters/inputs/zerohead/initialweights/PPOprior/KL0.1,target/optimizer/shuffle/2seeds/64epochs. No new cutoff,guard,modules,teachers or RL.',
   'motivation':'GRID49 completely free residual improved one seed but overcorrected another. Test sublinear correction growth without reintroducing hard peak-mass ceiling. This is a bounded engineering hypothesis,not known power-grid theory or guaranteed calibration.',
   'evaluation':'Bothfixed fits then26olddevelopment cached checks under ORIGINALFASTGRID28. Bothknown1733regressions required before any bounded closedloop. Everyseed/oldmodel retained. No test-basedcheckpoint or scalar selection.',
   'scope':'One final activation-only followup;no activation/scale sweep. If mixed or failing,retain original and investigate training coverage instead. GRID44allfamilies/variants not trained or newly evaluated.'}
OUT.mkdir(exist_ok=False);(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=list(HERE.glob('*.py'))+[ROOT/'work/grid21/prior.py',ROOT/'work/grid20/components.py',ROOT/'outputs/grid49/pilot_audit.json',ROOT/'outputs/grid48/qualification.json',ROOT/'outputs/grid21/training_finished.json']
files += [ROOT/p for p in read(ROOT/'outputs/grid48/qualification.json')['source_hashes']]
for seed in [0,1]:files += [ROOT/f'outputs/grid21/models/gnn_seed{seed}/{name}.pt' for name in ['initial','final']]
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN2FITS:one final activation-only soft-compression followup;originalfastcontroller default;no GRID44reuse.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'fits':2,'scale_unchanged':2,'original_fast_guard_retained':True}))
