"""Freeze feature definitions and source hashes; no feature/label association fitted."""
from pathlib import Path
import json,hashlib,datetime,shutil
from feature_core import FEATURES
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid98_feature_enrichment'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'freeze.json').exists()
    old=ROOT/'outputs/grid97_effect_learning'
    for n,h in read(old/'freeze.json')['sha256'].items():assert sha(ROOT/n)==h,n
    assert read(old/'review.json')['passed'] and read(old/'readout_verification.json')['passed']
    cases=read(ROOT/'outputs/grid96_value_learning/design.json')['cases'];assert len(cases)==56
    paths=[p for p in W.glob('*.py')]
    paths += [old/n for n in ['dataset.json','design.json','freeze.json','review.json','predictions.json','results.json']]
    paths += [ROOT/n for n in ['outputs/grid96_value_learning/design.json','outputs/grid96_value_learning/labels.json','outputs/grid28/offline.json','work/grid94_predictability/data.py','work/grid91_mechanism/policies.py','work/grid89_review/reference_logic.py','work/grid01/trace_agent.py','work/grid08/common.py','work/grid02/ljn_nn/assets/action_N1_unsafe.npz']]
    for c in cases:
        p=ROOT/c['source'];paths += [p,p.with_name(p.stem+'_state.npz'),p.parent/'metadata.json',ROOT/c['environment']/'grid.json']
    paths=list(dict.fromkeys(paths))
    d=dict(stage='GRID98_PRE_COMPLETION_FEATURE_ENRICHMENT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User: ok来补一下; augment meaningful pre-completion features, no refit in this stage',
        samples=56,new_features=FEATURES,original_features=read(old/'design.json')['features'],original_feature_count=15,new_feature_count=len(FEATURES),
        labels='All 56 and original economic/physical labels preserved in separate file; no model fitting, no feature selection by labels, no prediction or new physical episodes.',
        status='Exploratory feature design after GRID97 exposed results; not independent evidence or proven improvement.',
        structure='Use current state, masked/deduplicated public action commands and known candidate membership. Intended bus assignment fills unspecified commands from current topology, applies explicit bus changes and line disconnects; partition features describe action intentions, not solved future power flow. Same-bus pair relation is invariant to swapping bus1/bus2.',
        missing='Absent runner-up: gap0 plus runnerup_missing1. No legal feedback: std0 plus legal_fraction0. Other-station best gap missing:0, original station_support_missing supplies indicator. No overloaded live lines: overload_share0.',
        duplicated_information='Two clearance features are deterministic transforms of original rho/limit, not new information. Keep original constant prefix_valid for exact compatibility; explicitly list constants after extraction.',
        reconstruction='One existing Grid2Op template for vector decoding only; no reset, step, simulate or optimizer calls. Existing library/base mask and alias hashes verified for each case.',
        leakage_checks=['extractor accepts only redacted pre-completion prefix and public metadata','delete all hidden feedback: identical features','poison all hidden rho/reward/admissibility: identical features','shuffle visible rows back to chronological prefix required; invalid appended row rejected','bus1/2 swap invariance','independent arithmetic checks and original15 exact reproduction'],
        prohibitions=['no new physical forecasts','no training or deployment','no changed labels/splits/cases','no target-based feature ranking/selection','no claim new features improve accuracy before separate evaluation'])
    OUT.mkdir(exist_ok=True)
    (OUT/'design.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');paths.append(OUT/'design.json')
    freeze=dict(utc=d['utc'],sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},historical_grid97_freeze_sha256=sha(old/'freeze.json'))
    (OUT/'freeze.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    archive=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID98_feature_enrichment/frozen';archive.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+[OUT/'design.json',OUT/'freeze.json']:shutil.copy2(p,archive/p.name)
    print('Frozen feature definitions and',len(paths),'source items; no fits or predictions')
if __name__=='__main__':main()
