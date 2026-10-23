"""Qualify the revised proposal without optimizer, backward, or fitting."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import pickle
import sys

R = Path('D:/Codex/GRL研究')
W = Path(__file__).parent
O = R / 'outputs/grid118_fixed_base_plan'
sys.path.insert(0, str(R / 'work/grid113_relational_learning'))
from common import read, save, sha, configure, collate, digest, verify, sealed
from fixed_model import load_backbone, FixedBaseCorrection, BRANCH_SEED_OFFSET
import numpy as np
import torch


def main():
    assert not O.exists(), 'Preserve existing proposal receipts'
    O.mkdir(parents=True)
    configure()
    verify()
    sealed()
    assert read(R / 'outputs/grid117_module_diagnosis/review.json')['passed']
    paths = list(W.glob('*.py'))
    paths += [R / f'outputs/grid107_direct_value/{n}' for n in
              ['design.json', 'fits.json', 'train.npz', 'state_predictions.jsonl.gz',
               'validation_results.json']]
    paths += [R / f'outputs/grid113_relational_learning/{n}' for n in
              ['code_freeze.json', 'data_freeze.json', 'preflight.json',
               'train_public.pkl', 'train_metadata.json', 'train_targets.npz']]
    paths += [R / f'outputs/grid117_module_diagnosis/{n}' for n in
              ['decision.json', 'review.json', 'GRID117_report.md']]
    paths += [R / 'outputs/grid115_matched_module_plan/plan.json']
    paths += [R / f'work/{p}' for p in ['grid115_matched_module_plan/matched_model.py',
               'grid113_relational_learning/model.py', 'grid113_relational_learning/common.py',
               'grid112_relational_plan/schema.py']]
    for seed in range(3):
        paths += [R / f'outputs/grid107_direct_value/models/direct_seed{seed}_final.zip',
                  R / f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz']
    save(O / 'input_manifest.json', {p.relative_to(R).as_posix(): sha(p) for p in paths})

    rows = pickle.loads((R / 'outputs/grid113_relational_learning/train_public.pkl').read_bytes())
    meta = read(R / 'outputs/grid113_relational_learning/train_metadata.json')
    assert len(rows) == len(meta) == 3582 and all(x['partition'] == 'train' for x in meta)
    z = np.load(R / 'outputs/grid113_relational_learning/train_targets.npz')
    target = z['target']; prob = z['sampling_probability']
    original = np.load(R / 'outputs/grid107_direct_value/train.npz')
    X = np.stack([x['global41'] for x in rows])
    assert np.array_equal(X, original['X'])
    assert np.array_equal(target, original['target'])
    assert np.array_equal(prob, original['sampling_probability'])
    Y = np.column_stack([np.zeros_like(target), target])
    fits = read(R / 'outputs/grid107_direct_value/fits.json')['records']
    old_stream = read(R / 'outputs/grid113_relational_learning/preflight.json')['sampling_streams']
    old_predictions = {}
    # Historical rows only: no new validation model inference or label evaluation.
    with gzip.open(R / 'outputs/grid107_direct_value/state_predictions.jsonl.gz', 'rt', encoding='utf8') as f:
        for line in f:
            a = json.loads(line)
            if a['partition'] == 'train' and a['method'] in ('DIRECT0', 'DIRECT1', 'DIRECT2'):
                old_predictions[(a['method'], a['sample_id'], a['position'])] = [a['q0'], a['q1']]
    checks = {}; sampling = {}; target_cache = {}
    probe = collate(rows[:32])
    for seed in range(3):
        base = load_backbone(seed).eval()
        assert digest(base) == fits[seed]['final_q_hash']
        assert sum(p.numel() for p in base.parameters()) == 6978
        with torch.no_grad():
            qb = torch.cat([base(torch.from_numpy(X[j:j+32])) for j in range(0, len(X), 32)]).numpy()
        historical = np.asarray([old_predictions[(f'DIRECT{seed}', m['sample_id'], m['position'])] for m in meta], dtype='f')
        old_error = float(np.max(np.abs(qb - historical)))
        assert np.allclose(qb, historical, rtol=1e-6, atol=2e-6)
        # Independent affine/ReLU reference over all training inputs.
        state = {k: v.detach().numpy() for k, v in base.state_dict().items()}
        n = X
        for layer in (0, 2, 4):
            n = n @ state[f'q_net.{layer}.weight'].T + state[f'q_net.{layer}.bias']
            if layer != 4: n = np.maximum(n, 0)
        numpy_error = float(np.max(np.abs(qb - n)))
        assert np.allclose(qb, n, rtol=1e-5, atol=5e-6)
        j = FixedBaseCorrection(seed, 'JOINT')
        f = FixedBaseCorrection(seed, 'FACTOR')
        j.train(); f.train()
        assert not j.base.training and not f.base.training
        assert digest(j.base) == digest(f.base) == digest(base)
        assert digest(j.delta) == digest(f.delta)
        assert all(not p.requires_grad for p in j.base.parameters())
        assert len(j.trainable_parameters()) == len(tuple(j.delta.parameters()))
        assert {id(p) for p in j.trainable_parameters()} == {id(p) for p in j.delta.parameters()}
        assert sum(p.numel() for p in j.trainable_parameters()) == 5938
        assert sum(p.numel() for p in j.parameters()) == 12916
        before_j, before_f = digest(j), digest(f)
        with torch.no_grad():
            b = base(probe['global41'])
            assert torch.equal(j(**probe), b) and torch.equal(f(**probe), b)
            assert not j.correction_targets(probe['global41'], torch.from_numpy(Y[:32])).requires_grad
            assert torch.count_nonzero(j.delta(**probe)) == 0
            # Verify old Huber objective equals a correction-to-fixed-target objective.
            y = torch.from_numpy(Y)
            b = torch.from_numpy(qb)
            d = torch.from_numpy(np.random.default_rng(118 + seed).normal(size=Y.shape).astype('f'))
            old_loss = torch.nn.functional.smooth_l1_loss(b + d, y, beta=1., reduction='none')
            correction_loss = torch.nn.functional.smooth_l1_loss(d, y - b, beta=1., reduction='none')
            loss_error = float((old_loss - correction_loss).abs().max())
            assert torch.allclose(old_loss, correction_loss, rtol=1e-5, atol=2e-6)
        assert digest(j) == before_j and digest(f) == before_f
        target_cache[f'base_q_seed{seed}'] = qb
        target_cache[f'correction_target_seed{seed}'] = Y - qb
        # Draw-only qualification: first-stage hash exact, then a fresh common continuation stream.
        rng = np.random.default_rng(20261022 + seed)
        streams = []
        for stage in (1, 2):
            h = hashlib.sha256(); counts = np.zeros(3582, np.int64)
            for _ in range(7250):
                ix = rng.choice(3582, 32, replace=True, p=prob)
                h.update(ix.astype('<i8').tobytes()); np.add.at(counts, ix, 1)
            streams.append(dict(index_sha256=h.hexdigest(), state_draws=int(counts.sum()),
                scalar_targets=int(counts.sum() * 2), states_never_drawn=int((counts == 0).sum())))
            if stage == 1:
                assert h.hexdigest() == old_stream[str(seed)]['index_sha256']
                assert np.array_equal(counts, np.load(R / f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws'])
            else: target_cache[f'extra_stage_exposure_seed{seed}'] = counts
        assert streams[0]['index_sha256'] != streams[1]['index_sha256']
        sampling[str(seed)] = dict(original_stage=streams[0], proposed_extra_stage=streams[1])
        checks[str(seed)] = dict(base_final_hash=digest(base), full_initial_hash=before_j,
            paired_initial_tensors_identical=before_j == before_f,
            correction_initial_hash=digest(j.delta), historical_train_prediction_max_error=old_error,
            independent_numpy_max_error=numpy_error, huber_translation_max_error=loss_error,
            base_frozen_and_eval=True, initial_total_equals_base_exact=True,
            correction_parameter_ids_only=True)
    np.savez_compressed(O / 'training_target_and_exposure_check.npz', **target_cache)
    previous = read(R / 'outputs/grid115_matched_module_plan/plan.json')
    conditions = dict(previous['unchanged'])
    conditions['sampling'] = 'Per seed, reproduce original 7250x32 stream then use its next7250x32 draws for all three new arms; same probabilities, no class reweighting.'
    conditions['train_mode'] = 'Stage2 from each original GRID107 final base checkpoint. Fixed base for JOINT/FACTOR; BASE_EXTRA updates only original backbone. Fresh Adam for all new arms.'
    plan = dict(id='GRID118_FIXED_BASE_CORRECTION_PLAN',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='qualified_proposal_no_training',
        authorization='User: 好的，再改改. Revise architecture/target/budget and report before training.',
        rationale='Fix the previously jointly moving base so the learned correction has a fixed numerical reference; match extra optimization exposure separately.',
        groups=[dict(name='BASE_FIXED',new_fits=0,total_parameters=6978,trainable_parameters=0,source='Reuse GRID107 final seeds0/1/2'),
                dict(name='BASE_EXTRA',new_fits=3,total_parameters=6978,trainable_parameters=6978,source='Same GRID107 final checkpoints; fresh Adam,7250 additional updates'),
                dict(name='FIXED_JOINT',new_fits=3,total_parameters=12916,trainable_parameters=5938,source='Same fixed pretrained base plus zero-output JOINT correction'),
                dict(name='FIXED_FACTOR',new_fits=3,total_parameters=12916,trainable_parameters=5938,source='Same fixed pretrained base plus zero-output FACTOR correction')],
        mathematics=dict(base='b(x)=frozen GRID107 two-action Q prediction for this seed.',
            original_target='y=[0, immediate normalized net continuation] from original GRID107/113.',
            correction_reference='r(x)=y-b(x), detached; no new outcome label or changed reward.',
            output='q(x,z)=b(x)+delta(x,z); CONTINUE iff q[1]>q[0], tie STOP.',
            optimization='Keep original smooth_l1(q,y,beta=1,mean) in prospective trainer. In exact arithmetic this equals smooth_l1(delta,y-b,beta=1,mean).',
            identifiability='Base cannot absorb an arbitrary opposite shift while frozen. This fixes the output reference, not the internal neural representation or causal meaning of features.'),
        initialization=dict(base='Exact original trained GRID107 final weights, not random and not selected from GRID116.',
            correction=f'Original GRID1135938 architecture; seed={BRANCH_SEED_OFFSET}+s, JOINT/FACTOR identical tensors; last16->2 layer zero.',
            optimizer='All prospective extra-stage optimizers start fresh; never import SB3 Adam moments. BASE_EXTRA is reset-optimizer additional training, not uninterrupted continuation.',
            gradient='Frozen base excluded by parameter identity; first correction head update inside7250, upstream gradients initially zero. Check flow after first actual step, no warmup fitting.'),
        unchanged=conditions,
        future_fits=dict(count=9, order=[dict(seed=s,arm=a) for s in range(3) for a in ('BASE_EXTRA','FIXED_JOINT','FIXED_FACTOR')],
            updates_per_fit=7250, states_per_batch=32, scalar_targets_per_update=64,
            new_updates_total=65250, new_state_draws_total=2088000, new_scalar_targets_total=4176000,
            original_pretraining_updates_per_seed=7250, extra_updates_per_new_fit=7250,
            new_base_pretraining_fits=0, maximum_total_training_wall_s=1800,
            endpoint='Fixed final only; evaluate validation only after all9 fits. Keep every seed and partial failure; no automatic extension or refit.'),
        comparisons=dict(primary='FIXED_JOINT vs FIXED_FACTOR: same base, architecture, trainable count, initialization, optimizer, inputs and sampling; feedback association differs.',
            practical='Each corrected family vs BASE_FIXED: incremental addition, includes extra7250 training.',
            additional_training='Each corrected family vs BASE_EXTRA: equal original pretraining and extra update/target exposure; not equal architecture, trainable count, FLOPs or wall time.',
            historical='GRID116 and other historical offline outcomes stay visible; no claim that changing freezing alone is identified, because the new proposal also introduces staged pretraining.'),
        evaluation=dict(previous['evaluation']),
        limitations=['Correction can still worsen decisions or ignore action-feedback pairing; freezing does not guarantee improvement.',
            'Correction retains global41 input; ordinary summary fitting remains possible and is tested by FACTOR control, not removed by assertion.',
            'Four already exposed validation dates, correlated prefixes, same grid and fixed FULL teacher path; no whole-week quality, latency or cross-grid claim.',
            'Additional train exposure may overfit; no threshold, fee, loss, seed or dataset selection to rescue outcomes.'],
        before_training=['User reviews revised concrete plan; this stage does not run training.',
            'Write/freeze exact trainer and all code/data/checkpoint hashes, verify original independent schema and masks and public/target separation, push before first optimizer.',
            'Synthetic backward without optimizer; count actual first head update and subsequent upstream gradient check inside7250.',
            'Recheck no generator/benchmark, sealed final8 test absent; no environment or forecast in fit/evaluation.'])
    plan['evaluation']['incremental'] = 'Report all paired seeds against BASE_FIXED and BASE_EXTRA, not only mean; preserve BATCH1 and original gamma0 DQN references.'
    plan['evaluation']['progression'] = 'Mechanism signal as before AND JOINT family mean above BASE_FIXED, BASE_EXTRA, BATCH1 and original gamma0DQN; no automatic scaling or physical replay.'
    save(O / 'plan.json', plan)
    save(O / 'matching_check.json', dict(passed=True, scope='Model/reference/target/budget engineering only; not trained performance.',
        seeds=checks, sampling_streams=sampling, train_rows=3582,
        fixed_base_parameters=6978, correction_trainable_parameters=5938, corrected_total_parameters=12916,
        original_inputs_targets_probabilities_exact=True, optimizer_instances=0, backward_calls=0, optimizer_steps=0,
        new_fits=0, new_validation_model_predictions=0, historical_predictions_read=True,
        frozen_base_train_reference_forwards=3*3582, new_physical_calls=0, final_test_materialized=False))
    assert all(sha(R/p) == h for p,h in read(O / 'input_manifest.json').items())
    verify(); sealed()
    print(json.dumps(dict(passed=True,new_fits=0,planned_fits=9,checks=checks), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
