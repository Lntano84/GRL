"""LG01 solver interface. This module cannot read action outcome tables."""
from __future__ import annotations
import dataclasses, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, vstack

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'outputs/LG01_training_plan.md').exists())
OUT = ROOT / 'outputs/lg01'
SRC = Path(__file__).resolve().parent / 'source'
sys.path.insert(0, str(SRC))
import highspy
from lsp_model import MODE_AUDITED, Solution, build_model, unpack, enumeration_space
from lsp_gen import build_instance, apply_disruption, Disruption
from lsp_checker import check_solution
from lsp_release import vector_of, stability_row_local, RINS_TOL
from lsp_repair3 import repair_minbatch_v3
from lsp_highs import build_highs

ACTIONS = ['RINS', 'RC-EXPAND', 'LOSS-EXPAND', 'RANDOM-EXPAND', 'FULL']
KEYS = ['X', 'Y', 'Z', 'I', 'L']
TAU, KAPPA = 6, 4

def diskpath(p):
    # State IDs retain their canonical '|'; Windows filenames use reversible percent escape.
    p=Path(p)
    return p.with_name(p.name.replace('|','%7C'))

def json_number(x):
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,np.ndarray): return x.tolist()
    raise TypeError(f'Unsupported JSON value: {type(x).__name__}')

def sha(p):
    h = hashlib.sha256()
    with diskpath(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def save(p, obj):
    p = diskpath(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=1,default=json_number)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def read(p):
    return json.loads(diskpath(p).read_text(encoding='utf-8'))

def pack(s): return {k: np.asarray(getattr(s, k)).tolist() for k in KEYS}

def solution(rec):
    return Solution(status=rec.get('status', 'saved'), objective=float(rec['objective']),
                    mode=MODE_AUDITED, **{k: np.asarray(rec['plan'][k], float) for k in KEYS})

def from_x(bm, x, obj, status):
    return Solution(status=status, objective=float(obj), mode=MODE_AUDITED,
                    **dict(zip(KEYS, unpack(x, bm.idx))))

def stable_seed(*args):
    return int.from_bytes(hashlib.sha256(json.dumps([20261006, *args], separators=(',', ':')).encode()).digest()[:8], 'little')

def col_map(bm):
    out = {}
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, bm.inst.T + 1):
                for kind in ['X', 'Y', 'Z']: out[getattr(bm.idx, kind)(i, j, t)] = (kind, i, j, t)
            out[bm.idx.Z(i, j, 0)] = ('Z', i, j, 0)
        for t in range(1, bm.inst.T + 1):
            for kind in ['I', 'L']: out[getattr(bm.idx, kind)(i, t)] = (kind, i, -1, t)
    return out

def fixes_ok(sol, fixes, slots):
    # Includes ALL short/far Y and Z; old evaluate_candidate omitted far Y.
    bad = []
    for c, want in fixes.items():
        kind, i, j, t = slots[c]
        got = float(getattr(sol, kind)[i, j, t - 1]) if j >= 0 else float(getattr(sol, kind)[i, t - 1])
        if abs(got - want) > 1e-6: bad.append([c, got, want])
    return bad

def verify(inst, sol, anchor=None, fixes=None, slots=None):
    if any(not np.isfinite(getattr(sol, k)).all() for k in KEYS):
        return dict(ok=False, problems=['nonfinite plan'])
    chk = check_solution(inst, sol, MODE_AUDITED)
    flips = 0 if anchor is None else int(np.count_nonzero(np.abs(sol.Y[:, :, :TAU] - anchor.Y[:, :, :TAU]) > .5))
    bad = fixes_ok(sol, fixes or {}, slots or {})
    cost = float(chk.cost_recomputed['total'])
    obj_ok = abs(cost - sol.objective) <= 1e-6 * (1 + abs(cost))
    return dict(ok=bool(chk.ok and flips <= KAPPA and not bad and obj_ok),
                cost=cost, flips=flips, fixed_violations=bad[:8],
                objective_ok=bool(obj_ok), max_violation=float(chk.max_violation),
                problems=chk.failed_checks())

def system(bm, stab):
    co, rhs = stab
    a = coo_matrix((list(co.values()), ([0]*len(co), list(co))), shape=(1, bm.idx.n)).tocsr()
    return vstack([bm.A, a], format='csr'), np.r_[bm.row_lb, -np.inf], np.r_[bm.row_ub, rhs]

def solve(bm, fixed, stab, deadline, log, seed=0, start=None, relax=False, cap=None):
    """Deadline applies AFTER transfer and MIP start; never increases remaining time."""
    t0 = time.perf_counter()
    model = dataclasses.replace(bm, integrality=np.zeros_like(bm.integrality)) if relax else bm
    h, lp, n, nr = build_highs(model, fixed, stab)
    log = diskpath(log); log.parent.mkdir(parents=True, exist_ok=True)
    h.setOptionValue('output_flag', True)
    h.setOptionValue('log_to_console', False)
    h.setOptionValue('log_file', str(log))
    h.setOptionValue('threads', 1)
    h.setOptionValue('random_seed', int(seed))
    assert h.passModel(lp) == highspy.HighsStatus.kOk
    start_status = None
    if start is not None:
        start_status = str(h.setSolution(n, np.arange(n, dtype=np.int32), np.asarray(start, np.float64)))
    remaining = max(0., deadline - time.perf_counter())
    if cap is not None: remaining = min(remaining, float(cap))
    prep = time.perf_counter() - t0
    if remaining <= 0:
        return dict(status='deadline_before_solve', x=None, objective=None, solution=None,
                    prep_s=prep, solve_s=0., allocated_s=0., optimal=False, dual_valid=False,
                    col_dual=None, row_dual=None, primal_valid=False, start_status=start_status)
    h.setOptionValue('time_limit', remaining)
    t1 = time.perf_counter(); h.run(); ts = time.perf_counter() - t1
    info, hs = h.getInfo(), h.getSolution()
    status = h.modelStatusToString(h.getModelStatus())
    primal = bool(hs.value_valid and info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible)
    x = np.asarray(hs.col_value, float) if primal else None
    if x is not None and (len(x) != n or not np.isfinite(x).all()): raise RuntimeError('invalid primal interface')
    obj = float(info.objective_function_value) if x is not None else None
    if obj is not None and not np.isfinite(obj): raise RuntimeError('nonfinite objective')
    dual = bool(relax and hs.dual_valid and info.dual_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible)
    cd = np.asarray(hs.col_dual, float) if dual else None
    rd = np.asarray(hs.row_dual, float) if dual else None
    if dual:
        if not (len(cd) == n and len(rd) == nr and np.isfinite(cd).all() and np.isfinite(rd).all()):
            raise RuntimeError('invalid dual interface')
        # Raw constraint coefficients include appended stability and fixing rows.
        a, lo, up = system(bm, stab)
        if fixed:
            cols = list(fixed)
            af = coo_matrix((np.ones(len(cols)), (np.arange(len(cols)), cols)), shape=(len(cols), n))
            a = vstack([a, af], format='csr')
        residual = np.max(np.abs(bm.c - a.T @ rd - cd))
        if residual > 1e-5 * (1 + np.max(np.abs(bm.c))): raise RuntimeError(f'dual KKT stationarity {residual}')
    else: residual = None
    return dict(status=status, x=x, objective=obj,
                solution=None if x is None or relax else from_x(bm, x, obj, status),
                prep_s=prep, solve_s=ts, allocated_s=remaining, optimal=status.lower().startswith('optimal'),
                dual_valid=dual, col_dual=cd, row_dual=rd, primal_valid=primal,
                dual_residual=residual, start_status=start_status,
                gap=float(info.mip_gap) if not relax and np.isfinite(info.mip_gap) else None,
                dual_bound=float(info.mip_dual_bound) if not relax and np.isfinite(info.mip_dual_bound) else None)

def lp_summary(r):
    return {k: r.get(k) for k in ['status','objective','prep_s','solve_s','allocated_s','optimal','dual_valid','primal_valid','dual_residual','gap','dual_bound','start_status']}

def actions(bm, anchor, nominal, rlx, state):
    slots = col_map(bm); xr = vector_of(anchor, bm)
    binary = [c for c, s in slots.items() if s[0] in ['Y', 'Z'] and bm.var_lb[c] != bm.var_ub[c]]
    usable = bool(rlx['optimal'] and rlx['primal_valid'] and rlx['dual_valid'])
    fr = {c: float(round(xr[c])) for c in sorted(binary) if usable and abs(rlx['x'][c] - xr[c]) <= RINS_TOL}
    strata = {h: [] for h in ['short_Y','far_Y','short_Z','far_Z']}
    for c in fr:
        kind, i, j, t = slots[c]; strata[('short_' if t <= TAU else 'far_') + kind].append(c)
    quota = {h: math.ceil(.1 * len(cs)) for h, cs in strata.items()}
    loss = np.maximum(0., anchor.L - nominal.L) * np.asarray(bm.inst.l)[:, None]
    future = np.flip(np.cumsum(np.flip(loss, axis=1), axis=1), axis=1)
    released = [set() for _ in ACTIONS]
    for h, cs in strata.items():
        tie = lambda c: stable_seed(state, list(slots[c]))
        def score(c):
            kind, i, j, t = slots[c]; u = t if kind == 'Y' else t + 1
            return float(future[i, u-1]) if u <= bm.inst.T else 0.
        released[1].update(sorted(cs, key=lambda c: (abs(rlx['col_dual'][c]), tie(c)))[:quota[h]])
        released[2].update(sorted(cs, key=lambda c: (-score(c), tie(c)))[:quota[h]])
        released[3].update(random.Random(stable_seed('random', state, h)).sample(sorted(cs), quota[h]))
    released[4] = set(fr)
    fixes = [{c: v for c, v in fr.items() if c not in rel} for rel in released]
    masks = np.zeros((5, bm.idx.n), np.float32)
    # Masks describe the COMPLETE free binary neighborhood, including variables free in A0.
    for a in range(5): masks[a, [c for c in binary if c not in fixes[a]]] = 1.
    meta = dict(quota=quota, strata_counts={h:len(cs) for h,cs in strata.items()},
                fr_count=len(fr), n_binary=len(binary),
                fixed_counts=[len(f) for f in fixes], fallback=not usable,
                rc_zero_fraction=float(np.mean(np.abs(rlx['col_dual'][list(fr)]) < 1e-10)) if fr else None,
                distinct_actions=len({tuple(sorted(f)) for f in fixes}),
                structural_pins=int(np.count_nonzero((bm.integrality == 1) & (bm.var_lb == bm.var_ub))))
    return fixes, masks, fr, slots, meta

def online(rec, policy, seed, tag, graph_save=None, scorer=None, nofault=False):
    # Input deserialization/model imports are service loading. All state calculations start here.
    t0 = time.perf_counter(); deadline = t0 + 19.8
    marks = {}; last = t0
    def mark(name):
        nonlocal last
        now=time.perf_counter(); marks[name]=now-last; last=now
    inst, _ = build_instance('large', rec['rho'], rec['family'])
    d = rec['disruption']; dis = Disruption(d['name'], {int(j):ts for j,ts in d['down'].items()}, d.get('note',''))
    pert = inst if nofault else apply_disruption(inst, dis)
    anchor, nominal = solution(rec['repair']), solution(rec['nominal'])
    bm = build_model(pert, MODE_AUDITED); slots=col_map(bm); xr=vector_of(anchor,bm)
    stab = stability_row_local(bm, anchor, TAU, KAPPA)
    mark('build_s')
    pin = {c: float(round(xr[c])) for c in slots if bm.integrality[c] == 1}
    lp = solve(bm, pin, stab, deadline, OUT/'logs'/f'{tag}.fixedlp.log', seed, relax=True, cap=5.)
    slp = from_x(bm,lp['x'],lp['objective'],lp['status']) if lp['x'] is not None else None
    mark('fixed_lp_s')
    # Fixed FULL needs no relaxed LP or graph; all learning policies compute their required inputs.
    need_rlx = policy != 'FULL' or graph_save is not None
    rlx = solve(bm,{},stab,deadline,OUT/'logs'/f'{tag}.relax.log',seed,relax=True,cap=5.) if need_rlx else dict(optimal=False,primal_valid=False,dual_valid=False,status='not_needed',x=None,objective=None)
    mark('relax_s')
    fixes,masks,fr,slots,meta=actions(bm,anchor,nominal,rlx,tag.split('__')[0])
    mark('actions_s')
    features=None; inference=0.; pred=None
    if scorer is not None:
        from features import make_features
        features=make_features(bm,anchor,nominal,rlx,fr,masks,slots,stab,inst,
                               with_edges=scorer.requires_edges)
    mark('graph_s')
    if scorer is not None and not meta['fallback']:
        ti=time.perf_counter(); a,pred=scorer.choose(features); inference=time.perf_counter()-ti
    else: a = ACTIONS.index(policy) if policy in ACTIONS else 0
    fixed=fixes[a]; mark('select_s')
    # Validate reference and own pinned LP BEFORE submission; reserve accounts for final validation too.
    candidates=[]
    for name,s in [('repair',anchor),('lp',slp)]:
        if s is None: continue
        v=verify(pert,s,anchor,fixed,slots)
        if not v['ok']: raise RuntimeError(f'{tag}: invalid {name} {v}')
        candidates.append((name,s,v))
    mark('start_verify_s')
    start=vector_of(min(candidates,key=lambda c:c[2]['cost'])[1],bm)
    mip=solve(bm,fixed,stab,deadline,OUT/'logs'/f'{tag}.mip.log',seed,start=start)
    mark('mip_s')
    if 'infeasible' in mip['status'].lower(): raise RuntimeError(f'{tag}: infeasible despite valid reference')
    if mip['solution'] is not None:
        v=verify(pert,mip['solution'],anchor,fixed,slots)
        if not v['ok']: raise RuntimeError(f'{tag}: invalid solver incumbent {v}')
        candidates.append(('solver',mip['solution'],v))
    best=min(candidates,key=lambda c:c[2]['cost']); wall=time.perf_counter()-t0; mark('delivery_s')
    # Disk archival is measured separately AFTER effective delivery, never hidden as solving.
    result=dict(state=tag.split('__')[0], family=rec['family'], rho=rec['rho'], fault=d['name'],
                policy=policy, action=ACTIONS[a], seed=seed, repair_cost=float(anchor.objective),
                D=max(1.,abs(anchor.objective)), objective=best[2]['cost'], source=best[0],
                plan=pack(best[1]), validation=best[2], timing=marks, wall_s=wall,
                timing_ok=wall<=20.2, inference_s=inference,
                scores=None if pred is None else [float(x) for x in pred],
                fallback=meta['fallback'] if need_rlx else False, fallback_delivery=best[0] != 'solver',
                lp=lp_summary(lp), relax=lp_summary(rlx), mip=lp_summary(mip), action_meta=meta,
                fixed_sets=[{str(c):v for c,v in f.items()} for f in fixes],
                fixed_slots=[{str(c):list(slots[c]) for c in f} for f in fixes],
                FR={str(c):v for c,v in fr.items()}, anchor_objective=float(anchor.objective),
                candidate_plans={name:dict(plan=pack(s),objective=v['cost'],validation=v) for name,s,v in candidates})
    if graph_save is not None:
        from features import make_features
        tc=time.perf_counter()
        # Label action A0 must NOT pay a graph-only tax that the other fixed actions avoid.
        # Reconstruct legal raw features AFTER effective delivery from its own recorded LP.
        features=make_features(bm,anchor,nominal,rlx,fr,masks,slots,stab,inst,with_edges=True)
        graph_save=diskpath(graph_save)
        graph_save.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(graph_save, **features)
        result['offline_feature_cache_s']=time.perf_counter()-tc
    return result
