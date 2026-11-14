"""GRID332: independent cost-aware BO on the GRID330 local search task."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import log_ndtr
from scipy.linalg import cholesky, solve_triangular

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
OUTPUT = ROOT / "outputs/grid332_independent_logeipc"
ARCHIVE = HERE.parent / "GRID330_remaining19/results/raw_records_manifest.json"
MEMBERS = HERE.parent / "GRID331_staged_verification/fixed_scope.json"
FEES = HERE.parent / "GRID102_search_rl/complete/design.json"
LIBRARY = ROOT / "work/grid02/ljn_nn/assets/action_N1_unsafe.npz"
PARENT = "40f3bb60"
NUGGET = 1e-8
MAX_SECONDS = 600


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, expected=None):
    data = path.read_bytes()
    if expected is not None and sha(data) != expected:
        raise ValueError(f"Input hash mismatch: {path}")
    return json.loads(data)


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze():
    provenance = read(ARCHIVE)
    assert provenance["passed"] and provenance["independent_object_readback"]
    hashes = {r["path"]: r["sha256"] for r in provenance["entries"]}
    prior = read(MEMBERS)
    assert prior["parent_archive_manifest_sha256"] == sha(ARCHIVE.read_bytes())
    members = prior["members"]
    assert len(members) == len({m["path"] for m in members}) == 21
    metadata = {}
    for member in members:
        assert hashes[member["path"]] == member["sha256"]
        path = Path(member["path"]).with_name("metadata.json").as_posix()
        metadata[path] = hashes[path]
    assert len(metadata) == 4
    fee = read(FEES)["cost_per_query"]
    assert fee == 1.2442348640059487e-05
    save(HERE / "fixed_scope.json", dict(
        parent_commit=PARENT, source_sha256=sha(Path(__file__).read_bytes()),
        references={p.relative_to(ROOT).as_posix(): sha(p.read_bytes())
                    for p in (ARCHIVE, MEMBERS, FEES, LIBRARY)}, metadata_sha256=metadata,
        calendar=prior["calendar"], members=members, cost_per_query=fee,
        paper="https://proceedings.mlr.press/v306/xie26c.html",
        algorithm="Xie et al. equations 5 and 10, finite-set Gaussian LogEIPC. Model conditions on all revealed global and local observations. Next argmax with lower ID as tie-break. Stop when max LogEI/fee <= 0.",
        scope="All 21 GRID330 mixed-order protected 261 calls, single f0_s0 pool and four exposed 2039 dates. Shared original global prefix and station/pending IDs. LogEIPC owns all subsequent local acquisition and stopping, with no 208/261/0.9 local decisions.",
        feasibility="Reward for a strictly admissible result is the original float32 score. Encode nonadmissible outcomes as float32(2-rho_limit), below or equal to any admissible score. The legal incumbent alone can be delivered. Zero pending means no local search.",
        gp="Matérn 5/2 on public raw command vectors, fixed Euclidean length 1 command unit, standardized with original global revealed mean and population SD (minimum SD equals fee), fixed dimensionless jitter 1e-8. No optimizer or hyperparameter sweep. Fixed explicit modelling adaptation, not the authors' exact benchmark prior.",
        support="Same pending pool and hard budget len(pending) for both. If the independent arm requests an unsaved result, preserve trace and mark unobservable. No eligibility filter using rewards, no imputation of missing evaluations.",
        comparison="Keep all calls and dates. Count candidate queries, strict action match, immediate reward, reward minus the same historical per-query proxy charge with the common initial state. Controller or physical week timing/cost cannot be inferred.",
        caps=dict(wall_s=MAX_SECONDS, new_grid_forecasts=0, new_model_fits=0, new_208_261_forwards=0)))


def key(row):
    return float(np.float32(row["reward"])), -row["pool_id"]


def best(rows):
    return max((r for r in rows if r["strict_admissible"]), key=key, default=None)


def utility(row, floor):
    return key(row)[0] if row["strict_admissible"] else floor


def kernel(left, right):
    distance = np.sqrt(5 * cdist(left, right, metric="sqeuclidean"))
    return (1 + distance + distance**2 / 3) * np.exp(-distance)


def log_ei(mean, sd, incumbent):
    z = (mean - incumbent) / sd
    log_phi = -z**2 / 2 - np.log(2 * np.pi) / 2
    result = log_phi.copy()
    positive = z > 0
    result[positive] = np.logaddexp(log_phi[positive], np.log(z[positive]) + log_ndtr(z[positive]))
    negative = z < 0
    # Normal EI is sd*(phi(z)+z*Phi(z)); use subtraction in log space.
    with np.errstate(divide="ignore"):
        ratio = np.log(-z[negative]) + log_ndtr(z[negative]) - log_phi[negative]
        result[negative] = log_phi[negative] + np.log(-np.expm1(np.minimum(ratio, 0)))
    return np.log(sd) + result


def acquire(vectors, prefix, pending, floor, fee, reveal, deadline):
    start = time.perf_counter()
    incumbent = best(prefix)
    assert incumbent is not None and pending
    prefix_ids = [r["pool_id"] for r in prefix]
    x = vectors[prefix_ids + pending]
    cov = kernel(x, x)
    cov[np.diag_indices_from(cov)] += NUGGET
    initial = len(prefix)
    factor = cholesky(cov[:initial, :initial], lower=True)
    values = np.asarray([utility(r, floor) for r in prefix])
    offset, scale = float(values.mean()), max(float(values.std()), fee)
    observed = list(range(initial))
    hidden = list(range(initial, len(x)))
    trace, reason, max_log_ratio = [], "pool_exhausted", None
    for _ in pending:
        if time.monotonic() >= deadline:
            raise TimeoutError("GRID332 fixed 600-second window")
        proj = solve_triangular(factor, cov[np.ix_(observed, hidden)], lower=True)
        standardized = solve_triangular(factor, (values - offset) / scale, lower=True)
        mean = offset + scale * (proj.T @ standardized)
        sd = scale * np.sqrt(np.maximum(1 - np.sum(proj**2, axis=0), NUGGET))
        score = log_ei(mean, sd, key(incumbent)[0]) - np.log(fee)
        assert not np.any(np.isnan(score))
        index = max(range(len(hidden)), key=lambda j: (score[j], -pending[hidden[j] - initial]))
        max_log_ratio = float(score[index]) if np.isfinite(score[index]) else None
        if score[index] <= 0:
            reason = "ei_not_above_fee"
            break
        position = hidden.pop(index)
        cid = pending[position - initial]
        row = reveal(cid)
        if row is None:
            reason = "missing_saved_feedback"
            trace.append(dict(id=cid, log_ei_over_fee=max_log_ratio, available=False))
            break
        trace.append(dict(id=cid, log_ei_over_fee=max_log_ratio, available=True))
        values = np.append(values, utility(row, floor))
        if row["strict_admissible"] and key(row) > key(incumbent):
            incumbent = row
        if hidden:
            new = len(observed)
            updated = np.zeros((new + 1, new + 1))
            updated[:new, :new] = factor
            updated[new, :new] = proj[:, index]
            delta = cov[position, position] - proj[:, index] @ proj[:, index]
            assert delta > 0
            updated[new, new] = np.sqrt(delta)
            factor = updated
            observed.append(position)
    return incumbent, trace, reason, max_log_ratio, time.perf_counter() - start


def evaluate_member(member, vectors, fee, deadline):
    call = read(ROOT / member["path"], member["sha256"])
    assert call["order_mode"] == "STATIC64_DC64" and call["learning_method"] == "BOUNDED261"
    pending = call["completion_candidate_ids"]
    outcomes = call["outcomes"]
    cut = len(outcomes) - call["local_queries"]
    prefix, tail = outcomes[:cut], outcomes[cut:]
    original, initial = best(outcomes), best(prefix)
    assert len(outcomes) == call["public_queries"] == len({r["pool_id"] for r in outcomes})
    assert [r["pool_id"] for r in tail] == call["extra_selected_ids"]
    assert (None if original is None else original["pool_id"]) == call["chosen_pool_id"]
    assert (None if initial is None else initial["pool_id"]) == call["pre_extra_choice"]
    assert pending == list(dict.fromkeys(pending)) and len(pending) == call["completion_budget"]
    id_set = {r["pool_id"] for r in prefix}
    assert not id_set.intersection(pending) and set(pending).issubset(call["order"])
    assert set(r["pool_id"] for r in tail).issubset(pending)
    floor = float(np.float32(2 - call["rho_limit"]))
    for row in outcomes:
        legal = (0 < row["rho"] < call["rho_limit"] and not any(
            row[k] for k in ("done", "illegal", "ambiguous", "exceptions")))
        assert bool(legal) == row["strict_admissible"]
        if legal:
            assert key(row)[0] == float(np.float32(2 - row["rho"])) >= floor
    exact = {r["pool_id"]: r for r in tail}
    winner, trace, reason, log_ratio, seconds = initial, [], "no_pending", None, 0.0
    if pending:
        winner, trace, reason, log_ratio, seconds = acquire(
            vectors, prefix, pending, floor, fee, exact.get, deadline)
    observed = sum(t["available"] for t in trace)
    supported = reason != "missing_saved_feedback"
    query_delta = observed - len(tail) if supported else None
    reward_delta = (0.0 if original is None else key(winner)[0] - key(original)[0]) if supported else None
    original_id = None if original is None else original["pool_id"]
    new_id = None if winner is None else winner["pool_id"]
    assert observed <= len(pending)
    return dict(date=member["date"], source=member["path"], step=call["step"],
                state_hash=call["state_hash"], shared_queries=cut, pending=len(pending),
                saved_feedback=len(tail), original_local=len(tail),
                independent_local=observed, query_delta=query_delta,
                original_winner=original_id, independent_winner=new_id if supported else None,
                original_rho=None if original is None else original["rho"],
                independent_rho=winner["rho"] if supported and winner is not None else None,
                original_reward=None if original is None else key(original)[0],
                independent_reward=key(winner)[0] if supported and winner is not None else None,
                winner_match=original_id == new_id if supported else None,
                reward_delta=reward_delta,
                net_delta=reward_delta - fee * query_delta if supported else None,
                original_guard=bool(call.get("load_guard")), independent_stop=reason,
                final_log_ei_over_fee=log_ratio, independent_compute_s=seconds, trace=trace)


def summarize(rows):
    paired = [r for r in rows if r["query_delta"] is not None]
    return dict(calls=len(rows), paired=len(paired), local_opportunities=sum(bool(r["pending"]) for r in paired),
                missing=len(rows) - len(paired),
                original_local_queries=sum(r["original_local"] for r in paired),
                independent_local_queries=sum(r["independent_local"] for r in paired),
                original_total_queries=sum(r["shared_queries"] + r["original_local"] for r in paired),
                independent_total_queries=sum(r["shared_queries"] + r["independent_local"] for r in paired),
                fee_stops=sum(r["independent_stop"] == "ei_not_above_fee" for r in paired),
                changed_winners=sum(not r["winner_match"] for r in paired),
                immediate_losses=sum(r["reward_delta"] < 0 for r in paired),
                immediate_wins=sum(r["reward_delta"] > 0 for r in paired),
                reward_delta=sum(r["reward_delta"] for r in paired),
                net_delta=sum(r["net_delta"] for r in paired),
                independent_compute_s=sum(r["independent_compute_s"] for r in paired))


def evaluate():
    start = time.monotonic()
    scope_path = HERE / "fixed_scope.json"
    scope = read(scope_path)
    assert scope["source_sha256"] == sha(Path(__file__).read_bytes())
    for name, expected in scope["references"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    with np.load(LIBRARY, allow_pickle=False) as data:
        library = data["action_space"]
    assert library.shape == (909, 1605) and library.dtype == np.float32 and np.isfinite(library).all()
    library_hashes = [sha(row.tobytes()) for row in library]
    for name, expected in scope["metadata_sha256"].items():
        assert read(ROOT / name, expected)["library_hashes"] == library_hashes
    reduced = library[:, np.any(library != library[0], axis=0)].astype(np.float64)
    rows = [evaluate_member(m, reduced, scope["cost_per_query"], start + MAX_SECONDS)
            for m in scope["members"]]
    assert sum(r["shared_queries"] + r["original_local"] for r in rows) == 6094
    assert time.monotonic() < start + MAX_SECONDS
    summary = dict(all=summarize(rows),
                   by_date={d["date"]: summarize([r for r in rows if r["date"] == d["date"]])
                            for d in scope["calendar"]},
                   cost_per_query=scope["cost_per_query"], scope_sha256=sha(scope_path.read_bytes()),
                   new_forecasts=0, new_model_fits=0, new_208_261_forwards=0,
                   wall_s=time.monotonic() - start)
    OUTPUT.mkdir(exist_ok=False)
    save(OUTPUT / "rows.json", rows)
    save(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    {"freeze": freeze, "evaluate": evaluate}[parser.parse_args().stage]()

