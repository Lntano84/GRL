"""Frozen experimental staged fallback; takes only dispatched feedback."""
import struct


def key(row):
    value = struct.unpack("f", struct.pack("f", row["reward"]))[0]
    return value, -row["pool_id"]


def staged(order, observe, policy, safe=.9):
    assert policy in ("PROTECTED", "FIXED256", "PATIENCE64")
    assert len(order) == len(set(order))
    results, incumbent, events = [], None, []

    def purchase(ids, phase):
        nonlocal incumbent
        before = None if incumbent is None else key(incumbent)
        for i in ids:
            row = observe(i, phase)
            assert row["pool_id"] == i
            results.append(row)
            if row["strict_admissible"] and (incumbent is None or key(row) > key(incumbent)):
                incumbent = row
        after = None if incumbent is None else key(incumbent)
        improved = after is not None and (before is None or after > before)
        events.append({"queries": len(results), "phase": phase,
                       "chosen": None if incumbent is None else incumbent["pool_id"],
                       "improved": improved})
        return improved

    initial = min(128, len(order))
    purchase(order[:initial], "shortlist")
    prefix = None if incumbent is None else incumbent["pool_id"]
    fallback = initial < len(order) and (incumbent is None or incumbent["rho"] >= safe)
    reason = "initial_safe" if not fallback else "full_exhaustion"
    if fallback:
        if policy == "PROTECTED":
            purchase(order[initial:], "fallback")
        elif policy == "FIXED256":
            end = min(256, len(order))
            purchase(order[initial:end], "fallback")
            if incumbent is None:
                purchase(order[end:], "fallback_no_admissible")
            elif end < len(order):
                reason = "fixed256_experimental_stop"
        else:
            stale = 0
            for offset in range(initial, len(order), 64):
                improved = purchase(order[offset:offset+64], "fallback")
                stale = 0 if improved else stale + 1
                if incumbent is not None and incumbent["rho"] < safe:
                    reason = "verified_below_author_safe"; break
                if incumbent is not None and len(results) >= 256 and stale >= 2 and len(results) < len(order):
                    reason = "patience64_experimental_stop"; break
    return {"outcomes": results, "chosen_pool_id": None if incumbent is None else incumbent["pool_id"],
            "shortlist_choice": prefix, "fallback": fallback, "stop_reason": reason,
            "events": events, "public_queries": len(results)}


def hand_checks():
    # A late optimum is deliberately missed: plateau is NOT a safety certificate.
    def record(i, phase, rho=1.):
        return {"pool_id": i, "reward": 1. if i != 399 else 2., "rho": rho,
                "strict_admissible": True, "phase": phase}
    order = list(range(400))
    a = staged(order, record, "PATIENCE64")
    assert a["public_queries"] == 256 and a["chosen_pool_id"] == 0
    b = staged(order, record, "PROTECTED")
    assert b["public_queries"] == 400 and b["chosen_pool_id"] == 399
    # No admissible feedback cannot trigger either experimental stop.
    def invalid(i, phase):
        r = record(i, phase); r["strict_admissible"] = False; return r
    for p in ["FIXED256", "PATIENCE64"]:
        r = staged(order, invalid, p)
        assert r["public_queries"] == 400 and r["chosen_pool_id"] is None
    # Each query is dispatched once; tie uses the smallest action ID.
    r = staged(order[::-1], record, "FIXED256")
    assert len({x["pool_id"] for x in r["outcomes"]}) == r["public_queries"]
    assert r["chosen_pool_id"] == 399
    r = staged(order, lambda i,p: record(i,p,.8), "PATIENCE64")
    assert r["public_queries"] == 128
    # Exact fp32 tie rather than invented higher precision.
    assert key({"reward": 1.+1e-9, "pool_id": 4}) < key({"reward": 1., "pool_id": 3})
    return 8
