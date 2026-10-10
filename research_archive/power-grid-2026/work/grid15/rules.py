"""Cheap candidate sets: legal observation and action metadata only, no outcome labels."""
from collections import deque
import hashlib
import numpy as np


def shortlist_rules(rho, live, line_or, line_ex, n_sub, areas, subs, ids, state_key):
    ids = np.asarray(ids, dtype=int)
    subs = np.asarray(subs, dtype=int)
    assert len(ids) == len(subs) and len(set(ids.tolist())) == len(ids)
    graph = [[] for _ in range(n_sub)]
    pressure = np.zeros(n_sub)
    for l in np.flatnonzero(live):
        u, v = int(line_or[l]), int(line_ex[l])
        graph[u].append(v); graph[v].append(u)
        pressure[u] = max(pressure[u], float(rho[l]))
        pressure[v] = max(pressure[v], float(rho[l]))
    live_lines = sorted(np.flatnonzero(live), key=lambda l: (-float(rho[l]), int(l)))
    roots = {int(x) for l in live_lines[:3] for x in (line_or[l], line_ex[l])}
    distance = np.full(n_sub, n_sub+1, dtype=int)
    queue = deque(sorted(roots))
    for node in roots: distance[node] = 0
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if distance[v] > distance[u]+1:
                distance[v] = distance[u]+1; queue.append(v)
    local = sorted(range(len(ids)), key=lambda i: (int(distance[subs[i]]), -float(pressure[subs[i]]), int(ids[i])))
    random_order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f"grid15-seed0:{state_key}:{ids[i]}".encode()).digest())
    if len(rho):
        # Mirrors author's ZoneBasedTopoSearchModule: origin of most-loaded line.
        area = areas[int(line_or[int(np.argmax(rho))])]
        zone = [i for i, sub in enumerate(subs) if areas[int(sub)] == area]
    else:
        zone = []
    result = {"ALL": list(range(len(ids))), "ZONE": zone}
    for k in (32, 128):
        result[f"LOCAL{k}"] = local[:k]
        result[f"RANDOM{k}"] = random_order[:k]
        result[f"PREFIX{k}"] = list(range(min(k, len(ids))))
    return result, {"distance": distance.tolist(), "pressure": pressure.tolist(), "roots": sorted(roots)}
