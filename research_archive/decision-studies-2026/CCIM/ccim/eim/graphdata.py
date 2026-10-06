"""The frozen empirical contact network, with the roster and its pre-registered attribute.

Source
------
SocioPatterns high-school contact data (Mastrandrea, Fournet & Barrat, *PLoS ONE* 2015,
doi:10.1371/journal.pone.0136497): 327 participants in nine class groups, wearable-sensor contacts
with a 20-second resolution, five consecutive school days in December 2013.  Files are stored under
``_data/highschool``.  ``HighSchool2013_proximity_net.csv.gz`` and the PLOS supplementary
``pone.0136497.s002.csv`` are byte-for-byte the same measurements (188,508 rows, 327 nodes, identical
timestamps), so the download used does not affect any number.

What is frozen before looking at outcomes
-----------------------------------------
* **Day.**  The day retained is the one with the largest number of distinct participant pairs observed
  (a completeness proxy), ties broken by the earlier date.  This is fixed in advance so that no
  outcome-driven date choice is possible.  Under it the retained day is 2013-12-03.
* **Edge.**  A pair is an edge if at least one 20-second contact between them was recorded on that
  day.  Direction is ignored and self-loops are dropped.
* **Attribute.**  The class group from ``HighSchool2013_metadata.txt``.  It is an administrative
  roster field that exists *before* any contact is observed, which is exactly the setting being
  studied; it is not a community label computed from the hidden graph.
* **Roster.**  Every participant recorded anywhere in the five-day file, whether or not they had a
  contact on the retained day.  Seventeen participants have no contact on the retained day and are
  legitimate (isolated) roster members -- they are kept, because dropping them would be using the
  hidden graph to clean the roster.

Known limitation, stated in advance: the metadata file contains two participant ids that never appear
in the contact file.  They are excluded, because the roster is defined as recorded participants.
"""

from __future__ import annotations

import gzip
import os
from collections import defaultdict
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "_data", "highschool")
CONTACT_FILE = os.path.join(DATA_DIR, "HighSchool2013_proximity_net.csv.gz")
META_FILE = os.path.join(DATA_DIR, "HighSchool2013_metadata.txt")

CONTACT_SECONDS = 20          # the sensor resolution recorded in the file


def _rows(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            part = line.split()
            if len(part) != 5:
                continue
            yield int(part[0]), part[1], part[2]


def day_of(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()


def day_statistics() -> dict:
    """Distinct-pair count per day, computed **without** applying any duration filter."""
    pairs = defaultdict(set)
    for ts, u, v in _rows(CONTACT_FILE):
        if u == v:
            continue
        pairs[day_of(ts)].add((u, v) if u <= v else (v, u))
    return {d: len(p) for d, p in sorted(pairs.items())}


def select_day(stats: dict | None = None) -> str:
    """Frozen rule: most distinct pairs, ties to the earlier date."""
    stats = day_statistics() if stats is None else stats
    return max(sorted(stats), key=lambda d: stats[d])


def load_roster(path: str = META_FILE) -> dict:
    """``{participant_id: class_group}`` for every id that has a metadata row."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            part = line.split()
            if len(part) >= 2:
                out[part[0]] = part[1]
    return out


def load_network(day: str | None = None, min_contacts: int = 1):
    """Return ``(nodes, edges, meta)`` for the retained day.

    ``nodes`` is the sorted roster intersecting the recorded participants (string ids, roster order);
    ``edges`` is a sorted list of index pairs.  ``meta`` records every frozen choice so the report can
    be checked against it.
    """
    stats = day_statistics()
    chosen = select_day(stats) if day is None else day

    counts = defaultdict(int)
    participants = set()
    for ts, u, v in _rows(CONTACT_FILE):
        participants.add(u)
        participants.add(v)
        if u == v or day_of(ts) != chosen:
            continue
        key = (u, v) if u <= v else (v, u)
        counts[key] += 1

    roster = load_roster()
    nodes = sorted(participants)
    index = {v: i for i, v in enumerate(nodes)}
    edges = sorted((index[a], index[b]) for (a, b), c in counts.items()
                   if c >= min_contacts and a in index and b in index)

    meta = {
        "chosen_day": chosen,
        "day_statistics": stats,
        "contact_seconds": CONTACT_SECONDS,
        "min_contacts": int(min_contacts),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_participants_all_days": len(participants),
        "n_metadata_rows": len(roster),
        "n_isolated_on_day": len(nodes) - len({i for e in edges for i in e}),
        "classes": sorted({roster[v] for v in nodes if v in roster}),
        "n_without_attribute": sum(1 for v in nodes if v not in roster),
        "within_class_edges": sum(
            1 for a, b in edges
            if roster.get(nodes[a]) is not None and roster.get(nodes[a]) == roster.get(nodes[b])),
        "attribute_source": os.path.basename(META_FILE),
        "contact_source": os.path.basename(CONTACT_FILE),
    }
    attr = [roster.get(v, "") for v in nodes]
    return nodes, edges, attr, meta


def mean_degree(edges, n: int) -> float:
    return 2.0 * len(edges) / n if n else 0.0
