"""On-demand without-replacement candidate scanning: a sparse-map lazy Fisher-Yates shuffle.

Why
---
A 52-seed state on ca-GrQc has ``52 * (5241 - 52) = 269,828`` possible 1-swaps.  The original scanner
built that whole list of pairs and shuffled it, then stopped at the first improving move -- so most of
the permutation was never used, and building it cost ~85 ms per state, roughly 57% of the search's
wall time.  This module draws the next candidate only when one is actually needed.

The permutation rule is unchanged
---------------------------------
Forward Fisher-Yates over ``0 .. M-1``::

    for i in 0 .. M-1:
        j = randrange(i, M)
        swap(a[i], a[j])

which is uniform over the ``M!`` permutations.  ``Lazy`` implements it with a sparse map so only the
positions actually touched are stored; ``Explicit`` keeps the whole array.  Both draw from the caller's
RNG in exactly the same order, so with the same RNG they emit **identical** prefixes -- which is what
makes an end-to-end comparison between them a test of the implementation rather than of luck.

Sampling is **without replacement over the full range** and the scan still exhausts: no candidate is
dropped and no limited random subset is declared a local optimum.  With-replacement sampling would
change the search rule and is not used.

The candidate numbering also avoids building anything of size ``M``::

    id -> u = selected[id // len(outside)],  v = outside[id % len(outside)]

so a candidate is two integer divisions, not a stored tuple.
"""

from __future__ import annotations


class LazyShuffle:
    """Sparse-map Fisher-Yates: ``next_id()`` costs O(1) time and O(1) amortised memory."""

    __slots__ = ("M", "rng", "pos", "moved")

    def __init__(self, M: int, rng) -> None:
        self.M = int(M)
        self.rng = rng
        self.pos = 0
        self.moved: dict[int, int] = {}   # only touched positions are stored

    def next_id(self) -> int | None:
        if self.pos >= self.M:
            return None
        p = self.pos
        j = self.rng.randrange(p, self.M)
        vj = self.moved.get(j, j)
        vp = self.moved.get(p, p)
        if j != p:
            self.moved[j] = vp
            self.moved[p] = vj
        self.pos += 1
        return vj

    def __len__(self) -> int:
        return self.M


class ExplicitShuffle:
    """The same rule with the full array materialised; used ONLY to verify ``LazyShuffle``.

    It costs O(M) memory and a full ``list(range(M))`` up front, so it is not the production scanner.
    """

    __slots__ = ("M", "rng", "pos", "a")

    def __init__(self, M: int, rng) -> None:
        self.M = int(M)
        self.rng = rng
        self.pos = 0
        self.a = list(range(self.M))

    def next_id(self) -> int | None:
        if self.pos >= self.M:
            return None
        p = self.pos
        j = self.rng.randrange(p, self.M)
        a = self.a
        a[p], a[j] = a[j], a[p]
        self.pos += 1
        return a[p]

    def __len__(self) -> int:
        return self.M


def make_scanner(mode: str, M: int, rng):
    if mode == "lazy":
        return LazyShuffle(M, rng)
    if mode == "explicit":
        return ExplicitShuffle(M, rng)
    raise KeyError(f"unknown scanner mode {mode!r}; expected 'lazy' or 'explicit'")


def decode(candidate_id: int, selected: list[int], outside: list[int]) -> tuple[int, int]:
    """``id -> (u, v)`` with the fixed-order mapping, allocating nothing of size M."""
    k = len(outside)
    return selected[candidate_id // k], outside[candidate_id % k]


def candidate_count(selected: list[int], outside: list[int]) -> int:
    return len(selected) * len(outside)
