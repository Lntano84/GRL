"""Information sets for native PAR-10; no learner or hidden-runtime interface."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    lo: float
    hi: float
    lo_open: bool = False
    hi_open: bool = False

    def contains(self, x):
        return ((x > self.lo if self.lo_open else x >= self.lo)
                and (x < self.hi if self.hi_open else x <= self.hi))


def score_set(kind, value, cutoff=600.0, penalty=10.0):
    """kind 0 unknown, 1 premature censor, 2 completion, 3 native timeout."""
    kind, value = int(kind), float(value)
    if kind == 2:
        assert 0 < value <= cutoff
        return [Segment(value, value)]
    if kind == 3:
        assert value >= cutoff
        return [Segment(penalty * cutoff, penalty * cutoff)]
    assert kind in (0, 1)
    lower = value if kind == 1 else 0.0
    assert lower >= 0
    if lower >= cutoff:
        return [Segment(penalty * cutoff, penalty * cutoff)]
    return [Segment(lower, cutoff, True, False),
            Segment(penalty * cutoff, penalty * cutoff)]


def signed_difference(ka, va, kb, vb, cutoff=600.0):
    """All feasible components of score(B)-score(A), not just their envelope."""
    return [Segment(b.lo-a.hi, b.hi-a.lo,
                    b.lo_open or a.hi_open, b.hi_open or a.lo_open)
            for a in score_set(ka, va, cutoff)
            for b in score_set(kb, vb, cutoff)]


def preference(parts):
    lo = min(s.lo for s in parts)
    hi = max(s.hi for s in parts)
    zero_possible = any(s.contains(0.0) for s in parts)
    if lo == hi == 0:
        return 'tie'
    if lo >= 0:
        return 'a_no_worse' if zero_possible else 'a_better'
    if hi <= 0:
        return 'b_no_worse' if zero_possible else 'b_better'
    return 'unidentified'


def regret_parts(parts):
    pref = preference(parts)
    assert pref != 'unidentified', 'A preference is required for a regret weight'
    if pref.startswith('b'):
        return [Segment(-s.hi, -s.lo, s.hi_open, s.lo_open) for s in parts]
    return list(parts)


def describe(ka, va, kb, vb, cutoff=600.0):
    signed = signed_difference(ka, va, kb, vb, cutoff)
    pref = preference(signed)
    out = {'preference': pref,
           'signed_lower': min(s.lo for s in signed),
           'signed_upper': max(s.hi for s in signed)}
    if pref != 'unidentified':
        parts = regret_parts(signed)
        lo, hi = min(s.lo for s in parts), max(s.hi for s in parts)
        out.update(regret_lower=lo, regret_upper=hi, regret_exact=(lo == hi),
                   regret_segments=[s.__dict__ for s in parts])
    return out


def proxy_possible(info, weight):
    return any(Segment(**s).contains(float(weight)) for s in info['regret_segments'])


def update_visible(old, kind, value):
    """Audit replay of feedback, preserving the largest censor bound."""
    ok, ov = old
    kind, value = int(kind), float(value)
    if ok in (2, 3):
        assert (kind, value) == old
        return old
    if kind == 1:
        return (1, max(ov if ok == 1 else 0.0, value))
    if kind == 2:
        assert ok != 1 or value > ov
        return (2, value)
    if kind == 3:
        return (3, value)
    raise ValueError('Only a revealed observation can update visible state')
