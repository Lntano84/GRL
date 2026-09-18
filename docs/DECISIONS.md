# GRL Decision Log

## 2026-09-02 — Use marginal gain as a candidate signal

The intended learning target is conditional marginal gain `Delta(v|S)`, evaluated by within-state
ranking and downstream decision quality rather than aggregate regression error.

## 2026-09-05 — Abandon the static-IC learning-versus-RIS claim

The same-protocol OPIM-C comparison showed that the old static-IC route did not establish a quality
or runtime advantage over mature RIS. That claim is permanently abandoned. No pre-fix static-IC
number may be reused as evidence for the current paper.

## 2026-09-15 — Change the setting to threshold-dependent overexposure

Human-authorized route change: study the threshold-window overexposure setting and target DASFAA
2027. The motivation is that the standard non-negative seed-independent coverage representation
does not match the objective, so the method must be evaluated with an explicit state-tracking
oracle. RL and an additional predictor architecture remain out of scope until the core evidence
exists.

## 2026-09-18 — Current post-fix paper decision

The state-machine correction and MC=300 sweep changed the evidence boundary:

- keep the exact coverage counterexample;
- keep the corrected 8-graph regime table;
- withdraw the old negative-share, baseline, calibration, surrogate-gap, and learned-scorer numbers;
- do not claim a general sample-efficient algorithm until post-fix quality--cost experiments pass;
- treat the measurement protocol as a valid contribution, not as a substitute for missing algorithmic
  evidence.

The current DASFAA submission remains conditional. If corrected sequential experiments cannot show
near-oracle quality with a reproducible reduction in expensive oracle work, remove the
sample-efficiency framing and retarget the manuscript.

## 2026-09-18 — Reference audit decision

Only bibliographic records verified against publisher/official proceedings pages or DBLP may remain
in the submission. The audit corrected five materially wrong records, removed one unsupported
related-work claim, one duplicate entry and one uncited entry, and aligned the prose with the
published methods. Short venue names are retained to keep the fully verified bibliography inside
the 16-page LNCS limit; DOI, volume/issue, article number and page data remain in BibTeX.
