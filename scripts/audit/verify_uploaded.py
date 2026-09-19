"""Verify that every Go/No-Go deliverable is on the GitHub remote, and identical to local.

Answers "did you upload it?" by comparing blob hashes rather than by trusting a push message: a
push that reported success and a push that actually landed look the same in a log line, and git
writes progress to stderr so an exit code is not evidence either.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DELIVERABLES = [
    "docs/VALIDATION_TARGETED.md",
    "docs/VALIDITY_FIXES.md",
    "docs/GO_NO_GO_REPORT.md",
    "docs/WITHDRAWN_RESULTS.md",
    "docs/P0_COMPLETION_REPORT.md",
    "docs/results/validation_small.json",
    "docs/results/validation_large.json",
    "docs/results/scorer_diagnosis.json",
    "docs/results/gate_calibration.json",
    "docs/results/raw_weight_profiles.json",
    "docs/results/quality_cost.json",
    "docs/results/gate1b_sequential_vs_static.json",
    "src/grl/oracle/targeted_mc.py",
    "src/grl/evaluation/gate.py",
    "src/grl/scoring/exposure.py",
    "src/grl/baselines/random_pruning.py",
    "scripts/experiments/go_no_go.py",
    "scripts/audit/run_validation.cmd",
    "scripts/audit/diagnose_scorer.py",
    "scripts/audit/print_scorer_diagnosis.py",
    "scripts/audit/summarise_validation.py",
    "scripts/audit/verify_go_no_go_artifacts.py",
    "tests/test_targeted_oracle.py",
    "tests/test_stream_independence.py",
]


def git(*args: str) -> tuple[int, str]:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="lntano-src")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    ref = f"{args.remote}/{args.branch}"

    git("fetch", args.remote, args.branch, "--quiet")
    _, local_head = git("rev-parse", "HEAD")
    _, remote_head = git("rev-parse", ref)
    _, counts = git("rev-list", "--left-right", "--count", f"HEAD...{ref}")
    _, dirty = git("status", "--porcelain")
    _, url = git("remote", "get-url", args.remote)

    print("=" * 96)
    print("GITHUB UPLOAD VERIFICATION")
    print("=" * 96)
    print(f"  remote            : {url}")
    print(f"  local  HEAD       : {local_head}")
    print(f"  {ref:<17} : {remote_head}")
    print(f"  ahead / behind    : {counts}   (left = local only, right = remote only)")
    print(f"  working tree      : {'clean' if not dirty else 'DIRTY'}")
    if dirty:
        for line in dirty.splitlines()[:10]:
            print(f"      {line}")
    print()

    failures = 0
    print(f"  {'deliverable':<54}{'tracked':>8}{'remote':>8}{'content':>12}")
    for name in DELIVERABLES:
        code_tracked, _ = git("ls-files", "--error-unmatch", name)
        tracked = code_tracked == 0
        code_remote, _ = git("cat-file", "-e", f"{ref}:{name}")
        on_remote = code_remote == 0
        content = "-"
        if tracked and on_remote:
            _, local_blob = git("hash-object", name)
            _, remote_blob = git("rev-parse", f"{ref}:{name}")
            content = "identical" if local_blob == remote_blob else "DIFFERS"
            if content != "identical":
                failures += 1
        else:
            failures += 1
        print(f"  {name:<54}{'yes' if tracked else 'NO':>8}"
              f"{'yes' if on_remote else 'NO':>8}{content:>12}")

    print()
    synced = local_head == remote_head and not dirty
    print(f"  HEAD in sync      : {synced}")
    print(f"  deliverables ok   : {len(DELIVERABLES) - failures}/{len(DELIVERABLES)}")
    if failures:
        print()
        print("  NOT UPLOADED.  Re-run scripts/git_push.cmd; it decides success by comparing the")
        print("  remote ref to local HEAD rather than by an exit code.")
    return 1 if (failures or not synced) else 0


if __name__ == "__main__":
    raise SystemExit(main())
