"""Driver: wait for the 12 selection chunks, then assemble -> confirm -> merge.

Run detached.  Every stage is idempotent and resumable, so if this driver is killed it can simply be
started again: completed chunks are skipped, completed confirmations are skipped.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\windows\Desktop\_grl_merge\merged")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

import selection_budget_curve as sbc  # noqa: E402

LOG = ROOT / "docs" / "results" / "sbc_driver.log"


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def chunks_complete() -> tuple[int, int]:
    done = total = 0
    for index in range(len(sbc.NEW_SEEDS)):
        for start, end in sbc.CHUNKS:
            total += 1
            path = sbc.SEL_CHUNK.with_name(sbc.SEL_CHUNK.name.format(index=index, start=start, end=end))
            if path.exists():
                try:
                    if json.loads(path.read_text(encoding="utf-8")).get("complete"):
                        done += 1
                except Exception:
                    pass
    return done, total


def main() -> int:
    say("driver started")
    while True:
        done, total = chunks_complete()
        if done == total:
            break
        say(f"waiting: {done}/{total} selection chunks complete")
        time.sleep(120)
    say(f"all {total} selection chunks complete")

    # ---- stage: assemble + freeze (fast, in-process) ----
    for index in range(len(sbc.NEW_SEEDS)):
        if sbc.assemble(index) != 0:
            say(f"assemble failed for cfg{index}")
            return 1
    say("all four configurations assembled and frozen")

    # ---- stage: confirm, four detached processes in parallel ----
    pending = []
    for index in range(len(sbc.NEW_SEEDS)):
        path = sbc.CONFIRM.with_name(sbc.CONFIRM.name.format(index=index))
        done_already = False
        if path.exists():
            try:
                done_already = bool(json.loads(path.read_text(encoding="utf-8")).get("complete"))
            except Exception:
                done_already = False
        if done_already:
            say(f"cfg{index} confirmation already complete")
            continue
        out = ROOT / "docs" / "results" / f"sbc_confirm_cfg{index}.log"
        p = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "audit" / "selection_budget_curve.py"),
             "--confirm", str(index)],
            cwd=str(ROOT), stdout=out.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT)
        pending.append((index, p))
        say(f"launched confirmation for cfg{index} pid={p.pid}")

    while pending:
        still = []
        for index, p in pending:
            if p.poll() is None:
                still.append((index, p))
            else:
                say(f"cfg{index} confirmation exited with {p.returncode}")
        pending = still
        if pending:
            time.sleep(60)
    say("all confirmations finished")

    # ---- stage: merge ----
    rc = sbc.merge()
    say(f"merge exited with {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
