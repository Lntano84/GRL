# Reproduction scope and entry points

## Summary tables (Python standard library only)

After GRID62 completes and passes the saved-data audit:

```powershell
python work/grid_paper/reproduce_tables.py --source-root . --out outputs/grid_paper_table_reproduction
```

The script independently calculates date-level ratios from `finished.json` and checks the reported summaries. It does not call the simulator, use a GNN, or rerun training. Each output folder must be new.

## Saved-data audit in the original qualified workspace

```powershell
& work/grid02/.venv/Scripts/python.exe -X utf8 work/grid62/audit.py
```

This checks vectors, public feedback, causal policy replay, cost/timing accounting, physical repeated paths, and power events. It does not independently re-solve AC power flow.

## Actual trajectory reproduction

The original runner is `work/grid62/run.py`. It verifies the frozen code hashes and refuses to repeat a completed stage. Do not run it against the archived completed directory to regenerate evidence. A future independent reproduction must use a **new output directory**, the publicly sourced environment and LJN assets at the recorded versions, compatible installed packages, and the full dependency source closure listed in `code_freeze.json`.

The Git archive preserves small stage scripts/configs/summaries and local payload hash receipts. A single archived stage folder alone is **not** a standalone runnable full simulator distribution: it depends on earlier qualified controller/cache modules and third-party assets. `source_receipts.json`, the archive manifests, and `software_environment.json` identify these prerequisites. Public LJN/SoftGNN source and data provenance are credited separately. No copied checkpoint or huge trace payload is silently implied to be present in the lightweight clone.

The original raw `vectors.npz`, `steps.jsonl.gz`, candidate calls, and state vectors remain in `outputs/grid62/runs/` locally and are identified by `payload_manifest.json` after audit. A publication artifact should either expose these payloads via a documented large-file release or explain the retrieval procedure and licenses before claiming one-command complete reproduction.
