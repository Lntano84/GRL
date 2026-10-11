# Audit-only performance correction

The first complete audit repeatedly accessed `z['action']` and `z['observation']` inside the per-step loop. NumPy NPZ member access decompresses the complete member each time. The verified audit process alone was stopped before producing a report; all 48 controller runs had already completed. The archived V0 source is retained.

The corrected audit reads each member once, then performs exactly the same per-step vector hash and cost checks. No control source, frozen experiment, vector, learned weight, outcome, or timer changed; no controller was replayed to obtain different data.
