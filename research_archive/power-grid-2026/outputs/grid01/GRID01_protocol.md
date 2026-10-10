# GRID01 — LJN heuristic hybrid qualification

Frozen 2026-10-08 before downloading action assets or executing LJN.

## Purpose

Qualify the non-NN LJNAgent returned by make_agent_challenge at commit ca0637eab9f098be7f206ed0e46a3900cd4deec0 on bundled l2rpn_idf_2023. Record compute use, module triggers and execution/forecast flags. No comparative algorithm claim, learning, teacher generation or full research dataset download.

## Execution envelope

- Engineering limit half a day; stop with a concrete qualification report if code needs decision-logic changes.
- Add-on dependency/code/action downloads <=150 MiB. Use a new layered venv reading the unchanged GRID00 package directory; install additions locally and constrain all existing versions.
- Fetch two original action assets: action_12_unsafe.npz and action_N1_unsafe.npz. Do not fetch NN assets/models.
- Preserve original source. Allow only explicitly documented compatibility/import changes. Verify the selected LJNAgent class AST is identical (apart from named NumPy compatibility substitutions if present).
- Native IDF settings, LightSimBackend, author MaxRhoReward, seed 0, fresh env+agent per trajectory. No changed thermal limits, faults, permissions or reward parameters.
- Two bundled scenarios in lexicographic order, up to 288 actual steps each; stop on native done. Replay the first 24 steps of each with a fresh env+agent. Maximum executed steps 624.
- Per trajectory process cap 600 seconds after imports, whole simulation allocation 1800 seconds; checkpoint each completed step. A timeout or exception is a qualification limitation, not a scientific failure.
- Non-mutating telemetry wraps module get_act, OptimModule._solve_problem, BaseObservation.simulate. Forward identical arguments/results; no extra forecast calls. Count and time actual calls.

## Checks and reporting

1. Source/action/dependency hashes; original asset schema and vector dimension; reconstruct every released action against native action_space. Loading a candidate does not certify context-specific legality.
2. Selected class and factory semantics; NumPy/import adaptations; no artificial agent fallback.
3. Per step: legal input obs hash, returned action hash/vector, rho, native illegal/ambiguous/done/exception flags; module call/return flags and duration; forecast call horizon and cost; continuous solver status and configured solver/fallback order.
4. Exact replay equality of actions/observations/flags; timing is not required equal. Disclose solver nondeterminism if any.
5. Save same-field rho/p_or forecasts only as diagnostics when already requested by the agent. Actual future never enters agent inputs. Raw observation vectors of different classes must not be subtracted.
6. Save raw records and vectors, verify hashes and telemetry/accounting. This is artifact and native API validation, not an independent AC feasibility solver.

A pass means baseline integration for these short public test prefixes. It does not establish a performance gap, strong-baseline failure frequency, learning opportunity, or novelty. Next research hypothesis must be justified by observed behavior and current primary literature.

## Compatibility amendment before the replacement batch

The first integration run completed 60 actual steps then failed during decision 61, before the physical step, at CVXPY integer-Parameter float-dtype indexing. Keep that prefix and failed-decision telemetry as an integration failure. A checked cast accepts only finite exactly integral indices and changes only read indexing expressions. Fractional IDs raise an error; no rounding or model/constraint changes are permitted. Inverse-normalized OptimModule AST must equal the original after the enumerated import/NumPy substitutions.

To preserve the original maximum of 624 executed steps including the failed attempt, the replacement batch uses two fixed 258-step prefixes and two 24-step replays: 60 + 2*258 + 2*24 = 624. This is a resource amendment, not a performance selection; no scenario, seed, control parameter, asset or evaluation metric changes. All times/steps in the failed attempt remain part of costs. No further replacement batch is authorized within this qualification protocol.

## Recording correction and disclosed resource amendment

Offline review found that telemetry's calls to BaseAction.to_vect() populated a cache before the continuous optimizer modified the action. Some setters do not invalidate that cache. Consequently the first 564-step batch cannot certify final action vectors or matched-action forecast diagnostics. Archive it in INVALID_action_vector_cache; its previous PASS/residual summary is superseded and must not support conclusions.

This corrective amendment explicitly supersedes the preceding no-replacement restriction and 624-step cap for one identical rerun of 564 steps, making the total allowance 1188 actual steps (60 failed integration +564 invalid recording +564 corrected recording). No scenario, policy parameter, asset, seed, deadline or optimization objective changes. The extra computation is a disclosed protocol deviation/correction, not an independent experiment. Half-day engineering limit remains. No additional candidate method, training or scenario expansion is added.

Action telemetry now directly reads the fields listed in attr_list_vect, exactly matching a fresh native vectorization while leaving _vectorized untouched. Before rerun, regression checks must detect stale native cached vectors after a continuous setter, verify fresh vector fields, round-trip reconstruction, and non-mutation of the live cache. Saved per-step continuous-action magnitudes must match vector slices in the final audit.
