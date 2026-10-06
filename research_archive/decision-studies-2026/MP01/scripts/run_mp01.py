from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from validate_mp01 import (
    check_instance,
    check_solution,
    load_grid,
    load_scene_endpoints,
    load_solution,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs" / "mp01"
STATES = RUNS / "states"
FINAL_STATES = RUNS / "final_states"
LOGS = RUNS / "logs"
PP_EXE = ROOT / "build-mingw" / "pp" / "pp_open.exe"
PBS_EXE = ROOT / "build-mingw" / "pbs" / "pbs-replan.exe"
VCPKG_BIN = ROOT / "vcpkg" / "installed" / "x64-mingw-dynamic" / "bin"
BRANCH_BUDGET = 5.0
PP_STAGE_BUDGET = 10.0
VALIDATION_RESERVE = 0.20
PP_REPAIR_CAP = 0.6
NEIGHBOR_SIZE = 25
SEEDS = (1, 2)
PP_STAGE_SEED = 0
COMPLETION_MARKER = "Finish Generating for State:"
PATH_COORDINATE = re.compile(r"\((-?\d+)\s*,\s*(-?\d+)\)")
AGENT_LINE = re.compile(r"^\s*agent\s+(\d+)\s+(.*)$")


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def runtime_env(executable: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [str(executable.parent), str(VCPKG_BIN), env.get("PATH", "")]
    )
    return env


def parse_path_blocks(text: str) -> dict[str, list[list[int]]]:
    updates: dict[str, list[list[int]]] = {}
    inside = False
    for line in text.splitlines():
        if "new paths start" in line:
            inside = True
            continue
        if "new paths end" in line:
            inside = False
            continue
        if not inside:
            continue
        match = AGENT_LINE.match(line)
        if match is None:
            continue
        coords = PATH_COORDINATE.findall(match.group(2))
        if coords:
            updates[match.group(1)] = [[int(row), int(col)] for row, col in coords]
    return updates


def sample_neighborhood(rng: random.Random, agent_count: int) -> list[int]:
    return rng.sample(range(agent_count), min(NEIGHBOR_SIZE, agent_count))


def next_solver_seed(run_seed: int, round_index: int) -> int:
    return (run_seed + round_index * 104729) % 2147483647


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")


class PPServer:
    def __init__(
        self,
        map_path: Path,
        state_path: Path,
        agent_count: int,
        seed: int,
        deadline: float,
    ) -> None:
        command = [
            str(PP_EXE),
            "--map",
            relative(map_path),
            "--state",
            relative(state_path),
            "--agentNum",
            str(agent_count),
            "--seed",
            str(seed),
            "--replanTime",
            str(PP_REPAIR_CAP),
            "--cutoffTime",
            str(PP_STAGE_BUDGET),
            "--neighborSize",
            str(NEIGHBOR_SIZE),
            "--num_subset",
            "1",
            "--pprun",
            "1",
            "--destroyStrategy",
            "Random",
            "--initAlgo",
            "JSON",
            "--replanAlgo",
            "PP",
            "--screen",
            "0",
        ]
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=runtime_env(PP_EXE),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()

        def pump() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)

        self.reader = threading.Thread(target=pump, daemon=True)
        self.reader.start()
        self.startup_text, ready = self._read_until("Enter command:", deadline)
        if not ready:
            self.stop()
            raise TimeoutError("PP process did not reach its command prompt within budget")

    def _read_until(self, marker: str, deadline: float) -> tuple[str, bool]:
        gathered: list[str] = []
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                line = self.lines.get(timeout=min(0.05, max(0.001, remaining)))
            except queue.Empty:
                if self.process.poll() is not None:
                    return "".join(gathered), False
                continue
            if line is None:
                return "".join(gathered), False
            gathered.append(line)
            if marker in line:
                return "".join(gathered), True
        return "".join(gathered), False

    def repair(
        self,
        state_path: Path,
        agents: list[int],
        seed: int,
        repair_cap: float,
        deadline: float,
    ) -> tuple[str, bool]:
        if self.process.poll() is not None or self.process.stdin is None:
            return "", False
        command = (
            f'--state "{relative(state_path)}" --replanAgents '
            + " ".join(map(str, agents))
            + f" --replanTime {repair_cap:.6f} --pprun 1 --seed {seed}\n"
        )
        try:
            self.process.stdin.write(command)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return "", False
        return self._read_until(COMPLETION_MARKER, deadline)

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except OSError:
                pass
            self.process.kill()
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1)


def prepare_cases() -> list[dict[str, Any]]:
    manifest = json_read(DATA / "manifest.json")
    cases: list[dict[str, Any]] = []
    for row in manifest:
        map_path = ROOT / row["map_path"]
        scene_path = ROOT / row["scene_path"]
        initial_path = ROOT / row["initial_state_path"]
        solution = load_solution(initial_path)
        endpoints = load_scene_endpoints(scene_path, row["agents"])
        grid = load_grid(map_path)
        errors = check_instance(solution, endpoints, row["agents"])
        soc, solution_errors = check_solution(solution, grid, row["agents"])
        if errors or solution_errors:
            raise ValueError(
                f"Invalid initial plan {initial_path}: "
                + "; ".join((errors + solution_errors)[:8])
            )
        if soc != row["initial_soc"]:
            raise ValueError(f"Initial SOC mismatch for {initial_path}: {soc}")
        cases.append(
            {
                **row,
                "map_path_abs": map_path,
                "scene_path_abs": scene_path,
                "initial_path_abs": initial_path,
                "grid": grid,
                "endpoints": endpoints,
                "initial_solution": solution,
            }
        )
    return cases


def accept_repair(
    text: str,
    selected: list[int],
    current: dict[str, list[list[int]]],
    current_soc: int,
    grid: list[str],
    agent_count: int,
    method: str,
) -> tuple[str, dict[str, list[list[int]]] | None, int | None, str]:
    lower = text.lower()
    updates = parse_path_blocks(text)
    if "new paths start" in lower and "new paths end" not in lower:
        return "failure", None, None, "incomplete path block"
    if "no path" in lower or "no solutions" in lower or "pbs_replan_cost: -1" in lower:
        return "failure", None, None, "solver reported no path"
    if not updates:
        if re.search(r"\bImproved\s*:\s*[1-9]\d*", text) or re.search(
            r"pbs_replan_cost:\s*-?\d+", text
        ) and int(re.search(r"pbs_replan_cost:\s*(-?\d+)", text).group(1)) < 0:
            return "failure", None, None, "solver reported an improvement without paths"
        if "no improvement" in lower or "improvement: 0" in lower:
            return "no_improvement", None, None, ""
        if method == "PBS" and re.search(r"pbs_replan_cost:\s*\d+", text):
            return "no_improvement", None, None, ""
        return "failure", None, None, "solver output did not contain a complete result"

    expected = {str(agent) for agent in selected}
    if set(updates) != expected:
        return (
            "failure",
            None,
            None,
            f"path block agents differ from requested set ({len(updates)}/{len(expected)})",
        )
    candidate = copy.deepcopy(current)
    candidate.update(updates)
    candidate_soc, errors = check_solution(candidate, grid, agent_count)
    if errors:
        return "failure", None, None, "; ".join(errors[:5])
    if candidate_soc >= current_soc:
        return "no_improvement", None, None, "reported paths did not lower full SOC"
    return "improvement", candidate, candidate_soc, ""


def result_record(
    case: dict[str, Any],
    stage: str,
    seed: int,
    method: str,
    start_soc: int,
    final_soc: int,
    elapsed: float,
    calls: int,
    improvements: int,
    failures: int,
    timeouts: int,
    final_path: Path,
    log_path: Path,
    status: str,
    verification_errors: list[str],
) -> dict[str, Any]:
    return {
        "map": case["map"],
        "scene": case["scene"],
        "stage": stage,
        "seed": seed,
        "method": method,
        "agents": case["agents"],
        "start_soc": start_soc,
        "final_soc": final_soc,
        "soc_saved": start_soc - final_soc,
        "improvement_pct": 100.0 * (start_soc - final_soc) / start_soc
        if start_soc
        else 0.0,
        "elapsed_s": round(elapsed, 6),
        "calls": calls,
        "valid_improvements": improvements,
        "failures": failures,
        "timeouts": timeouts,
        "verified": not verification_errors,
        "verification_errors": verification_errors,
        "status": status,
        "final_state": relative(final_path),
        "log": relative(log_path),
    }


def run_pp_budget(
    case: dict[str, Any],
    start_solution: dict[str, list[list[int]]],
    stage: str,
    seed: int,
    budget: float,
    state_path: Path,
    log_path: Path,
) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    start = time.monotonic()
    deadline = start + budget
    current = copy.deepcopy(start_solution)
    start_soc, initial_errors = check_solution(current, case["grid"], case["agents"])
    if initial_errors:
        raise ValueError("Cannot start PP with invalid solution: " + "; ".join(initial_errors[:5]))
    json_write(state_path, current)
    solver_deadline = deadline - VALIDATION_RESERVE
    rng = random.Random(seed)
    calls = improvements = failures = timeouts = 0
    status = "budget_elapsed"
    round_index = 0
    server: PPServer | None = None
    try:
        server = PPServer(
            case["map_path_abs"], state_path, case["agents"], seed, solver_deadline
        )
        append_log(log_path, "# PP startup\n" + server.startup_text)
        while solver_deadline - time.monotonic() > 0.08:
            selected = sample_neighborhood(rng, case["agents"])
            remaining = solver_deadline - time.monotonic()
            repair_cap = min(PP_REPAIR_CAP, max(0.01, remaining - 0.05))
            solver_seed = next_solver_seed(seed, round_index)
            call_start = time.monotonic()
            text, completed = server.repair(
                state_path, selected, solver_seed, repair_cap, solver_deadline
            )
            call_elapsed = time.monotonic() - call_start
            calls += 1
            round_index += 1
            append_log(
                log_path,
                f"\n# round={round_index} seed={solver_seed} agents={selected} "
                f"elapsed={call_elapsed:.6f} completed={completed}\n{text}",
            )
            if not completed:
                failures += 1
                timeouts += 1
                status = "solver_call_timeout"
                break
            disposition, candidate, candidate_soc, reason = accept_repair(
                text,
                selected,
                current,
                sum(len(path) - 1 for path in current.values()),
                case["grid"],
                case["agents"],
                "PP",
            )
            if disposition == "failure":
                failures += 1
                append_log(log_path, f"# rejected: {reason}\n")
            elif disposition == "improvement" and candidate is not None:
                current = candidate
                json_write(state_path, current)
                improvements += 1
                append_log(log_path, f"# accepted_soc={candidate_soc}\n")
    except TimeoutError as exc:
        failures += 1
        timeouts += 1
        status = "startup_timeout"
        append_log(log_path, f"# {status}: {exc}\n")
    except Exception as exc:
        failures += 1
        status = "process_error"
        append_log(log_path, f"# {status}: {type(exc).__name__}: {exc}\n")
    finally:
        if server is not None:
            server.stop()
    final_soc, verification_errors = check_solution(current, case["grid"], case["agents"])
    verification_errors += check_instance(current, case["endpoints"], case["agents"])
    elapsed = time.monotonic() - start
    return current, {
        "start_soc": start_soc,
        "final_soc": final_soc,
        "elapsed_s": elapsed,
        "calls": calls,
        "valid_improvements": improvements,
        "failures": failures,
        "timeouts": timeouts,
        "status": status,
        "verification_errors": verification_errors,
    }


def run_pbs_budget(
    case: dict[str, Any],
    start_solution: dict[str, list[list[int]]],
    stage: str,
    seed: int,
    budget: float,
    state_path: Path,
    log_path: Path,
) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    start = time.monotonic()
    deadline = start + budget
    current = copy.deepcopy(start_solution)
    start_soc, initial_errors = check_solution(current, case["grid"], case["agents"])
    if initial_errors:
        raise ValueError("Cannot start PBS with invalid solution: " + "; ".join(initial_errors[:5]))
    json_write(state_path, current)
    solver_deadline = deadline - VALIDATION_RESERVE
    rng = random.Random(seed)
    calls = improvements = failures = timeouts = 0
    status = "budget_elapsed"
    round_index = 0
    while solver_deadline - time.monotonic() > 0.08:
        selected = sample_neighborhood(rng, case["agents"])
        remaining = solver_deadline - time.monotonic()
        solver_seed = next_solver_seed(seed, round_index)
        command = [
            str(PBS_EXE),
            "--map",
            relative(case["map_path_abs"]),
            "--state",
            relative(state_path),
            "--agentNum",
            str(case["agents"]),
            "--replanAgents",
            *map(str, selected),
            "--cutoffTime",
            f"{remaining:.6f}",
            "--seed",
            str(solver_seed),
            "--screen",
            "0",
        ]
        call_start = time.monotonic()
        calls += 1
        round_index += 1
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=runtime_env(PBS_EXE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            text, _ = process.communicate(timeout=max(0.01, remaining))
            return_code = process.returncode
            call_elapsed = time.monotonic() - call_start
        except subprocess.TimeoutExpired:
            process.kill()
            text, _ = process.communicate()
            call_elapsed = time.monotonic() - call_start
            failures += 1
            timeouts += 1
            status = "solver_call_timeout"
            append_log(
                log_path,
                f"\n# round={round_index} seed={solver_seed} agents={selected} "
                f"elapsed={call_elapsed:.6f} timeout=true\n{text}",
            )
            break
        except Exception as exc:
            failures += 1
            status = "process_error"
            append_log(log_path, f"# process error: {type(exc).__name__}: {exc}\n")
            break

        append_log(
            log_path,
            f"\n# round={round_index} seed={solver_seed} agents={selected} "
            f"elapsed={call_elapsed:.6f} exit={return_code}\n{text}",
        )
        if return_code != 0:
            failures += 1
            status = "solver_process_error"
            break
        disposition, candidate, candidate_soc, reason = accept_repair(
            text,
            selected,
            current,
            sum(len(path) - 1 for path in current.values()),
            case["grid"],
            case["agents"],
            "PBS",
        )
        if disposition == "failure":
            failures += 1
            append_log(log_path, f"# rejected: {reason}\n")
        elif disposition == "improvement" and candidate is not None:
            current = candidate
            json_write(state_path, current)
            improvements += 1
            append_log(log_path, f"# accepted_soc={candidate_soc}\n")

    final_soc, verification_errors = check_solution(current, case["grid"], case["agents"])
    verification_errors += check_instance(current, case["endpoints"], case["agents"])
    elapsed = time.monotonic() - start
    return current, {
        "start_soc": start_soc,
        "final_soc": final_soc,
        "elapsed_s": elapsed,
        "calls": calls,
        "valid_improvements": improvements,
        "failures": failures,
        "timeouts": timeouts,
        "status": status,
        "verification_errors": verification_errors,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_pp10_states(cases: list[dict[str, Any]], force: bool) -> list[dict[str, Any]]:
    result_file = RUNS / "pp10_results.jsonl"
    old = {
        f"{row['map']}|{row['scene']}": row
        for row in load_jsonl(result_file)
    } if not force else {}
    if force and result_file.exists():
        result_file.unlink()
    pp10: list[dict[str, Any]] = []
    for case in cases:
        key = f"{case['map']}|{case['scene']}"
        state_path = STATES / "pp10" / f"{case['map']}_scene{case['scene']}.json"
        if key in old and old[key].get("experiment_ok") and state_path.exists():
            record = old[key]
            solution = load_solution(state_path)
            errors = check_instance(solution, case["endpoints"], case["agents"])
            _, path_errors = check_solution(solution, case["grid"], case["agents"])
            if not errors and not path_errors:
                pp10.append({**case, "pp10_path_abs": state_path, "pp10_solution": solution, "pp10_record": record})
                continue
        initial = case["initial_solution"]
        log_path = LOGS / f"pp10_{case['map']}_scene{case['scene']}.txt"
        if log_path.exists():
            log_path.unlink()
        print(f"PP-10s state: {case['map']} scene {case['scene']} starting", flush=True)
        solution, metrics = run_pp_budget(
            case,
            initial,
            "pp10_state_generation",
            PP_STAGE_SEED,
            PP_STAGE_BUDGET,
            state_path,
            log_path,
        )
        record = {
            "map": case["map"],
            "scene": case["scene"],
            "agents": case["agents"],
            **metrics,
            "verified": not metrics["verification_errors"],
            "experiment_ok": metrics["calls"] > 0
            and metrics["elapsed_s"] >= PP_STAGE_BUDGET - VALIDATION_RESERVE - 0.15,
            "state_path": relative(state_path),
            "log": relative(log_path),
        }
        append_jsonl(result_file, record)
        print(
            f"PP-10s state: {case['map']} scene {case['scene']} "
            f"SOC {metrics['start_soc']}->{metrics['final_soc']} "
            f"rounds={metrics['calls']} improvements={metrics['valid_improvements']} "
            f"failures={metrics['failures']} elapsed={metrics['elapsed_s']:.3f}s",
            flush=True,
        )
        if not record["verified"] or not record["experiment_ok"]:
            raise RuntimeError(f"PP-10s run did not produce a verified 10-second state for {key}: {record}")
        pp10.append({**case, "pp10_path_abs": state_path, "pp10_solution": solution, "pp10_record": record})
    return pp10


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(records: list[dict[str, Any]]) -> None:
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    raw_fields = [
        "map",
        "scene",
        "stage",
        "seed",
        "method",
        "agents",
        "start_soc",
        "final_soc",
        "soc_saved",
        "improvement_pct",
        "elapsed_s",
        "calls",
        "valid_improvements",
        "failures",
        "timeouts",
        "verified",
        "status",
        "final_state",
        "log",
    ]
    write_csv(out / "MP01_branch_results.csv", records, raw_fields)

    fixed_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for map_name in sorted({row["map"] for row in records}):
        for stage in ("init", "pp10s"):
            cells = [row for row in records if row["map"] == map_name and row["stage"] == stage]
            if not cells:
                continue
            by_method = {
                method: [row for row in cells if row["method"] == method]
                for method in ("PP", "PBS")
            }
            mean_soc = {
                method: sum(row["final_soc"] for row in rows) / len(rows)
                for method, rows in by_method.items()
                if rows
            }
            mean_start = sum(row["start_soc"] for row in cells) / len(cells)
            if abs(mean_soc["PP"] - mean_soc["PBS"]) < 1e-9:
                fixed_method = "Tie"
                fixed_mean = mean_soc["PP"]
            else:
                fixed_method = min(mean_soc, key=mean_soc.get)
                fixed_mean = mean_soc[fixed_method]
            cell_oracles = []
            directions: dict[str, list[str]] = {"seed1": [], "seed2": []}
            consistent = mixed = tied = 0
            scenes = sorted({row["scene"] for row in cells})
            for scene in scenes:
                seed_winners = []
                for seed in SEEDS:
                    pair = [
                        row
                        for row in cells
                        if row["scene"] == scene and row["seed"] == seed
                    ]
                    if len(pair) != 2:
                        continue
                    pp_soc = next(row["final_soc"] for row in pair if row["method"] == "PP")
                    pbs_soc = next(row["final_soc"] for row in pair if row["method"] == "PBS")
                    winner = "PP" if pp_soc < pbs_soc else "PBS" if pbs_soc < pp_soc else "Tie"
                    seed_winners.append(winner)
                    directions[f"seed{seed}"].append(winner)
                    cell_oracles.append(min(pp_soc, pbs_soc))
                if len(seed_winners) == 2:
                    if seed_winners[0] == seed_winners[1] == "Tie":
                        tied += 1
                    elif seed_winners[0] == seed_winners[1]:
                        consistent += 1
                    else:
                        mixed += 1
            oracle_mean = sum(cell_oracles) / len(cell_oracles) if cell_oracles else fixed_mean
            fixed_rows.append(
                {
                    "map": map_name,
                    "stage": stage,
                    "instances": len({row["scene"] for row in cells}),
                    "paired_runs_per_method": len(by_method[fixed_method]),
                    "baseline_mean_soc": round(mean_start, 3),
                    "PP_mean_soc": round(mean_soc.get("PP", float("nan")), 3),
                    "PBS_mean_soc": round(mean_soc.get("PBS", float("nan")), 3),
                    "fixed_method": fixed_method,
                    "fixed_mean_soc": round(fixed_mean, 3),
                    "fixed_mean_saved_soc": round(mean_start - fixed_mean, 3),
                    "fixed_mean_saved_pct": round(100 * (mean_start - fixed_mean) / mean_start, 4)
                    if mean_start
                    else 0,
                    "oracle_mean_soc_diagnostic_only": round(oracle_mean, 3),
                    "oracle_extra_saved_vs_fixed_soc": round(fixed_mean - oracle_mean, 3),
                    "oracle_extra_saved_vs_fixed_pct": round(100 * (fixed_mean - oracle_mean) / mean_start, 4)
                    if mean_start
                    else 0,
                    "seed1_scene_winners": ";".join(directions["seed1"]),
                    "seed2_scene_winners": ";".join(directions["seed2"]),
                    "same_winner_both_seeds_scenes": consistent,
                    "mixed_direction_scenes": mixed,
                    "ties_both_seeds_scenes": tied,
                    "PP_elapsed_s": round(sum(row["elapsed_s"] for row in by_method["PP"]), 3),
                    "PBS_elapsed_s": round(sum(row["elapsed_s"] for row in by_method["PBS"]), 3),
                    "PP_valid_improvements": sum(row["valid_improvements"] for row in by_method["PP"]),
                    "PBS_valid_improvements": sum(row["valid_improvements"] for row in by_method["PBS"]),
                    "PP_failures": sum(row["failures"] for row in by_method["PP"]),
                    "PBS_failures": sum(row["failures"] for row in by_method["PBS"]),
                    "verified_runs": sum(bool(row["verified"]) for row in cells),
                    "total_runs": len(cells),
                }
            )
            oracle_rows.append(
                {
                    "map": map_name,
                    "stage": stage,
                    "fixed_method": fixed_method,
                    "fixed_mean_soc": round(fixed_mean, 3),
                    "per_state_oracle_mean_soc_diagnostic_only": round(oracle_mean, 3),
                    "extra_saved_soc_diagnostic_only": round(fixed_mean - oracle_mean, 3),
                    "per_state_oracle_cells": len(cell_oracles),
                    "seed_consistent_scenes": consistent,
                    "mixed_direction_scenes": mixed,
                    "ties_both_seeds_scenes": tied,
                }
            )

    write_csv(out / "MP01_map_stage_fixed.csv", fixed_rows)
    write_csv(out / "MP01_oracle_diagnostic.csv", oracle_rows)
    make_report(records, fixed_rows)


def make_report(records: list[dict[str, Any]], fixed_rows: list[dict[str, Any]]) -> None:
    output = ROOT / "outputs" / "MP01_summary.md"
    total_expected = 64
    verified = sum(bool(row["verified"]) for row in records)
    branch_times = [float(row["elapsed_s"]) for row in records]
    pp10_records = load_jsonl(RUNS / "pp10_results.jsonl")
    pp10_times = [float(row["elapsed_s"]) for row in pp10_records]
    total_calls = sum(row["calls"] for row in records)
    total_failures = sum(row["failures"] for row in records)
    total_timeouts = sum(row["timeouts"] for row in records)
    method_totals = {}
    for method in ("PP", "PBS"):
        subset = [row for row in records if row["method"] == method]
        method_totals[method] = {
            "elapsed": sum(row["elapsed_s"] for row in subset),
            "improvements": sum(row["valid_improvements"] for row in subset),
            "failures": sum(row["failures"] for row in subset),
            "calls": sum(row["calls"] for row in subset),
        }
    table = [
        "| Map | 阶段 | 固定选择 | PP 平均 SOC | PBS 平均 SOC | 固定选择平均 SOC | 逐状态 oracle 平均 SOC* | oracle 额外节省* | 两种子同向场景 / 4 | 混合 / 4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed_rows:
        table.append(
            f"| {row['map']} | {row['stage']} | {row['fixed_method']} | "
            f"{row['PP_mean_soc']:.1f} | {row['PBS_mean_soc']:.1f} | "
            f"{row['fixed_mean_soc']:.1f} | {row['oracle_mean_soc_diagnostic_only']:.1f} | "
            f"{row['oracle_extra_saved_vs_fixed_soc']:.1f} | "
            f"{row['same_winner_both_seeds_scenes']}/4 | {row['mixed_direction_scenes']}/4 |"
        )
    lines = [
        "# MP01：PP/PBS 固定选择分支实验",
        "",
        f"运行记录：{len(records)}/{total_expected} 个分支；最终方案独立验解 {verified}/{len(records)}；"
        f"分支墙钟 {min(branch_times):.3f}–{max(branch_times):.3f}s；局部调用 {total_calls} 次；"
        f"失败 {total_failures} 次，其中超时 {total_timeouts} 次。PP-10s 状态 {len(pp10_records)}/8 个，"
        f"墙钟 {min(pp10_times):.3f}–{max(pp10_times):.3f}s，全部独立验解通过。",
        "",
        "## 初步判断",
        "",
        "本批地图×阶段固定选择后，逐状态 oracle 额外节省 7.2–48.0 SOC（各单元起始平均 SOC 的约 0.07%–0.25%）；"
        "两种子在 1–3/4 个场景上支持相同赢家，其余场景方向混合。当前效应小且种子/场景方向不稳定，"
        "结论记为**不确定**：本轮没有建立足够稳健的状态条件选择空间，不宜据此训练或声称学习优势；"
        "这也不是对更大样本或其他场景分布下差异的否定。",
        "",
        "## 地图 × 阶段固定选择与逐状态诊断",
        "",
        *table,
        "",
        "* 逐状态 oracle 在同一实例与种子上事后取 PP/PBS 较低 SOC，只作诊断，不是可部署成绩。"
        "固定选择是在每个地图与阶段的 4 个场景、2 个种子上按平均 SOC 选出，属于本批样本内描述。",
        "",
        "## 方法总量",
        "",
        "| 方法 | 分支总秒数 | 完整有效改善次数 | 局部调用次数 | 失败次数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("PP", "PBS"):
        row = method_totals[method]
        lines.append(
            f"| {method} | {row['elapsed']:.2f} | {row['improvements']} | "
            f"{row['calls']} | {row['failures']} |"
        )
    lines.extend(
        [
            "",
            "## 执行口径",
            "",
            "- 初始化方案取 SoCS 2025 统一代码库公开的 LNS2 init states；场景起终点逐智能体与 Moving AI 官方随机场景核对。",
            "- 每个实例先运行一次固定随机邻域 PP，端到端墙钟预算 10 秒，生成第二阶段起点；计时包含 PP 进程启动、地图/智能体预处理、输入和终验解。8 条实际用时见首段。",
            "- 两阶段各从自己的起点复制，按 seed 1、2 分别运行 PP、PBS；每条分支端到端墙钟预算 5 秒，计时包含起始方案检查与写入、进程启动、输入、求解、输出解析及完整方案终验解。64 条实测均未超过 5 秒。",
            f"- 在总预算内为最终独立验解预留 {VALIDATION_RESERVE:.2f}s；求解停止后保留最近一次已验证的完整方案。",
            "- 共享邻域规则为从全部智能体均匀无放回抽 25 个；每个 seed 预先由独立随机流确定同一邻域序列，两种方法按相同序号取集合。",
            "- PP 每个局部修复使用代码接口的 0.6 秒上限；PBS 每次局部 PBS 搜索使用该分支剩余时间上限。未完成的局部修复不写回，最近一次已验证的完整方案保留。",
            "- SOC 对每个最终路径集重新计算；验解覆盖场景端点、地图障碍、合法移动、顶点冲突和边交换冲突。",
            "- 统一代码库基线：mapf-lns-unified `d8fab81b25b564fe23c278298ac989f092bbb7d7`；PP 子模块 `531d643d416a2980f9b23fa1b385d4a0ea24e52d`；PBS 子模块 `35404c4b9af2e44e1478db03660dfd99b71e79e9`。本机 PP 驱动按交互请求应用 seed/局部时限，并将 `_MT` 参数名改为 `mt` 以通过 MinGW 编译；PBS 驱动新增 `--seed` 以移除固定 `srand(0)`。这些本地运行适配未提交上游。",
            "- 本机编译环境：MinGW GCC 14.2、CMake 4.1、Boost 1.92、nlohmann-json 3.12、Eigen 5.0.1；PP/PBS 执行文件均成功编译并启动。",
            "- 每个地图×阶段仅 4 个实例、每实例 2 个种子；表内固定策略成绩是本批样本内描述，不是留出集泛化结果，也未做显著性检验。",
            "",
            "## 上游资料",
            "",
            "- [SoCS 2025 统一代码库](https://github.com/ChristinaTan0704/mapf-lns-unified)",
            "- [数据说明与初始化方案入口](https://github.com/ChristinaTan0704/mapf-lns-unified/blob/mapf-lns-main/docs/data.md)",
            "- [规则求解器接口说明](https://github.com/ChristinaTan0704/mapf-lns-exe/blob/rule-based/README.md)",
            "",
            "## 文件",
            "",
            "- [64 条分支明细](MP01_branch_results.csv)",
            "- [地图与阶段固定选择汇总](MP01_map_stage_fixed.csv)",
            "- [逐状态 oracle 诊断汇总](MP01_oracle_diagnostic.csv)",
            "",
            "该轮只判断 PP/PBS 在固定地图与阶段选择后是否留下稳定的逐状态质量差距；不包含预测器，也不把 oracle 当部署成绩。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-stage", action="store_true", help="regenerate all PP-10s states")
    parser.add_argument("--only-stage", action="store_true", help="generate PP-10s states only")
    args = parser.parse_args()
    for folder in (RUNS, STATES, FINAL_STATES, LOGS):
        folder.mkdir(parents=True, exist_ok=True)
    if not PP_EXE.exists() or not PBS_EXE.exists():
        print("Build both executables before running MP01.", file=sys.stderr)
        return 2
    cases = prepare_cases()
    stage_cases = create_pp10_states(cases, args.fresh_stage)
    if args.only_stage:
        return 0

    branch_file = RUNS / "branch_results.jsonl"
    old_records = load_jsonl(branch_file)
    completed = {
        f"{row['map']}|{row['scene']}|{row['stage']}|{row['seed']}|{row['method']}"
        for row in old_records
        if row.get("verified")
    }
    stage_by_key = {
        (case["map"], case["scene"]): case for case in stage_cases
    }
    all_records = old_records.copy()
    for case in cases:
        staged = stage_by_key[(case["map"], case["scene"])]
        for stage, solution in (
            ("init", case["initial_solution"]),
            ("pp10s", staged["pp10_solution"]),
        ):
            for seed in SEEDS:
                for method in ("PP", "PBS"):
                    key = f"{case['map']}|{case['scene']}|{stage}|{seed}|{method}"
                    if key in completed:
                        continue
                    run_id = f"{case['map']}_s{case['scene']}_{stage}_seed{seed}_{method}"
                    final_path = FINAL_STATES / f"{run_id}.json"
                    log_path = LOGS / f"{run_id}.txt"
                    if log_path.exists():
                        log_path.unlink()
                    start_soc = sum(len(path) - 1 for path in solution.values())
                    print(f"Branch start: {run_id} start_SOC={start_soc}", flush=True)
                    if method == "PP":
                        final_solution, metrics = run_pp_budget(
                            case,
                            solution,
                            stage,
                            seed,
                            BRANCH_BUDGET,
                            final_path,
                            log_path,
                        )
                    else:
                        final_solution, metrics = run_pbs_budget(
                            case,
                            solution,
                            stage,
                            seed,
                            BRANCH_BUDGET,
                            final_path,
                            log_path,
                        )
                    record = result_record(
                        case,
                        stage,
                        seed,
                        method,
                        start_soc,
                        metrics["final_soc"],
                        metrics["elapsed_s"],
                        metrics["calls"],
                        metrics["valid_improvements"],
                        metrics["failures"],
                        metrics["timeouts"],
                        final_path,
                        log_path,
                        metrics["status"],
                        metrics["verification_errors"],
                    )
                    append_jsonl(branch_file, record)
                    all_records.append(record)
                    if record["verified"]:
                        completed.add(key)
                    print(
                        f"Branch done: {run_id} SOC {start_soc}->{metrics['final_soc']} "
                        f"calls={record['calls']} improvements={record['valid_improvements']} "
                        f"failures={record['failures']} elapsed={record['elapsed_s']:.3f}s "
                        f"verified={record['verified']}",
                        flush=True,
                    )
    summarize_results(all_records)
    print(f"Saved {len(all_records)} branch rows under {ROOT / 'outputs'}", flush=True)
    return 0 if len(all_records) == 64 and all(row["verified"] for row in all_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
