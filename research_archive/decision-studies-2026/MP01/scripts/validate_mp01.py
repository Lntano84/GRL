from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_grid(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        map_row = lines.index("map") + 1
    except ValueError as exc:
        raise ValueError(f"{path}: missing MovingAI map header") from exc
    height_line = next((line for line in lines if line.startswith("height ")), None)
    width_line = next((line for line in lines if line.startswith("width ")), None)
    if height_line is None or width_line is None:
        raise ValueError(f"{path}: missing map dimensions")
    height = int(height_line.split()[1])
    width = int(width_line.split()[1])
    rows = lines[map_row : map_row + height]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError(f"{path}: map rows do not match declared dimensions")
    return rows


def load_solution(path: Path) -> dict[str, list[list[int]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: solution root must be a JSON object")
    return value


def load_scene_endpoints(path: Path, expected_agents: int) -> list[tuple[int, int, int, int]]:
    endpoints: list[tuple[int, int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 9 or not fields[0].isdigit():
            continue
        # MovingAI stores x,y; the JSON input stores row,column.
        sx, sy, gx, gy = (int(fields[i]) for i in (4, 5, 6, 7))
        endpoints.append((sy, sx, gy, gx))
    if len(endpoints) < expected_agents:
        raise ValueError(f"{path}: expected {expected_agents} scenario rows")
    return endpoints[:expected_agents]


def check_instance(
    solution: dict[str, list[list[int]]],
    endpoints: list[tuple[int, int, int, int]],
    expected_agents: int,
) -> list[str]:
    errors: list[str] = []
    if len(solution) != expected_agents:
        errors.append(f"expected {expected_agents} paths, got {len(solution)}")
    for agent in range(expected_agents):
        path = solution.get(str(agent))
        if not isinstance(path, list) or not path:
            errors.append(f"agent {agent}: missing nonempty path")
            continue
        if path[0] != list(endpoints[agent][:2]):
            errors.append(f"agent {agent}: start does not match scenario")
        if path[-1] != list(endpoints[agent][2:]):
            errors.append(f"agent {agent}: goal does not match scenario")
    return errors


def check_solution(
    solution: dict[str, list[list[int]]], grid: list[str], expected_agents: int
) -> tuple[int, list[str]]:
    errors: list[str] = []
    if len(solution) != expected_agents:
        errors.append(f"expected {expected_agents} paths, got {len(solution)}")
    width = len(grid[0])
    height = len(grid)
    paths: list[list[tuple[int, int]]] = []
    total_cost = 0
    for agent in range(expected_agents):
        raw_path = solution.get(str(agent))
        if not isinstance(raw_path, list) or not raw_path:
            errors.append(f"agent {agent}: missing nonempty path")
            paths.append([])
            continue
        path: list[tuple[int, int]] = []
        for timestep, coordinate in enumerate(raw_path):
            if (
                not isinstance(coordinate, list)
                or len(coordinate) != 2
                or not all(isinstance(value, int) for value in coordinate)
            ):
                errors.append(f"agent {agent}: malformed coordinate at t={timestep}")
                break
            row, col = coordinate
            if not (0 <= row < height and 0 <= col < width):
                errors.append(f"agent {agent}: out-of-bounds coordinate at t={timestep}")
                break
            if grid[row][col] in "@T":
                errors.append(f"agent {agent}: obstacle at t={timestep}")
                break
            if path and abs(row - path[-1][0]) + abs(col - path[-1][1]) > 1:
                errors.append(f"agent {agent}: illegal move at t={timestep}")
                break
            path.append((row, col))
        if not path:
            errors.append(f"agent {agent}: no usable path")
        else:
            total_cost += len(path) - 1
        paths.append(path)

    if any(not path for path in paths):
        return total_cost, errors

    makespan = max(map(len, paths))
    for timestep in range(makespan):
        positions = [path[min(timestep, len(path) - 1)] for path in paths]
        if len(set(positions)) != expected_agents:
            errors.append(f"vertex conflict at t={timestep}")
            break
        if timestep:
            previous = [path[min(timestep - 1, len(path) - 1)] for path in paths]
            edges = set()
            for first, (source, target) in enumerate(zip(previous, positions)):
                if source == target:
                    continue
                if (target, source) in edges:
                    errors.append(f"edge swap at t={timestep}")
                    return total_cost, errors
                edges.add((source, target))
    return total_cost, errors


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value
