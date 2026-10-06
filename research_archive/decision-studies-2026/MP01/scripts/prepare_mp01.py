from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

from validate_mp01 import check_instance, check_solution, load_grid, load_scene_endpoints


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INSTANCES = [
    ("empty-32-32", 350),
    ("warehouse-10-20-10-2-1", 200),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_member(archive: zipfile.ZipFile, member: str, target: Path) -> None:
    try:
        data = archive.read(member)
    except KeyError as exc:
        raise SystemExit(f"Missing archive member: {member}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def main() -> None:
    for subdir in ("map", "scene", "initial_states"):
        (DATA / subdir).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ROOT / "lns2_init_states.zip") as states_zip:
        with zipfile.ZipFile(ROOT / "mapf-map.zip") as maps_zip:
            with zipfile.ZipFile(ROOT / "mapf-scen-random.zip") as scenes_zip:
                manifest = []
                for map_name, agent_count in INSTANCES:
                    map_member = f"{map_name}.map"
                    map_path = DATA / "map" / map_member
                    extract_member(maps_zip, map_member, map_path)
                    for scene_id in range(1, 5):
                        state_name = (
                            f"map-{map_name}-scene-{scene_id}-agent-{agent_count}.json"
                        )
                        state_member = f"lns2_init_states/{state_name}"
                        state_path = DATA / "initial_states" / state_name
                        extract_member(states_zip, state_member, state_path)

                        scene_name = f"{map_name}-random-{scene_id}.scen"
                        scene_member = f"scen-random/{scene_name}"
                        scene_path = DATA / "scene" / scene_name
                        extract_member(scenes_zip, scene_member, scene_path)

                        state = json.loads(state_path.read_text(encoding="utf-8"))
                        grid = load_grid(map_path)
                        endpoints = load_scene_endpoints(scene_path, agent_count)
                        input_errors = check_instance(state, endpoints, agent_count)
                        initial_soc, validity_errors = check_solution(state, grid, agent_count)
                        if input_errors or validity_errors:
                            errors = input_errors + validity_errors
                            raise SystemExit(f"{state_name}: " + "; ".join(errors[:5]))
                        manifest.append(
                            {
                                "map": map_name,
                                "scene": scene_id,
                                "agents": agent_count,
                                "map_path": str(map_path.relative_to(ROOT)),
                                "scene_path": str(scene_path.relative_to(ROOT)),
                                "initial_state_path": str(state_path.relative_to(ROOT)),
                                "initial_soc": initial_soc,
                                "map_sha256": sha256(map_path),
                                "scene_sha256": sha256(scene_path),
                                "initial_state_sha256": sha256(state_path),
                            }
                        )

    (DATA / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with (DATA / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Prepared {len(manifest)} instances in {DATA}")
    for row in manifest:
        print(
            f"{row['map']} scene {row['scene']} agents={row['agents']} "
            f"initial_SOC={row['initial_soc']}"
        )


if __name__ == "__main__":
    main()
