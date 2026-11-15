"""Render the paper's paired results table from the included CSV."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABEL = {
    "BOUNDED261_GUARD_DC": "Guarded mixed 261",
    "PARENT208_GUARD_DC": "Guarded mixed 208",
    "FRAME_GUARD": "Original FRAME",
    "LOGEIPC_DC": "LogEIPC",
}
HEADER = "| Week | Arm vs. reference | Steps per arm | Candidate delta | Cost delta | Controller time delta |"
SEPARATOR = "| --- | --- | ---: | ---: | ---: | ---: |"

def percent(value):
    if not value:
        return "—"
    number = float(value)
    return "0.00%" if number == 0 else f"{number:+.2f}%"

def render(rows):
    lines = [HEADER, SEPARATOR]
    for row in rows:
        title = f'{LABEL[row["arm"]]} vs. {LABEL[row["reference"]]}'
        cells = (row["date"], title, row["arm_steps"], row["candidate_delta"],
                 percent(row["full_week_cost_delta_pct"]), percent(row["controller_delta_pct"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

if __name__ == "__main__":
    with (ROOT / "paired_results.csv").open(encoding="utf8", newline="") as stream:
        print(render(list(csv.DictReader(stream))))
