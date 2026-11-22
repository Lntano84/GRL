"""Rebuild the closed-loop table from the paired results."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {
    "BOUNDED261_GUARD_DC": "Mixed guarded 261",
    "FRAME_GUARD": "Original FRAME",
    "PARENT208_GUARD_DC": "Mixed guarded 208",
    "LOGEIPC_DC": "LogEIPC",
}


def percentage(value):
    if not value:
        return "---"
    number = float(value)
    return ("0.00" if number == 0 else f"{number:+.2f}") + r"\%"


def render(rows):
    table = [r"""\begin{table*}[t]
\centering
\caption{Paired closed-loop outcomes on the four 2039 development horizons. Each count cell shows reference / evaluated arm. Cost and controller time changes are relative to the reference. A dash denotes unavailable full-week cost or comparable timing.}
\label{tab:closed}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Week & Evaluated arm vs.\ reference & Steps/arm & Candidate queries & Total forecasts & Cost change & Time change\\
\midrule"""]
    previous = None
    for row in rows:
        if previous is not None and previous != row["reference"]:
            table.append(r"\addlinespace")
        previous = row["reference"]
        title = f'{LABELS[row["arm"]]} vs. {LABELS[row["reference"]]}'
        counts = [
            rf"{row[f'reference_{metric}']} / {row[f'arm_{metric}']}"
            for metric in ("candidate_queries", "total_forecasts")
        ]
        cells = [
            row["date"], title, row["arm_steps"], *counts,
            percentage(row["full_week_cost_delta_pct"]),
            percentage(row["controller_delta_pct"]),
        ]
        table.append(" & ".join(cells) + r" \\")
    table.append(r"""\bottomrule
\end{tabular}
\end{table*}""")
    return "\n".join(table)


if __name__ == "__main__":
    with (ROOT / "paired_results.csv").open(encoding="utf-8", newline="") as data:
        print(render(csv.DictReader(data)))

