"""One-command entry point for stage 03.

    python run.py

Equivalent to `python stage03_run.py`. Reads the frozen twelve disruption states from
`../stage02_repair_reopt/stage02_solutions.json` and writes the local release-benefit table.
"""

import sys

from stage03_run import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
