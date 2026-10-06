"""One-command entry point for stage 04.

    python run.py --phase all        # regenerate instances, solve, verify
    python run.py --phase report     # rebuild the report tables from stored results

Pipeline (each step is resumable; run in this order on a fresh checkout):

    python lsp_gen.py                  # freeze the 12 instances to JSON BEFORE solving
    python stage04_repair_tests.py     # repair_minbatch_v2 regression tests (R1, R2)
    python stage04_run.py --phase all  # nominal -> reference -> timing -> report
    python stage04_savecheck.py        # re-read everything and re-verify

`run.py` forwards to `stage04_run.py`. The full pipeline is intentionally NOT chained into a
single command: the nominal and reference phases cost roughly 15 and 8 minutes of solves, and
the brief asks for the instance parameters to be written out before any solve happens.
"""

import sys

from stage04_run import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
