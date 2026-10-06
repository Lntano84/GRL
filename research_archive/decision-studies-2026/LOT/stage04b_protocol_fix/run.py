"""One-command entry point for stage 04b.

    python run.py --outdir .

Pipeline, in order (each step is quick except the retest):

    python stage04b_selection_test.py   # output-selection interface, 5 cases, no solver
    python stage04b_repair_tests.py     # corrected R1 / R2 regression tests
    python stage04b_run.py              # 8 states x 4 methods x 20 s  (~11 min)
    python stage04b_savecheck.py        # re-read and re-verify every stored run
    python stage04b_offline.py          # PROVISIONAL scalar regrade of stage 04

`run.py` forwards to `stage04b_run.py`.
"""

import sys

from stage04b_run import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
