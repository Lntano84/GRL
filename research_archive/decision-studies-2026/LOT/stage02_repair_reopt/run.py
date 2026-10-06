"""One-command entry point for stage 02.

    python run.py

Equivalent to `python stage02_run.py`. Afterwards run `python stage02_savecheck.py` to
verify that everything written to disk reads back consistently.
"""

import sys

from stage02_run import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
