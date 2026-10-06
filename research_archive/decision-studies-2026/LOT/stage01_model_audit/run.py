"""One-command entry point.

    python run.py            # full audit (default)
    python run.py --quick    # same checks, smaller enumeration cap

Equivalent to running `python stage01_run.py`. Kept as a stable, short entry point so the
reproduction command does not depend on the internal module layout.
"""

import sys

from stage01_run import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        argv = []
    sys.exit(main(argv))
