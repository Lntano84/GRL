"""One-command entry point for stage 01b.

    python run.py                 # full 01b audit + independence falsification tests
    python run.py --skip-selfcheck

Equivalent to running `python stage01b_run.py` followed by
`python selfcheck_independence.py`. Kept as a short, stable command so the reproduction
instruction does not depend on the internal module layout.
"""

import subprocess
import sys

from stage01b_run import main

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--skip-selfcheck"]
    rc = main(args)
    if "--skip-selfcheck" not in sys.argv:
        print("\n" + "#" * 90)
        print("# independence falsification tests (selfcheck_independence.py)")
        print("#" * 90, flush=True)
        sub = subprocess.run([sys.executable, "selfcheck_independence.py"],
                             stdout=sys.stdout, stderr=sys.stderr)
        print("selfcheck return code: %d" % sub.returncode, flush=True)
        rc = rc or sub.returncode
    sys.exit(rc)
