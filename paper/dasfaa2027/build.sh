#!/usr/bin/env bash
# Build the DASFAA 2027 submission.
#
# Toolchain note (2026-09-17): this machine has no TeX Live / MiKTeX, so we use a downloaded
# Tectonic binary, which is self-contained and fetches packages on demand.
#
#   tectonic  C:\Users\windows\tools\latex\tectonic\tectonic.exe   (v0.17.0)
#   llncs.cls, splncs04.bst   already copied into src/dasfaa2027/ from the CTAN llncs bundle
#                             (https://mirrors.ctan.org/macros/latex/contrib/llncs.zip)
#
# Measured: 17 pages, all citations resolved, no undefined references.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src/dasfaa2027"
OUT="$HERE/build"
mkdir -p "$OUT"

TECTONIC="${TECTONIC:-/c/Users/windows/tools/latex/tectonic/tectonic.exe}"
if [ ! -x "$TECTONIC" ] && ! command -v "$TECTONIC" >/dev/null 2>&1; then
  echo "tectonic not found at $TECTONIC; set TECTONIC=/path/to/tectonic" >&2
  exit 2
fi

cd "$SRC"
"$TECTONIC" -X compile paper.tex --outdir "$OUT"

cp -f "$OUT/paper.pdf" "$OUT/dasfaa2027_draft.pdf"
echo "Built $OUT/dasfaa2027_draft.pdf"
