#!/usr/bin/env bash
# Build the DASFAA 2027 submission.
#
# Requires an LNCS toolchain. This machine has none of pdflatex / latexmk / xelatex /
# bibtex, and paper/iclr2027/tools/tectonic is not present in this working copy, so the
# paper has NOT been compiled here. Run this on a machine that has TeX Live with
# llncs.cls (texlive-publishers) or MacTeX.
#
# What you still need to obtain before this can build:
#   - llncs.cls            (Springer LNCS, from the DASFAA 2027 author kit or Springer's
#                           "Proceedings Authors" page)
#   - splncs04.bst         (Springer LNCS bibliography style)
#   - algorithmicx/algpseudocode, booktabs, multirow, xcolor  (standard in texlive-full)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src/dasfaa2027"
OUT="$HERE/build"
mkdir -p "$OUT"

cd "$SRC"

if [ ! -f llncs.cls ]; then
  echo "Missing llncs.cls in $SRC" >&2
  echo "Download the LNCS author kit from Springer and place llncs.cls and splncs04.bst here." >&2
  exit 2
fi

pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex

cp -f paper.pdf "$OUT/dasfaa2027_draft.pdf"
echo "Built $OUT/dasfaa2027_draft.pdf"
