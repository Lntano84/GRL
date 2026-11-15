"""Build a local anonymous ACM source from the verified English draft."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
KEYS = ("", "marot2021", "ljn", "hassouna", "xie2026")
ESCAPE = str.maketrans({
    "\\": r"\textbackslash{}", "%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#",
    "$": r"\$", "{": r"\{", "}": r"\}", "\u2014": "---", "\u2013": "--", "\u2212": "-",
})
INLINE = re.compile(r"\\\(.+?\\\)|\[[1-4](?:[–,-][ 1-4]+)*\]")
PHRASES = {
    "The earlier 2035 deletion ablation is reported separately.": "The earlier 2035 deletion ablation is reported separately. All runs use a LightSimBackend, and the saved metadata records one PyTorch intra-op thread. Component records identify OSQP for DC flow and continuous optimization in the newly timed 2039 runs.",
    "GRID330 evaluates": "The ordering and residual experiment evaluates",
    "GRID334 compares": "The paired stopping experiment compares",
    "the GRID330 pair": "that comparison",
    "the four GRID330 horizons": "the four horizons in the ordering experiment",
    "within GRID330": "within the ordering experiment",
    "In GRID334": "In the paired stopping experiment",
}

def prose(line):
    line = line.replace(chr(96), "")
    for old, new in PHRASES.items():
        line = line.replace(old, new)
    parts = INLINE.split(line)
    # Preserve the exact math text and map manuscript citation numbers to bibitems.
    matches = list(INLINE.finditer(line))
    result = [parts[0].translate(ESCAPE)]
    for match, rest in zip(matches, parts[1:], strict=True):
        token = match.group()
        if token.startswith(r"\("):
            result.append(token)
        else:
            labels = [int(value) for value in re.findall(r"[1-4]", token)]
            if "–" in token or "-" in token:
                labels = range(labels[0], labels[-1] + 1)
            result.append(r"\cite{" + ",".join(KEYS[i] for i in labels) + "}")
        result.append(rest.translate(ESCAPE))
    return "".join(result)

def tabular(lines):
    assert len(lines) == 14
    rows = [" & ".join(prose(cell.strip()) for cell in line[1:-1].split("|")) + r" \\"
            for line in (lines[0], *lines[2:])]
    return [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-week paired outcomes. Candidate deltas count the first arm minus the reference. Cost and time deltas are percentages relative to the reference.}",
        r"\label{tab:paired}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule", rows[0], r"\midrule", *rows[1:], r"\bottomrule",
        r"\end{tabular}", r"\end{table*}",
    ]

def main():
    md = ROOT / "manuscript_en.md"
    receipt = json.loads((ROOT / "evidence_review.json").read_bytes())
    key = f"paper/{ROOT.name}/{md.name}"
    assert receipt["passed"] and receipt["evidence_hashes"][key] == hashlib.sha256(md.read_bytes()).hexdigest()
    lines = md.read_text(encoding="utf8").splitlines()
    title = lines[0].removeprefix("# ")
    assert lines[2] == "## Abstract"
    start = lines.index("## 1 Introduction")
    end = lines.index("## References")
    output = [
        r"\documentclass[sigconf,review,anonymous]{acmart}",
        r"\settopmatter{printacmref=false}",
        r"\setcopyright{none}",
        r"\acmISBN{}",
        r"\acmDOI{}",
        r"\acmConference[ACM e-Energy 2027]{ACM e-Energy 2027}{June 2027}{Atlanta, GA, USA}",
        r"\setlength{\emergencystretch}{2em}",
        r"\title{" + prose(title) + "}",
        r"\begin{document}",
        r"\begin{abstract}", *(prose(line) for line in lines[3:start]), r"\end{abstract}",
        r"\maketitle",
    ]
    table = []
    for line in lines[start:end]:
        if line.startswith("| "):
            table.append(line)
            continue
        if table:
            output.extend(tabular(table))
            table.clear()
        if line.startswith("## "):
            output.append(r"\section{" + prose(re.sub(r"^## \d+ ", "", line)) + "}")
        elif line.startswith("### "):
            output.append(r"\subsection{" + prose(line[4:]) + "}")
        else:
            paragraph = prose(line).replace("Table 1", r"Table~\ref{tab:paired}")
            paragraph = paragraph.replace(
                r"strict loading bound \(0<\rho<\min(\rho_t,\theta_t)\),",
                r"strict loading bound \[0<\rho<\min(\rho_t,\theta_t),\]",
            )
            output.append(paragraph)
    assert not table
    output.append(r"\begin{thebibliography}{4}")
    for line in filter(None, lines[end + 1:]):
        match = re.fullmatch(r"([1-4])\. (.*?)\*(.*?)\*(.*?)\[(.*?)\]\((.*?)\)", line)
        assert match, line
        idx, leading, title, trailing, label, url = match.groups()
        output += [r"\bibitem{" + KEYS[int(idx)] + "}",
                   prose(leading) + r"\emph{" + prose(title) + "}" + prose(trailing)
                   + r"\href{" + url + "}{" + prose(label) + "}."]
    output += [r"\end{thebibliography}", r"\end{document}"]
    rendered = "\n".join(output) + "\n"
    assert not re.search(r"GRID330|GRID334|Lntano84|D:/|\\author(?:\{|\\)|\\affiliation\{", rendered, re.I)
    assert len(re.findall(r"^2039-\d\d-\d\d & ", rendered, re.M)) == receipt["table_rows"] == 12
    assert rendered.count(r"\bibitem{") == 4
    folder = Path(sys.argv[1])
    folder.mkdir(exist_ok=False)
    (folder / "manuscript.tex").write_text(rendered, encoding="utf8", newline="\n")

if __name__ == "__main__":
    main()
