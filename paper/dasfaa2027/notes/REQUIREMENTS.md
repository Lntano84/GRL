# DASFAA 2027 submission requirements

> Snapshot taken 2026-09-17. Base facts were verified against the official site
> (`https://dasfaa2027.github.io/index.html`) on 2026-09-15 and recorded in
> `../../../../论文整理_三条理由/_三条理由的推论与课题启示/CCFB投稿窗口与IM友好度评估.md`.
> **Items marked ⚠️ must be re-checked on the official site before submission.**
> The official site could not be re-fetched on 2026-09-17 from this machine (the hostname
> resolves to a non-public address under the current network policy).

## Verified dates

| milestone | date |
| --- | --- |
| Paper submission | **2026-11-25** |
| Notification | 2027-01-25 |
| Camera-ready | 2027-02-20 |
| Conference | 2027-05-27 to 05-30, Shenyang, China |

Countdown from 2026-09-17: **69 days**.

## Format (⚠️ confirm the template version on the official site)

| item | value |
| --- | --- |
| Proceedings style | **LNCS** (Springer), `llncs.cls` |
| Page limit | **16 pages** including references ⚠️ |
| Review | **double-blind** |
| Submission system | CMT ⚠️ |
| Topics | official list includes "Graph and social network analytics", "Data mining and knowledge discovery", "Neural networks and deep learning" |

## Consequences for this paper

1. **Double-blind** means no author names, affiliations, acknowledgements, or self-identifying
   repository links in the submitted PDF. Our artefact links must be anonymised.
2. **LNCS 16 pages** is a comfortable budget for 5 claims; the ICLR skeleton's 9-page limit is
   not binding here.
3. **LNCS** requires a different document class from the ICLR skeleton. The `math_commands.tex`
   and figure sources carry over; `paper.tex` and the style files do not.
4. Direction ① reports **negative and inapplicability results**. LNCS/DASFAA has no
   "reproducibility statement" slot, so the honesty constraints in `CLAIMS.md` must be handled
   inside the normal sections (a dedicated "Threats to validity" subsection).

## ⚠️ Open items to verify manually

- [ ] Confirm the exact page limit and whether references count.
- [ ] Confirm the CMT submission URL and whether an abstract-only pre-registration is required.
- [ ] Download `llncs.cls` and the official `sample-authordraft.tex` from Springer.
- [ ] Confirm whether a supplementary/artefact track exists.

## Build status

**The paper compiles.** Toolchain installed 2026-09-17:

| component | source |
| --- | --- |
| Tectonic 0.17.0 | `C:\Users\windows\tools\latex\tectonic\tectonic.exe` (GitHub release, self-contained, fetches packages on demand) |
| `llncs.cls` | CTAN bundle `https://mirrors.ctan.org/macros/latex/contrib/llncs.zip`, copied into `src/dasfaa2027/` |
| `splncs04.bst` | same bundle |

Command (see `build.sh`):

```powershell
cd paper\dasfaa2027\src\dasfaa2027
& "C:\Users\windows\tools\latex\tectonic\tectonic.exe" -X compile paper.tex --outdir ..\..\build
```

**Latest build: 18 pages, 0 undefined citations, 0 overfull hboxes.** That is 2 pages over the
16-page limit, so trimming is required (see `WRITING_STATUS.md`).

`tectonic` writes a harmless `Fontconfig error: Cannot load default config file` line to stderr;
the build succeeds and the exit code is non-zero only because of that. Ignore it.

The LNCS bundle also ships `llncsdoc.pdf`/`llncsdoc.tex` (the class documentation) and a sample
paper; `samplepaper.tex` was not included in the CTAN zip, so the anonymous-double-blind author
block is our own construction and should be checked against the official DASFAA author kit
before submission.
