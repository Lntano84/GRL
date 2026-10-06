# Exploratory experiment archive (2026)

This archive preserves the scripts, protocols, reports, inputs with explicit redistribution licenses, raw run records, and audit artifacts from the MP01, SB01, BC01–BC08, and GP00–GP01 research screens. These were exploratory studies across several solver domains; they are not one integrated GRL algorithm and should not be presented as results of the GRL influence-maximization method.

## Study map

| Track | Contents | Status |
|---|---|---|
| MP01 | PP/PBS branching, reports, run records, input provenance, and licensed source patches | Historical screening; see deliverables |
| SB01 | SAT runtime matrix, replay script, sampling manifest, estimator results | Paired192 correction recorded; fixed ridge extension closed |
| BC01–BC08 | Baleen baseline, lower-bound and intervention records, snapshot/ranking analyses | BC05 privileged witness; online legal rules and NEW ranking not supported |
| GP00–GP01 | Mt-KaHyPar engineering records, licensed SuiteSparse inputs, score tables, partitions and audits | GP01 joint gate failed; no new learner trained |

## Reproduction and provenance

Start with each track's report and protocol files. `source_provenance.json` pins external source commits and explains which licensed inputs are included. `file_manifest.csv` contains the SHA-256 and byte length of each archived file other than the manifest itself. Use the original project URLs and recorded hashes to retrieve inputs that are not redistributed.

The original solver source repositories are not vendored. Local source changes used by MP01 and GP01 are preserved as patches with their source revision and applicable license notice. Build products, dependency caches, and the embedded Python runtime are omitted; build manifests and source pins remain.

## Data attribution and modifications

- SAT benchmark runtime data: `mathefuchs/al-for-sat-solver-benchmarking-data`, CC BY 4.0; cite Fuchs, Bach, and Iser, “Active Learning for SAT Solver Benchmarking” (TACAS 2023), and retain the upstream README attribution.
- Baleen traces: CMU PDL's Meta Tectonic traces, offered under the same Apache 2.0 terms as CacheLib; cite Baleen (FAST 2024) and the trace page.
- SuiteSparse matrices: CC BY 4.0. GP01 converts each source matrix to an unweighted, loop-free, duplicate-free undirected structural graph while retaining all vertices; source and normalized hashes plus removal counts are in `GP01/outputs/graph_manifest.json`.
- MAPF input files: not redistributed here because the upstream repository does not declare a top-level data license. The original manifest, source references, and hashes are retained under `MP01/input-provenance/`.

## Large files

Compressed trace and snapshot artifacts are tracked through Git LFS. Ordinary GitHub Git blocks files over 100 MiB; the largest preserved BC06 log requires LFS. If the destination account's LFS quota is unavailable, the LFS objects must be stored in an approved artifact store and referenced by hash rather than dropped.

See `source_provenance.json` and the per-track reports for interpretation limits, protocol deviations, and frozen decisions.
