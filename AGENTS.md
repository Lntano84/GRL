# GRL Agent Guide

## Project purpose
GRL is a research codebase for Influence Maximization, with the current research direction centered on learning conditional marginal gain Δ(v|S) and using the learned signal in sequential decision making / a learning-augmented oracle.

## First-read order
Whenever starting a fresh Codex/agent session in this repository, read in this order:
1. `AGENTS.md`
2. `docs/RESEARCH_STATE.md`
3. `docs/NEXT_STEPS.md`
4. `docs/DECISIONS.md` when making technical-route changes
5. `docs/EXPERIMENT_LOG.md` when interpreting previous experiments
6. Existing technical docs such as `docs/EXPERIMENT_PROTOCOL.md`, `docs/CODEBASE_GUIDE.md`, and `docs/PAPER_CODE_MAPPING.md` as needed

## Workspace rules
- Primary workspace: `/workspace/GRL`.
- Do not create GRL temporary worktrees, clones, test folders, or experiment scratch directories directly under `/workspace`.
- Keep all GRL-specific temporary material inside `/workspace/GRL` (or a documented project-local subdirectory).
- Do not delete or overwrite experiment outputs merely to make the Git working tree look clean.
- Before modifying code, inspect the current branch, working tree, running processes, and relevant experiment directories.
- Avoid committing large datasets, checkpoints, copied worktrees, temporary logs, or one-off smoke-run directories unless explicitly requested.

## Research-state discipline
After a meaningful experiment or technical decision:
- update `docs/RESEARCH_STATE.md` if the confirmed state changed;
- append a concise item to `docs/EXPERIMENT_LOG.md` if the experiment is worth retaining;
- update `docs/DECISIONS.md` for durable technical choices and rejected alternatives;
- keep `docs/NEXT_STEPS.md` limited to current actionable priorities.

## Coding and experiment discipline
- Prefer configuration-driven, reproducible experiment entry points.
- Preserve deterministic seeds/configs where applicable.
- Report both predictive metrics and downstream decision-quality metrics when evaluating learned marginal gains.
- Treat ranking quality under a fixed seed set/state as important; aggregate MAE/MSE alone is insufficient for validating a marginal-gain oracle.
- Do not claim a result as confirmed until the command/config/result artifact can be located.

## Git discipline
- Main repository remote: **`Lntano84/GRL`**, branch **`main`** (local branch is `master`;
  push with `HEAD:main`).  Git remote name in this working copy: `lntano-src`.
- **Push after every meaningful unit of work, without being asked.**  A stage, a fix, a corrected
  test, an experiment result — if it is worth committing it is worth pushing immediately.  Do not
  leave work only on the local machine.
- **Use `scripts\git_push.cmd "<message>"` (or a message-file path) to commit and push.**  Plain
  `git push` writes progress to stderr, so shells report a non-zero exit code and a red error line
  even on success; that has already produced a false "did you push?" check.  The helper decides
  success by comparing the remote ref to local HEAD and prints `PUSHED OK <sha>` or
  `PUSH FAILED` with both shas.  It is a `.cmd` rather than a `.ps1` because PowerShell execution
  policy blocks local scripts on this machine.
- Do not include unrelated untracked project-local experiment directories in commits.
- Make focused commits for code/config/state-document changes.
- Never leave the working tree dirty at the end of a piece of work: either commit it or say
  explicitly why it is uncommitted.
