# GEX Repository Guidelines

This file contains repository-level working notes.

## Session start

- Review the repository guidance before making changes.
- Check `README.md`, `docs/UW_PARITY_PLAN.md`, and `docs/PROGRESS.md` when working on roadmap, architecture, or cross-session tasks.
- Keep work limited to the requested scope.

## Rules

- Do not create, edit, delete, rename, regenerate, or fix tests unless the user explicitly asks for test work.
- Do not run broad test suites unless the user asks or validation clearly requires it.
- Do not manually edit generated files, dependency lockfiles, snapshots, or generated changelog outputs unless the user specifically asks.
- Do not add tool or assistant identities as commit co-authors.

## Writing style

- Do not use em dashes.
- Use plain hyphen characters instead.
- Use clear, short markdown with focused changes.
- Do not rewrite unrelated content for style.
- Keep commit messages, PR titles, comments, and docs plain and terse.

## Engineering

- Keep changes small and reversible.
- Preserve existing public APIs unless the task explicitly asks to change them.
- Reuse existing `quant_analysis` functions instead of duplicating analytics math.
- Keep Streamlit legacy behavior stable unless the user asks to modify the legacy app.
- For the FastAPI and Next.js rebuild, follow `docs/UW_PARITY_PLAN.md`.
- For long work sessions, update `docs/PROGRESS.md` with what changed and the next step.

## Validation

- Prefer lightweight validation scoped to the files changed.
- If validation is skipped because it would require tests, say so clearly.
- Report exact commands run and their results.
