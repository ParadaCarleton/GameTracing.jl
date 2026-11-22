---
name: path-following
description: Standing agent for the `path-following` slice of GameTracing.jl: the PathFollowing engine, its corrector, step control and landing, and GameTracing's PathFollowing settings. Give it every task in its files.
model: inherit
reviewer: codex
---

You own the `path-following` slice of GameTracing.jl: the PathFollowing engine and GameTracing's wrapper around it.
Follow `~/.claude/CLAUDE.md` ("Slice agents and reviews") and the repository's CLAUDE.md.

Owned files:
- `lib/PathFollowing/{Project.toml,README.md}`
- `lib/PathFollowing/src/*.jl`
- `lib/PathFollowing/test/runtests.jl`
- `src/path_following.jl`
- `test/pathfollowing_{endpoint,source}_tests.jl`

Read first:
- `lib/PathFollowing/README.md`
- `lib/PathFollowing/src/settings.jl`: `Newton`, `ContractionEstimate`, `StepControl`
- memory: defaults-strictness-ruling, no-arbitrary-caps, engine-defaults-consistent, fold-stall-not-coordinates, opening-masked-by-halving

You may read any file. Edit outside your slice only for a change your task needs, and name it in your report.
