# Private GitHub research snapshot — 2026-10-27

Repository: `nanjinyao/tactile-posterior-guided-engagement` (private).
Supplementary release: `research-snapshot-2026-10-27`.

## Scope and provenance

This is a curated snapshot of the active project, authorized by the author for
private GitHub storage. It includes simulation source/configs/models/tests and
all retained files under `outputs/current`, hardware source/configs/CAD/data/
learning/runtime/tests, the current manuscript and figures, and two selected
presentation decks. Scientific artifacts are copied without content changes.
NPZ samples, PT checkpoints, MP4s and files >=40 MiB are distributed as private
Release ZIPs; other selected files are committed directly. Every selected
research file has a size and SHA-256 entry in `SNAPSHOT_MANIFEST.json`.

Excluded: copied environments, caches, editor/agent/sync metadata, all nested
Git histories, third-party reference paper collections and independent
reference projects, temporary/archive/backup/conflict files, historical
top-level traces, unselected meeting drafts, and the two identical 1.32 GB
experimental slide decks. These remain in the original local workspace.
The original manuscript repository and its history remain intact locally;
the GitHub repository contains its current file snapshot, not a gitlink.

No project-wide license, public redistribution clearance, final publication
metadata, or new hardware qualification is inferred from this private upload.
IEEE template notices remain in the supplied class file. Private installation
addresses and original provenance paths may remain in scientific records.

## Documentation and validation

The root README adds a Chinese research explanation, Mermaid workflow,
implementation links, method/result table, limitations, and download commands.
The earlier README is preserved in `REPOSITORY_PREPARATION_20260917.md`.
`PUBLICATION_AUDIT.md` describes the historical 2026-09-17 audit and its then
unresolved issues; its statements about no commits/uploads are historical.

The snapshot builder verifies copied files and archived members against source
SHA-256, checks ZIP CRCs and scans selected text for common credential patterns.
The restore tool validates archive/member hashes and paths, rejects conflicting
local files, and restores original relative paths. Manifest checks exclude the
manifest itself and Git metadata. New experimental results are not generated.

See `VALIDATION_20261027.md` for checks actually completed during this upload.
Native-library/full-physics limitations in the previous audit remain applicable
unless explicitly superseded by a recorded check. No robot was connected.

## Documentation follow-up — 2026-10-27

The main branch now also includes `AI_ASSISTED_RESEARCH.md` and README sections
on documented AI assistance, a proposed repeat-project workflow, and learning
through derivations, small examples, counterexamples and evidence tracing.
Historical attribution is distinguished from new prompt/workflow suggestions.
The current main-branch manifest is updated for these documentation files.
The original Release tag remains fixed at the initial upload commit and its
initial manifest; the six scientific archives and their hashes are unchanged.

## Rebuild packaging

From the source workspace with Python 3.10+:

```bash
python tools/package_github.py --output D:/path/to/a/new/snapshot-directory
```

Output must be a new directory outside the source workspace. Packaging does not
initialize or modify any source Git history. It creates `repository/` and
`assets/` and updates the source `docs/SNAPSHOT_MANIFEST.json`.
