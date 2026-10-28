# GitHub research snapshot — 2026-10-27

Repository: `nanjinyao/tactile-posterior-guided-engagement` (public).
Supplementary release: `research-snapshot-2026-10-27`.

## Scope and provenance

This is a curated snapshot of the active project, initially uploaded privately
and subsequently authorized by the author for public access. It includes simulation source/configs/models/tests and
all retained files under `outputs/current`, hardware source/configs/CAD/data/
learning/runtime/tests, the current manuscript and figures, and, in the initial snapshot, two selected
presentation decks. On 28 October 2026, the two decks were removed from the current
branch at the author's request; all distinct project presentation videos are now
available directly in the README. The current manifest excludes the removed decks. Scientific artifacts are copied without content changes.
NPZ samples, PT checkpoints, MP4s and files >=40 MiB are distributed as
Release ZIPs; other selected files are committed directly. Every selected
research file has a size and SHA-256 entry in `SNAPSHOT_MANIFEST.json`.

Excluded: copied environments, caches, editor/agent/sync metadata, all nested
Git histories, third-party reference paper collections and independent
reference projects, temporary/archive/backup/conflict files, historical
top-level traces, unselected meeting drafts, and the two identical 1.32 GB
experimental slide decks. These remain in the original local workspace.
The original manuscript repository and its history remain intact locally;
the GitHub repository contains its current file snapshot, not a gitlink.

Public access makes the snapshot and Release assets available for inspection.
No project-wide license has been added; existing third-party notices remain in
place. Final publication metadata and new hardware qualification are not implied.
Installation-specific settings and original provenance paths remain in scientific
records; configure the hardware interfaces for the installation where they are used.

## Documentation and validation

The root README now provides an illustrated English research narrative,
Mermaid workflows, implementation links, method/result tables, limitations,
and download commands. It also explains AI-assisted research, physical CAD
development, hardware interfaces, failed hypotheses, and a proposed learning plan.
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

The main branch includes an English `AI_ASSISTED_RESEARCH.md` workbook and an
English README tracing the full research cycle. Seven existing project images
and four Mermaid diagrams connect the physical apparatus, inference, simulation,
CAD, hardware integration, and learning plan. The workbook supplies suggested
prompts, a proposed read-only hardware interface, independent exercises, and
a contribution ledger. Existing scientific files retain their original languages.
Historical attribution is distinguished from new prompt/workflow suggestions.
The current main-branch manifest is updated for these documentation files.
The original Release tag remains fixed at the initial upload commit and its
initial manifest; the six scientific archives and their hashes are unchanged.

## Public access and research narrative

The README presents the author's role through concrete project decisions,
implementation, experiments, and engineering artifacts. AI assistance is disclosed
once near the beginning and explained through the research workflow. Proposed
derivations, measurements, and experiments are identified as next steps.
The technical results, comparison limits, and scientific archives are unchanged.

The public download instructions include both authenticated GitHub CLI use and
ordinary Git plus manually downloaded Release archives. The latter requires no
GitHub account. See the validation record for the scope of pre-publication checks.

## Rebuild packaging

From the source workspace with Python 3.10+:

```bash
python tools/package_github.py --output D:/path/to/a/new/snapshot-directory
```

Output must be a new directory outside the source workspace. Packaging does not
initialize or modify any source Git history. It creates `repository/` and
`assets/` and updates the source `docs/SNAPSHOT_MANIFEST.json`.
