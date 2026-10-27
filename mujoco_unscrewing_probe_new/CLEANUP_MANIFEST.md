# Cleanup Manifest

Updated: 2026-07-15

The new path itself has not had active evidence removed. On 2026-07-11, its
historical predecessor was retired from the active `group_meeting` tree and
preserved as a complete verified archive at:

```text
/home/njy/njy_ws/archive/group_meeting_legacy/
  mujoco_unscrewing_probe_legacy_2026-07-11.tar.zst
  mujoco_unscrewing_probe_legacy_2026-07-11.tar.zst.sha256
```

Archive verification before removal of the source directory:

- `zstd -t`: passed;
- `tar --compare`: passed with no differences;
- archive/source entry count: `2910` / `2910`;
- SHA-256:
  `e097bf5faf3fcbe25cddbf2082ec03098c493c48f8bad53e900abcb9571d3dd2`.

Current root files are active:

- `README.md`
- `PROJECT_RULES_AND_LESSONS.md`
- `DESIGN_DECISIONS.md`
- `PROJECT_TRACE.md`
- `CLAIM_BOUNDARY.md`
- `CLEANUP_MANIFEST.md`

Current generated model artifacts are active:

- `configs/allen_*_source.json`
- `models/allen_*_source.xml`
- `docs/M8_M10_M12_MODEL_INVENTORY.md`
- `outputs/current/reports/m8_m10_m12_model_inventory.json`

Current M8 visualization artifacts are active:

- `outputs/current/media/videos/stages/` contains exactly the eight files indexed by
  `outputs/current/reports/m8_sampling_video_manifest.json`;
- `outputs/current/media/figures/m8_sampling_video_contact_sheet.png` is the synchronized
  eight-video overview;
- `outputs/current/media/keyframes/stages/` contains exactly one active representative
  frame per video plus `index.json`;
- `src/extract_m8_video_keyframes.py` reproducibly regenerates those eight
  frames and the contact sheet from the video manifest;
- the three active spiral routes are outer partial, double-overhang, and
  supported search to rim/chamfer capture.

Current paper comparison organization:

- `outputs/current/active_comparison/strict_fingerprint_direct_matrices/` is the
  current paper-facing matrix entry point. It contains the frozen muted/short
  `sobol_016` and clear/long `sobol_000` cases, both ordered Proposed+,
  Proposed, pure Outward and fingerprint Direct-first under the same strict
  130 s contract;
- `outputs/current/active_comparison/proposed_plus_strict_explainer/` remains
  the selected measured-path Proposed+ mechanism page;
- `outputs/current/active_comparison/strict_four_methods/` is superseded by the
  two new matrices and retained only for artifact regression; the older
  `four_methods/` and `four_methods_clear_fingerprint/` folders are historical
  capture-era mechanism examples;
- the preceding six-method and race files are under
  `diagnostics/historical_comparisons/`; they are historical and no longer the
  current baseline ladder;
- the fresh distribution-level original-style comparison is
  `outputs/current/evaluations/m8_jin_style_vs_improved/`, not any selected
  video. The Jin arm is an M8 engineering proxy and not a faithful paper
  reproduction;
- the selected muted/clear labels are traceable to
  `outputs/current/evaluations/m8_strict24_fingerprint_quality_stratification/`;
  this is a post-hoc descriptive artifact with source hashes, not a calibrated
  quality classifier or probability estimate;
- `outputs/current/evaluations/m8_finite_outward_boundary_audit/` is the active
  read-only geometry check that prevents the finite Outward comparator from
  being mislabeled as a complete visual-disk coverage certificate;
- `outputs/current/diagnostics/` contains selected failure/stress artifacts that
  are explicitly excluded from active baseline claims. The scripted
  `+22.5/-22.5/0 deg` direct-retry video lives here.

The renderer prunes obsolete MP4 files that are not in its current case list.
The stale single-spiral key frame and temporary overlay crop were removed on
2026-07-10.

The 2026-07-12 cleanup changed documentation ownership rather than deleting
evidence: `README.md`, `CLAIM_BOUNDARY.md`, and `outputs/current/README.md` now
contain only the current interface and route historical metrics to
`PROJECT_TRACE.md`. No dataset, model, evaluation, trace, or media artifact was
deleted.

The stale root-level `MUJOCO_LOG.TXT` was moved, without changing its content,
to `outputs/current/diagnostics/legacy_runtime/MUJOCO_LOG_2026-07-10.TXT`.
It records two 2026-07-10 development-time unstable-QACC warnings and is not a
current run result.
