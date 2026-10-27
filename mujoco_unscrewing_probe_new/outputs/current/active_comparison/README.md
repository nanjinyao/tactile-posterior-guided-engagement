# Active M8 Comparison Media

Sixteen composite MP4s are retained in the current output tree. Five are
current paper-facing mechanism videos; eleven are downgraded historical or
diagnostic media. `../VIDEO_RETENTION.json` is the machine-readable registry.

## Current Mechanism Videos

| role | artifact | interpretation |
|---|---|---|
| Muted fingerprint / short shortcut | `m8_retry3_selected_media/sobol_016/comparison_fuzzy_retry3_2x2.mp4` | Proposed+, Proposed, Outward and improved direct x3 on one same-start case; direct pays for all three physical attempts and still misses |
| Clear fingerprint / long shortcut | `m8_retry3_selected_media/sobol_000/comparison_clear_retry3_2x2.mp4` | same four tiles and contract; improved direct captures on its first ray |
| Improved 4-channel vs Jin-style 3-channel | `m8_retry3_selected_media/sobol_005/comparison_improved4_vs_jin3_retry3.mp4` | same raw first probe and physical direct-x3 contract; correct estimator-specific dynamic interfaces |
| Actual M8 entropy-grid top view | `strict_entropy_grid_topview_sobol_003/m8_strict_entropy_grid_topview_sobol_003.mp4` | five selected actions and measured MuJoCo tip trace, with offline direct/chamfer capture overlays |
| Actual M8 entropy-grid top view (English) | `strict_entropy_grid_topview_sobol_003_en/m8_strict_entropy_grid_topview_sobol_003_en.mp4` | English-only overlay rendered from byte-identical audited trace/result artifacts; no physics rerun |

These are selected mechanism explanations. Method-level success/time evidence
comes only from
`../evaluations/m8_five_method_retry3_r3_closed_loop/`.

## Contracts Visible In The New Media

- Fingerprint methods share the same frozen task, nuisance, sensor seed and
  first raw conical scan.
- Improved display: dynamic $F_z,\tau_x,\tau_y,z_{tip}$ plus a direction
  posterior.
- Jin-style display: three paper-inspired polar binary channels
  $X_z,M_x,M_y$ plus its sector direction.
- A direct miss triggers physical unload/lift/shift/full re-probe in the same
  rollout, with at most three rays and no simulator reset.
- After sensor capture, every method enters the same tip-RCC/safety/R3 and
  terminal verifier.
- Selected outcomes do not estimate success probabilities.

The paired direct-x3 result is mixed: improved/Jin-style CAD-bottom is 17/24
versus 16/24, while any-ray capture is 19/24 for both. Improved has lower
median first-probe executed-direction error; Jin-style has lower p90 and more
within-15-degree cases. The `sobol_005` clip cannot establish superiority.

## Entropy Top-View Boundary

The `sobol_003` replay uses the actual M8 MuJoCo geometry and measured tip path.
It displays the initial pose, five chosen action endpoints, online belief
updates, contact/lift-shift segments, sensor capture and CAD-bottom completion.
The Chinese and English assets render the same copied `trace.csv/result.json`;
their manifests record the shared hashes and whether physics was rerun.

The true center, capture disks and eligible-node coloring are offline
explanatory overlays. The video distinguishes 1201 belief hypotheses from 673
inside-support actions and 76 boundary-padding actions. It does not claim that
all 749 nodes were traversed or that node geometry guarantees tracking,
capture, finite-time completion or hardware behavior.

## Retained Diagnostic Media

The following files remain present but are explicitly downgraded in
`../VIDEO_RETENTION.json`:

- `strict_fingerprint_direct_matrices/`: previous one-ray muted/clear 2x2
  matrices;
- `strict_four_methods/`: superseded strict entropy matrix;
- `proposed_plus_strict_explainer/`: selected older strict-contract P+ replay;
- `entropy_grid_planar_explainer/`: historical endpoint-level entropy
  explanation;
- `funnel_guard/`: older defense clip; `funnel_extreme_recovery/`: a corrected
  severe-fault safe-rejection clip that does not execute a full recovery;
  neither is current R3 distribution evidence;
- `four_methods/` and `four_methods_clear_fingerprint/`: capture-era matrices;
- `../diagnostics/historical_comparisons/`: former six-method matrix and
  three-method race.

The current R3 completion evidence is the fixed paired report and poster under
`../evaluations/m8_recoverable_funnel_r3_ablation/`; there is no new R3 video
whose selected outcome substitutes for that report.

Component renders are deleted only after each composite, poster, keyframe and
manifest are verified. Do not recreate deleted components simply to infer a
claim; use each composite manifest and its frozen source trial.
