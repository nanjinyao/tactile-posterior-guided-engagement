# Current Outputs

This tree is organized by evidence role. The current claim is defined by the
root `CLAIM_BOUNDARY.md`, not by the newest timestamp or an MP4 filename.

## Paper-Facing M8 Evidence

| question | current artifact | allowed interpretation |
|---|---|---|
| How do the five complete loops compare? | `evaluations/m8_five_method_retry3_r3_closed_loop/` | paired 24-pose MuJoCo success, time, path, wear, safety and failure categories under one retry-x3/R3 contract |
| Does bounded post-capture recovery help? | `evaluations/m8_recoverable_funnel_r3_ablation/` | R3 improves completion on the fixed 12-case, 60 s command-side pose-stress set; no guarantee or hardware/passivity claim |
| What does the entropy-grid controller actually do? | `active_comparison/strict_entropy_grid_topview_sobol_003/` and `strict_entropy_grid_topview_sobol_003_en/` | bilingual renders of the same actual-M8 measured replay, with offline capture-region overlays |
| What changes between a muted and clear fingerprint? | `active_comparison/m8_retry3_selected_media/sobol_016/` and `sobol_000/` | two selected same-start 2x2 mechanisms; not distribution estimators |
| Are the improved and Jin-style displays/controllers represented correctly? | `active_comparison/m8_retry3_selected_media/sobol_005/` | improved four-channel versus Jin-style three-channel direct-x3 mechanism example |
| How do shortcut and residual search connect? | `active_comparison/m8_shortcut_residual_sobol_008/` | same-start measured paths; task-aware ordering, not monotone entropy convergence or causal ablation |

The exact active MP4 list and all downgraded retained MP4s are registered in
`VIDEO_RETENTION.json`.

## Current Five-Method Contract

All five methods share the same paired tasks, contact/sensor nuisance, 130 s
non-extendable active-time budget, sensor-only handoff, CAD-bottom verifier and
post-capture tip-RCC/safety/R3 layer. The two direct arms receive at most three
same-rollout rays; every miss charges unload/lift/shift/re-probe motion. There
are no simulator resets or time extensions.

| method | CAD-bottom | capture | hard stop |
|---|---:|---:|---:|
| Proposed | 20/24 | 23/24 | 3 |
| Proposed+ | 19/24 | 23/24 | 4 |
| Pure Outward | 19/24 | 23/24 | 4 |
| improved four-channel + direct x3 | 17/24 | 19/24 | 3 |
| Jin-style three-channel + direct x3 | 16/24 | 18/24 | 3 |

Jin 2021 uses a 12-channel contact pattern; the current three-channel arm is an
M8 engineering proxy, not a full reproduction. The improved-versus-Jin-style
result is mixed. Any-ray capture is 19/24 for both and
paired CAD-bottom discordance is 4 versus 3 ($p=1$). Improved has lower median
executed-direction error; Jin-style has lower p90 and more within-15-degree
cases. Do not turn the selected `sobol_005` video into a superiority claim.

## Physical Funnel Boundary

The funnel is shared by Proposed, Proposed+, Outward and both direct baselines
after sensor capture. It changes compliance, supervisory intervention and the
consequence of a recoverable contact; it does not make every capture terminally
valid.

On the fixed paired stress set, bottom verification is 2/12 without funnel,
0/12 with shield/R0 and 9/12 with shield/R3. The isolated recovery comparison
is R3 versus R0: nine R3-only completions, zero regressions, paired sign
$p=0.00390625$. R0 makes ten non-hard outcomes safe aborts; R3 recovers nine.
R3 still has two hard stops and one safe abort. The older
`m8_funnel_guard_ablation/`, `funnel_guard/`, and `funnel_extreme_recovery/`
artifacts are historical/diagnostic only. The regenerated extreme clip now
shows a named CAD-depth safe abort with recovery count zero; it is not a
successful-recovery example.

## Entropy-Grid Boundary

`active_comparison/strict_entropy_grid_topview_sobol_003/` separates:

- the 1201 Bayesian center hypotheses;
- the 673 action nodes inside visual support;
- 76 offline padding nodes used only by the separate 749-node geometry
  certificate;
- the five actions actually selected and the measured MuJoCo tip path.

True center, the 0.10 mm direct-clearance disk, the 0.25 mm chamfer-eligible
disk, and eligible-node coloring are offline overlays. They are not online
observations. A finite node-spacing certificate does not guarantee that the
controller traverses all nodes or captures/completes before the budget.
The English directory copies the audited Chinese `trace.csv/result.json`
byte-for-byte and rerenders visible annotations only. Its manifest records
`physics_rollout_rerun_for_this_render=false` and the source hashes.

The older `entropy_grid_planar_explainer/` uses a historical selected rollout
and endpoint-level explanation; it is retained only as a diagnostic.

## Hardware-Easy-v2 Shared M8/M10/M12 Direction Model

`models/hardware_easy_v2_multisize_robust_direction_v4/` is the current shared
three-seed fingerprint-direction ensemble. Its size-normalized inputs use each
config's bit radius. M10 contributes 496 grouped training and 128 calibration
fingerprints; M12 contributes 462 and 140. Captured/incomplete scans were
filtered only after complete physical C6-orbit splitting and contribute no
direction labels. APS calibration uses an explicit 95% target (`alpha=0.05`).

The final untouched evaluation is
`evaluations/hardware_easy_v2_multisize_robustness_fresh6/`: all independent
M8, M10 and M12 gates pass on four 180-task domains, with zero hard stops and
zero accepted direction errors above 90 degrees. On the M10/M12 easy-v2 valid
fingerprints, the shared model has p90 errors of 2.55/3.16 degrees and accepted
within-7.5-degree rates of 1.000/1.000. M8 source/easy non-forgetting gates also
pass relative to the promoted M8-v5 ensemble.

This is geometry-conditioned MuJoCo evidence for the direction/shortcut stage,
not real Bota F/T fine-tuning, common 42 mm sensor-to-tip alignment, printed
compliance/friction identification, socket engagement completion or full-loop
success.

## Cross-Size S4b: Development Closure, Untouched M10 Safety Failure

`datasets/cross_size_s4b_adaptation/` contains disjoint train, validation and
calibration C6 orbits; `models/cross_size_s4b_state_direction_heads/` updates
only state/direction heads. Deployment uses the validation-selected residual
adapter $p_A=p_B+0.92(p_F-p_B)$ and abstains from a directed shortcut when the
fine-tuned concentration is below 0.60 or the parent means disagree by more
than 90 degrees. Abstention retains finite residual search; it does not read
the true center.

The current reports are:

- development: `diagnostics/cross_size_s4b_full_loop_development_v5/`;
- first and permanently retired untouched test:
  `evaluations/cross_size_s4b_full_loop_untouched_v1/`.

The trial-level physical/provenance audit is
`../../docs/CROSS_SIZE_S4B_FULL_LOOP_AUDIT.md`.

Both use 12 poses per size, five paired arms, the same size-conditioned
admittance/R3 funnel, unchanged 20 N / 60 mN m raw hard limits, a 130 s active
budget, zero extension and zero simulator reset.

| phase / fine-tuned arm | M10 strict / hard / gate | M12 strict / hard / gate |
|---|---:|---:|
| development-v5 | 12/12 / 0 / PASS | 10/12 / 0 / PASS |
| untouched-v1 | 10/12 / 1 / **FAIL** | 10/12 / 0 / PASS |

In untouched-v1, M10 zero-shot and calibration-only are both 12/12 with no hard
stop; M12 calibration-only is 11/12 with no hard stop. Fine-tuning necessity is
therefore not identified and the M10 safety failure means independent
cross-size closure is not established. The two non-bottom M12 fine-tuned exits
are named low-force requalification safe aborts, not successes. Untouched-v1
is forbidden for any later training, calibration, controller/gate tuning or
rerun-as-confirmation.

Strict success uses offline true-center/contact-tag scoring unavailable to the
deployed controller. Privileged straight-to-center is an execution diagnostic,
not a deployable policy or guaranteed upper bound. These are MuJoCo results,
not real F/T data, and cannot establish hardware-data fine-tuning necessity.
The older S3-v3 report remains a historical development diagnostic and was not
used as S4b adaptation data.

## Historical And Diagnostic Evidence

The following remain on disk for reproducibility but are not pooled with the
current five-method report:

- `evaluations/m8_strict_24_complete_loop/`: prior seven-role, one-ray-era
  complete-loop snapshot;
- `active_comparison/strict_fingerprint_direct_matrices/`: prior one-ray
  muted/clear matrices;
- `active_comparison/strict_four_methods/`: superseded strict entropy matrix;
- `active_comparison/four_methods*/`: historical capture-era matrices;
- `active_comparison/funnel_guard/` and `funnel_extreme_recovery/`: selected
  defense/recovery mechanisms under older contracts;
- `evaluations/m8_jin_style_vs_improved/`: prior estimator-only 48-pose study,
  not the current same-rollout direct-x3 comparison;
- `diagnostics/`: development failures and historical comparisons.

Large raw debug traces are governed by `DIAGNOSTIC_RETENTION.json`; older
archived MP4s and every retained current-tree MP4 are governed by
`VIDEO_RETENTION.json`.

## Stable Directory Roles

- `datasets/`: role-separated simulation data and adaptation preparation;
- `models/`: frozen model/interface contracts and adaptation preparation;
- `evaluations/`: distribution-level or paired experiment reports;
- `active_comparison/`: selected current and retained diagnostic visual media;
- `media/`: static figures and contact sheets;
- `reports/`: compact cross-module inventories;
- `diagnostics/`: development evidence that cannot support active claims.

Every new evaluation should include a task contract, report, explicit data
role, hashes, and claim boundary before it is promoted here.
