# MuJoCo Unscrewing Probe New

M8-first benchmark for low-force Allen bit/socket engagement under a bounded
visual residual. The current complete loop is:

```text
guarded touchdown
-> one conical fingerprint when top contact is ambiguous
-> bounded shortcut and residual search
-> sensor-only capture handoff
-> shared upright tip-RCC / safety observer / recovery R3
-> CAD-depth plus force/wrench dwell
```

The implemented endpoint is geometric engagement in MuJoCo, not motor coupling
or unscrewing. Current claim wording is owned by [CLAIM_BOUNDARY.md](CLAIM_BOUNDARY.md).

## Current Evidence Hierarchy

| role | source of truth |
|---|---|
| Current five-method M8 success/time comparison | `outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop/` |
| Recoverable physical-funnel R3 ablation | `outputs/current/evaluations/m8_recoverable_funnel_r3_ablation/` |
| Actual M8 entropy-grid top-view replay (Chinese/English) | `outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003/` and `strict_entropy_grid_topview_sobol_003_en/` |
| Muted/clear retry-x3 2x2 mechanism matrices | `outputs/current/active_comparison/m8_retry3_selected_media/sobol_016/` and `sobol_000/` |
| Improved four-channel vs Jin-style three-channel video | `outputs/current/active_comparison/m8_retry3_selected_media/sobol_005/` |
| Shortcut then residual-search figure | `outputs/current/active_comparison/m8_shortcut_residual_sobol_008/` |
| Active/downgraded video registry | `outputs/current/VIDEO_RETENTION.json` |
| Cross-size S4b development / first untouched status | `outputs/current/diagnostics/cross_size_s4b_full_loop_development_v5/` and `outputs/current/evaluations/cross_size_s4b_full_loop_untouched_v1/` |
| Historical trace and design evolution | `PROJECT_TRACE.md`, `DESIGN_DECISIONS.md` |

The older strict-24 seven-role report, one-ray 2x2 matrices, strict entropy
matrix, funnel-guard videos, and capture-only matrices remain reproducible but
are historical/diagnostic. They are not the current comparison contract.

## Five Methods Under One Contract

All five methods share the same 24 task realizations, visual/contact nuisance,
sensor seed, 130 s non-extendable active-time budget, sensor-only handoff,
terminal verifier, and post-capture physical funnel with at most three
recoveries. Simulator reset and budget extension counts are zero.

1. **Proposed**: improved four-channel conical fingerprint, bounded posterior
   shortcut, residual/full Outward search.
2. **Proposed+**: the same shortcut followed by measured-path task-aware
   information-grid actions and conditional Outward.
3. **Pure Outward**: geometry-only finite Archimedean search.
4. **Improved fingerprint + direct x3**: at most three physical same-rollout
   rays, with unload/lift/shift/full re-probe after a miss.
5. **Jin-style three-channel + direct x3**: the same direct controller and
   retry budget driven by a paper-inspired M8 proxy.

The direct retry count is pre-capture. The shared R3 count is post-capture.
Neither is a simulator restart or an independent repeated trial.

| 24 paired M8 poses | CAD-bottom | offline-valid capture | hard stop | successful bottom-time median | successful XY-path median |
|---|---:|---:|---:|---:|---:|
| Proposed | 20/24 | 23/24 | 3 | 13.87 s | 14.52 mm |
| Proposed+ | 19/24 | 23/24 | 4 | 13.74 s | 6.76 mm |
| Pure Outward | 19/24 | 23/24 | 4 | 34.63 s | 71.27 mm |
| Improved four-channel + direct x3 | 17/24 | 19/24 | 3 | 9.22 s | 6.07 mm |
| Jin-style three-channel + direct x3 | 16/24 | 18/24 | 3 | 8.96 s | 5.67 mm |

Efficiency medians are conditional on different successful subsets. Proposed+
does not outperform Proposed on paired success (19 both-success, one
Proposed-only, four both-failed).

## Improved Versus Jin-Style Is Mixed

The improved learner consumes phase-aligned
$F_z,\tau_x,\tau_y,z_{tip}$ plus dimensionless context. Jin 2021 describes a
12-channel contact pattern (3D force, 3D torque, and 6D pose); the current
Jin-style arm deliberately retains only three polar binary channels
$X_z,M_x,M_y$ as an M8 engineering proxy. It is not a faithful reproduction of
the original full input, shape task, unspecified network, or retry controller.

In the current direct-x3 audit, improved versus Jin-style reaches CAD-bottom
17/24 versus 16/24 and first-ray capture 18/24 versus 16/24, but any-ray capture
is 19/24 for both. First-probe executed-direction median/p90 error is
4.75/31.45 deg versus 7.41/13.79 deg, and within-15-deg is 19/24 versus 22/24.
This is a mixed tradeoff, not uniform accuracy or success superiority.

## Shared Physical Funnel

Search policies differ only before sensor capture. After handoff, every current
method uses the same upright tip-RCC, injected-energy/progress/CAD observer,
bounded unload/retract/upright/retouch recovery, and terminal dwell. Recovery XY
is rebased from measured tip pose; true socket center is not supplied.

On a fixed paired 12-case, 60 s, command-side post-capture pose-stress set,
CAD-bottom changes
from 2/12 without a funnel and 0/12 with shield/R0 to 9/12 with shield/R3.
The strict recovery causal comparison is R3 versus R0: those two arms differ
only in the recovery allowance, and R3 uniquely completes nine cases with zero
regressions (exact paired sign $p=0.00390625$). R0 turns ten non-hard-stop
outcomes into safe aborts; R3 recovers nine of them. R3 still produces two hard
stops and one safe abort. The result supports a completion benefit inside this
tested MuJoCo stress envelope only; it is not a success guarantee or passivity
proof.

## Entropy Grid And Selected Visuals

The current top-view entropy video replays the actual measured MuJoCo tip trace
for strict `sobol_003`. It shows five selected actions, online belief updates,
the offline 0.10 mm direct-clearance and 0.25 mm chamfer-eligible disks, sensor
capture, and CAD-bottom completion. The true center/capture disks/node
eligibility are explanatory overlays and are not online observations.
The English render copies the audited Chinese render's `trace.csv/result.json`
byte-for-byte and rerenders annotations only; its manifest explicitly records
that no physics rollout was rerun.

Keep the finite sets distinct:

- 1201 center hypotheses belong to the Bayesian belief;
- 673 action nodes lie inside the visual support;
- 76 offline boundary-padding nodes complete the separate 749-node geometry
  certificate;
- only five actions are selected in the displayed rollout.

The certificate is geometric only. It does not guarantee tracking, cue
validity, scheduling, finite-time capture, or terminal completion.

The two active 2x2 videos are complementary selected cases from the current
retry-x3 audit: `sobol_016` has a muted cue, short shortcut, and three failed
direct attempts; `sobol_000` has a clear cue, long shortcut, and first-ray
capture. The `sobol_005` 1x2 video shows the correct improved four-channel and
Jin-style three-channel interfaces. Selected videos explain mechanisms; the
24-pose report estimates method outcomes.

## Cross-Size S4b

S4b uses disjoint train, validation and calibration $C_6$ orbits under
`outputs/current/datasets/cross_size_s4b_adaptation/`. The encoder/shared trunk
stays frozen; only the 387-parameter state head and 6,192-parameter direction
head may update. A byte-identical calibration-only arm changes temperatures and
APS but not checkpoint weights. The validation-selected runtime adapter is

$$
p_A=p_B+0.92(p_F-p_B),
$$

and abstains from directed transport when fine-tuned concentration is below
0.60 or the base/fine-tuned means disagree by more than 90 degrees. Abstention
keeps finite residual search; it does not insert a true direction.

The frozen development-v5 and first untouched-v1 reports each contain 12 poses
per size and five paired arms. Every arm shares size-conditioned commissioning,
the unchanged 20 N / 60 mN m raw hard limits, a 130 s active budget, zero
extension, zero reset and offline strict scoring.

| phase / fine-tuned arm | M10 strict / hard / gate | M12 strict / hard / gate |
|---|---:|---:|
| development-v5 | 12/12 / 0 / pass | 10/12 / 0 / pass |
| untouched-v1 | 10/12 / 1 / **fail** | 10/12 / 0 / pass |

Trial-level mechanisms and provenance are audited in
`docs/CROSS_SIZE_S4B_FULL_LOOP_AUDIT.md`.

Development closure therefore does not establish independent two-size
closure. Untouched M10 zero-shot/calibration-only are both 12/12 with zero hard
stops; untouched M12 calibration-only is 11/12 with zero hard stops. Fine-tuning
necessity and monotonic benefit are not identified. The two M12 fine-tuned
non-bottom outcomes are named low-force requalification safe aborts, not
successes. Untouched-v1 is permanently test-only and cannot be used for later
training, calibration, model/controller/gate selection, or a tuned rerun.

Straight-to-center reads true center and is only a privileged execution
diagnostic. Strict success also uses offline true-center/contact-tag scoring
unavailable to the deployed controller. These are MuJoCo results, not real F/T
data or evidence that hardware fine-tuning is necessary. The older S3-v3 and
frozen-M8 cross-size reports remain historical diagnostics and were not S4b
adaptation data.

## Model Boundary

The improved three-seed circular CNN predicts a 48-bin center-direction
distribution, APS angular set, circular concentration, ensemble disagreement,
and soft contact-support state. It does not predict a center coordinate,
millimeter distance, capture event, force, stiffness, velocity, or terminal
success. Geometry and the visual prior bound the search; classical control owns
physical execution, safety, recovery, and terminal verification.

## Main Reproduction Commands

```bash
# Current paired five-method complete-loop audit
MUJOCO_GL=egl python3 -m src.run_m8_five_method_retry3_audit \
  --poses 24 --workers 24 --no-resume

# Fixed paired post-capture recovery ablation
MUJOCO_GL=egl python3 -m src.run_m8_recoverable_funnel_r3_ablation \
  --workers 12 --no-resume

# Actual M8 entropy-grid replay and top-view explainer
MUJOCO_GL=egl python3 -m src.render_m8_strict_entropy_topview_explainer

# English-only annotations from the same audited replay; no MuJoCo rerun
MUJOCO_GL=egl python3 -m src.render_m8_strict_entropy_topview_explainer \
  --language en \
  --reuse-replay-from outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003

# Current selected retry-x3 visualization (repeat with the pose/mode recorded
# by each selected manifest)
MUJOCO_GL=egl python3 -m src.render_m8_retry3_selected_media \
  --mode matrix --pose-id sobol_016 --overwrite

# Same-start shortcut/residual-search figure
MUJOCO_GL=egl python3 -m src.render_m8_shortcut_residual_comparison

# Write/read back the S4b data contract only (default; no MuJoCo sampling)
python3 -m src.sample_cross_size_s4b_adaptation

# Explicit opt-in S4b dynamic sampling, only after reviewing a fresh contract
MUJOCO_GL=egl python3 -m src.sample_cross_size_s4b_adaptation \
  --sample --workers 24

# Valid only after both S4b size manifests report status=complete
python3 -m src.finetune_cross_size_s4_state_direction_heads \
  --data outputs/current/datasets/cross_size_s4b_adaptation \
  --output outputs/current/models/cross_size_s4b_state_direction_heads

# Full-loop runner defaults to freeze-only. Use a new development output;
# never execute or tune against the retired untouched-v1 tasks again.
python3 -m src.run_cross_size_s4_full_loop_evaluation \
  --phase development --output outputs/current/diagnostics/new_s4b_development

# Full regression
python3 -m unittest discover -s tests -v
```

Use the per-artifact manifest for exact arguments and hashes if a module's CLI
changes. Never infer a current claim from an MP4 filename alone.

## Hardware Gate

Before enabling posterior transport on an iiwa:

1. measure the camera residual distribution and declare the visual support;
2. calibrate flange/F-T/tip frames, gravity, bias, covariance, delay, and sign;
3. commission low-force conical scans and return-upright behavior in shadow
   mode;
4. fit only a small normalization/adapter/head on role-separated real data;
5. test on an untouched hardware set with no threshold or gate reuse;
6. commission transport, recovery, and terminal verification in capped stages;
7. add a motor-coupling verifier before any unscrewing claim.

Large raw diagnostics and old videos are retained according to
`outputs/current/DIAGNOSTIC_RETENTION.json` and
`outputs/current/VIDEO_RETENTION.json`; they are not pooled with active
evidence.
