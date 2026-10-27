# Current Claim Boundary

Updated: 2026-07-15

This is the current paper-facing claim snapshot. Older experiments remain
traceable, but they do not override this file or the manifests linked below.

## Implemented Task

```text
declared bounded visual residual over an M8 internal-hex socket
-> low-force search and sensor-only capture handoff
-> shared post-capture physical funnel
-> CAD-depth plus force/wrench dwell verification
```

The implemented endpoint is geometric M8 bit/socket engagement in MuJoCo. It
does not include motor-current coupling, reverse rotation, threaded
unscrewing, extraction, or KUKA iiwa hardware validation. The loop starts from
a benchmark-declared `ambiguous_top_contact`; the upstream contact-state router
is a deployment contract, not an evaluated classifier.

Online policies receive measured F/T, end-effector/tip pose, commanded action,
conical phase, the visual prior, and approximate CAD. Except for the Oracle
diagnostic, they do not receive the true socket center or MuJoCo contact tags.
True center, contact tags, and capture disks appear only in offline scoring and
explanatory overlays.

## Current Complete-Loop Contract

The fresh paired source of truth is:

`outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop/`

All five methods use the same 24 M8 task realizations, nuisance parameters,
sensor seeds, 130 s non-extendable active-controller budget, sensor-only
capture handoff, terminal verifier, and post-capture physical funnel. Every
active-time extension and simulator-reset count is zero.

| method | search before capture | post-capture contract |
|---|---|---|
| Proposed | improved four-channel conical fingerprint, bounded shortcut, residual/full Outward | shared tip-RCC, safety observer, at most three recoveries, terminal dwell |
| Proposed+ | same fingerprint, shortcut, measured-path task-aware information grid, conditional Outward | same shared contract |
| Pure Outward | geometry-only finite Archimedean search | same shared contract |
| Improved fingerprint + direct x3 | up to three physically executed rays from the improved four-channel estimator | same shared contract |
| Jin-style three-channel + direct x3 | up to three physically executed rays from an M8 paper-style proxy | same shared contract |

There are two distinct bounded retry counts:

- `direct_ray_attempts <= 3` is a pre-capture direct-search policy used only by
  the two direct arms. A miss pays for unload, lift/shift, a complete new probe,
  and the next ray in the same rollout.
- `post_capture_recoveries <= 3` belongs to the shared physical funnel after
  sensor capture. It unloads, retracts, returns upright, rebases XY from the
  measured tip pose, retouches, and re-enters the terminal verifier.

Neither count means three independent trials, and neither guarantees success.

## Fresh Five-Method Result

Strict success requires an offline-valid capture, a declared CAD-bottom dwell,
no hard stop, active time at or below 130 s, and zero budget extension.
Efficiency medians are conditioned on each method's successful subset.

| 24 paired M8 poses | CAD-bottom | offline capture | hard stop | median successful bottom time | median successful XY path | median successful wear proxy |
|---|---:|---:|---:|---:|---:|---:|
| Proposed | 20/24 | 23/24 | 3 | 13.87 s | 14.52 mm | 28.48 mJ |
| Proposed+ | 19/24 | 23/24 | 4 | 13.74 s | 6.76 mm | 9.54 mJ |
| Pure Outward | 19/24 | 23/24 | 4 | 34.63 s | 71.27 mm | 120.38 mJ |
| Improved fingerprint + direct x3 | 17/24 | 19/24 | 3 | 9.22 s | 6.07 mm | 13.67 mJ |
| Jin-style three-channel + direct x3 | 16/24 | 18/24 | 3 | 8.96 s | 5.67 mm | 13.22 mJ |

Proposed versus Proposed+ is 19 both-success, one Proposed-only and four
both-failed (exact McNemar $p=1$). This batch supports a path/wear tradeoff, not
Proposed+ superiority. Exact-center transport is not sufficient for terminal
completion, so success cannot be inferred from search geometry alone.

## Improved Four-Channel Versus Jin-Style Three-Channel

The improved interface consumes phase-aligned
$F_z,\tau_x,\tau_y,z_{tip}$ plus context. Jin 2021 describes a 12-channel
contact pattern (3D force, 3D torque, and 6D pose). The current Jin-style M8
proxy deliberately retains only polar binary images of $X_z,M_x,M_y$. It is an
engineering comparator, not a faithful reproduction of the original full
input, task, labels, rasterization, network, or controller.

Under the fresh same-rollout direct-x3 contract the comparison is mixed:

| direct-x3 arm | first-ray capture | any-ray capture | CAD-bottom | first-probe executed-direction median / p90 | within 15 deg |
|---|---:|---:|---:|---:|---:|
| improved four-channel | 18/24 | 19/24 | 17/24 | 4.75 / 31.45 deg | 19/24 |
| Jin-style three-channel | 16/24 | 19/24 | 16/24 | 7.41 / 13.79 deg | 22/24 |

CAD-bottom discordance is 4 improved-only versus 3 Jin-only
($p=1$); any-ray capture is 3 versus 3 ($p=1$). The improved arm has the lower
median error and one more bottom, while the Jin proxy has the lower p90 and
more within-15-degree cases. This does not establish uniform estimator or
closed-loop superiority.

The selected comparison video at
`outputs/current/active_comparison/m8_retry3_selected_media/sobol_005/`
correctly shows the improved four dynamic channels and posterior on one side,
and the Jin-style three polar channels on the other. It is a mechanism example,
not a probability estimate.

## Physical Funnel: What It Changes

After sensor capture, every current method uses the same physical funnel. The
funnel changes the contact response and failure consequence; it cannot make an
arbitrary capture state terminally valid. Residual XY offset, friction,
clocking, tilt, lateral force/torque, noise, or insufficient dwell can still
cause a terminal rejection, safe abort, or hard stop.

The paired fixed-stress source uses a dedicated 60 s contract and
controller-commanded tilt/lateral pose stresses that recovery can cancel:

`outputs/current/evaluations/m8_recoverable_funnel_r3_ablation/`

| 12 fixed moderate post-capture stresses | CAD-bottom | hard stop | safe abort | recoveries |
|---|---:|---:|---:|---:|
| no funnel | 2/12 | 2 | 0 | 0 |
| tip-RCC + shield, R0 | 0/12 | 2 | 10 | 0 |
| tip-RCC + shield, R3 | 9/12 | 2 | 1 | 11 |

The strict causal recovery contrast is R3 versus R0 because these two arms
differ only in their recovery allowance. R3 uniquely completes nine paired
cases and regresses zero ($p=0.00390625$). The no-funnel comparison (seven
R3-only, zero no-funnel-only, $p=0.015625$) mixes recovery with tip-RCC/shield
changes and is descriptive, not the isolated recovery effect. R0 turns ten
non-hard-stop outcomes into safe aborts; R3 recovers nine of them. R3 still has
two hard stops and one safe abort. The causal statement is limited to this
pre-programmed MuJoCo stress envelope and current controller; it is not a
passivity proof, a hardware result, or a guarantee.

The older `m8_funnel_guard_ablation` (5/12 versus 4/12) did not exercise the
current recovery loop and is historical defense-only evidence. The older
extreme selected video is also mechanism-only.

## Information Grid And Actual M8 Top View

The current entropy artifact is:

`outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003/`

It is a replay of the actual M8 MuJoCo geometry and measured tip trace for the
frozen `sobol_003` strict trial. The controller chooses five adaptive actions;
posterior entropy over the first four observations changes
$5.95\rightarrow5.15\rightarrow3.71\rightarrow2.24$ nats, the measured tip
enters the nominal 0.10 mm direct-clearance disk, sensor capture occurs at
10.034 s, and CAD-bottom is verified at 12.746 s.

The visualization keeps three sets separate:

- 1201 Bayesian center hypotheses are belief states, not action nodes;
- 673 nodes lie inside the visual support and 76 are offline boundary-padding
  nodes, giving the separate 749-node finite geometry certificate;
- the true center, 0.10 mm direct-clearance disk, 0.25 mm chamfer-eligible disk,
  and node eligibility are offline explanatory overlays only.

The certificate bounds nearest-node geometry. It does not prove that all nodes
are visited, measured tracking, cue validity, scheduling within the hard
budget, capture, terminal completion, or hardware success.

The same-start `sobol_008` figure under
`outputs/current/active_comparison/m8_shortcut_residual_sobol_008/` shows
shortcut then task-aware grid versus shortcut then local Outward, with pure
Outward as a faint reference. Proposed+ entropy is not monotone in that case;
the correct claim is task-aware action ordering and sensor termination, not
monotone convergence or an isolated causal ablation.

## Selected Videos Versus Distribution Evidence

The current active videos are registered in
`outputs/current/VIDEO_RETENTION.json`:

- `sobol_016`: muted fingerprint, short shortcut, direct consumes all three
  physical attempts and still misses;
- `sobol_000`: clear fingerprint, long shortcut, direct captures on its first
  ray;
- `sobol_005`: improved four-channel versus Jin-style three-channel direct-x3;
- `sobol_003`: actual M8 entropy-grid top-view replay.

These selected cases explain mechanisms only. The five-method report above is
the sole current success/time ranking source. The old strict-24 seven-role
audit, old one-ray matrices, old entropy matrix, old guard videos, and older
capture-only matrices are retained only as historical or diagnostic artifacts.

## Cross-Size S4b Status

The current development report is
`outputs/current/diagnostics/cross_size_s4b_full_loop_development_v5/report.json`;
the first permanently retired untouched report is
`outputs/current/evaluations/cross_size_s4b_full_loop_untouched_v1/report.json`.
Both use 12 poses per size, five paired arms, 130 s zero extension, shared R3,
unchanged 20 N / 60 mN m raw hard limits and zero simulator resets. The S4b
model updates only state/direction heads. Its validation-selected deployment
adapter is $p_A=p_B+0.92(p_F-p_B)$; it abstains when fine-tuned concentration is
below 0.60 or the two parent means disagree by more than 90 degrees.

Development-v5 closes both registered fine-tuned gates: M10 reaches 12/12 with
zero hard stops and M12 reaches 10/12 with zero hard stops. This is development
evidence only. On untouched-v1, M12 fine-tuned reaches 10/12 with zero hard
stops and passes, but M10 fine-tuned reaches 10/12 with one primary hard stop
and fails the zero-hard-stop gate. Its other non-bottom trial is a named
post-release requalification safe abort. Thus independent two-size closure is
not established.

Fine-tuning necessity is also not identified. Untouched M10 zero-shot and
calibration-only both reach 12/12 with zero hard stops; untouched M12
calibration-only reaches 11/12 with zero hard stops. Untouched-v1 is forbidden
for all later training, calibration, controller/gate selection, and cannot be
rerun as a fresh confirmation after tuning.

Strict success is offline-audited with true-center/contact-tag capture scoring;
that scorer is unavailable to the four non-oracle controllers and is not a
deployable terminal signal. Straight-to-center reads true center and is a
privileged execution diagnostic, not a guaranteed upper bound. Both contracts
freeze the local Python source tree, CAD/XML, dataset/model manifests,
checkpoints and calibration before rollout; worker-init, worker-completion,
per-size and final hashes agree. These are MuJoCo results and cannot establish
hardware-data fine-tuning necessity. The older S3-v3 and frozen-M8 cross-size
reports remain historical diagnostics and were not reused as S4b adaptation
data.

## Allowed Claim

```text
In the paired M8 MuJoCo benchmark, a four-channel conical fingerprint can be
used as a bounded direction shortcut, while residual search, sensor-only
capture, and a shared recoverable physical funnel form a complete but
non-guaranteed engagement loop. The task-aware grid is a transparent adaptive
search component; the current improved-vs-Jin result is mixed.
```

## Not Yet Proved / Not Allowed

- hardware camera/F/T calibration, iiwa safety, or sim-to-real robustness;
- a passivity guarantee or universal benefit of the R3 physical funnel;
- that a finite spiral or entropy grid must capture within a fixed time;
- that the improved fingerprint uniformly outperforms the Jin-style proxy;
- distribution-level superiority of Proposed+ over Proposed;
- a passed two-size untouched S4b gate, monotonic benefit of fine-tuning, or
  necessity of hardware-data fine-tuning;
- bit/socket rotational coupling, screw loosening, or autonomous unscrewing;
- use of offline true center, contact tags, or capture disks by an online
  non-oracle controller.

For historical stage-by-stage reasoning, use `PROJECT_TRACE.md`; for the active
artifact registry, use `README.md` and `outputs/current/README.md`.
