# M8 Proposed Force And Transport Audit

Updated: 2026-07-15

> Current-status note (2026-07-15): this is a historical mechanism/force audit.
> The current closed-loop contract is the seven-method strict 24-pose report in
> `outputs/current/evaluations/m8_strict_24_complete_loop/`. Older privileged
> evaluator and 36-pose capture-only numbers below remain development history.
> In particular, any independently added fallback budget is superseded by the
> current accumulated 130 s hard budget with zero extension.

## Question

The selected P2c rollout reported a 22.43 N force peak despite a 3 N search
target, while the fixed 1.50 mm posterior transport looked unnecessarily
conservative. Before using either setting on hardware, the force peak and the
distance rule must be separated and audited.

## Peak-force diagnosis

The original learned rollout was replayed without rendering and logged at every
0.5 ms MuJoCo physics step. The 22.43 N / 141.02 mN m peak occurred in exactly
one sample at the transition from `progress_dwell` to `insertion_handoff`.
It was not representative of the search stages:

| stage | raw force maximum |
|---|---:|
| conical probe | 4.35 N |
| posterior transport | 5.23 N |
| local outward spiral | 7.07 N |
| progress dwell | 4.87 N |
| old insertion handoff | 22.43 N |

The 25 Hz filtered force maximum was only 4.62 N. The old implementation
instantaneously changed the force target and reset the admittance state while
the bit was transitioning from rim/chamfer support to wall guidance. The spike
was therefore a control/contact-mode handoff artifact, not evidence that the
whole proposed search required a large contact force.

The handoff now continuously releases the target from the current search force
to 0.8 N over 0.30 s, then rebases the insertion controller and opens bounded XY
compliance. On the identical rollout:

| quantity | old handoff | continuous handoff |
|---|---:|---:|
| CAD-bottom result | success | success |
| capture time | 26.72 s | 26.72 s |
| total active time | 31.62 s | 28.07 s |
| raw force peak | 22.43 N | 7.07 N |
| filtered force peak | not recorded | 4.62 N |
| torque peak | 141.02 mN m | 20.89 mN m |

Raw wrench safety is now inspected at every physics step rather than only at
the slower 2 ms admittance update. This mirrors a hardware architecture in
which a fast safety monitor is independent of the force-control loop.

## Transport-distance diagnosis

The fixed 1.50 mm cap, rather than the fused posterior, was the active limit in
the selected case. The posterior radial quantiles were 2.51 / 3.59 / 4.69 mm.
The current planner keeps a conservative projected quantile but changes it
continuously with direction quality and reverse posterior mass:

$$
q_{dir}=\kappa\cos\left(\frac{\Delta\phi_{seed}}{2}\right),
\qquad
q_{eff}=\operatorname{clip}\left(q_{dir}(1-2P[s_i\le0]),0,1\right),
$$

$$
\alpha=0.10+0.15q_{eff},\qquad
d_{raw}=\max\left(0,Q_{\alpha}(s_i)-0.25r_{eff}\right),
$$

$$
d_{move}=\min\left(d_{raw},(0.35+0.30q_{eff})R_{99},d_{hardware}\right).
$$

The local spiral still covers the residual posterior:

$$
R_{local}=Q_{0.99}(\|c_i-p-d_{move}u\|)-r_{eff},
\qquad \Delta r_{turn}\le2r_{eff}.
$$

Single-case ablation:

| distance policy | transport | local radius | capture time | contact path | wear proxy | raw peak |
|---|---:|---:|---:|---:|---:|---:|
| fixed cap 1.50 mm | 1.500 mm | 2.945 mm | 26.72 s | 45.56 mm | 83.76 mJ | 7.07 N |
| fixed cap 2.00 mm | 2.000 mm | 2.445 mm | 20.84 s | 32.34 mm | 61.54 mJ | 7.09 N |
| legacy projected-Q10 | 2.257 mm | 2.188 mm | 15.91 s | 21.82 mm | 43.21 mJ | 7.01 N |
| continuous quality-adaptive | 2.880 mm | 1.800 mm | 12.28 s | 13.73 mm | 29.42 mJ | 7.78 N |

This is evidence that the 1.50 mm cap and the later binary APS gate were
unnecessarily conservative in selected cases. It is not enough to freeze the
continuous rule for hardware. The rule must
retain its lower quantile, residual-coverage spiral, force/rate limits, and a
commissioning cap until a nuisance-paired distribution audit and then hardware
shadow-mode trials show no loss of capture or safety.

## Hardware interpretation

The research policy and the commissioning policy should be distinguished:

- research policy: posterior-limited transport plus residual outward coverage;
- initial hardware commissioning: clamp the same policy to a lower distance
  and speed while logging what the unclamped policy would have commanded;
- release the clamp in stages only after measured visual covariance, F/T bias,
  frame, latency, and effective capture radius are calibrated.

The distance is not a learned center regression. The learner contributes a
set-valued direction and support-state belief; vision and approximate CAD
provide radial probability; the outward spiral handles the remaining error.

## Evidence

- `outputs/current/diagnostics/proposed_force_audit/current_policy_trace.csv`
- `outputs/current/diagnostics/proposed_force_audit/optimized_handoff_trace.csv`
- `outputs/current/evaluations/m8_transport_distance_ablation/report.json`

All results above are M8 MuJoCo evidence with synthetic vision and privileged
contact tags in the capture evaluator. They are not hardware force limits or
wear measurements.

## Frozen fair-policy audit

The first nuisance-paired run exposed a second control bug: after a persistent
50 ms safe-progress cue, the FSM performed an unnecessary fixed-XY descent
confirmation. That extra motion could wedge at the rim/wall transition. The
FSM now treats persistent progress only as a mode-switch cue, continuously
hands force ownership to upright compliant insertion, and leaves task
completion to the later depth/wrench verifier. The same gate is used by all
three search policies.

The posterior policy was then frozen as:

1. one shared conical fingerprint;
2. posterior-limited transport with no arbitrary 1.50 mm cap;
3. 99% residual local outward coverage;
4. if local coverage is exhausted, return to the visual anchor and run the
   full visual-support outward spiral.

A new untouched Sobol seed (`20260714`) generated 36 M8 positions inside the declared
visual 95% radius. Each position shared friction in 0.55--0.95, contact time
constant in 0.04--0.07 s, and down-force in 2.5--3.5 N across methods.
Proposed and Baseline B used byte-identical first fingerprints and the same
frozen P1v3 ensemble. The corrected simulation evaluator also required the bit
center to enter the CAD socket capture region, preventing screw-head outer-edge
chamfer drop from being counted as socket capture.

| method | capture | hard stop | time median | contact path median | wear median | raw force p90 |
|---|---:|---:|---:|---:|---:|---:|
| Proposed posterior + local + global fallback | 34/36 | 2 | 16.00 s | 21.72 mm | 45.72 mJ | 9.74 N |
| Baseline A full outward | 31/36 | 5 | 44.32 s | 82.85 mm | 151.37 mJ | 14.79 N |
| Baseline B shared direction + direct/retry | 34/36 | 0 | 7.46 s | 5.52 mm | 13.50 mJ | 7.18 N |

Only one Proposed trial used the global fallback. Proposed and Baseline B each
had two exclusive successes (`McNemar p=1.0`); on their 32 common successes,
Baseline B was faster, shorter, and lower-work in every pair. Versus Baseline A,
Proposed had four exclusive successes and A had one (`p=0.375`). On 30 common
successes, Proposed was faster in 80.0%, shorter in 96.7%, and lower-work in
93.3%, but its peak force was lower in only 40.0%.

The honest conclusion is therefore conditional:

- the fixed 1.50 mm cap was too conservative;
- the corrected Proposed policy and direct/retry have different failure sets,
  but no net success-rate difference in this audit;
- Baseline B remains the best action when its direction ray captures;
- Proposed is not yet a universal speed, wear, force, or success-rate winner;
- the strongest current role is a robust hybrid: direct posterior shortcut,
  local residual coverage, then a rare full outward fallback.

Final artifacts:

- `outputs/current/evaluations/m8_fair_policy_final_audit/report.json`;
- `outputs/current/evaluations/m8_fair_policy_final_audit/summary.md`;
- pre-fix and viewed development runs remain under
  `outputs/current/diagnostics/` and were not reused as final evidence.

## Historical Pre-Hardware Observation Robustness

A separate historical 36-pose audit inserted a shared stateful F/T observation model
between MuJoCo physics and every consumer of wrench data. It includes residual
bias, random-walk drift, colored noise, gain/cross-axis error, and 2.0--7.87 ms
delay. Proposed and Baseline B still receive byte-identical first fingerprints.

The frozen clean-trained policies with continuous posterior distance and the
shared sensor-only capture handoff reach 36/36 Proposed, 35/36 full outward,
and 29/36 direct/retry captures; hard-stop counts are 0/1/0. Proposed and B
have 7/0 exclusive successes with no joint failure (`McNemar p=0.015625`). The
finite result reflected the then-current independently budgeted
physical-reset/full-outward fallback after its shortcut; that budget policy is
not used by the strict controller. It is not a universal
success-set guarantee or evidence that Proposed is always faster: B remains
the faster, shorter, lower-work direct shortcut on common successes.

The observation parameters are engineering priors, not measured hardware
noise. They are useful for exposing brittle interfaces and planning the real
calibration set. Hardware claims require replacing them with measured sensor
bias/gravity residuals, covariance, frame calibration, and end-to-end delay.
