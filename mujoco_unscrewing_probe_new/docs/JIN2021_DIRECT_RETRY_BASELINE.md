# Jin 2021 Direction-Only Control: Fair Baseline and Scripted Recovery Diagnostic

> Current alignment (2026-07-15): the paper comparison below remains useful,
> but every shifted-reset/full-reprobe result in this file is historical
> capture-only evidence. The strict 24-pose table retains only `direct_first`:
> one shared conical fingerprint, one bounded direct ray, no simulator reset or
> physical retry claim, the common dual-channel terminal verifier, and the
> non-extendable 130 s active-time budget.

## Current Frozen M8 Proxy Comparison

The current same-domain comparator is no longer the historical scripted-retry
diagnostic. It is a preregistered `Jin-2021-style M8 proxy` with the
paper-specified $X_z,M_x,M_y$ polar-image interface, 2000 resampled points,
20-point moving average, `3x20x20` CNN input, and 2Conv--2Pool--1FC structure.
Because the PDF does not specify a hex label set, normalization/rasterizer,
filter widths, optimizer, direct-slide distance, or finite-time terminal, the
missing choices are frozen engineering adaptations and not attributed to Jin
et al.

Both arms consume byte-identical probe scans and use the same one-ray strict
controller, sensor-only handoff, CAD-bottom verifier, and 130 s zero-extension
budget. On 48 fresh paired M8 poses:

| estimator | first-ray capture | strict CAD-bottom |
|---|---:|---:|
| improved circular fingerprint | 37/48 | 32/48 |
| Jin-2021-style M8 proxy | 24/48 | 18/48 |

The exact McNemar values are 0.00443 and 0.00131. The structured P1v6
direction audit is deliberately retained even though it is mixed: the improved
model has better median/within-15, while the proxy has better p90,
within-7.5, common 12-sector accuracy, and fewer $>90^\circ$ errors. Thus the
paired base-pose cluster bootstrap for the structured-set median absolute-error
difference is -0.805 deg with 95% interval [-2.139, 0.663] deg, including zero.
The cluster-bootstrap endpoint was declared before evaluation, but its
deterministic seed and resample count were fixed post-run during the disclosed
consistency repair; it is not presented as a fully preregistered implementation.
The allowed conclusion is therefore higher closed-loop success against this frozen M8 proxy,
not uniformly higher direction accuracy and not superiority to the paper's
reported square/pentagon percentages.

The fresh strict-48 task-layer direction summary now uses the ray actually
executed by the controller. Improved versus proxy control-direction median/p90
is 3.34/20.55 deg versus 8.18/17.36 deg; within 7.5 deg is 38/48 versus 22/48,
within 15 deg is 42/48 versus 40/48, and errors above 90 deg are 0/48 versus
3/48. This is still mixed because the improved p90 is worse. The frozen
`direction_error_deg` field measures the ensemble top class and is retained as
a diagnostic only, not mislabeled as the executed ray.

The original frozen task contract and all 48 trial records remain unchanged.
`contract_amendment_derived_rebuild.json` is an explicitly non-preregistered,
post-completion correction that permits hash-chained rebuilding of reports,
tables and the figure only; it does not permit retraining, resampling, simulator
reruns or trial edits. The original preregistration `created_utc` is retained as
a date-level placeholder, while observed filesystem mtimes record only that the
preregistration file preceded the first proxy checkpoint by 17.536 s.

Active evidence:

- `outputs/current/evaluations/m8_jin_style_vs_improved/`
- `outputs/current/evaluations/p1v6_m8_jin_comparison_direction_audit/`
- `outputs/current/models/m8_jin_2021_style_direction_proxy/`

Local source:

- `reference/锥形探索.pdf`
- Shiyu Jin et al., *Contact Pose Identification for Peg-in-Hole Assembly under Uncertainties*

## Paper Contract

The local PDF supports the following implementation boundary:

1. Section III-A applies a full $15^\circ$ tilt-then-rotate scan under constant
   downward force.
2. Section III-B/C converts the sequence into polar channels and classifies a
   discrete error direction. The selected CNN inputs are $X_z$, $M_x$, and
   $M_y$.
3. The controller guides the peg along the predicted error direction and uses
   admittance plus a small yaw oscillation for insertion.
4. The method does not estimate error magnitude. The paper lists this as a
   limitation.
5. Section III-D attributes failures to a wrong direction prediction or
   insufficient admittance compensation. Recovery changes the initialization
   slightly and repeats the complete tilt-then-rotate procedure.
6. The evaluation declares failure after more than three attempts. Table I and
   Table II report nonzero multi-attempt and failure counts.

The paper does not specify an online angle-correction optimizer after failure.
This proxy therefore uses shifted re-initialization and full re-probing rather
than inventing such a module.

## Historical Fair M8 Direct/Retry Proxy

The historical Baseline B used exactly the same M8 conical F/T trace and frozen P1v3
direction estimator as the proposed method. It changes only the action after
inference:

```text
same 15 deg probe -> same frozen direction output
-> direct admittance ray up to the declared visual bound
-> stop immediately on the common capture gate
-> if insertion fails: shifted reset and full re-probe, max 3 attempts
```

It does not use the true center distance, CAD radial annuli, or a hand-selected
direction error. However, each retry reconstructs the simulator and the old
summary does not charge the reset motion/time. It is therefore excluded from
the current strict table. The retained historical clips under
`outputs/current/active_comparison/` explain the mechanism only.

The current `direct_first` diagnostic stops after the first bounded ray and is
scored by the same sensor-only handoff, offline capture check, CAD-bottom dwell,
hard-stop and 130 s zero-extension contract as all other strict methods.

## Scripted Recovery Diagnostic, Not an Active Baseline

```text
contact settle
-> full tilt-then-rotate
-> scripted direction class
-> direct slide under admittance
-> down-press + yaw insertion check
-> if no capture: unload, lift, shift reset, and repeat
```

The older selected stress sequence is:

| attempt | reset XY mm | direction error | result |
|---:|---:|---:|---|
| 1 | $(4.00, 0.00)$ | $+22.5^\circ$ | no capture |
| 2 | $(4.00, 0.25)$ | $-22.5^\circ$ | no capture |
| 3 | $(4.00, -0.25)$ | $0^\circ$ | capture |

The $\pm22.5^\circ$ values represent a selected discrete-class boundary stress.
They are not outputs from the paper CNN or from the local frozen P1v3 model.

## Historical Scripted Numbers, Not a Baseline Comparison

| method | attempt | capture time | actual XY path |
|---|---:|---:|---:|
| proposed posterior transport + local spiral | 1 | 15.91 s | 21.70 mm |
| Jin 2021-inspired direct + retry | 3 | 28.85 s | 26.56 mm |

This diagnostic illustrates a possible recovery mechanism, but it is not a fair
performance comparison because its direction errors are scripted and its old
proxy used the true initial offset magnitude to choose travel. It must not be
used to claim proposed superiority.

The conceptual mechanism contrast is:

```text
direction-only direct slide:
  residual angular error can miss the capture set and require a complete retry

posterior transport + finite local outward search:
  residual error is searched inside a bounded region after one probe
```

Pitch below the capture diameter controls adjacent-turn spacing, but the
implemented finite spiral has no padded outer boundary and therefore no
complete-domain coverage certificate. It remains an empirical local search,
not a global or hardware guarantee.

## Historical Frozen Pre-Hardware Observation Audit

The first non-scripted Baseline-B failure visualization now comes from a full
36-pose paired audit. Residual F/T bias, colored noise, drift, same-unit
cross-axis/gain error, and 2.00--7.87 ms delay are applied before the shared
frozen learner and controller. Proposed and B receive byte-identical first
noisy fingerprints.

Baseline B succeeds on 29/36 after up to three attempts; the physically reset
Proposed policy succeeds on 36/36. Their paired counts are 29 both, 7
Proposed-only, and no B-only or joint failures (`McNemar p=0.015625`). This is a
finite pre-hardware audit result, not a guarantee that Proposed contains B's
hardware success set. B remains the faster and shorter direct shortcut whenever
its ray captures.

The selected `sobol_002` video is generated by a frozen rule: first
lexicographic audit trial where Proposed succeeds and B succeeds only after a
retry. The shared circular-mean direction error is 18.98 deg and APS size is 8.
Proposed uses a risk-bounded 0.991 mm transport plus residual outward search;
B attempts one and two exhaust their rays. The retained historical video then shows low-force unload,
vertical retract, 0.5 mm shifted reset, guarded retouchdown, a new full conical
probe, and the next direct ray through attempt three. No angle error is injected
by the renderer.

The two rendered reset motions take 7.47 s. The frozen distribution audit still
excludes reset travel and time, so the reported Baseline-B timing remains a
favorable lower bound rather than the complete physical cycle time.
The three attempts remain independent rollout episodes: at each video boundary,
the contact solver state is reconstructed at the same final XY and released
nominal surface height before dynamically rendering retract, shift, and
retouchdown before attempt three captures. Thus the action sequence is complete, but solver-state continuity
across cuts is not a claim.

These observation parameters are pre-hardware engineering priors. The video is
a mechanism/failure illustration, not measured failure-rate evidence.

Diagnostic artifacts:

- `outputs/current/diagnostics/paper_direct_scripted_stress/scripted_direct_retry_to_inner_wall.mp4`
- `outputs/current/diagnostics/paper_direct_scripted_stress/scripted_stress_vs_assumed_proposed.mp4`
- `outputs/current/diagnostics/paper_direct_scripted_stress/result.json`

Historical robustness artifacts:

- `outputs/current/evaluations/m8_pre_hardware_noise_audit/`
- `outputs/current/diagnostics/historical_comparisons/robustness_noise/`
