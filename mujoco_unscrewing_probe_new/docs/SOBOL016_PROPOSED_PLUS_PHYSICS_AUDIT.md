# `sobol_016` Proposed+ physics audit

This is a read-only audit of the frozen selected-mechanism video. It does not
rerun MuJoCo and it does not read any cross-size untouched task.

## Verdict

The rollout is consistent with the frozen paired M8 trial, and its online
action path does not receive the true center or MuJoCo contact tags. However,
the mechanism must **not** be described as entropy convergence or as a
successful shortcut.

- The circular-mean control direction is `5.04372 deg`; the offline true-center
  direction is `49.84098 deg`, hence the first shortcut error is `44.79726 deg`.
- The first executed shortcut is `0.903266 mm`. It reduces the offline radial
  distance from `3.29754 mm` to `2.73175 mm`, but it does not capture.
- The later serialized `0.542486 mm` value is an unexecuted replan after the
  fourth belief update, not a second shortcut.
- The entropy increases from `6.06478` to `6.44827 nat`, and the credible radius
  increases from `3.93058` to `4.53990 mm`. After repeated noncontracting
  updates, Proposed+ downweights information gain and becomes a
  capture-weighted coverage policy.
- Probe 5 triggers a persistent sensor-only inner-wall cue during the smooth
  segment at `15.4475 s`; the shared physical funnel then reaches the CAD-bottom
  terminal set at `22.1925 s` with no hard stop.

The command endpoints drawn by the grid are not measured-tip samples. Contact,
low planar stiffness, admittance, and the chamfer constrain the measured path:
the summed commanded task-grid travel is about `18.895 mm`, while the complete
measured XY path is `8.052 mm` (`7.815 mm` in contact). The frozen trial stores
measured-path sample counts and aggregate lengths but not the full polyline, so
the exact tracking-error history cannot be reconstructed without a diagnostic
replay.

## Claim boundary

The supported statement is:

> In this selected M8 case, a short but directionally inaccurate shortcut is
> recovered by contact-limited, task-aware coverage; a sensor-only capture cue
> hands off to the shared physical funnel, which reaches the terminal set.

Unsupported statements include "the shortcut enters the hole," "the entropy
grid converges to the true center," and "the yellow grid line is the measured
tip trajectory."

The selected-media manifest freezes the source trial, audit report, rendered
video, and 504 substantive replayed fields with zero mismatch. It does not
freeze the generating source hashes; current source files postdate the video,
so current hashes are diagnostic rather than proof of source identity at render
time.
