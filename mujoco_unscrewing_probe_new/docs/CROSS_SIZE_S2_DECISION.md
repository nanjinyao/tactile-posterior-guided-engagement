# M10/M12 Frozen-M8 Full-Loop S2 Decision

Updated: 2026-07-12

## Question

Can the frozen M8 conical-fingerprint learner, combined with size-conditioned
CAD geometry and coverage, complete the full M10/M12 geometric engagement loop
without any M10/M12 training or calibration?

## Frozen design

- 30 unseen Sobol poses per size inside the declared synthetic visual 95% disk;
- shared nuisance and pre-hardware F/T observation prior across Proposed,
  ordinary outward and oracle;
- frozen M8 P1v3 ensemble, preprocessing and learning interface;
- size-conditioned CAD, capture radius, visual support, transport cap and
  spiral pitch;
- common sensor-only capture handoff, physical-funnel guard and CAD-bottom
  terminal;
- oracle uses the true center only as an execution-feasibility upper bound.

The 60 trials are evaluation-only and cannot be recycled into training or
threshold selection.

## Result

| size | Proposed / outward / oracle bottom | Proposed hard stop | executed direction median/p90 | executed direction >90 deg | common-success Proposed minus outward time/path |
|---|---:|---:|---:|---:|---:|
| M10 | 23/30 / 21/30 / 27/30 | 3 | 2.58/21.78 deg | 1 | -15.88 s / -46.16 mm |
| M12 | 21/30 / 19/30 / 30/30 | 4 | 3.58/24.76 deg | 1 | -17.86 s / -50.54 mm |

Both sizes fail the preregistered full-loop gate. The efficiency mechanism is
still visible on paired common successes, but zero-shot complete-loop transfer
is rejected.

## Failure attribution

| source among Proposed failures | M10 | M12 | response |
|---|---:|---:|---|
| control-direction error >45 deg | 2 | 2 | fresh size-specific learner calibration and OOD gating |
| error <=45 deg, no capture/bottom | 3 | 4 | inspect planner, capture and terminal; do not relabel as CNN failure |
| comparatively accurate direction, hard stop | 2 | 3 | size-specific capture-release/reset/spiral commissioning |

M10 oracle reaches bottom in 27/30 and M12 in 30/30, so the fixtures are
executable. M10 still has three oracle capture-without-bottom cases, which is
direct evidence that terminal/control work is required independently of the
learner.

## Training decision

M10 and M12 should participate in **small calibration and independent test**,
but not yet in full multi-size backbone training.

1. Freeze the current S2 as evaluation-only.
2. Collect 12--24 fresh calibration scans per size, concentrated on supported
   search and outer double-overhang bands across all six physical sectors.
3. Tune input affine normalization, temperature, or a small adapter/head while
   replaying M8 data.
4. Separately run oracle-guided capture/release and terminal commissioning so
   controller fixes do not contaminate learning evidence.
5. Rerun 30 new untouched Sobol poses per size with a new seed.
6. Admit full multi-size backbone training only if the small adapter still
   leaves systematic directional aliasing.

## Claim boundary

This is simulation-only evidence for geometric engagement. It does not prove
real iiwa transfer, manufacturing tolerance, probability calibration,
passivity, first-turn bit/socket coupling, threaded unscrewing, or extraction.
