# MuJoCo Modeling And Experiment Design

Updated: 2026-07-15

> Current-status note (2026-07-15): sections below that describe a privileged
> online chamfer/wall tag or CAD capture-region gate are historical diagnostics.
> The strict controller uses depth trend + measured wrench online and keeps
> MuJoCo geometry/contact tags for offline scoring only; current success also
> requires a dual-channel CAD-bottom dwell under a 130 s zero-extension budget.
> See
> `FULL_CLOSED_LOOP_SCIENTIFIC_AUDIT.md`.

## Scope

This document records the clean-path MuJoCo modeling plan after the meeting and
the ChatGPT-roadmap discussion.

The implementation supports parameterized M8/M10/M12 Allen socket-head models,
but the first evidence boundary remains:

```text
M8-first fixed Allen socket-head engagement under bounded visual residual error.
```

M10/M12 are model-ready scale variants. They are not a generalization claim until
scale experiments are run and summarized.

## Geometry Family

The modeled object is:

```text
workpiece
  -> fixed circular socket-head screw
       -> Allen hex socket opening
       -> sloped micro-chamfer
       -> vertical hex socket walls
       -> socket bottom
  -> matching Allen bit above the recess
```

Not modeled:

- screw threads
- extraction torque
- screw rotation
- tool changer
- visual detector
- full autonomous unscrewing

## Dimension Convention

The size table uses nominal Allen key / bit across-flats values:

```text
M8  -> 6 mm Allen bit
M10 -> 8 mm Allen bit
M12 -> 10 mm Allen bit
```

The socket wall across-flats is modeled as:

```text
W_wall = W_bit + 2c
```

where `c` is a single-side clearance chosen as a controlled simulation variable.
The micro-chamfer expands the top opening by `a_ch` on each side:

```text
W_top = W_wall + 2a_ch
```

The aligned top and wall capture radii are:

```text
r_cap_wall = c
r_cap_top  = c + a_ch
```

Clearance and chamfer values are not claimed as ISO manufacturing tolerances.
They are physically plausible simulation variables for capture-basin and
tolerance studies.

## Generated Artifacts

Run:

```bash
python3 -m src.allen_socket_model --sizes m8,m10,m12
```

Generated files:

```text
configs/allen_m8_source.json
configs/allen_m10_source.json
configs/allen_m12_source.json
models/allen_m8_source.xml
models/allen_m10_source.xml
models/allen_m12_source.xml
docs/M8_M10_M12_MODEL_INVENTORY.md
outputs/current/reports/m8_m10_m12_model_inventory.json
```

## Current Model Table

See `docs/M8_M10_M12_MODEL_INVENTORY.md` for the generated table. The intended
values are:

| size | bit AF | socket wall AF | clearance | chamfer w/d | top AF | r_cap wall/top | socket depth | head D/H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M8 | 6.00 mm | 6.20 mm | 0.10 mm | 0.15/0.15 mm | 6.50 mm | 0.10/0.25 mm | 4.0 mm | 13.0/8.0 mm |
| M10 | 8.00 mm | 8.24 mm | 0.12 mm | 0.18/0.18 mm | 8.60 mm | 0.12/0.30 mm | 5.0 mm | 16.5/10.0 mm |
| M12 | 10.00 mm | 10.30 mm | 0.15 mm | 0.22/0.22 mm | 10.74 mm | 0.15/0.37 mm | 6.0 mm | 18.5/12.0 mm |

## Experiment Design

### E0: Geometry sanity

Goal: verify generated model dimensions and derived capture radii.

Metrics:

- `bit_af`
- `socket_wall_af`
- `socket_top_af`
- `clearance`
- `r_cap_wall`
- `r_cap_top`
- `head_diameter`
- `socket_depth`

Pass criterion:

```text
Generated XML/config inventory matches the table above.
```

### E1: M8 source-style baseline

Goal: establish baseline under the same initial-offset distribution.

Conditions:

- size: M8 only
- initial visual residual: start with 1/3/5 mm rings or bounded area-uniform sampling
- controller: source-style guarded spiral / threshold baseline

Metrics:

- inserted / engagement rate
- time-to-insert
- max force
- contact work
- path length
- failure reason

### E2: M8 action-conditioned trigger dataset

Goal: make the meeting pipeline concrete.

Before building the full trigger dataset, run a conical/nutation fingerprint
smoke as described in `docs/M8_CONICAL_PROBE_STATE_TRANSITION_DESIGN.md`.
This tests whether a small phase-structured probe can provide a reliable
inward-direction prior for the guarded spiral.

Then run the signed-distance observability pilot in
`docs/M8_FINGERPRINT_OBSERVABILITY_AND_DATASET.md`. It must test the local
Jacobian/Fisher information and fingerprint collisions before the project
chooses direction-only or direction-plus-radius labels. The first model should
not regress the true center by default.

Scope boundary: conical/CNN data covers only the three ambiguous flat-support
states (`outer_partial`, `double_overhang`, and `supported_search`).
`rim_chamfer`, `inner_wall`, `inserted`, and `bottom_reached` use separate
white-box transition/terminal gates calibrated from short force-ramp or descent
traces of `z`, `Fxy`, `tau_xy`, and `Fz`; they are not conical-pattern classes.

Actions:

- `press_read`
- `press_rotate_small`
- `conical_probe_small` as a candidate active-sensing primitive, if the smoke
  shows interpretable phase structure
- `lateral_probe_px`
- `lateral_probe_nx`
- `lateral_probe_py`
- `lateral_probe_ny`

Labels:

- press labels: `no_contact`, `plane`, `near_edge`, `inserted`, `ambiguous_jam`, `unsafe`
- probe labels: `slide`, `block`, `drop`, `unsafe`

Required outputs:

- `rows.csv`
- `features.csv`
- raw 6D F/T sequence files
- `summary.md`
- `results.json`
- confusion matrices
- observability/Fisher maps and fingerprint-collision report
- grouped train/validation/test split manifest created before symmetry augmentation

### E3: Classifier and conformal gate

Goal: compare physics features, KNN/small MLP, and a polar phase CNN. The
learning output should be a circular direction distribution `q(phi)`, a
finite-coverage angular set, calibrated confidence, search state, and unsafe
score.

Claim boundary:

```text
classifier reads contact event; it does not predict true center or output low-level control.
```

Model ladder: physical/Fourier phase features, paper-style polar-image CNN,
then a compact 1D circular CNN. Use only physically consistent $C_6$
augmentation, rotating phase, CAD/yaw, direction labels, $F_{xy}$, and
$\tau_{xy}$ together. Rotated copies cannot enter a different split from their
base rollout.

### E4: M8 gated recovery smoke

Goal: connect the classifier output to conservative action choices.

Routing:

```text
unsafe in set -> retreat
outer partial -> CAD-filtered medium biased spiral
double-overhang / wide angular set -> slow dense spiral or re-probe
supported search / narrow angular set -> short fast biased spiral
safe downward progress -> end spiral expansion, retain low down-force, and begin guarded upright descent; do not classify chamfer online
inner-wall / inserted -> bypass learning and use the white-box depth+wrench gate
```

Historical representative M8 visualization schedules used:

| Initial state | Confidence | Angular half-width | Turns | Speed scale | Video |
|---|---:|---:|---:|---:|---|
| outer partial | 0.63 | 24 deg | 5.00 | 0.75 | `m8_spiral_outer_partial_to_rim_chamfer.mp4` |
| double-overhang | 0.42 | 46 deg | 5.50 | 0.50 | `m8_spiral_double_overhang_to_rim_chamfer.mp4` |
| supported search | 0.86 | 13 deg | 3.50 | 1.00 | `m8_spiral_supported_search_to_rim_chamfer.mp4` |

All three keep roll/pitch at zero and stop after depth drop plus a privileged
MuJoCo chamfer/wall contact tag persists for 50 ms at the 2 ms controller rate.
The tag is a simulation proxy for generic safe downward progress, not an online state label. The
spiral phase uses only the posterior transport direction, not the true hole
center. These are assumed-learning-output visualizations, not learned results.

The additional same-start outer-partial stress comparison uses one continuous
dynamic rollout per policy rather than concatenating V1 and V6a. At `(4.00,
0.00) mm`, both policies use 3 N, 0.40 mm/rev, 2.25 mm/s, a 70 s active budget,
and the same below-rim gate plus privileged M8 CAD capture-region evaluator.
Proposed reaches that endpoint at 15.91 s with 21.70 mm actual XY path,
including 5.80 s probe time. The uninterrupted source-style outward baseline
reaches it at 50.59 s with 109.86 mm actual XY path. For communication, the
comparison stays at 1x until proposed completes and then displays baseline at
8x while preserving the physical simulation clock. See
`outputs/current/reports/m8_outer_partial_search_comparison.json`. This is one assumed-
posterior, privileged-evaluator, fixed-phase stress case, not a distribution-
level speed, path, or wear result.

### E4b: Controller and threshold calibration

Sweep the M8 simulation-initial values recorded in
`outputs/current/reports/m8_sampling_video_manifest.json`:

- nominal down-force proxy: 3 N;
- soft speed gate: 8 N force / 20 mN m tilt torque;
- hard abort: 20 N force / 60 mN m tilt torque;
- search-to-descent candidate: 0.07 mm absolute depth drop plus 0.04 mm recent
  descent over 50 ms while raw wrench stays below 8 N / 20 mN m;
- capture confirmation: freeze XY, ramp to 2 N over 150 ms, reduce planar
  impedance, and require another 0.04 mm descent;
- pre-hard search escape: 12 N / 40 mN m; hard abort remains 20 N / 60 mN m;
- inner-wall gate: depth -0.30 mm plus 0.25 N lateral force or 1.5 mN m tilt torque;
- bottom completion uses a method-independent dual channel inside the 0.15 mm CAD-depth band: either expected-sign axial force of at least 0.70 N, or low absolute axial force with stable depth; both channels require bounded lateral force/tilt torque and dwell. All thresholds remain simulation-initial.

The first real-robot experiment must calibrate the implemented force-ramp
sensor-only safe-progress event using `z`, `Fz`, `Fxy`, `tau_xy`, and
persistence; no chamfer class is required. It must also recalibrate these values with sensor bias, sample
rate, stiffness, friction, hysteresis, and independent emergency-stop logic.

### E5: Scale sweep after M8 is stable

Only after E1-E4 are traceable:

- M10 nominal
- M12 nominal
- optional clearance/chamfer sweeps

Use dimensionless pose, depth, force, torque, and CAD ratios. Test both
zero-shot transfer and leave-one-size-out transfer before per-size fine-tuning.
Each real size still requires independent calibration/test trajectories because
manufacturing clearance, chamfer, concentricity, sensor lever arm, and contact
control do not scale exactly.

This is a scale-variation experiment, not a shape-generalization claim.
