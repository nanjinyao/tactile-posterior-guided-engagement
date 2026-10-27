# Hardware-easy-v2 shared M8/M10/M12 simulation adaptation

Date: 2026-07-21

## Frozen outcome

M10 and M12 are now included with M8 in one shared, three-seed direction
ensemble:

`outputs/current/models/hardware_easy_v2_multisize_robust_direction_v4/`

The load-bearing post-freeze audit is:

`outputs/current/evaluations/hardware_easy_v2_multisize_robustness_fresh6/`

This promotes a geometry-conditioned **simulation direction interface**, not a
complete insertion controller. An accepted posterior may drive only bounded
transport; abstention routes to guarded outward search or a new probe. Capture,
compliant insertion and terminal engagement verification remain separate.

## Data and leakage control

Each canonical easy-v2 size set contains 768 MuJoCo tasks: eight radii, eight
local angles in one 60-degree fundamental sector, all six C6 rotations, and two
nuisance repeats. Splitting is performed before capture filtering. The physical
group key removes the nuisance suffix, so every nuisance repeat and all six C6
copies stay together; canonical data therefore form 64 physical orbits of 12
tasks, split into 51 training and 13 calibration orbits. This prevents rotated
or nuisance-correlated copies of the same contact condition from leaking across
the split.

| direction domain | canonical tasks | complete direction fingerprints | captured/incomplete, excluded from direction labels | final train | final calibration |
|---|---:|---:|---:|---:|---:|
| M8 easy-v2 | 768 | 675 | 93 | 669 | 152 |
| M10 easy-v2 | 768 | 624 | 144 | 496 | 128 |
| M12 easy-v2 | 768 | 602 | 166 | 462 | 140 |
| M8 source replay | legacy source replay | legacy source replay | see note below | 1,984 | 254 |

The M8 easy totals include 124 train and 22 calibration examples from one
failed 180-task audit that was explicitly reclassified as development. The M8
source totals include 132/31 and 136/31 train/calibration examples from two
failed 180-task audits, also explicitly reclassified as development; those two
sets contained 17 and 13 capture/incomplete outcomes. M10 and M12 consume no
failed-audit development sets. Across the four final domains, fitting uses
3,611 direction examples and calibration uses 674. Captured or incomplete
traces used for direction supervision: **zero**.

Capture exclusion is semantic, not data cleaning: a trace captured during the
scan is a successful mechanics-routing outcome, but it no longer supplies the
complete circular fingerprint required by the direction estimator.

## Size-aware, dimensionless input contract

The shared network retains four 360-bin phase channels
`[Fz, Tx, Ty, tip_z]`. Its shape branch standardizes each channel over phase.
Its context branch uses the size-aware nondimensional quantities

$$
\left(
\frac{F_z}{F_d},\;
\frac{T_x}{F_d r_b},\;
\frac{T_y}{F_d r_b},\;
\frac{z}{r_b}
\right),
\qquad r_b=\frac{\mathrm{AF}_{\mathrm{peg}}}{\sqrt{3}},
$$

with mean, standard deviation, minimum and maximum from each channel. Runtime
must pass the correct `bit_radius_m`: M8 `0.003464101615`, M10
`0.004618802154`, and M12 `0.005773502692` m. Passing the M8 radius for another
size violates the frozen inference contract.

## v4 adaptation and calibration

- Initialization: the promoted M8 easy-v2 v5 three-seed ensemble.
- Adaptation mode: `tail`; trainable modules are the last convolution, shared
  block, state head and direction head.
- Frozen safeguards: per-seed M8-v5 normalization, batch-normalization running
  statistics, L2-SP anchors and the inherited maximum APS set size of three.
- Domain objective: M8 source replay weight 0.60; M8, M10 and M12 easy-v2 each
  weight 0.1333, with an independent batch from every domain per optimizer
  step.
- Runtime calibration: one global, equal-domain temperature/APS/transport gate,
  rather than a hidden size-specific gate.

The earlier v3 contract used a looser APS quantile and fresh5 obtained only
0.937 coverage on M8 source, below the predeclared 0.95 promotion gate. v4 fixes
the contract mismatch by exposing `aps_alpha=0.05`, recalculating the
finite-sample 95% APS quantile (`0.9872068167`), freezing the model/calibration,
and then generating fresh6. No fresh6 sample was seen during fitting or
calibration.

## Fresh6 frozen audit

Every fresh6 domain contains 180 tasks and retains
`training_use=forbidden`. Captured/incomplete tasks are reported as routing
outcomes and receive no direction error, APS score or transport-acceptance
score.

| shared v4 domain | valid / total | routed capture | median / p90 error | within 7.5 deg | APS coverage / mean size | gate accept / accepted within 7.5 deg | accepted >90 deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| M8 source | 170 / 180 | 10 | 2.17 / 5.00 deg | 0.947 | 0.994 / 2.90 | 0.876 / 0.993 | 0 |
| M8 easy-v2 | 162 / 180 | 18 | 1.39 / 3.80 deg | 0.994 | 1.000 / 2.41 | 0.988 / 0.994 | 0 |
| M10 easy-v2 | 162 / 180 | 18 | 1.22 / 2.55 deg | 1.000 | 1.000 / 2.46 | 1.000 / 1.000 | 0 |
| M12 easy-v2 | 160 / 180 | 20 | 1.46 / 3.16 deg | 1.000 | 1.000 / 2.51 | 0.994 / 1.000 | 0 |

Hard-stop outcomes are zero in all four rows. Independent M8, M10 and M12
promotion gates pass, including the finite-sample radial screen; overall
geometry-conditioned multisize simulation promotion is **PASS**. The M8 source
row contains two latent raw errors above 90 degrees, but the frozen gate rejects
both. The promoted object is therefore the guarded interface, not every raw
top-1 prediction.

## Physical claim boundary

The audit establishes only that one size-aware network can map matched MuJoCo
contact fingerprints to guarded direction posteriors across M8/M10/M12 under
the sampled geometry/noise prior. It does **not** establish real-hardware
fine-tuning, chamfer capture probability, compliant insertion success, terminal
engagement, or a full-loop task success rate.

The Bota measurement origin and sensor-to-tip transform, the common 42 mm
hardware lever, printed mass/COM/inertia, FDM stiffness and friction, burrs and
layer ridges, real noise/latency, robot compliance and fixture registration are
still commissioning quantities. Moreover, the enlarged easy-v2 walls and
lead-ins intentionally enlarge the mechanical basin and change yaw free play;
they must not be described as improved estimator accuracy. Real trials must
first run in shadow mode and keep capture and terminal verification independent
from the learned direction cue.

No visualization, video, PPT, CAD or robot-command path was changed by this
documentation update.

## Verification

The focused Python suite covering multisize contracts, easy-v2 geometry and
artifacts, direction training/runtime, conical capture routing and hardware
tooling passes 35/35 tests. The final artifact test pins v4's three checkpoints,
`alpha=0.05`, zero final-audit fitting samples, all four forbidden fresh6
contracts, all three size gates and the zero hard-stop/accepted-catastrophe
conditions.
