# Next Random-Deployment And Scale-Transfer Experiments

Updated: 2026-07-15

> Current alignment (2026-07-15): the frozen simulation reference is the
> seven-method strict complete-loop audit in
> `outputs/current/evaluations/m8_strict_24_complete_loop/`. Proposed is the
> paper-facing method; Proposed+ is an adaptive research branch with no
> superiority claim; outward, entropy-grid, direct-first, random and Oracle
> have distinct comparator/diagnostic roles. The 36-pose four-method report is
> historical capture-only evidence, and its simulator-reset direct/retry is not
> an active strict baseline.

## 1. Frozen Decision

The next learner remains the frozen P1v3 three-seed direction ensemble.

- learned output: circular direction posterior, APS angular set, concentration,
  ensemble disagreement, and transport gate;
- geometric fusion: measured visual prior plus CAD feasible set;
- action: conservative posterior transport plus finite outward search;
- strict comparators: ordinary outward search, repeated entropy-grid contact
  probing, one shared-probe direct-first ray with no reset claim, and seed-fixed
  random waypoints; privileged Oracle is an upper-bound diagnostic;
- Proposed+ may reorder finite-width grid actions and conditionally reserve a
  global-outward fallback, but it receives no time extension beyond the shared
  130 s accumulated active-controller budget;
- no exact radial-distance head;
- no tuning on P1v5 or on any new final test set;
- safety and terminal verification remain outside the neural network.

## 2. Why Two M8 Test Sets Are Required

P1v5 is an untouched direction audit, but its positions are structured radial
strata and angles rather than random draws from the deployment visual error.
P2b is also structured and uses the older P1 runtime. Therefore neither is a
deployment-weighted random closed-loop evaluation of frozen P1v3.

### R0: Measure the visual prior

Estimate the camera-to-socket residual covariance from real repeated
localization rather than choosing a convenient simulation radius:

$$
\delta_i=c_{CAD}-\hat c_{vision},\qquad
\hat\Sigma_v=\operatorname{Cov}(\delta_i).
$$

The declared 95% acceptance ellipse is

$$
\mathcal E_{95}=\left\{\delta:
\delta^T\hat\Sigma_v^{-1}\delta\le\chi^2_2(0.95)=5.991
\right\}.
$$

Until this measurement exists, the current isotropic 2 mm Gaussian remains a
simulation assumption and cannot support a real deployment claim.

### R1: Deployment-weighted random M8 benchmark

Freeze all model, gate, planner, controller, and verifier parameters before
sampling.

- draw 120 initial XY positions independently from the measured visual-error
  distribution conditioned on $\mathcal E_{95}$;
- pair frozen Proposed / Proposed+ / ordinary outward / entropy-grid /
  shared-probe direct-first / seeded random / privileged Oracle at every point,
  while preserving their declared method roles rather than presenting all seven
  as equal paper contributions;
- pair the same friction, contact time constant, down-force, yaw, and start
  phase nuisance across methods;
- include all sensing time and contact work; outward, entropy-grid and random do
  not receive a conical trace, while Proposed, Proposed+ and direct-first use a
  byte-identical first conical trace; Oracle's privileged input is explicit;
- preregister the random seed before rollout and use either three random seeds
  per pose or a blocked pose-seed design, because one lucky random chord is not
  a method-level result;
- use no test-set threshold or planner tuning;
- save raw F/T, pose, posterior, APS, gate, path, event trace, and terminal
  result for every episode.

Primary reports:

- direction APS coverage, accepted angular accuracy, rejected fraction, and
  accepted error above 90 degrees;
- strict CAD-bottom success and paired McNemar interval, with sensor handoff,
  offline capture and terminal mode reported separately;
- named failures: `capture_without_bottom`, `hard_stop`, `safe_abort`,
  `time_budget_exhausted`, and `search_radius_exhausted`;
- paired time, XY path, work proxy, peak wrench, and hard-stop count;
- conditional efficiency only on equal-endpoint pairs;
- penalized or survival-style aggregate including failures.

The intended paper claim does not require Proposed success or time to dominate
every comparator. Direct-first should often be fastest when its direction ray
is correct. The required result is narrower: the learned shortcut should reduce
coverage path relative to unstructured outward/random search, and its local/
global fallback should reduce the consequence of a wrong direction or radial
estimate relative to direction-only direct motion.

### R2: Boundary challenge set

Random Gaussian draws under-sample difficult boundaries. Keep a separate,
non-deployment-weighted challenge set:

- 30 positions near the 3.036 mm full-support boundary and the measured visual
  95% boundary;
- 12 physical directions across the set;
- independent friction/stiffness/yaw/down-force nuisance;
- frozen learned / baseline / oracle triplets;
- report by radial/CAD signed-distance stratum, never merge these results into
  the R1 deployment average.

## 3. Direction Plus Vision/CAD, Not Learned Exact Distance

For center offset $\delta=c-p_0$, the current hybrid posterior is

$$
p(\delta\mid y,vision,G)\propto
q_\psi(\phi(\delta)\mid y,G)
p_v(\delta)
\mathbf 1[\delta\in\mathcal C_{CAD}].
$$

If the training direction prior is not uniform, divide by the corresponding
training prior before deployment fusion. In the current strict controller, the
conical posterior may shorten or reorder search, but it does not create a new
success guarantee. Proposed+ uses measured-path belief updates and a
finite-width task-aware lattice; before shortcut/grid/commit it checks a
conditional nominal reserve for guarded return, full-radius outward search,
candidate handling and terminal verification. If the assumptions fail or the
reserve is insufficient, it switches early or exits with a named failure. The
accumulated active-controller budget remains

$$
T_{active}\le 130\ \mathrm{s},\qquad T_{extension}=0.
$$

The padded M8 lattice has 749 actions at 0.3359 mm spacing and a certified
geometric worst gap of 0.2375 mm below the CAD-derived
$r_{eff}=0.25$ mm. This certificate concerns finite-domain intersection only;
tracking, dwell, scheduling, safety and CAD-bottom verification remain separate
conditions.

The padded lattice alone produces a conditional node-geometry coverage
certificate, not an unconditional guarantee of insertion. It still depends on the true center lying in the
declared posterior set, adequate chamfer/capture width, compliant down-force,
and absence of jamming or hard-limit events.

Only consider an ordinal radial-band head if all three conditions hold:

1. measured vision plus CAD still leaves a task-limiting local search radius;
2. a separate nuisance-conditioned observability audit shows stable radial-band
   information;
3. the radial band improves a new held-out random benchmark without increasing
   accepted unsafe moves.

Do not train direct millimeter center-distance regression from the current
phase fingerprint.

## 4. M10/M12: Test Before Training

M8/M10/M12 share Allen topology but are not exact scaled copies. Camera error
is approximately an absolute metric error, while clearance, chamfer, torque
lever arm, stiffness, and manufacturing eccentricity change with size.
Therefore M8 does not prove scale transfer.

Use dimensionless channels and CAD ratios:

$$
\tilde z=z/a,\qquad
\tilde F=F/F_d,\qquad
\tilde\tau=\tau/(F_da),
$$

with clearance/$a$, chamfer/$a$, and depth/$a$ as geometry conditioning.

### S1: Zero-shot direction test - completed in simulation

The executed design uses 3 states x 2 CAD-normalized radial bands x 2 off-grid
angles x 6 physical sectors = 72 new scans per size. The M8 P1v3 model,
preprocessing, 15 deg action, APS and thresholds were frozen; M8 received the
same 72-scan reference design.

| size | complete/hard stop | within 15 deg | APS coverage | state acc | context >5 sigma |
|---|---:|---:|---:|---:|---:|
| M8 | 72/72 / 0 | 1.000 | 1.000 | 1.000 | 0.014 |
| M10 | 72/72 / 0 | 0.986 | 0.986 | 1.000 | 0.014 |
| M12 | 72/72 / 0 | 0.972 | 0.972 | 0.986 | 0.153 |

The frozen direction gate passes with no greater-than-90-deg errors, so S2 is
scientifically justified. The M12 context shift and higher raw wrench forbid
interpreting this as transferred calibration or control safety.

### S2: Zero-shot task subset - completed, gate failed

Each size used 30 unseen Sobol initial positions inside its declared synthetic
visual 95% disk. Proposed, ordinary outward and oracle used the same nuisance,
sensor observation model, CAD-bottom terminal and physical-funnel guard. The
M8 learner was frozen; size-specific CAD, capture radius, coverage pitch and
visual support were recomputed.

| size | Proposed / outward / oracle bottom | Proposed hard stop | executed direction median/p90 | executed direction >90 deg | common-success Proposed minus outward time/path |
|---|---:|---:|---:|---:|---:|
| M10 | 23/30 / 21/30 / 27/30 | 3 | 2.58/21.78 deg | 1 | -15.88 s / -46.16 mm |
| M12 | 21/30 / 19/30 / 30/30 | 4 | 3.58/24.76 deg | 1 | -17.86 s / -50.54 mm |

Both sizes fail the preregistered gate: oracle bottom >=0.90, Proposed bottom
>=0.80, zero Proposed hard stops and zero executed circular-mean direction
errors above 90 deg. The efficiency mechanism remains visible on paired common
successes, but complete zero-shot scale transfer is rejected.

Failure attribution is mixed. Each size has two Proposed failures with control
direction error above 45 deg. The other M10/M12 failures occur with less severe
direction error in search/terminal (3/4) or accurate-direction hard-stop
stages (2/3). Training alone therefore cannot repair S2.

### S3: Calibration only if zero-shot fails

Because S2 fails, collect a new 12--24-scan learner calibration set per size and
tune only input affine normalization, temperature, a small adapter, or the
final head. In parallel, use a separate oracle-guided set to commission
capture-release/reset, force/torque limits and CAD-bottom dwell. Preserve the
current 30+30 S2 records as evaluation-only and use a new random seed for the
next untouched test. Full per-size backbone retraining remains the last option.

Real-robot scale transfer requires independent real scans on every claimed
size. Dimensionless normalization is an inductive bias, not evidence. A
low-cost hardware screen may start with 6 sectors x 3 radial bands = 18
structured zero-shot scans plus 12 untouched random scans per size. Use the
full 60-scan design when publication-grade sector/nuisance coverage is needed.

## 5. Execution Order

1. Finished: freeze the genuine P1v3 learned closed-loop M8 result.
2. Measure real M8 visual residual covariance and F/T bias/frame/delay.
3. Run R1 deployment-random M8 simulation with the measured covariance.
4. Run R2 M8 boundary challenge without changing R1 parameters.
5. Run M8 hardware shadow-mode direction audit, then gated short transport.
6. Finished: run M10/M12 S1 zero-shot direction scans.
7. Finished: run S2; both sizes fail the frozen complete-loop gate.
8. Next scale step: separate S3 learner calibration from oracle-guided
   controller/terminal commissioning, then rerun a new untouched S2 test.

This order keeps the paper's main contribution narrow: active tactile probing
provides a direction belief, analytical geometry and vision provide radial
support, and conservative coverage plus chamfer/compliance completes insertion.

For every P2 distribution comparison, direct-first must consume the same stored
conical trace and frozen P1v3 direction output as Proposed. Do not inject
direction errors or true radial distance. The strict primary result stops after
one bounded ray. A retry study is separate and valid only if unload, retract,
reset, retouchdown and re-probe are all physically simulated and charged to the
same hard budget. The selected historical same-start cases already show that a
correct direct ray can be substantially better than Proposed.

## 6. Minimum-Cost M8 Hardware Adaptation

The first real campaign should not repeat the simulation grid.

1. **H0, no-contact calibration:** repeated vision localization, six safe F/T
   gravity/bias poses, frame transforms, covariance and latency.
2. **H1, 12 safety scans:** six sectors x two radial bands at low force, shadow
   mode only. Verify scan completion, upright return, peak wrench and phase
   ordering.
3. **H2, 24 adaptation scans:** six sectors x three radial bands plus six
   boundary/day repeats. Freeze the backbone and first tune normalization,
   temperature, or a small adapter/head.
4. **H3, 12 untouched random scans:** sampled from the measured visual 95%
   ellipse; never used to change the model, gate, planner, or threshold.
5. **Active add only on failure:** at most 12 high-entropy, APS-disagreement,
   or action-boundary scans before another frozen test.

This gives 48 first-round contact scans, but only 24 enter fine-tuning. Labels
come from the calibrated jig/CAD offset and are offline only; the online
controller must still not see the true center. Keep all raw 6D F/T, pose,
phase, command, force target, geometry ID and timestamps in the same schema as
simulation.
