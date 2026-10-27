# M8 Fingerprint Observability And Dataset Design

Updated: 2026-07-15

> Current alignment: this long document preserves the chronological
> observability/data-design record. Sections that call direct/retry or
> entropy-grid `Baseline B/C`, grant fallback an independent budget, or describe
> the 36-pose capture-only audit as active are historical. The current truth is
> the seven-method strict 24-pose contract in
> `FULL_CLOSED_LOOP_SCIENTIFIC_AUDIT.md`: direct-first has no reset claim,
> entropy-grid is a strong named diagnostic, random is separate, and every
> method shares a non-extendable 130 s active-time budget plus CAD-bottom dwell.

## 1. Decision

The first learning target is not the true socket center and not a control
action. It is a calibrated likelihood over the socket-center direction for the
three ambiguous flat-support states:

```text
outer_partial
double_overhang
supported_search
```

The conical fingerprint is fused with the measured visual prior and CAD
feasibility. The controller then computes a bounded transport and a covering
local outward spiral. A radial-distance head is added only if a separate
observability audit shows that distance survives nuisance variation.

This keeps the learning bottleneck narrow:

```text
physical probe owns observability
CNN owns a calibrated contact likelihood
CAD and the visual prior own geometric feasibility
the posterior planner owns distance and search coverage
admittance and hard limits own contact safety
```

The reference conical-probe paper predicts discrete error directions. It does
not establish continuous center-distance recovery. The current project must
therefore test radial observability rather than assuming it.

## 2. Observation Model And Identifiability

Let the current bit center be $p_0$, socket center be $c$, and relative offset
be $\delta=p_0-c$. A complete conical scan produces

$$
y(\theta)=h(\delta,\xi;G,a(\theta))+\epsilon,
$$

where $G$ is CAD geometry, $a(\theta)$ is the commanded probe phase, and
$\xi$ contains nuisance variables such as friction, contact stiffness,
clearance, down-force, yaw, controller parameters, sensor bias, delay, and
virtual-pivot error.

In an ideal circular-boundary approximation, a probe point at radius $a$
crosses a CAD boundary of radius $R$ when

$$
\|\delta+a[\cos\theta,\sin\theta]^T\|^2
=\rho^2+a^2+2a\rho\cos(\theta-\phi)=R^2.
$$

The event phase encodes the direction $\phi$. Under ideal geometry, the angular
width between two crossings can also constrain $\rho$. Real Allen contact is a
hybrid multi-contact problem, so this inverse relation is only motivation, not
a proof of global injectivity.

Before training a CNN, estimate the mean fingerprint $\mu(\delta)$ and noise
covariance $\Sigma_y$ from repeated MuJoCo rollouts. Compute the finite-
difference Jacobian and Fisher information:

$$
J(\delta)=\frac{\partial\mu}{\partial(\delta_x,\delta_y)},
\qquad
\mathcal I(\delta)=J(\delta)^T\Sigma_y^{-1}J(\delta).
$$

Local two-dimensional estimation is useful only where $J$ has rank two and its
smallest singular value remains above the nuisance/noise scale. Also run a
nearest-neighbor collision audit: two fingerprints that are close after the
deployment preprocessing must not correspond to materially different center
directions or safe actions.

Decision rule:

| Audit result | Learning scope |
|---|---|
| Direction separable, radius aliased | learn `search_state` and $q(\phi)$ only; derive movement distance from visual prior + CAD |
| Direction and radius both stable | add a radial quantile or joint polar-posterior head |
| Direction also aliased in a region | report it as unobservable; use $d_{move}=0$, ordinary outward coverage, or a second probe |

No network architecture can repair a physical alias that remains after adding
the available action and CAD context.

## 3. Posterior And Planner Interface

If a network is trained under sampling prior $p_{train}(\delta)$, its output
must be corrected before combining it with the deployment visual prior:

$$
p(\delta\mid y,G,vision)\propto
q_\psi(\delta\mid y,G)
\frac{p_{vision}(\delta)}{p_{train}(\delta)}
\mathbf 1[\delta\in\mathcal C_{CAD}].
$$

If the first model outputs only $q(\phi)$, the radial posterior is inherited
from the visual prior after conditioning on the angular likelihood and CAD
state compatibility. The tactile model therefore does not need to predict an
exact distance to make a useful conservative movement.

The planner consumes posterior center samples. With selected unit direction
$u$ and $s_i=u^T(c_i-p_0)$, the existing first rule is

$$
d_{move}=\operatorname{clip}\left(
Q_{0.10}(s_i)-r_{eff},0,\min(0.5R_{95},d_{max})
\right).
$$

A hardware version must additionally reject any transport whose posterior CAD
corridor leaves the valid supported-contact region. Low concentration,
multimodality, or OOD sets $d_{move}=0$. The local outward spiral then covers
the remaining posterior with pitch no greater than $2r_{eff}$.

## 4. Geometry-Aware Sampling Coordinates

Do not sample raw $x,y$ uniformly over a large square. Parameterize initial
conditions by CAD signed distances:

- signed distance to the circular head support boundary;
- signed distance to the hex socket/chamfer boundary;
- polar direction in the sensor/CAD frame;
- state-transition margin between `outer_partial`, `double_overhang`, and
  `supported_search`;
- normalized yaw mismatch.

This makes the important narrow transition bands explicit and keeps most
samples near places where a different action might be chosen.

### P0 observability pilot

Start with approximately:

$$
24\ \text{directions}\times
8\ \text{signed-distance bands}\times
5\ \text{nuisance repeats}=960\ \text{rollouts}.
$$

Use adjacent polar/signed-distance sites to estimate finite differences, and
add targeted $\pm x/\pm y$ neighbors only where that grid is ill-conditioned.
P0 produces phase plots, singular-value/Fisher maps, nearest-neighbor collision
maps, and a go/no-go decision for the radial head.

### Current P0a result

The first cheap symmetry stage is complete:

- 12 base poses across the three flat-support states;
- six $60^\circ$ rotations per base pose;
- 72/72 successful headless dynamic rollouts;
- 60 non-reference transformed comparisons;
- primary normalized-shape correlation (`Fz`, `Tx`, `Ty`, `tip_z`) mean 0.945
  and p05 0.883;
- raw selected-channel mean NRMSE 0.382, with the largest mismatch in lateral
  wrench channels.

This supports provisional $C_6$ augmentation only for the normalized
direction/shape branch. It does not support treating rotated raw wrench
amplitudes as independent radial-distance or real-robot evidence. P0b and P0d
subsequently added nuisance repeats and paired physical-sector anchors so this
nominal symmetry residual could be checked against deployment-like variation.
Evidence is stored in
`outputs/current/datasets/p0a_m8_symmetry/`.

### Current P0b result

The bounded-nuisance stage is also complete:

- 3 states x 5 signed-distance levels x 8 angles in the fundamental sector x
  3 Sobol nuisance repeats = 360/360 successful rollouts;
- varied sliding friction in [0.55, 0.95], contact time constant in
  [0.040, 0.070] s, down-force in [2.5, 3.5] N, and yaw in
  [-1.5, +1.5] deg;
- 48D shape feature: state accuracy 0.911, angular error median 0 deg / p90
  7.5 deg, radial MAE 0.209 mm, and 23/120 action-conflict nearest neighbors;
- 64D shape plus dimensionless absolute feature: state accuracy 0.936,
  angular error median 0 deg / p90 7.5 deg, radial MAE 0.157 mm, and 12/120
  action-conflict nearest neighbors;
- finite-difference relative Fisher diagnostics cover 54 interior grid sites;
  shape+absolute whitened $\sigma_{min}$ median 43.37 / p05 25.43.

The direction result is strong enough to proceed with `search_state + q(phi)`
as the first learner. The remaining collisions are mainly same-direction radial
or state-boundary pairs (`supported_search/double_overhang` and
`double_overhang/outer_partial`). Therefore radial distance remains an
auxiliary quantile, never a direct center command. The next efficient data step
is a targeted independent repeat set on the 18 unique collision groups, not a
uniform expansion of all 120 sites.

These Fisher values are relative to the pilot feature whitening and nuisance
model. They are comparative diagnostics, not physical information units or a
formal global identifiability proof. Evidence is in
`outputs/current/datasets/p0b_m8_observability/`.

### Current P0c result

P0c selected the 18 unique groups involved in the 12 P0b shape+absolute
action-conflict pairs and collected five new independent Sobol nuisance repeats
per group:

- 90/90 dynamic rollouts completed without hard stop;
- P0b's 120 centroids and nuisance scale were frozen before evaluating P0c;
- shape-only direction acceptable rate: 0.989;
- shape+absolute state accuracy: 0.933;
- shape+absolute direction acceptable rate: 0.989;
- shape+absolute radial MAE: 0.083 mm;
- shape+absolute state+direction acceptable rate: 0.922;
- shape+absolute full-action proxy rate: 0.911.

Only one of 90 trials exceeds the 7.5 deg direction tolerance; the remaining
failures are predominantly same-direction aliases across radial/contact-state
boundaries. This independently confirms the direction-first decision. It does
not justify commanding the predicted radial distance directly.

No further uniform P0 expansion was justified. The next check therefore used
paired physical-sector anchors rather than increasing every grid cell. Evidence
is stored in `outputs/current/datasets/p0c_m8_collision_followup/`.

### Current P0d result: symmetry is a prior, not free data

P0d samples three representative radial levels and three local angles in each
flat-support state. For every local pose and Sobol nuisance draw, the same
friction, contact time constant, down-force, and yaw are reused across all six
physical sectors:

- 3 states x 3 radial levels x 3 local angles x 2 nuisance draws x 6 sectors =
  324/324 successful rollouts;
- 54 complete paired six-sector orbits;
- normalized primary-shape correlation mean 0.930 / p05 0.834;
- raw selected-channel NRMSE mean 0.412 / p95 0.522.

The paired design separates sector residual from nuisance residual. It shows
that $C_6$ augmentation is useful as an inductive bias, but not accurate enough
to replace physically rolled-out sector anchors. This result is especially
important for real data, where cables, gravity compensation, manufacturing
eccentricity, and sensor cross-axis terms will break symmetry further. Evidence
is in `outputs/current/datasets/p0d_m8_sector_anchors/`.

### Initial P1 result: direction-first posterior

The first trained baseline uses 360 phase bins with four normalized channels
(`Fz`, `Tx`, `Ty`, `tip_z`) and 16 dimensionless context statistics. Data is
split by physical base pose before augmentation. Training combines:

- 1296 P0b examples after physically consistent $C_6$ augmentation;
- 180 real MuJoCo P0d sector-anchor examples;
- 144 held-out P0b/P0d examples for early stopping, temperature scaling, APS,
  and confidence-gate selection.

Three CNN initializations use the same frozen split. Their calibrated class
probabilities are averaged. Independent direction results are:

| split | within 7.5 deg | APS coverage / mean set size | >90 deg top-1 error |
|---|---:|---:|---:|
| P0b grouped test | 1.000 | 0.958 / 3.21 | 0 |
| P0d physical-sector grouped test | 0.972 | 1.000 / 3.01 | 0 |
| P0c hard nuisance set | 1.000 | 1.000 / 2.63 | 0 |
| P0a nominal full-angle diagnostic | 0.958 | 0.986 / 3.51 | 0 |

Single-seed variation is non-negligible, which is why the ensemble is reported
instead of selecting the best initialization. This initial P1 stage kept the
state head diagnostic because P0c showed state-boundary ambiguity. The frozen
P1v5 calibration and the later action-interface audit now allow one narrower
use: state probabilities may multiply an approximate-CAD **soft radial
likelihood with a nonzero floor**. They still cannot become a hard state gate,
direct radial command, capture trigger, or terminal verifier. Radial action
remains bounded by posterior geometry and outward coverage.

This is simulation calibration, not a calibrated real posterior. The current
confidence threshold does not reject the independent simulation sets, so it is
not evidence of OOD detection. Reports and checkpoints are in
`outputs/current/models/p1_m8_direction_models/`.

### Current P2 result: correct direction is not sufficient without distance support

P2 connects the frozen three-seed ensemble to a new dynamic rollout. Each
learned trial executes the full 15 deg conical scan, resamples the measured
`Fz/Tx/Ty/tip_z` trace, infers $q(\phi)$, fuses it with an isotropic synthetic
visual prior, and passes posterior samples to the coverage-bounded planner.
The ordinary outward baseline and true-center oracle use the same initial pose,
3 N down-force, 2.25 mm/s XY speed, 0.40 mm/rev pitch, and 30 s budget. The
learned probe time and contact work are included.

The pilot uses 18 poses from the frozen P0d grouped test set and three methods,
for 54 dynamic rollouts:

| method | success | penalized-time median | contact-path median | wear-proxy median |
|---|---:|---:|---:|---:|
| learned posterior + local outward | 7/18 | 30.00 s | 30.51 mm | 55.27 mJ |
| continuous ordinary outward | 6/18 | 30.00 s | 56.08 mm | 91.25 mJ |
| true-center oracle | 18/18 | 2.09 s | 4.71 mm | 7.85 mJ |

All 18 learned directions are within 7.5 deg, yet overall learned success is
only 38.9%. Stratification explains the apparent contradiction:

- inside the synthetic 2 mm visual prior's 95% radius (4.90 mm): learned 7/9,
  ordinary outward 6/9; learned penalized-time median 15.51 s versus 17.11 s;
- outside that visual support: learned 0/9 and ordinary outward 0/9, while the
  oracle is 9/9.

Thus conical learning solves a direction bottleneck, not arbitrary radial
localization. If the visual/CAD distance prior excludes the center, the planner
correctly cannot infer how far to transport from direction alone. The next
scientific step is to measure the real visual error covariance and sample from
that deployment distribution, not to add an unconstrained learned distance
head. Five poses currently have both learned and baseline success, but this
small nominal-dynamics pilot and its one-sample success difference do not
support a general speed, success-rate, or wear-reduction claim.

This historical P2 search-to-capture verifier used privileged MuJoCo contact
tags. It has been superseded by the current depth-trend + bounded-wrench online
handoff; geometry/tags are now offline scorers only. A centered low-wrench
below-rim insertion remains a valid alternative to wall-guided capture, which
avoids falsely requiring lateral wall force near the center. Historical
evidence is in
`outputs/current/evaluations/p2_m8_learned_search_benchmark/`.

### P2b result: remove the 30 s cutoff and test the declared visual domain

P2b treats 4.895 mm as a deployment acceptance radius rather than mixing
inside/outside samples. Five existing P0d radii at or below this bound and all
six physically rolled-out sectors give 30 initial poses. Learned, continuous
outward, and true-center oracle are paired at every pose, for 90/90 rollouts.

The ordinary spiral centerline covers to

$$
R_{\mathrm{cov}}=R_{\mathrm{vis}}-r_{\mathrm{eff}}=4.645\ \mathrm{mm},
$$

with 0.40 mm/rev pitch. Its exact generated path is 169.65 mm and its nominal
coverage time at 2.25 mm/s is 75.40 s. The 175.45 s program guard is derived
from 2.25 times that coverage time plus probe overhead; it is not the success
definition. Runs terminate on capture, path exhaustion, hard stop, or this
non-performance guard.

| method | success | successful-time median | all-trial path median | all-trial work median |
|---|---:|---:|---:|---:|
| learned direction posterior | 21/30 | 15.08 s | 31.29 mm | 59.77 mJ |
| continuous ordinary outward | 24/30 | 20.48 s | 50.37 mm | 86.02 mJ |
| corrected true-center oracle | 30/30 | 1.23 s | 2.68 mm | 4.31 mJ |

The overall medians above mix endpoints, so the valid efficiency comparison is
the 21-pose subset where learned and baseline both reach the same gate:

- learned time median 15.08 s versus baseline 17.11 s; paired difference
  median -3.66 s, with learned faster on 13/21;
- learned path median 19.76 mm versus 35.24 mm; learned is shorter on 21/21;
- learned work median 38.56 mJ versus 61.05 mJ; learned is lower on 20/21;
- learned peak force/torque are not lower, so this is not a lower-wrench claim.

The radius split is the more important result. Both methods succeed throughout
1.518--2.171 mm. Learned is 5/6 and 4/6 at 2.823 and 3.404 mm while baseline is
6/6. At 4.875 mm both are 0/6: baseline reaches six hard stops; learned has five
safe radius exhaustions and one program-guard exhaustion. All learned angular
errors remain within 7.5 deg (median and p90 are 0 deg), so adding search time
or another direction CNN cannot solve this boundary.

The current isotropic visual prior factorizes into radius and angle. Reweighting
only by $q(\phi)$ therefore sharpens direction but does not create radial
information. P2c now implements the previously missing narrow interface without
retraining P1v3: fuse the already predicted support-state probabilities with
soft CAD-derived radial annuli, then recompute conservative transport and
residual coverage. This is a new same-start interface demonstration and does not
revise the frozen P2b distribution result. P2b evidence remains in
`outputs/current/evaluations/p2b_m8_in_domain_coverage/`; the P2c same-case artifact is
`outputs/current/reports/m8_outer_partial_cad_fused_full_closed_loop_comparison.json`.

For approximate concentric M8 geometry,

$$
r_{full}=R_{head}-R_{bit}=3.036\ \mathrm{mm},\qquad
r_{overlap}=a_{socket}+R_{bit}=6.714\ \mathrm{mm}.
$$

The implementation does not hard-assign a radius from the top state. It forms

$$
p(\delta\mid y,v,G)\propto
p_v(\delta)q(\phi(\delta)\mid y)
\sum_s p(s\mid y)\ell_s(\|\delta\|;G),
$$

where $\ell_s$ uses logistic CAD transitions with 0.35 mm width and a 0.02
floor. This preserves manufacturing/model tolerance and non-top state mass. It
uses no true-center label, MuJoCo contact point, or learned radial head.

On the existing `[4,0]` mm case, P1v3 reports `double_overhang` with probability
approximately one. The fused radial q10/q50/q95 is 2.51/3.59/4.69 mm. The
planner's unchanged 1.50 mm cap, not the posterior lower bound, now limits the
transport; residual outward radius falls from 3.434 to 2.945 mm. CAD-bottom time
falls from 44.67 to 31.62 s relative to direction-only and remains above the
20.78 s assumed-radial illustration. Peak force rises to 22.43 N, so this is not
a lower-wrench result and requires a distribution/transition-stress audit before
hardware transport.

### Trace-driven nominal FSM stabilization

The physically cleaner nominal boundary is not the Gaussian 4.895 mm radius.
For full support on the M8 head,

$$
R_{\mathrm{nominal}}
=R_{\mathrm{head}}-R_{\mathrm{bit}}
=6.5-\frac{6}{\sqrt{3}}
=3.036\ \mathrm{mm}.
$$

The 1.518, 2.171, and 2.823 mm strata are nominal; 3.404 mm is transition and
4.875 mm is stress. The only nominal learned hard stop was instrumented at the
physics rate with FSM stage, nominal/actual pose, raw and filtered 6D wrench,
candidate decisions, and offline contact tags. The trace showed:

```text
first candidate freeze: z -0.096 -> -0.988 mm
50 ms chamfer cue not continuously true
FSM resumes planar spiral below the -0.30 mm CAD depth band
three candidate attempts -> 74.10 mNm hard stop
```

The minimal fix does not change the network or force threshold. A sensor-only
depth commit latches once the tip crosses the CAD below-rim band under hard
wrench limits. Planar search is then locked out and the FSM enters guarded
descent. On the same smoke case, the result becomes one-attempt centered
insertion with 19.89 mNm peak torque and no hard stop.

The post-fix nominal development regression contains 18 poses x learned and
baseline = 36 rollouts:

| metric | learned | baseline |
|---|---:|---:|
| capture / hard stop | 18/18 / 0 | 18/18 / 0 |
| capture-time median | 12.93 s | 17.11 s |
| contact-path median | 12.18 mm | 35.19 mm |
| work-proxy median | 28.03 mJ | 60.87 mJ |
| force / torque p90 | 8.87 N / 24.50 mNm | 7.92 N / 21.75 mNm |

Paired results are intentionally narrower: learned is faster on 10/18, has a
shorter path on 18/18, and lower work proxy on 17/18. Peak wrench remains
higher. These are reused development poses after a trace-driven FSM change,
not an independent test. The next test must use unseen local angles and applied
friction/stiffness/yaw/down-force nuisance. Evidence is in
`outputs/current/diagnostics/p2b_nominal_failure_diagnostic/` and
`outputs/current/evaluations/p2b_nominal_fsm_acceptance/`.

### P1 training distribution

The pilots show that immediately copying the reference paper's 5000-rollout
count is unnecessary. Keep the 1476-example P1 training baseline as the first
learning-curve point, then add samples only when grouped validation, physical
sector anchors, or real calibration expose a gap. Any later expansion should
follow this deployment-weighted distribution:

| Stratum | Initial share | Purpose |
|---|---:|---|
| measured visual-error deployment distribution | 60% | optimize the cases that will actually occur |
| CAD state boundaries and mixed-support bands | 25% | resolve action-changing ambiguities |
| measured-range nuisance/domain randomization | 10% | cover sim-to-real-sensitive contact parameters |
| active-learning collision and uncertainty points | 5% | spend new samples where the current model is weak |

Use Sobol or Latin-hypercube samples for continuous nuisance variables. Do not
randomize parameters over arbitrary huge ranges; use measured or defensible
engineering intervals and keep the sampled values in metadata.

## 5. Symmetry Without Data Leakage

The Allen socket has a $C_6$ rotational symmetry. A $60^\circ$ rotation can
provide an exact simulation augmentation only when every coupled quantity is
transformed consistently:

$$
\delta' = R_\psi\delta,
\quad
[F_x',F_y']^T=R_\psi[F_x,F_y]^T,
\quad
[\tau_x',\tau_y']^T=R_\psi[\tau_x,\tau_y]^T,
\quad
\theta'=\theta+\psi.
$$

The CAD frame, yaw label, direction label, contact points, and visual
covariance must rotate by the same $\psi=k\pi/3$. $F_z$, $\tau_z$, and relative
$z$ remain unchanged under the ideal in-plane rotation.

Important boundaries:

- symmetry augmentation increases effective training coverage but is not an
  independent test sample;
- split base physical rollouts into train/validation/test before augmentation;
- arbitrary-angle rotation is not an exact symmetry of a hex socket;
- cable routing, gravity compensation error, robot anisotropy, sensor cross-
  axis coupling, and asymmetric head/socket manufacturing break ideal $C_6$ on
  hardware;
- keep an unaugmented full-angle validation set to measure those effects.

Use circular phase shifts and $C_6$ transforms as data augmentation and as an
equivariance-consistency loss, not as a substitute for physical coverage.

## 6. Saved Data And Preprocessing Contract

Every rollout must save the same schema in simulation and hardware:

```text
trial_id, geometry_id, size_id, base_pose_id, repeat_id
timestamps, commanded phase, commanded pose, measured pose
raw 6D F/T, bias/gravity-corrected 6D F/T
controller state, down-force target, admittance parameters
CAD signed distances, visual-prior mean/covariance
offline-only true offset, contact labels and nuisance parameters
termination reason, hard-limit flags
```

Preprocessing:

1. synchronize command, pose, and F/T timestamps;
2. transform wrench to the same tool/CAD frame;
3. subtract F/T bias and gravity/tool-load contribution;
4. segment valid contact and resample by probe phase to 256 or 512 bins;
5. use circular interpolation/padding;
6. save both raw-scale and dimensionless channels.

Recommended dimensionless quantities for scale transfer are

$$
\tilde\delta=\delta/a,
\quad \tilde z=z/a,
\quad \tilde F=F/F_d,
\quad \tilde\tau=\tau/(F_da),
$$

where $a$ is a CAD bit/socket characteristic radius. Retain one shape-
normalized branch for direction and one absolute/context branch for any radial
information. Per-trial min-max normalization alone can remove amplitude cues
that may be useful for distance and sim-to-real diagnostics.

## 7. Model And Evaluation

Run a staged model ladder:

1. Fourier/phase features with logistic regression, KNN, or a small MLP;
2. paper-style polar-image CNN reproduction;
3. compact 1D circular CNN with circular padding and scalar CAD/context input.

Recommended first outputs:

- three-state logits;
- circular angular-bin distribution or a multimodal circular density;
- calibrated angular set;
- OOD score.

Add radial quantiles only after P0. Report state accuracy, angular error,
angular-set coverage and size, calibration error, and OOD detection. The task-
level comparison must also include probe time, transport time, time-to-safe-
progress, success/abort rate, contact path, accumulated contact-work proxy,
and peak wrench.

The required baseline ladder is:

```text
source-style outward spiral
direction-only posterior + conservative transport + outward spiral
direction-and-radius posterior, only if P0 supports it
oracle center/direction upper bound
```

## 8. Real-Robot Fine-Tuning

Use a calibration fixture or a successfully inserted reference pose to define
the socket center. Real labels must come from known commanded offsets, not from
the learner's own prediction.

An efficient first M8 anchor set is approximately:

$$
6\ \text{directions}\times3\ \text{radial bands}
\times3\ \text{repeats}\times2\ \text{down-force levels}
=108\ \text{scans}.
$$

Then add 30--60 active-selected hard cases if calibration or coverage fails.
Keep independent real validation points that are never used for fine-tuning.

Fine-tuning order:

1. fit sensor bias, tool-frame transform, timing, and dimensionless scales;
2. calibrate the direction posterior without changing the backbone;
3. tune a small adapter or output head while mixing simulation replay;
4. enable the model in shadow mode;
5. enable only bounded transport after real angular coverage is valid.

Do not fine-tune the complete network first. Most sim-to-real discrepancy is
expected in wrench scale, delay, contact stiffness, and controller response;
the geometric phase pattern should be preserved as much as possible.

## 9. M8 To M10/M12

M8, M10, and M12 share the Allen topology and have similar normalized
clearance/chamfer ratios, but they are not exact scaled copies. The first shared
model should therefore be conditioned on dimensionless CAD parameters rather
than hiding size from the network.

Validation ladder:

1. train and close the M8 loop;
2. run M10/M12 simulation without fine-tuning;
3. perform leave-one-size-out tests, for example train M8+M10 and test M12;
4. compare a shared conditional model against per-size heads;
5. collect at least 36--60 independent real calibration/test scans per new
   size before claiming hardware scale transfer;
6. fine-tune only normalization/adapter/head if zero-shot calibration fails.

Symmetry augmentation and dimensionless scaling reduce required data. They do
not remove the need to test manufacturing clearance, chamfer, head/socket
concentricity, sensor lever arm, and controller behavior on each real size.

## 10. Compute Plan

The current machine exposes six NVIDIA RTX A5000 24 GB GPUs and CUDA-enabled
PyTorch. Use one GPU for the small CNN and the remaining devices for independent
seeds, folds, or ensembles. Generate the first datasets with headless C MuJoCo
CPU multiprocessing because the current rollout implementation is already
validated there.

MJX/JAX is not currently installed. Port to MJX-Warp only if rollout generation
is the measured bottleneck, and accept GPU-generated data only after matched
C-MuJoCo/MJX fingerprints agree on phase, wrench distribution, contact events,
and termination outcomes.

## 11. P0 Gate Status

The P0 gate is complete enough to begin deployment-weighted sampling:

- direction is observable on the tested bounded-nuisance simulation domain;
- radius remains less reliable across state boundaries and is not a direct
  learned command;
- fingerprint collisions have been localized and independently repeated;
- $C_6$ is useful as augmentation but requires physical sector anchors;
- the posterior can support bounded transport, while outward search remains
  the geometric fallback.

This is a go decision for a direction-first learner, not for unconstrained
center regression. The next dataset is an independent audit, not another broad
training expansion.

## 12. Reference-Paper Transfer Audit And Frozen Next Batch

The reference paper `reference/锥形探索.pdf` uses the following sampling and
learning pipeline:

- after flat contact removes roll, pitch, and z uncertainty, sample
  $x,y\in[-20,20]$ mm and yaw in $[-3^\circ,3^\circ]$;
- execute the same $15^\circ$ tilt-then-rotate scan for $2\pi$ under constant
  down-force and admittance control;
- record 12D pose and wrench for $N=2000$ time steps;
- normalize each channel, smooth with a 20-sample moving average, and form a
  $12\times200\times200$ polar binary pattern;
- select $X_z,M_x,M_y$, downsample to $3\times20\times20$, and train a small
  CNN on 5000 simulation trials with a random 80/20 split;
- add 5% noise to admittance parameters and use simulation pose as the
  self-supervised label;
- predict only a discrete error direction, not the error magnitude.

The transferable design principles are the fixed action template, phase-based
sequence representation, simulation self-supervision, wrench transformation to
a common peg/tool reference point, and selective use of informative channels.
The exact sample count, uniform square distribution, image rasterization, and
random sample-level split are not transferred because they do not match the
M8 deployment distribution and can hide neighboring-pose leakage.

For the current M8 geometry, the next batch is frozen as an independent audit:

$$
(3\ \text{nominal radial bands}+2\ \text{transition bands})
\times12\ \text{physical directions}
\times3\ \text{Sobol nuisance repeats}
=180\ \text{scans}.
$$

The nominal bands lie at or below the CAD full-support radius
$R_{nominal}=3.036$ mm. The transition bands remain inside the declared
4.895 mm visual 95% bound. In the current sampler, `outer_partial` begins near

$$
R_{outer\_partial}
=a_{socket,top}+R_{bit}
=3.25+\frac{6}{\sqrt{3}}
=6.714\ \text{mm},
$$

so it is outside the declared deployment bound and should be retained only as
a small stress/OOD set, not one third of the primary training distribution.

The 180 scans must not enter training. They use unseen local angles and new
friction, contact-time-constant, down-force, and yaw draws. Report angular
error, APS coverage/set size, calibration error, and physical-sector residual.
Only failed regions return to the training pool through ensemble-disagreement
or boundary-focused active sampling.

The CNN data contract remains:

- raw archive: synchronized 6D F/T, commanded/measured pose, phase, controller
  state, CAD distances, visual prior, nuisance values, action id, and terminal
  flags;
- primary input: 360 phase bins of $F_z,\tau_x,\tau_y,z$ plus 16D
  dimensionless context;
- primary target: 48-bin $q(\phi)$ at $7.5^\circ$ resolution plus a calibrated
  angular set;
- diagnostic target: support-state logits;
- forbidden direct targets: center distance, control command, chamfer class,
  or insertion completion.

Rim/chamfer, inner-wall, and inserted trajectories are collected separately as
short force-ramp/descent calibration data for white-box gates. They are not
mixed into the direction-CNN training domain.

## 13. P1v2 Frozen Independent Audit Result

The frozen batch completed 180/180 dynamic MuJoCo scans with zero hard stop in
approximately 68 seconds using 56 CPU workers. The pre-existing three-seed P1
ensemble, temperatures, APS quantile, and confidence gate were then evaluated
once without fitting or calibration on these data.

| Metric | Result |
|---|---:|
| angular error median / p90 | $3.75^\circ/11.25^\circ$ |
| within $7.5^\circ$ | 0.761 |
| APS coverage / mean set size | 0.972 / 3.68 bins |
| top-1 error above $90^\circ$ | 6/180 |
| catastrophic errors allowed by current transport gate | 5/6 |
| minimum APS size among catastrophic errors | 9 bins |
| current confidence-gate accept rate | 0.994 |
| diagnostic state accuracy | 0.983 |
| six-sector within-rate range | 0.233 |

The result changes the claim boundary. The phase fingerprint still carries a
useful set-valued direction signal, but maximum softmax confidence is not a
safe transport gate under unseen continuous angles and nuisance combinations.
The six catastrophic errors concentrate in two paired nuisance groups at the
innermost nominal radius and outermost transition radius. The $15^\circ$ local
angle has 0.589 within-$7.5^\circ$ rate, versus 0.933 at $45^\circ$, exposing
the limitation of training only on hard 48-bin direction centers.

P1v2 remains excluded from fitting, temperature scaling, and gate selection.
Its broad failure regimes informed a later targeted design, so it is a viewed
development audit rather than final independent evidence. The next model
iteration should:

1. use circular soft labels or a circular density loss for continuous angles;
2. add angle jitter and independent nuisance neighborhoods around the
   innermost nominal and outermost transition regimes;
3. gate transport using APS width/circular concentration and ensemble
   disagreement in addition to maximum probability;
4. select all thresholds on existing validation or a new calibration split;
5. evaluate once on a new untouched radial/angle/nuisance audit.

The project still does not need a learned radial-distance head. Low
concentration or a wide angular set must set $d_{move}=0$ and fall back to the
ordinary guarded outward spiral.

Artifacts are in `outputs/current/evaluations/p1v2_m8_independent_direction_audit/`:

- `sampling_contract.json`;
- `progress.json` and `manifest.json`;
- 180 raw/phase-aligned trial NPZ files;
- `independent_direction_report.json`;
- `predictions.csv`;
- `summary.md`.

## 14. P1v3 Targeted Training And P1v5 Final Audit

P1v3 adds 288 physical MuJoCo scans without copying P1v2 trials or nuisance
values:

- three inner nominal and three outer transition radii;
- four off-grid local angles across six physical sectors;
- two paired new Sobol nuisance repeats;
- 240 scans for training and 48 for grouped calibration, split by
  `base_pose_id` before nuisance repeats.

Three CNN seeds were trained in parallel on GPUs 0--2 with a circular Gaussian
soft target of $sigma=7.5^\circ$. The controller consumes the circular mean of
the predicted direction density rather than the quantized top-1 bin. A
structured gate was selected only on P0b/P0d validation plus P1v3 grouped
calibration:

$$
|\Phi_{APS}|\le3,
\qquad
\Delta\phi_{ensemble}\le7.5^\circ.
$$

Maximum softmax probability and concentration did not add a useful threshold
on the calibration split and are not load-bearing. Low-confidence behavior is
represented by a wide APS set or seed disagreement, which sets
$d_{move}=0$ and preserves outward search.

P1v4 was used to diagnose the benefit of circular-mean decoding and is therefore
development evidence. After freezing that interface, P1v5 generated a new
180-scan audit with unseen radii, 7/41 degree local angles, and Sobol seed
20260716. P1v5 is the current untouched direction result:

| Metric | P1v5 result |
|---|---:|
| scan completion / hard stop | 180/180 / 0 |
| circular-mean error median / p90 | $1.30^\circ/4.05^\circ$ |
| within $7.5^\circ$ | 0.989 |
| APS coverage / mean set size | 0.994 / 2.63 bins |
| error above $90^\circ$ | 0/180 |
| structured-gate accept rate | 179/180 = 0.994 |
| accepted samples within $7.5^\circ$ | 0.994 |
| six-sector within-rate range | 0.033 |
| diagnostic state accuracy | 0.989 |

The largest error is $38.1^\circ$ and is the only rejected sample; its APS set
has four bins, exceeding the frozen maximum of three. One accepted sample has
$8.24^\circ$ error, so the gate is not perfect. Maximum-class ECE remains
0.304 and must not be interpreted as a calibrated real-world success
probability. The valid claim is narrower: on this untouched simulation audit,
the set-valued direction interface rejects the largest error and provides a
useful bounded transport direction on 179/180 samples.

This does not establish hardware calibration, radial localization, terminal
verification, or superiority over outward search. The next scientific gate is
real F/T frame/bias/delay calibration and a small shadow-mode anchor set.

### Historical fair-action direct/retry correction

The earlier scripted direct-retry video is not an active baseline. It injected
`+22.5/-22.5/0 deg` errors and used the true initial offset magnitude to choose
travel, so it is retained only under `outputs/current/diagnostics/` as a recovery
illustration.

The historical 36-pose Baseline B consumed the same conical F/T trace and the same frozen
P1v3 circular-mean direction as proposed. It receives no CAD radial likelihood
and no true distance; it moves along that ray up to the visual bound and stops on
the common capture gate. On the selected `[4,0]` mm case, both methods command
179.59 deg, but direct reaches CAD bottom in 11.22 s with 6.25 mm XY path and
4.72 N peak force, versus proposed 31.62 s, 45.89 mm, and 22.43 N. Therefore this
case is evidence against a blanket proposed-superiority claim. Proposed residual
coverage must earn its value on frozen angular-error, multimodal, and boundary
distributions where a direct ray can miss the capture basin.

### Historical frozen downstream-policy audit

The downstream controller was subsequently audited independently of training.
The old 22.43 N peak was a one-step handoff transient, and a second nuisance
pilot exposed an unnecessary fixed-XY descent confirmation after a persistent
safe-progress cue. Both transitions now use continuous force ownership; the FSM
stops planar search on persistent progress and defers completion to upright
insertion and the depth/wrench verifier.

The frozen Proposed policy removes the arbitrary 1.50 mm transport cap, retains
the conservative projected lower quantile, covers 99% of residual posterior
radius locally, and falls back to full visual-support outward coverage only if
the local set is exhausted. A new 36-pose Sobol audit with shared bounded contact
nuisances produced:

| policy | capture | hard stop | time median | path median | work median |
|---|---:|---:|---:|---:|---:|
| Proposed | 34/36 | 2 | 16.00 s | 21.72 mm | 45.72 mJ |
| full outward | 31/36 | 5 | 44.32 s | 82.85 mm | 151.37 mJ |
| shared-direction direct/retry | 34/36 | 0 | 7.46 s | 5.52 mm | 13.50 mJ |

The evaluator requires the bit center to enter the CAD socket capture region,
so an outer-head chamfer drop is not counted as capture. Proposed used global
fallback once. Proposed and direct/retry each had two exclusive successes
($p=1.0$); direct/retry remained faster, shorter, and lower-work on all 32 common
successes. Against full outward, Proposed was faster in 80.0%, shorter in 96.7%,
and lower-work in 93.3% of 30 common successes, but had no peak-force advantage.

## 15. Historical Pre-Hardware Wrench-Observation Audit

The clean/contact-nuisance audit does not exercise the observation chain that
will dominate the first sim-to-real gap. A separate frozen audit therefore
applies a stateful six-axis observation model after MuJoCo contact physics and
before admittance, fingerprint preprocessing, CNN inference, and sensor gates:

$$
\tilde w_k=Cw(t_k-\tau_k)+b_0+b_k^{rw}+\epsilon_k^{colored}.
$$

The frozen engineering prior uses per-episode residual bias, gain and same-unit
cross-axis coupling, correlated noise with $\rho=0.92$, random-walk drift, and
2.0--7.87 ms delay. The force bias standard deviations are 0.06/0.06/0.08 N;
the torque values are 0.30/0.30/0.20 mN m. These are deliberately modest
pre-hardware values, not measurements from the future sensor. They must be
replaced by bias/gravity residuals, covariance or Allan statistics, frame error,
and latency measured on the real stack.

The 36 new Sobol poses remain inside the visual 95% radius and share position,
friction, contact time constant, down-force, and the first noisy fingerprint
between Proposed and Baseline B. The frozen P1v3 model is not retrained on this
noise, so this is a robustness diagnostic and data-design input:

| method | capture | hard stop | success-time median | contact path median | work median | force p90 |
|---|---:|---:|---:|---:|---:|---:|
| Proposed posterior + residual coverage | 36/36 | 0 | 14.06 s | 17.72 mm | 34.01 mJ | 9.48 N |
| Baseline A full outward | 35/36 | 1 | 42.84 s | 84.09 mm | 136.87 mJ | 9.44 N |
| Baseline B shared direction + direct/retry | 29/36 | 0 | 7.51 s | 6.02 mm | 14.19 mJ | 7.53 N |

Proposed/B paired counts are 29 both-success, 7 Proposed-only, and no B-only or
joint failures; McNemar $p=0.015625$. B remains the fastest and lowest-work
method whenever its ray captures. The observed success-set inclusion reflected
the then-current policy, in which Proposed retained a physically reset,
independently budgeted full-outward fallback after its posterior shortcut. That
budget policy is superseded by the strict 130 s zero-extension contract. It is a finite simulation result rather
than a hardware or mathematical guarantee.

The selected video is the first lexicographic audit trial satisfying the frozen
criterion "Proposed and Baseline B succeed, but B needs a retry." In
`sobol_002`, the shared circular-mean command has 18.98 deg offline angular error
and APS size 8. The legacy gate rejects transport, but the continuous planner
uses 0.991 mm transport and a 2.599 mm residual spiral; Baseline B attempts one
and two exhaust their rays, then the third captures after two physical resets.
The video is a conditional recovery illustration, not deployment-weighted evidence.

The Baseline-B video now renders the complete recovery sequence rather than
attempt one only: failed ray, low-force release, vertical retract, 0.5 mm
shifted re-initialization, guarded retouchdown, full conical re-probe, and the
next ray, repeated through attempt three. The two rendered resets take 7.47 s.
The frozen 36-pose audit still excludes reset travel/time, so its B timing
remains a favorable lower bound.

### Continuous replacement for the binary transport gate

For a known offline radius $r$ and angular error $\Delta\phi$, transport along
the estimated direction gives

$$
r'(d)^2=r^2+d^2-2rd\cos\Delta\phi.
$$

For `sobol_002`, $r=3.055$ mm and the circular-mean error is 18.98 deg. The
offline optimum is $d^*=2.889$ mm and leaves 0.994 mm residual error. Even a
0.5/1.0 mm move would reduce the true error to 2.587/2.134 mm. Therefore the
legacy rule `APS > 3 -> d_move = 0` is safe but over-conservative in this case.

This does not justify using the offline true error or commanding the optimum.
The implemented online replacement uses the already available joint center posterior:

$$
s_i=(c_i-p)^Tu,
\qquad
d_{move}=\operatorname{clip}(Q_\alpha(s_i),0,d_{max}),
$$

and recomputes residual local coverage after transport. Direction concentration
and ensemble span continuously select a lower quantile between Q10 and Q25;
posterior mass behind the tool discounts this quality and can still collapse
transport to zero. On `sobol_002`, the online values are quality 0.898, reverse
mass 0.001, $d_{move}=0.991$ mm, and residual radius 2.599 mm. The 36-pose
rerun reaches 36/36 with four global fallbacks, but its noise model remains a
pre-hardware engineering prior.

Artifacts:

- `outputs/current/evaluations/m8_pre_hardware_noise_audit/`;
- `outputs/current/diagnostics/historical_comparisons/robustness_noise/`;
- `src/wrench_observation.py`;
- `src/render_m8_noisy_failure_comparison.py`.

The next efficient sim-to-real step is not to widen randomization arbitrarily.
Collect a small real calibration set, fit the observation distribution, then
fine-tune normalization/temperature or a small adapter/head while retaining
simulation replay and an untouched real test set.

## 16. Historical 2026-07-12 Interface Alignment

The current interface supersedes the older "state is diagnostic only" and
"online privileged capture evaluator" descriptions above:

- P1v5 state accuracy/NLL/Brier/ECE are 0.989/0.0189/0.0105/0.0126;
- the state head is retained only as a soft action-relevant CAD radial
  likelihood, never as a discrete transition or distance target;
- the frozen action audit gives direction+visual versus state-CAD coverage
  0.994 versus 1.000 and post-transport true-residual medians 1.399 versus
  0.805 mm;
- the online search-to-insertion handoff uses depth history and measured
  wrench only; true center and MuJoCo contact tags are offline scorers;
- the then-active baseline ladder contained ordinary outward, shared-probe
  direct/retry, and a Chen-inspired maximum-entropy grid-contact policy in
  addition to Proposed; seeded random waypoints are appendix-only.

The 36-pose four-method audit under the unmeasured pre-hardware wrench prior
reports Proposed/A/B/entropy-C capture 34/30/28/22. Proposed uses four
physical-reset full-outward fallbacks. C uses a 0.25 mm visual/CAD belief grid,
a 0.75 mm action grid, guarded low-force touchdowns and maximum expected
entropy reduction; it has no conical learner and no spiral fallback. Its median
probe count is 7.5. On the 21 common successes C is often faster and uses less
contact path, but it has lower coverage. The supported claim is therefore that
one learned fingerprint plus residual coverage trades fewer active contacts for
higher capture coverage, not universal time/path dominance.

The wrist-F/T observation is explicitly signed: moment magnitude alone cannot
separate inner-socket and outer-head edge support. A 12-point nominal MuJoCo
cue audit found all four inner/outer pairs magnitude-ambiguous but opposite in
signed moment category, with zero hard stops. This is only a simulation cue
check; an iiwa/F/T likelihood still requires real calibration.

The minimum first real-data campaign is deliberately smaller than a copied
5000-trial grid: 12 low-force commissioning scans, 24 adaptation anchors, and
12 untouched random tests. Add at most 12 entropy/APS/action-boundary samples
only after a documented failure. Frozen M10/M12 S1 direction-interface tests
have since been run, while the S2 full-loop gates fail for both sizes. Therefore
the current claim is bounded direction-interface evidence plus failed full-loop
transfer—not successful cross-size engagement.
