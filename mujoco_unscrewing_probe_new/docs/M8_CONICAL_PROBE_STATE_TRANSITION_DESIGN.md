# M8 Conical Probe State-Transition Design

Updated: 2026-07-24

> **Current alignment (2026-07-15):** this is a chronological design record.
> Earlier planned dataset/classifier steps and privileged contact-tag
> visualizations are historical. The active evidence is the conditioned
> ambiguous-contact strict 24-pose loop with a 130 s zero-extension budget,
> sensor-only capture handoff and dual-channel CAD-bottom verifier. See
> `FULL_CLOSED_LOOP_SCIENTIFIC_AUDIT.md` and
> `PROPOSED_PLUS_ENTROPY_POLICY.md`.

## Purpose

This document records the current hypothesis for adding a small conical /
nutation contact probe before or during M8 Allen socket engagement.

The goal is not to claim that one probe recovers the true hole center. The goal
is narrower:

```text
Use a low-force phase-structured contact probe to turn an ambiguous vertical
touchdown cue into:

1. a macro contact state,
2. a calibrated circular direction distribution,
3. a finite-coverage angular set and confidence score,
4. a CAD-constrained feasible center region,
5. a posterior width that determines how much distance can be safely consumed,

then use those outputs to plan a bounded transport and a coverage-preserving
local outward spiral. Hard wrench and rate limits remain independent of the
learning output.
```

This is an active-sensing design. It must be validated before it becomes a
controller claim.

## 2026-07-24 Contact-Transition Closure

The conical probe is not assumed to remain on the top face for a complete
cycle.  In particular, a peg supported only by the inner chamfer can unload or
move downward when the favorable phase points into the aperture.  That event
is useful task progress, but the truncated trace is not a circular direction
fingerprint.

The online simulation policy therefore uses only measured wrench, measured
tip-Z trend, elapsed time, and the commanded normal-force target:

```text
CONTACT_SCAN
  |-- brief support reduction --> invalidate this fingerprint
  |                              (continue only while below stop dwell)
  |
  |-- sustained low support OR safe downward progress
          --> stop the remaining cone
          --> smoothly return upright from the actual last orientation
          --> physical capture requalification
                |-- confirmed --> shared compliant insertion funnel
                `-- rejected  --> direction-free global outward fallback
                                  or safe stop
```

No partial/incomplete cone is passed to a learned model, classical direction
estimator, posterior shortcut, or direct ray.  Capture requalification is a
fresh physical test and remains distinct from the low-force diagnostic cue.

The offline training-data policy is deliberately different because MuJoCo
offers privileged truth that hardware does not:

- sustained `contact_count == 0` for at least 6 ms is quarantined;
- full-chamfer-depth downward progress during the scan is quarantined;
- endpoint/early capture, incomplete scans, failures, and non-finite signals
  are quarantined;
- sustained low measured force with continuous MuJoCo contact and no downward
  progress is retained as a valid unloading fingerprint, while being tagged as
  a diagnostic.

Thus low force alone is neither an insertion certificate nor an automatic
training rejection.  Online safety uses observable evidence; offline hygiene
uses privileged contact truth only to prevent ambiguous supervision.

## Why Vertical Touchdown Is Not Enough

After visual localization, the bit descends approximately in the Z direction.
That first touchdown can produce a force signal, but the mapping is not
injective:

```text
same or similar F/T signal
<- outer partial support on screw-head edge
<- double-overhang / annular bridge support
<- supported search near inner socket rim
<- chamfer contact
<- bit already shallowly inside the socket
<- multipoint / frictional contact
```

Therefore:

- `Fz` alone can establish contact but cannot identify the true hole center.
- `tau_x, tau_y` can indicate an eccentric support center, but not whether the
  unsupported side is the internal hole or the outer air beyond the screw head.
- CAD geometry and the visual prior are required to reject impossible or unsafe
  interpretations.
- A short active probe can add directional information by making contact
  response vary with phase.

## Reference Logic

Two references motivate this design:

1. `reference/锥形探索.pdf`
   - Contact pose identification uses a tilt-then-rotate action.
   - The useful input is a sequence / pattern, not a single F/T sample.
   - This supports treating conical probing as a fingerprint generator.

2. `reference/漏斗效应.pdf`
   - Robust insertion comes from exploiting environmental constraints and
     compliant interaction, not from assuming perfect planned trajectories.
   - For this project, the full inclined-peg funnel theory is not the main first
     stage, because the Allen bit is assumed roughly parallel to the socket.
   - The useful part is later: once chamfer / wall contact is established, use
     compliance so the geometry can passively refine insertion rather than
     forcing a rigid path.

## Conical Probe Definition

Let the nominal tool axis be `e_z`. A small conical probe commands a slowly
rotating tool-axis direction:

$$
n(t)=R_z(\theta(t))R_x(\alpha)e_z,
\qquad \theta(t)=\Omega t
$$

The probe records phase-indexed signals:

$$
F_z(\theta),\quad F_x(\theta),\quad F_y(\theta),
\quad \tau_x(\theta),\quad \tau_y(\theta),\quad \tau_z(\theta),
\quad z(\theta)
$$

If the effective tool length from the wrist / compliance center to the contact
tip is $L_\mathrm{eff}$, the approximate lateral sweep caused by the tilt is:

$$
\rho \approx L_\mathrm{eff}\tan \alpha \approx L_\mathrm{eff}\alpha
$$

For M8, the current model has:

```text
r_cap_wall = 0.10 mm
r_cap_top  = 0.25 mm
```

Earlier notes proposed starting with a very small alpha. That is now superseded
for the main fingerprint line: to reduce variables, match the reference paper,
and improve real F/T signal-to-noise before fine-tuning, the nominal conical
fingerprint uses `alpha = 15 deg`.

For scale awareness, with `L_eff = 10 mm`:

| tilt | lateral sweep |
|---:|---:|
| 4.0 deg | 0.699 mm |
| 5.0 deg | 0.875 mm |
| 6.0 deg | 1.051 mm |
| 15.0 deg | 2.679 mm |

Current recommendation:

```text
nominal initial-contact fingerprint: alpha = 15 deg
phase sweep: 360 deg
normal force: low, impedance/admittance controlled, not hard pressing
motion requirement: virtual pivot / remote-center compensation near the tip
hard stop: force / torque / depth / slip cap
after probe: return roll/pitch to zero, then verify force and depth
```

This does not mean the probe is non-invasive. The command trajectory is closed
in roll/pitch, but contact with friction, chamfer, and compliance is not
guaranteed to be reversible. Therefore the 15 deg probe is a strong active
sensing action, not a zero-disturbance measurement.

## Macro Cases From The Current Top-View Figure

The practical cases are the top-view states 2-6 in
`outputs/current/media/figures/top_view_m8_physical_states_outside_to_inside.png`.

### Case 2: Outer Partial Support

Description:

```text
bit footprint partly on screw-head top, partly outside the screw-head support
region
```

Risk:

- The unsupported side may be outside air, not the internal socket.
- A naive rule may mistake outer-edge support for inner-rim support.

Expected conical fingerprint:

- phase-dependent support strength;
- eccentric torque;
- weak or no meaningful depth drop toward the true socket unless the phase also
  points toward the internal recess;
- strong dependence on CAD head boundary.

Recommended route:

```text
if inferred open side points outside screw-head boundary:
    inward recovery / re-detect, not insertion claim
else:
    low-confidence biased search
```

### Case 3: Double-Overhang / Annular Bridge

Description:

```text
outer side of the bit footprint is partly outside the screw-head top;
inner side is partly over the socket opening;
the remaining middle annular band carries the support
```

Risk:

- Both the outer side and the socket side can look like an open / weak-support
  direction in a conical fingerprint.
- This is the clearest case where a naive rule can mistake the outer edge for
  the inner hole, or the inner hole for the outer edge.
- Direction output should be treated as a weak prior unless it is compatible
  with CAD head/socket boundaries and phase consistency.

Expected conical fingerprint:

- two-sided or broadened unload pattern;
- pressure-center / torque phase may not point cleanly to one side;
- only the inner opening side should show useful `z_drop` or chamfer-like
  depth cue;
- outer-edge phases tend to show support loss without valid insertion depth.

Recommended route:

```text
if CAD compatibility says the open side is outside the screw head:
    recover / re-detect
elif inner-side drop cue is repeatable:
    weakly biased guarded spiral toward the inner side
else:
    small re-probe or ordinary guarded spiral
```

### Case 4: Supported Search

Description:

```text
bit footprint fully supported on screw-head top; socket may be nearby
```

Expected conical fingerprint:

- one or more phases may show lower lateral stiffness;
- a phase near the socket may show mild `Fz` unload or small `z_drop`;
- opposite phases may show stronger support / block.

Recommended route:

```text
use conical output as inward-direction prior
run biased guarded spiral
fall back to ordinary guarded spiral if confidence is low
```

### Case 5: Rim / Chamfer

Description:

```text
bit contacts rim or chamfer; geometry may already provide a mechanical lead-in
```

Expected conical fingerprint:

- stronger phase structure than case 3;
- a favorable phase may show depth drop, force unload, or torque transition;
- unfavorable phases show block / higher torque.

Recommended route:

```text
slow down
reduce spiral radius / pitch
use compliant insertion bias along the favorable phase
switch to verify if depth and force cues agree
```

This is likely the most useful regime for conical probing.

### Terminal Family: Inner Wall / Capture / Inserted

Description:

```text
bit is below the rim plane, either supported by an inner wall or already moving
inside the socket clearance toward the CAD bottom
```

Direct observable cues:

- `z_tip` is below the rim by a CAD-consistent amount;
- inner-wall capture has sustained `Fxy` and/or `tau_xy` plus wall-contact consistency;
- free insertion releases the lateral wrench while depth keeps increasing;
- bottom reach produces a bottom-contact / `Fz` cue near the CAD depth.

Recommended route:

```text
do not run conical probing or the direction classifier
route with a white-box depth+wrench gate
keep roll/pitch upright and execute wall-guided compliant descent
stop at the CAD bottom cue
```

## Learning Contract

The authoritative observability, symmetry, dataset-schema, real fine-tuning,
and M8/M10/M12 transfer plan is
`docs/M8_FINGERPRINT_OBSERVABILITY_AND_DATASET.md`. This section defines the
state-machine interface only.

The learning module reads a conical fingerprint. It should not output low-level
control commands directly.

Recommended outputs:

| Output | Meaning | Downstream use |
|---|---|---|
| `search_state` | outer partial / double-overhang / supported search | search-state routing for ambiguous flat support only |
| `q(phi)` | calibrated circular distribution over the likely socket-center direction | bias spiral direction without pretending the center is known |
| `Phi_(1-alpha)` | finite-coverage angular set, possibly top-2 / multimodal | preserve direction ambiguity for conservative control |
| `confidence` / `kappa` | reliability and concentration of the direction cue | schedule speed, turn count, radial pitch, and re-probe |
| `C_(1-alpha)` | CAD-constrained feasible center sector | restrict the search region, not regress the true center point |
| `unsafe_score` | jam / high-force / outer-edge risk | retreat or re-detect |
| `phase_features` | compact explanation of which phases carried the cue | diagnostics / plots |

`rim/chamfer / inner-wall / inserted / bottom` are not learning outputs. A
higher-priority white-box transition/terminal gate routes them from a short
force-ramp or descent trace before the conical classifier is called. A single
static wrench sample is not sufficient to infer the contact geometry, and the
online router does not need a chamfer label.

The center representation is set-valued:

$$
\mathcal C_{1-\alpha}=
\left\{p_0+r
\begin{bmatrix}\cos\phi\\ \sin\phi\end{bmatrix}:
\phi\in\Phi_{1-\alpha},\;
r\in[r_{min}(S,\phi),r_{max}(S,\phi)]\right\}.
$$

The conical fingerprint primarily identifies direction and contact state. The
radial interval should come from CAD compatibility, the current state, the
visual residual bound, and depth cues. One probe does not generally identify a
unique center distance.

Before adding a radial head, the P0 dataset must pass the finite-difference
Jacobian/Fisher and nearest-neighbor fingerprint-collision audits. If direction
is observable but radius is not, the intended interface remains valid:
`q(phi)` narrows the visual/CAD posterior and the posterior planner computes a
conservative movement distance.

Suggested raw representation:

```text
sequence: [theta, Fx, Fy, Fz, Tx, Ty, Tz, z, x, y]
```

Suggested feature representation:

- phase of minimum `Fz`;
- phase of maximum `z_drop`;
- phase of maximum / minimum lateral stiffness;
- phase of dominant torque vector;
- consistency between those phases;
- peak force and peak torque;
- whether inferred open side is compatible with CAD head/socket geometry.

## Spiral After The Probe

The conical probe should not replace spiral in the first version. It should bias
the spiral.

### Relation To The Task-Source Spiral

Section 4.4 of the task-source Unscrewing paper uses a surface sweep whose two
Cartesian sinusoidal amplitudes increase with time. Its geometric baseline is
therefore an outward spiral around the vision estimate:

$$
p_\mathrm{src}(\theta)=c_v+b_0\theta
\begin{bmatrix}\cos\theta\\ \sin\theta\end{bmatrix}.
$$

The source keeps about 5 N axial contact force, switches to insertion when the
axial reaction approaches zero, verifies coupling with a 0.1 N m motor-torque
threshold, and aborts when the search radius exceeds 10 mm.

The confidence-modulated method starts from the observed contact point and
contracts toward a belief center. Inward versus outward traversal alone is not
an efficiency contribution: with the same center, pitch, and radial interval,
the two paths are the same Archimedean curve traversed in opposite directions.
The potential gain comes from reducing the effective search region, changing
the turn count / pitch / speed, and terminating lateral search on an earlier
safe downward-progress cue without classifying the chamfer.

A fair timing claim must include the active probe:

$$
T_\mathrm{ours}=T_\mathrm{probe}+T_\mathrm{biased},
\qquad
T_\mathrm{ours}<T_\mathrm{src}
\iff
T_\mathrm{src}-T_\mathrm{biased}>T_\mathrm{probe}.
$$

For line spacing $p=2\pi b$, a many-turn spiral covering radius $R$ has the
approximate path length:

$$
L\approx\frac{\pi R^2}{p}.
$$

Therefore a calibrated narrow belief can reduce expected path length, but a
low-confidence or multimodal belief can erase that advantage. The exact
fallback condition is:

$$
q(\phi)\rightarrow\mathrm{Uniform},\qquad
\kappa\rightarrow0,\qquad
\Phi_{1-\alpha}\rightarrow[0,2\pi).
$$

Then remove directional bias, return the center to the visual estimate, restore
fixed pitch and speed, and run the source-style isotropic outward spiral.

Use an inward Archimedean spiral around an estimated center inside the feasible
set:

$$
r(\theta)=r_0-b(\theta-\theta_0),
\qquad p(\theta)=\hat c+r(\theta)
\begin{bmatrix}\cos\theta\\ \sin\theta\end{bmatrix}.
$$

The equivalent velocity decomposition is:

$$
\dot p =
v_r(\kappa)e_\mathrm{inward}
 +
v_t(\kappa)R_{90}e_\mathrm{inward}
$$

where:

- $e_\mathrm{inward}$ comes from the circular direction belief;
- $\kappa$ is calibrated confidence / concentration;
- high confidence permits fewer turns and a higher nominal speed;
- low confidence uses a wider angular set, more turns, a denser radial pitch,
  and a lower nominal speed;
- a wrench gate and hard abort always override the learned schedule.

The current representative M8 visualization profiles are:

| Profile | Assumed belief | Turns | Radial pitch | Nominal speed scale | Reason |
|---|---|---:|---:|---:|---|
| outer partial | `kappa=0.63`, angular half-width 24 deg | 5.00 | 0.40 mm/rev | 0.75 | directional posterior permits a 3.30 mm transport, then 2.00 mm local coverage |
| double-overhang | `kappa=0.42`, angular half-width 46 deg | 5.50 | 0.40 mm/rev | 0.50 | broad cue retains 2.20 mm radius and a lower tangential speed |
| supported search | `kappa=0.86`, angular half-width 13 deg | 3.50 | 0.40 mm/rev | 1.00 | narrow cue reduces transport residual and local radius to 1.40 mm |

These numbers are visualization profiles, not optimized controller gains. They
assume the learning output already exists.

The current speed gate is:

$$
v_{cmd}=s(\kappa)g_wv_0,
\qquad
g_w=\operatorname{clip}_{[0.25,1]}
\min\left(1,\frac{8\,\mathrm N}{\lVert F\rVert},
\frac{20\,\mathrm{mN\,m}}{\tau_{xy}}\right).
$$

Current M8 simulation-initial controller / termination values:

| Rule | Initial value | Action |
|---|---:|---|
| nominal downward force | 3 N | dynamic MuJoCo admittance initial value; maintain search contact |
| soft wrench gate | 8 N force or 20 mN m tilt torque | reduce progress speed, with floor 0.25 |
| hard abort | 20 N force or 60 mN m tilt torque | stop / retreat; learning cannot override |
| search-to-descent candidate | depth drop at least 0.07 mm and another 0.04 mm over 50 ms, raw wrench below 8 N / 20 mN m | freeze XY and begin low-force confirmation without outputting a chamfer class |
| capture confirmation | another 0.04 mm descent while down-force ramps to 2 N and planar impedance is reduced | accept safe progress; otherwise restore search parameters |
| pre-hard search escape | 12 N force or 40 mN m tilt torque | unload and physically reset before the unchanged hard abort |
| inner-wall gate | depth at most -0.30 mm and either lateral force at least 0.25 N or tilt torque at least 1.5 mN m | bypass CNN and descend upright |
| bottom completion | within 0.15 mm of CAD depth and either expected-sign axial force at least 0.70 N, or low absolute axial force + stable depth; both require bounded lateral force/tilt torque and dwell | method-independent simulation initial value; terminate geometric insertion |

These values must be swept in simulation and recalibrated on hardware. The
implemented sensor-only safe-progress event does not identify the contact as
chamfer and does not use the true center or MuJoCo contact tag. Real control
also requires hysteresis, sensor-bias handling, speed limits, and an independent
emergency stop.

Routing:

| Probe result | Spiral / action choice |
|---|---|
| high-confidence inward direction | biased guarded spiral |
| medium confidence | weakly biased guarded spiral |
| low confidence | ordinary guarded spiral or re-probe |
| unsafe / outer-edge risk | retreat / re-detect |
| double-overhang ambiguity | CAD compatibility check plus weak bias or re-probe |
| safe downward progress | stop XY expansion and start upright compliant insertion |
| inner-wall / inserted | white-box gate, then compliant descent |

## Completion Signal

The task is engagement / insertion, not full unscrewing.

A first-version success signal should combine CAD depth and force behavior:

```text
success =
    z_tip reaches expected engagement depth band
    AND Fz does not exceed the force cap
    AND lateral force / tilt torque is low or stable
    AND no unsafe / jam flag appears
```

An optional engineering assist is now implemented but disabled by default:

```text
posterior transport ends
-> local outward spiral + bounded yaw oscillation
-> persistent safe-progress or sensor-depth commit while yaw continues
-> freeze planar search, then guarded 0.15--0.8 N descent / insertion
-> continue the same +/-0.5 deg, 1.5 Hz yaw oscillation
-> disable yaw only when |tau_z| >= 10 mN m
```

The yaw motion is not active during the conical fingerprint, posterior
transport, or source-style baseline spiral. For the proposed/learned policy it
starts with the local spiral, so it can affect rotational alignment before
chamfer capture, and continues through capture dwell and guarded insertion. It
is an action-side capture aid, not a chamfer classifier and not a terminal
verifier. The existing six-axis 60 mN m hard stop remains independent. This is
inspired by the source task's use of force and torque cues, but the current new
path does not model threaded unscrewing or screw extraction, and no robustness
benefit is claimed without an ablation.

## First Experiments To Run

Do not start with a large classifier. Start with controlled observability smoke.

## Sampling Policy

Detailed P0/P1 sizes, signed-distance coordinates, nuisance strata, $C_6$
augmentation, saved-data schema, and hardware fine-tuning are specified in
`docs/M8_FINGERPRINT_OBSERVABILITY_AND_DATASET.md`.

Do not copy the reference paper's full uniform pose sampling directly. That
paper uses the tilt-then-rotate pattern to classify contact pose as the main
state estimator. This project instead uses a finite posterior-guided transport
followed by a local outward Archimedean spiral, so the conical probe only needs
to support action branching and a conservative search-region update.

Recommended sampling density:

| Region / condition | Density | Reason | Downstream meaning |
|---|---:|---|---|
| `outer_partial` | medium-high | outer edge can be mistaken for the socket opening | detect `outer_edge_risk` / recover |
| `double_overhang` | high | both outer air and inner opening can look like open side | CAD check, re-probe, weak bias |
| `supported_search` near socket | medium | a useful center-direction posterior may appear | short transport, then local outward spiral |
| physical `rim / chamfer` condition | short progress-gate calibration; no CNN full scan | useful calibration condition, not an online class | detect generic safe progress, end lateral search, then continue upright descent |
| `inner-wall capture` | no CNN sampling; short threshold calibration only | depth and sustained lateral wrench are explicit cues | white-box gate, then wall-guided descent |
| `captured / inserted` | no CNN sampling; short terminal calibration only | direction estimation is already finished | hold XY/upright and descend to CAD bottom |
| `no-contact / outside` | guard only | not a nominal sampling state | re-hover / re-detect |
| obvious middle of supported plane | low | ordinary spiral can handle it | avoid wasting samples |

Important missed cases to include:

- transition bands between `outer_partial`, `double_overhang`, and
  `supported_search`;
- transition from `rim / chamfer` to `inner wall capture`;
- small yaw mismatch between bit and socket;
- different downward force / contact depth levels;
- small clearance and friction perturbations.

Use the Allen $C_6$ symmetry only as a physically consistent augmentation:
rotate offset/direction labels, CAD/yaw, phase, $F_{xy}$, and $\tau_{xy}$
together by $k\pi/3$. Split base rollouts before augmentation. Rotated copies
are not independent physical validation data, and arbitrary-angle rotation is
not an exact symmetry of the hex socket.

The sampling rule is:

```text
sample for observability and action choice,
not for uniform geometric coverage.
```

## Offline Sampling vs Online Probing

Offline data collection and online control should not use the same rule.

Offline sampling should expose the trigger to all relevant states. Online
probing should happen only when a decision is needed.

| State | Offline sampling | Online full conical probe | Online action |
|---|---:|---:|---|
| `no-contact / no valid overlap` | not a training subject | no | re-hover / visual re-detect |
| `outer_partial` | medium-high, full 360 deg | yes at initial contact | classify outer-edge risk / recover |
| `double_overhang` | high, full 360 deg | yes or short re-probe | CAD check + weak bias |
| `supported_search` | medium, full 360 deg | yes initially or when confidence is low | biased or ordinary guarded spiral |
| physical `rim / chamfer` condition | short force-ramp/descent trajectories; not classifier training | no full conical probe | detect generic safe progress, end lateral search, continue upright descent |
| `inner-wall capture` | threshold-calibration trajectory only; not classifier training | no conical probe | depth+sustained lateral wrench gate, then compliant descent |
| `captured / inserted` | terminal-calibration trajectory only; not classifier training | no conical probe | hold XY/upright and descend to the CAD bottom depth |
| `unsafe / jam` | low-medium | no if force is high | unload / retreat first |

### Distinguishing `re-detect` From `captured / inserted`

Top-view state 1 and state 6 can both look uninformative in a single image, but
they are physically different in depth and contact validity.

```text
no-contact / no valid overlap:
    no valid screw/socket contact before the expected touchdown depth
    OR contact belongs to outside/base/invalid geometry

captured / inserted:
    z_tip is below the top plane by a CAD-consistent depth
    AND lateral force / tilt torque is low or stable
    AND contact is on chamfer / inner wall / socket-relevant geometry
```

The trigger should not infer state 6 from top view alone. It needs depth and
force/contact consistency.

### Online Re-Probe Policy

Only a still-ambiguous flat-support touchdown should use the full conical
fingerprint. During the spiral, re-probe should be event-triggered, not
continuous.

| Event during spiral | Re-probe? | Preferred action |
|---|---:|---|
| initial ambiguous flat-support touchdown | yes | full 15 deg conical fingerprint |
| no depth drop after a search segment | maybe | short re-probe or continue ordinary spiral |
| low confidence / inconsistent direction | yes | half-turn or small-angle re-probe |
| suspected `double_overhang` | yes | CAD compatibility check + weak bias |
| initial force-ramp already shows safe downward progress | no full fingerprint | do not classify chamfer; freeze XY and continue upright descent |
| safe downward progress appears during spiral | no large probe | stop spiral expansion, retain low down-force, and continue upright descent |
| `inner-wall capture` | no | direct depth+wrench gate, then compliant descent |
| `captured / inserted` | no direction probe | hold XY/upright and descend to the CAD bottom depth |
| force / torque spike or jam | no | unload / retreat first |
| outside / no contact | no | re-detect |

### Probe Force And Tilt

The reference paper uses a tilt-then-rotate pattern:

$$
roll=\alpha\sin\theta,\qquad pitch=\alpha\cos\theta,\qquad \theta\in[0,2\pi)
$$

with a constant downward force to maintain contact. For this M8 Allen task,
that principle is useful, and the main fingerprint line now keeps the same
15 deg angle for stronger sim/real phase patterns:

```text
use alpha = 15 deg for flat-support outer-partial, double-overhang, and supported-search states;
the reference paper assumes flat contact surfaces and does not validate chamfer fingerprints;
do not classify chamfer online; persistent safe downward progress switches to upright descent;
at inner-wall capture, do not start conical probing or call the CNN;
use direct depth + sustained lateral wrench + contact consistency instead;
after captured/inserted, descend upright to the CAD bottom;
return commanded roll/pitch to zero after the probe, then verify;
use a small downward force/admittance bias, not hard pressing;
always keep force / torque / depth hard stops active;
stop probing once inner-wall/capture cues are reliable.
```

### P0: Fingerprint smoke

Run M8 conical probes from representative cases:

- outer partial support;
- double-overhang / annular bridge;
- supported search;
- no-contact/outside guard only, not as a nominal fingerprint.

Collect physical `rim / chamfer` cases only as short force-ramp/descent
calibration trajectories for the generic progress gate, not as conical
fingerprints or CNN labels.

Outputs:

- phase plots of `Xz(theta)`, `Mx(theta)`, `My(theta)` plus raw 12D sequence;
- top-view contact-state label;
- predicted circular distribution `q(phi)` and angular set;
- ground-truth direction only for offline evaluation;
- safety summary.

Pass criterion:

```text
The conical fingerprint differs across macro cases in a physically interpretable
way, without exceeding force / torque caps.
```

The smoke is followed by the approximately 960-rollout observability pilot.
Its required outputs are a smallest-singular-value/Fisher map and a nearest-
neighbor collision map. Direction learning proceeds only in regions supported
by those maps; a radial head is optional and requires stronger evidence.

### P1: Probe-biased spiral vs ordinary spiral

Compare:

1. source-style ordinary guarded spiral;
2. conical-probe-biased guarded spiral;
3. oracle direction upper bound.

Same conditions:

- same initial offset distribution;
- same M8 geometry;
- same force and torque caps;
- same success signal.

Metrics:

- success / abort rate;
- time-to-insert;
- contact work;
- max force;
- max torque;
- path length;
- number of re-probes.

### P2: Learning baseline

Only after P0 is interpretable:

- feature baseline: rule / logistic / KNN / small MLP;
- sequence baseline: small temporal model or pattern image, if needed;
- polar phase CNN for the circular distribution `q(phi)`, if the smoke is
  physically interpretable;
- a 1D circular CNN with circular padding as the main candidate after the
  paper-style polar-image reproduction;
- conformal / finite-coverage angular set calibration for `Phi_(1-alpha)`;
- report angular error, angular-set coverage and size, calibration error,
  unsafe recall, and downstream time-to-safe-progress.

Separately calibrate the progress and terminal rules on short `plane / rim /
shallow-insert -> inner-wall -> bottom` trajectories using `z`, `Fxy`,
`tau_xy`, `Fz`, and persistence. These trajectories are not conical
fingerprints and are not CNN inputs; the online rule does not need a chamfer
label.

For real fine-tuning, first align sensor bias, gravity/tool load, tool/CAD
frames, timestamps, and dimensionless wrench scales. Fine-tune calibration or
a small adapter/head before changing the full backbone. Keep held-out hardware
poses and run the learner in shadow mode before enabling bounded transport.

## Claim Boundary

Allowed after design only:

```text
A conical/nutation probe is a plausible active-sensing primitive for M8 Allen
engagement. It may provide a directional prior for a guarded spiral, but this
must be validated with force/torque phase patterns and matched baselines.
```

Not allowed:

```text
one conical probe recovers the true center;
conical probing replaces spiral;
simulation proves real-robot performance;
the classifier should directly control the tool;
M10/M12 generalization is proven.
```

## Historical Decision Before Dataset/Classifier Completion

The clean path should insert this stage between vertical touchdown and the
action-conditioned dataset:

```text
visual-error random start
-> vertical touchdown
-> conical/nutation fingerprint smoke
-> biased guarded spiral or verify/retreat
-> only then build dataset/classifier around the observed sequences
```
