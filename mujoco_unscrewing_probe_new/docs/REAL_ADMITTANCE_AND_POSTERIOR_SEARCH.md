# Real Admittance And Posterior-Guided Search

Updated: 2026-07-15

> Current-status note (2026-07-15): the active runtime no longer consumes true
> center or MuJoCo contact tags online. The current contract is the strict
> seven-method, 24-pose audit summarized in
> `FULL_CLOSED_LOOP_SCIENTIFIC_AUDIT.md`. Any older statement granting fallback
> an independent additional budget is superseded: accumulated
> active-controller time is capped at 130 s with zero extension.

## 1. Decision

The proposed task flow is physically coherent under the explicit assumption
that guarded vertical touchdown has already established valid contact:

```text
visual center + measured covariance
-> guarded touchdown
-> short guarded force-ramp for safe downward progress / inner-wall / inserted cues
-> one 15 deg conical fingerprint only when the rim state is ambiguous
-> state + direction posterior + calibrated confidence
-> bounded posterior-guided transport
-> local outward Archimedean spiral
-> safe downward progress stops XY search
-> upright compliant insertion
-> F/T + CAD-depth terminal verification
```

The learning module does not command force, stiffness, or arbitrary motion.
It changes only the search prior. Classical control owns contact compliance,
rate limits, and force/torque limits.

The direct gate is dynamic, not a one-sample wrench classifier. A single F/T
sample remains non-injective under unknown friction, contact point, and
compliance. Only a persistent combination of downward depth evolution, bounded
wrench, and known commanded force-ramp can preempt the conical branch. The
controller does not need to decide whether that progress came from chamfer,
rim transition, or shallow insertion.

## 2. What The Chamfer Does And Does Not Guarantee

The bit-tip and socket chamfers create an inward lateral component under
down-force once their sloped faces overlap. This enlarges the capture basin and
makes the transition from search to insertion less fragile than point-center
geometry suggests.

It does **not** prove that any spiral pitch will cover the hole. Capture still
depends on clearance, chamfer width/depth, friction, residual roll/pitch/yaw,
controller compliance, and the finite bit footprint. The search must retain

$$
\Delta r_{\mathrm{turn}} \le 2r_{\mathrm{eff}},
$$

where $r_{\mathrm{eff}}$ is a conservative effective capture radius. For the
current M8 simulation, $r_{\mathrm{eff}}=0.25$ mm is derived from the modeled
CAD/chamfer geometry rather than measured hardware. A real-system value must be
identified experimentally. The corresponding conservative pitch is
$0.8\times2r_{\mathrm{eff}}=0.40$ mm/rev. This is a simulation starting value,
not a hardware claim.

The source unscrewing paper used an outward spiral, about 5 N surface force,
an axial-force drop to start Z advance, and motor torque above 0.1 N m to verify
coupling. Those values are empirical for that system. This project has no motor
current/torque observation, so it must not claim verified rotational coupling.
The strongest available terminal claim is narrower: the upright bit reached
the CAD depth band and satisfied the declared low-axial/lateral/tilt-wrench,
stable-depth dwell. This is geometric engagement, not rotational coupling.

## 3. Implemented Cartesian Admittance Core

`src/cartesian_admittance.py` implements the task-frame dynamics

$$
M_d\ddot x + D_d\dot x + K_d x = W_{meas}-W_{target}.
$$

The implementation includes:

- six independent translational/rotational axes in SI units;
- first-order measured-wrench filtering;
- semi-implicit integration;
- acceleration, velocity, and pose-correction limits;
- per-axis enable masks for probe/search/insertion modes;
- latched hard wrench stop and explicit reset;
- anti-windup at pose limits.

The initial M8 parameters are stored in
`configs/m8_admittance_initial.json`. They are intentionally labeled
simulation-initial. A real robot adapter must provide gravity-compensated,
time-aligned task-frame wrench and track the returned pose correction with the
robot's stable inner position/velocity loop.

`src/run_m8_admittance_smoke.py` now connects this outer loop to a MuJoCo
free-body Cartesian PD inner loop. In the current supported-contact smoke, a
3 N target converged to a 2.9998 N mean over the final 0.25 s without a hard
stop (`outputs/current/reports/m8_admittance_flat_contact_smoke.json`). The smoke uses
a 0.05 s contact time constant to isolate controller behavior.

`src/render_m8_dynamic_videos.py` now reuses the same controller through
`src/mujoco_admittance_rollout.py` for all eight current videos. V1--V3 keep
the nominal tip-center XY fixed, finish the 15 deg scan, and return upright;
their maximum actual XY deflections are 0.202, 0.191, and 0.225 mm, with final
residuals below 0.060 mm. V4 starts from a known physical chamfer condition and
uses an upright 0.15--0.8 N force ramp to reach the inner-wall gate. V5 reaches
the CAD bottom verifier. V6a--V6c remain upright, execute posterior transport
plus a 0.40 mm/rev local outward spiral, and latch the simulation safe-progress
proxy. None triggered the
hard stop. These remain bounded simulation results with assumed V6 posteriors.

The inner Cartesian PD loop tracks both setpoint pose and setpoint velocity.
Tracking zero velocity while commanding the cone created several millimeters
of artificial tip drift; that implementation has been removed. The conical
paper specifies roll/pitch and constant down-force, but does not guarantee a
numerical XY drift bound or exact physical return.

The current same-start M8 outer-partial stress comparison is stored in
`outputs/current/reports/m8_outer_partial_search_comparison.json`. Both policies start
at `(4.00, 0.00) mm` and use 3 N, 0.40 mm/rev, 2.25 mm/s, the same below-rim
gate plus privileged M8 CAD capture-region evaluator, and a common 70 s active
budget. The proposed rollout includes the full 5.80 s conical probe and reaches
the endpoint at 15.91 s with 21.70 mm actual XY path. The uninterrupted
source-style outward baseline reaches it at 50.59 s with 109.86 mm actual XY
path. Both are equal-endpoint simulation results; the 43.25 versus 182.74 mJ
work values remain proxies, not hardware wear measurements. The comparison
video stays synchronized at 1x until proposed completes, then holds proposed
and displays baseline at 8x while retaining the physical simulation clock. The
posterior and evaluator remain assumed/privileged, the baseline omits the
source paper's 20 rpm and motor-torque verifier, and this initial offset is
outside its reported 1--2 mm visual-error regime. A phase and parameter sweep
is required before any general superiority claim.

Recommended axis ownership:

| Phase | Nominal trajectory | Admittance axes |
|---|---|---|
| conical fingerprint | commanded roll/pitch phase | Z force; small roll/pitch compliance |
| posterior transport | commanded XY path, upright | Z force only; yaw assist remains zero |
| local outward spiral | commanded XY path plus optional bounded yaw | Z force; XY stays trajectory-controlled |
| insertion | upright guarded Z descent plus optional bounded yaw | Z force plus limited XY compliance; roll/pitch kept near zero |

`src/capture_yaw_assist.py` and `src/render_m8_search_comparison.py` also expose
a default-off capture/clocking assist. It begins at the first local-spiral step
after posterior transport and commands a smoothly ramped `+/-0.5 deg`,
`1.5 Hz` yaw oscillation while the planned XY trajectory remains unchanged.
The same command continues through the persistent-progress dwell and guarded
insertion. It is absent from probing, posterior transport, and the source-style
baseline spiral. A measured `|tau_z| >= 10 mN m` disables only this yaw command;
the existing spiral/descent trajectory and 60 mN m controller hard stop remain
active. These values are simulation starting points, not identified hardware
settings or evidence that clocking improves capture probability.

An elaborate anomaly policy is not required for the first experiment, but
hard wrench and rate limits remain mandatory hardware invariants. Removing
learned OOD/recovery from the first paper scope is reasonable; removing basic
contact limits is not.

## 4. How To Set The Short Transport

Direction confidence alone cannot determine a distance. The distance should
come from a posterior over possible centers:

$$
p(\delta\mid y,G,vision)\propto
q_\psi(\delta\mid y,G)\,
\frac{p_{vision}(\delta)}{p_{train}(\delta)}\,
\mathbf 1[\delta\in\mathcal C_{CAD}].
$$

The prior-ratio term matters when the simulation sampler is deliberately
boundary-biased rather than distributed like the real visual error. If the
learner outputs only `q(phi)`, radius remains supplied by the visual prior and
CAD compatibility. A radial output is enabled only after the observability
audit in `docs/M8_FINGERPRINT_OBSERVABILITY_AND_DATASET.md`.

Let $u$ be the learned inward direction and let posterior center samples be
$c_i$. Project them onto $u$:

$$
s_i=u^T(c_i-p).
$$

Use a quality-adaptive conservative lower distance bound:

$$
q_{dir}=\kappa\cos\left(\frac{\Delta\phi_{seed}}{2}\right),
\qquad
q_{eff}=\operatorname{clip}\left(q_{dir}(1-2P[s_i\le0]),0,1\right),
$$

$$
\alpha=0.10+0.15q_{eff},
\qquad
d_{raw}=\max\left(0,Q_{\alpha}(s_i)-0.25r_{\mathrm{eff}}\right),
$$

$$
d_{move}=\min\left(d_{raw},(0.35+0.30q_{eff})R_{99},d_{max}\right).
$$

Interpretation:

- $\alpha\in[0.10,0.25]$ spends more of the projected posterior only when
  circular concentration and seed agreement are high;
- posterior mass behind the selected direction continuously discounts that
  quality, and mass at or above 0.20 forces zero transport;
- the quality-scaled $R_{99}$ cap and $d_{max}$ prevent one inference from
  consuming the whole search;
- residual posterior coverage remains active even when transport is nonzero.

After transport to $p_1=p+d_{move}u$, set the local outward-spiral radius from
the remaining posterior:

$$
R_{max}=\max\left(0,
Q_{0.99}(\|c_i-p_1\|)-r_{\mathrm{eff}}
\right).
$$

This rule is implemented in `src/posterior_search_plan.py`. For the first M8
hardware trials, a separate $d_{max}=0.5$--$1.0$ mm and a low tangential speed
around 1 mm/s are reasonable conservative starting caps, but both must be
validated against force tracking and sensor latency. Safe downward progress
always stops the transport or spiral early.

### Approximate CAD and F/T state fusion

The planner now has a separate P2c input path that uses the frozen learner's
support-state probabilities from the same `Fz/Mx/My/tip-z` phase trace. Approximate
CAD contributes two interpretable radial transitions:

$$
r_{full}=R_{head}-R_{bit},\qquad
r_{overlap}=a_{socket}+R_{bit}.
$$

Candidate centers are weighted by a soft mixture of `supported_search`,
`double_overhang`, and `outer_partial` radial likelihoods. A 0.35 mm transition
sigma and 0.02 floor prevent an approximate CAD model or an overconfident state
head from becoming a hard distance oracle. The output remains a radial
distribution consumed by the conservative planner, not a point estimate.

The first same-start M8 rollout was clipped by a fixed 1.50 mm transport cap and
reported a 22.43 N one-step peak at the search-to-insertion handoff. A 0.5 ms
trace audit showed that probe/transport/spiral maxima were only 4.35/5.23/7.07 N.
Replacing the instantaneous admittance reset with a 0.30 s continuous force
release reduced the identical full-rollout peak to 7.07 N without delaying
capture. Raw wrench safety is now checked at every physics step, not only at the
2 ms admittance update.

The current continuous planner replaces the fixed Q10/0.5R95 choice with a
quality-adaptive Q10--Q25 lower quantile and quality-scaled posterior cap. In
the same case it selects 2.880 mm, reduces actual XY path to 13.97 mm, and
retains a 7.78 N raw / 4.70 N filtered peak. The
active distribution policy uses 99% residual local coverage and, if that local
set is exhausted, returns to the visual anchor and executes the full outward
coverage path. Thus posterior guidance is a shortcut, not the final guarantee.

## 5. F/T And CAD-Only Verification

Without motor torque, use a staged verifier rather than one threshold:

| Cue | Physical meaning | Action |
|---|---|---|
| persistent downward motion under bounded wrench | safe insertion progress; exact contact class is irrelevant | end lateral search, keep upright, and continue compliant descent; not task completion |
| below-rim depth plus sustained lateral force/moment | inner-wall guidance | guarded compliant descent |
| CAD-bottom depth band plus either expected-sign axial seating force or stable low axial wrench; both with bounded lateral force/tilt torque and dwell | geometric engagement at modeled depth | stop Z motion and report CAD-bottom verified |

The current search rollout no longer uses MuJoCo contact tags or the true-center
capture radius for its handoff. It requires an absolute 0.07 mm drop from the
surface baseline, a further 0.04 mm drop over the latest 50 ms, and raw wrench
inside 8 N / 20 mN m. It then freezes the commanded XY point, ramps the search
down-force to 2 N over 150 ms, reduces planar inner impedance from 12 kN/m and
420 N s/m to 4 kN/m and 280 N s/m, and accepts capture only after another
0.04 mm of downward motion. A rejected candidate smoothly restores force and
impedance. This is a sensor/depth trend gate that can be implemented on the
robot; the exact values are simulation-initial and still require hardware
calibration.

A 12 N / 40 mN m pre-hard search escape can unload and physically reset before
the unchanged 20 N / 60 mN m per-axis hard stop is reached. In the current
strict controller, guarded return and global outward coverage are charged to
the same accumulated 130 s active-time budget as the shortcut; fallback budget
extension is exactly zero. Proposed+ uses a conditional nominal reserve before
shortcut/grid/commit and switches early when it cannot reserve guarded return,
full outward motion, candidate handling and terminal dwell. Tracking error,
unexpected dwell or interruption can still consume the reserve and end in a
named failure. Thus fallback is executable under stated assumptions, not a
mathematical or hardware success guarantee.

The terminal insertion stage reduces the target down-force from 3.0 N search
contact to 0.8 N. The implemented method-independent verifier now accepts two
50 ms channels inside $|z-z_{CAD}|\le0.15$ mm: expected-sign axial seating
force $s_zF_z\ge0.70$ N, or $|F_z|$ below the configured low-wrench limit with
depth span at most 0.02 mm. Both channels also bound lateral force and tilt
torque; a reverse-sign axial force cannot pass the force channel. The low-wrench
channel prevents a centered free insertion from being mislabeled as failure.
All 126 strict successes in the current batch used this low-wrench branch, so
the signed force branch is implemented and unit-tested but not batch exercised.

These cues diagnose geometric insertion. They do not prove that the bit can
transmit unscrewing torque. A later first-turn verification or shaft torque
measurement is required for that stronger claim.

## 6. Immediate Validation Ladder

1. Unit-test the controller signs, bounds, hard-stop latch, and posterior planner.
2. Calibrate the implemented sensor-only safe-progress gate on force-ramp and
   shallow-insertion hardware traces; retain tags only for offline simulation scoring.
3. Perturb the completed dynamic conical/chamfer/wall/search rollouts across contact parameters.
4. Identify $r_{\mathrm{eff}}$ under friction, stiffness, clearance, pose, and delay variation.
5. Compare the seven declared strict roles: Proposed, Proposed+, outward,
   entropy-grid, direct-first, random and privileged Oracle. Proposed,
   Proposed+ and direct-first must share the exact first probe trace and frozen
   direction output; only downstream action may differ. No retry/reset is
   credited unless its physical motion and time are explicitly modeled.
6. On hardware, first run the learner in shadow mode; then commission capped
   transport before releasing the posterior-limited command.

The historical 36-pose capture-only nuisance audit gave Proposed / full
outward / shared-direction direct-retry capture of 36/36, 35/36, and 29/36 with
0/1/0 hard stops. Proposed uses global physical-reset/full-coverage fallback in
four trials. Proposed/B paired counts are 29 both, 7 Proposed-only, and no
B-only or joint failures ($p=0.015625$); Proposed/A counts are 35 both and one
Proposed-only ($p=1.0$). B remains the fastest direct shortcut when it succeeds.
These are results under an engineering noise prior, not measured hardware
statistics or a universal dominance claim.

The current frozen result is instead the 24-pose strict complete-loop audit in
which every method pays the same terminal and 130 s zero-extension contract;
its Proposed/Proposed+ strict CAD-bottom counts are both 19/24.

The clean paper hinge is:

> A standardized conical contact fingerprint converts an ambiguous rim contact
> into a calibrated direction posterior; a conservative admittance-controlled
> transport spends only the reliable part of that posterior, while a
> geometry-covering outward path retains a conditional geometric fallback.

## Post-capture iiwa funnel guard

The search hinge above is unchanged. After the sensor-only downward-progress
handoff, the rollout can now switch to a peg-tip impedance frame and an
independent `FunnelGuard`. The guard monitors injected contact work, depth
progress, tilt, and the CAD bottom envelope. First entry into the CAD bottom
tolerance band freezes the axial command for terminal dwell. Ordinary
energy-without-progress stalls permit one unload/retract/upright/retouch cycle;
crossing the CAD depth envelope unloads and terminates without retouch.

The frozen 12-pair MuJoCo stress ablation gives legacy/guard bottom completion
of 5/12 and 4/12, hard stops of 1/12 and 0/12, peak-force p90 of 17.31 N and
11.15 N, and maximum CAD-depth overrun of 1.00 mm and 0.49 mm. Six guard runs
end in explicit CAD-depth safe abort. Thus the current result supports bounded
defense, not a success-rate improvement. No ordinary stall recovery occurred
in this batch, and the parameters are not hardware calibrated. See
`docs/IIWA_FUNNEL_GUARD.md` for equations and the hardware boundary.
