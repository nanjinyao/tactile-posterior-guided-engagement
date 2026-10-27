# Proposed+ Task-Aware Belief-Grid Policy

Updated: 2026-07-15

## What each information source actually does

It is misleading to call every input a separate "prior" or "posterior". The
closed-loop variables have different roles:

- vision gives a bounded initial center density $p_v(\mathbf h)$;
- the fixed conical fingerprint $y_c$ gives a direction/state likelihood;
- CAD gives the valid geometry, capture tube, and observation model;
- a guarded contact gives a new F/T observation and, if capture did not occur,
  a negative swept-area observation;
- the controller converts the current belief into the next physical action.

For hidden socket center $\mathbf h$, the initial grid belief is

$$
b_0(\mathbf h)\propto p_v(\mathbf h)
\left[p(y_c\mid\mathbf h,\mathcal G_{CAD})\right]^{\eta_c},
\qquad 0<\eta_c\leq 1.
$$

A nonzero likelihood floor is retained. This prevents an imperfect simulated
fingerprint model from deleting the true center before real-robot fine tuning.

## Why pure maximum entropy is not the proposed controller

Pure active perception selects

$$
\arg\max_{\mathbf g} I(\mathbf h;y\mid\mathbf g).
$$

That point is often a diagnostic boundary point. It can be informative without
moving the peg toward capture. Proposed+ instead solves a one-step,
task-conditioned objective:

$$
\mathbf g_k^*=\arg\max_{\mathbf g\in G}
\left[
w_c P_{cap}([\mathbf x_k,\mathbf g]\mid b_k)
+w_I I(\mathbf h;y\mid\mathbf g)
-\lambda_l\|\mathbf g-\mathbf x_k\|-C_{contact}
\right].
$$

$P_{cap}$ is belief mass inside the CAD-derived capture tube swept by the whole
low-force segment, not just its endpoint. Information gain breaks ties and
targets ambiguity; it is not the task objective by itself. Wrench limits,
admittance, tip virtual RCC, and the energy/health shield are hard constraints,
not rewards that the optimizer may trade away.

For an infinite square lattice, the pitch obeys the local geometric condition

$$
\frac{h_g}{\sqrt 2}\leq r_{eff}.
$$

For the finite M8 visual disk, that inequality alone is insufficient because a
clipped lattice can leave an uncovered boundary crescent.  The audited action
set therefore pads the disk: visual support radius $R=4.8953$ mm,
$r_{eff}=0.25$ mm, action pitch $h_g=0.3359$ mm, 673 in-disk nodes, and 749
padded nodes.  Both an analytic cell/boundary check and a dense numerical check
give a worst support-to-node gap of 0.2375 mm $<r_{eff}$.  This certifies node
geometry only; it says nothing about which nodes are visited, measured tracking,
cue detection, time, safety interruption, or CAD-bottom completion.

## Observation update and sim-to-real restraint

Node-to-node motion is a low-force, minimum-jerk straight segment,

$$
\mathbf x(u)=\mathbf x_k+(10u^3-15u^4+6u^5)
(\mathbf g_k^*-\mathbf x_k),\qquad u\in[0,1].
$$

The endpoint is held quasi-statically before assigning an F/T class. If the
segment does not capture, that absence is useful evidence: hypotheses whose CAD
capture tube was just swept are softly downweighted rather than deleted. The
subsequent F/T likelihood is tempered when the observed class was surprising,

$$
b_{k+1}(\mathbf h)\propto b_k^-(\mathbf h)
p(y_k\mid\mathbf h,\mathbf g_k)^{T_k},
\qquad
T_k=\operatorname{clip}\left(
\frac{p(y_k\mid b_k^-)}{p_{ref}},T_{min},1
\right).
$$

This separates robust geometric evidence (the path did not capture) from the
less trusted simulated F/T class model. Real iiwa data can later recalibrate the
likelihood and temperature without changing the controller structure.

## Executed closed loop

```text
deployment router: no-contact / below-rim / ambiguous top contact
-> evaluated loop starts here, conditioned on ambiguous top contact
-> one fixed 15-degree conical fingerprint
-> vision + fingerprint likelihood + CAD belief initialization
-> bounded posterior shortcut
-> receding-horizon capture-probability + information grid segments
-> depth/F/T capture candidate + conical-direction consistency check
-> accepted capture preempts every tangential action
-> shared upright/RCC/energy-shield physical funnel
-> CAD-depth + (expected-sign axial-force dwell OR stable low-axial-wrench depth dwell); both bound lateral force/tilt torque
```

The upstream deployment router is an interface specification, not an evaluated
classifier in the frozen strict batch.  The strict rollouts begin after the
benchmark has declared `ambiguous_top_contact`.

There is no local spiral inside Proposed+. Before every optional shortcut,
task-grid segment, or MAP commit, a conditional nominal schedule reserves time
for a guarded return, full visual-support outward path, candidate contingency,
and terminal execution.  If the schedule no longer fits, the policy switches
early to the outward path.  This is not an eventual-coverage guarantee: it
assumes no unsafe interruption, bounded tracking error (0.25 mm in the frozen
contract), at most 6 s extra candidate handling, continuous-contact grid motion,
and the configured command speed.  Every fallback use is counted and can never
extend the hard watchdog.

The controller is considered a scientifically reasonable closed loop when all
of the following interfaces are explicit; it is not required to succeed on
every rollout:

1. each guarded segment produces capture, a contact observation, a swept-area
   miss, or an unsafe event;
2. capture preempts further tangential search;
3. a miss/contact observation changes the belief or is recorded as a
   deliberately tempered non-update;
4. the updated belief changes the next segment score and action ordering;
5. exhausted utility, active-time budget, and unsafe wrench each have a named
   fallback or terminal result;
6. search capture and CAD-bottom completion remain separate outcomes.

For the frozen M8 comparisons the watchdog is a non-extendable 130 s of
accumulated simulated active-controller time.  Reaching it is a named controlled
failure; `fallback_budget_extension_s` must remain zero.  Inactive display tails
do not consume this clock and are frozen out of terminal success/offline scoring,
although their raw physical peaks remain diagnostic.  The nominal schedule is
not a physical theorem. A finite coverage claim requires a bounded residual
region $\mathcal R$, a swept path $\Gamma_T$, and

$$
\mathcal R\subseteq\bigcup_{\mathbf x\in\Gamma_T}
B(\mathbf x,r_{eff}),
\qquad
T_{max}\geq T_{touch}+T_{probe}+T_{optional}+T_{return}
+L(\Gamma_T)/v+T_{candidate}+T_{handoff}+T_{terminal}+T_{recovery}.
$$

This establishes conditional intersection with the declared capture tube only
if the scheduled path is actually executed.  It does not guarantee that a
capture cue is detected or that CAD-bottom is verified under arbitrary
friction, false cues, tracking error, timeout, or safety interruption.

An outer-head edge can create a depth/F/T signature similar to the socket inner
wall. For methods that own a conical direction estimate, a candidate is therefore
accepted only if it is not materially behind the frozen top-1 direction:

$$
(\mathbf x_{cand}-\mathbf x_0)^\top\hat{\mathbf u}_{top1}
\geq-r_{eff}.
$$

This is a one-sided consistency check, not a center estimate. The CAD-scaled
$r_{eff}$ tolerance permits local capture and finite contact width. It uses no
true center or MuJoCo contact tag. A rejected reverse-side drop remains in the
search loop instead of being handed to insertion.

## Current evidence boundary

The frozen seed-20260723 audit contains 24 paired M8 poses and directly executes
capture handoff through CAD-bottom.  Strict success requires sensor-only handoff,
offline capture validation, a declared terminal dwell, no hard stop, active time
$\leq130$ s, and zero budget extension.

| method | strict bottom / offline capture | bottom-time median | success-path median | wear median | hard stop |
|---|---:|---:|---:|---:|---:|
| Proposed | 19/23 of 24 | 13.81 s | 14.29 mm | 27.94 mJ | 0 |
| Proposed+ | 19/23 of 24 | 13.74 s | 6.76 mm | 9.54 mJ | 1 |
| Outward | 19/23 of 24 | 34.63 s | 71.27 mm | 120.38 mJ | 0 |
| Entropy-grid | 22/23 of 24 | 17.53 s | 14.07 mm | 0.85 mJ | 0 |
| Direct-first | 15/18 of 24 | 9.11 s | 5.58 mm | 13.39 mJ | 0 |
| Random | 10/12 of 24 | 11.46 s | 20.97 mm | 28.91 mJ | 1 |
| Oracle | 22/24 of 24 | 3.16 s | 3.72 mm | 5.19 mJ | 0 |

The efficiency columns are medians only over each method's own strict-success
subset; their sample membership differs and they cannot be read independently
of success.  Proposed and Proposed+ have 18 both-success, one Proposed-only,
one Proposed+-only, and four both-failed pairs (exact McNemar $p=1.0$).  On the
18 common successes, Proposed+ is 0.40 s slower at the median, while its total
path, contact path, and wear are lower by 0.07 mm, 2.12 mm, and 6.09 mJ.  These
are tradeoffs, not a superiority result.

Proposed+ switches early to preserve the nominal fallback schedule in 7/24
rollouts and uses global outward in 8/24.  Its five non-successes are one search
exhaustion, three captures without a valid bottom dwell, and one raw-physical
torque hard stop.  It has zero timeout, zero budget extension, zero budget
contract violation, and zero guarded-motion bypass.  This is a reasonable
bounded failure-aware loop, not a guaranteed rescue mechanism.

Entropy-grid reaches strict bottom in 22/24 and has lower contact path/work on
all 18 common successes with Proposed.  It remains a strong transparent
diagnostic; the result cannot be hidden or relabeled as a weak baseline, but
24 simulation poses also do not prove universal entropy-policy superiority.
Even Oracle reaches only 22/24 CAD-bottom, which directly shows that perfect
center transport does not solve the terminal physics.

All 126 strict successes across the seven methods use
`cad_depth_low_wrench_dwell`; the signed axial-force channel remains implemented
and unit-tested but is not exercised by this batch.  The quantitative source is
`outputs/current/evaluations/m8_strict_24_complete_loop/report.json`.

The selected strict mechanism rollout `sobol_008` uses five grid observations,
captures at 11.944 s, and reaches CAD bottom at 16.209 s without global outward.
It is a causal observe-update-move example, not an independent distributional
sample.

## Claim boundary

The mathematics establishes the role of each signal and the finite-grid
coverage condition under the declared CAD capture model. It does not prove that
the simulated F/T likelihood is calibrated on hardware, that every contact path
is collision free, or that Proposed+ universally beats spiral, direct/retry, or
pure entropy baselines. The defensible hypothesis is narrower: useful tactile
direction information can shrink the region that must be covered, while
task-aware contacts retain a geometric fallback against imperfect direction and
distance estimates.

## Artifacts

- strict 24-pose complete-loop report:
  `outputs/current/evaluations/m8_strict_24_complete_loop/report.json`;
- strict audit overview:
  `outputs/current/media/figures/paper/m8_strict_complete_loop.png`;
- finite-disk padded-lattice certificate:
  `outputs/current/evaluations/m8_entropy_grid_coverage_certificate/report.json`;
- selected strict Proposed+ top view, video, and manifest:
  `outputs/current/active_comparison/proposed_plus_strict_explainer/planar_map.png`,
  `outputs/current/active_comparison/proposed_plus_strict_explainer/proposed_plus_strict_sobol_008.mp4`,
  `outputs/current/active_comparison/proposed_plus_strict_explainer/manifest.json`;
- historical six-method video:
  `outputs/current/diagnostics/historical_comparisons/six_methods/comparison_six_methods_3x2.mp4`;
- historical three-method race:
  `outputs/current/diagnostics/historical_comparisons/three_method_race/comparison_proposed_plus_proposed_entropy_race_1x3.mp4`;
- machine-readable selected-case gates:
  `outputs/current/diagnostics/historical_comparisons/six_methods/manifest.json`;
- physical-funnel safety ablation:
  `outputs/current/active_comparison/funnel_guard/comparison_post_capture_funnel_guard.mp4`.
