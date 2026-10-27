# M8 Entropy-Grid: Strong Transparent Active-Perception Diagnostic

Updated: 2026-07-15

> The filename is retained for artifact compatibility. `Baseline C` was a
> historical 2026-07-12 label and is no longer used: in the historical E3 table
> C means random, while the current strict table uses formal method names only.
> Entropy-grid is now a strong diagnostic, not a deliberately weak baseline.

## Scientific role

Entropy-grid asks whether repeated white-box CAD/F/T observations can outperform
one learned conical fingerprint in the same M8 simulation. It receives no
conical learner and no true center/contact tag online. Its hidden state is the
planar socket center $\mathbf h=(h_x,h_y)$ under the declared visual prior.

The policy repeatedly selects a guarded low-force point with maximum expected
information gain,

$$
p(y\mid\mathbf g)=\sum_i w_i p(y\mid\mathbf h_i,\mathbf g),
$$

$$
\mathbf g^*=\arg\max_{\mathbf g\in G}
\sum_{i,y}w_i p(y\mid\mathbf h_i,\mathbf g)
\log\frac{p(y\mid\mathbf h_i,\mathbf g)}{p(y\mid\mathbf g)}.
$$

Signed wrist moment is essential: moment magnitude alone cannot distinguish an
outer-head edge from an inner-socket edge. The support-centroid likelihood is a
simulation-initial CAD/F/T model and must be recalibrated before hardware use.

Entropy-grid is not literally an Archimedean spiral. It is a discrete
finite-width coverage generalization: the lattice specifies where a continuous
low-force path may be sampled, and entropy changes visit order. It is explicitly
information-seeking and is designed to improve visit order under the simulated
likelihood model, but neither an information advantage nor completion is
guaranteed.

## Finite M8 geometry certificate

The visual support is a disk of radius $R=4.8953$ mm. The CAD-derived effective
capture radius is $r_{eff}=0.25$ mm and the action pitch is
$h_g=0.3359$ mm. Clipping a square lattice at the disk boundary gives 673 nodes
and leaves an uncovered boundary crescent. Padding the action support by 0.25 mm
gives 749 nodes. Analytic cell/boundary reasoning and a dense numerical check
both give

$$
\max_{\mathbf h\in\mathcal R}\min_{\mathbf g\in G}
\|\mathbf h-\mathbf g\|=0.2375\ \mathrm{mm}<r_{eff}.
$$

This certifies only that visiting the complete padded lattice gives a geometric
node/capture-disk intersection under the declared CAD model. It does not certify
measured tracking, continuous contact, cue detection, terminal dwell, finite
execution time, hardware chamfer attraction, or survival of a hard stop. The
machine-readable source is
`outputs/current/evaluations/m8_entropy_grid_coverage_certificate/report.json`.

## Current strict execution contract

All seven methods share:

- the same initial pose, contact nuisance and sensor seed;
- sensor-only capture handoff; true center/contact tags are offline scorers;
- a common physical funnel and dual-channel CAD-bottom verifier;
- one post-step measured sensor update per physics step for controller and
  terminal, with raw physical wrench independently driving the fast hard stop;
- a non-extendable 130 s accumulated active-controller budget;
- frozen terminal outcome before the inactive 0.50 s display tail.

Strict success requires sensor handoff, offline capture validation, declared
CAD-bottom dwell, no hard stop, active time no greater than 130 s and zero
budget extension. Capture without bottom verification is a failure, not a
partial success relabeled as completion.

## Frozen 24-pose result

Entropy-grid reaches strict CAD-bottom on 22/24 poses and offline capture on
23/24. Its failures are one `capture_without_bottom` and one
`search_radius_exhausted`; it has zero hard stops, zero budget violations and
zero extensions. Over its own 22 strict successes, medians are 17.53 s to
bottom, 14.07 mm measured XY path, 2.25 mm contact path and 0.85 mJ wear proxy.

These conditional medians must not be compared without their success subsets.
On the 18 poses where both Entropy-grid and Proposed strictly succeed,
Entropy-grid has lower contact path and work on all 18; its strict count is
22/24 versus Proposed's 19/24, with exact paired $p=0.375$. This makes it a
strong transparent diagnostic. It does not prove universal entropy-policy
superiority from 24 simulated poses.

Even privileged Oracle reaches only 22/24 CAD-bottom. Thus locating the exact
center does not guarantee completion: capture, compliant descent, wrench
safety and terminal physics remain separate.

## Relation to Proposed+

Proposed+ is not pure maximum entropy. It initializes belief with vision, CAD
and one conical fingerprint, then scores capture probability, information,
travel and contact cost. Measured-path misses update belief, and a conditional
nominal reserve may trigger an early global-outward switch. There is no local
spiral inside Proposed+, and no fallback may extend the 130 s watchdog.

The defensible relationship is therefore:

```text
fixed-width spiral
-> discrete padded coverage lattice
-> entropy-grid changes diagnostic visit order
-> Proposed+ adds learned shortcut, task utility, measured-path updates,
   conditional scheduling and the shared terminal/safety loop
```

This is an algorithmic progression, not a proof that Proposed+ dominates
Entropy-grid. The frozen strict counts are 19/24 versus 22/24.

## Artifact boundary

- strict report:
  `outputs/current/evaluations/m8_strict_24_complete_loop/report.json`;
- finite-domain certificate and figure:
  `outputs/current/evaluations/m8_entropy_grid_coverage_certificate/report.json`,
  `outputs/current/media/figures/paper/m8_entropy_grid_coverage_certificate.png`;
- current selected Proposed+ mechanism explainer:
  `outputs/current/active_comparison/proposed_plus_strict_explainer/`;
- historical entropy-grid `sobol_002` clip:
  `outputs/current/active_comparison/entropy_grid_planar_explainer/`.

Selected videos explain causality and geometry; distribution claims come only
from frozen paired reports. None of these artifacts proves iiwa performance,
bit/socket rotational coupling, screw loosening, extraction or autonomous
unscrewing.
