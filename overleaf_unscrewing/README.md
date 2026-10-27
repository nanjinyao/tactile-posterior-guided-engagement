# Posterior-Guided Tactile Shortcut Search

Publication audit note (2026-09-17): `main.tex` now has the title
**Learning-Based Tactile Posterior-Guided Search for Robotic Hex-Socket Engagement**;
the included `main.pdf` has 14 pages. The older title, organization and
seven-page validation below describe an earlier draft. Use the parent
[README](../README.md), [reproducibility map](../REPRODUCIBILITY.md) and
[publication audit](../PUBLICATION_AUDIT.md) for current release status.

Self-contained IEEE double-column working-paper project for:

> Posterior-Guided Tactile Shortcut Search for Bounded-Time Hex-Socket Engagement

The current double-anonymous build is seven US-Letter pages.

## Paper logic

```text
ambiguous top contact
-> one low-force conical scan
-> circular direction/contact posterior
-> posterior-supported guarded shortcut
-> bounded residual search
-> compliant capture, finite recovery, and terminal verification
```

The learned component estimates direction and contact support. It does not
declare the socket center, insertion success, force, stiffness, or depth. Those
decisions stay with the geometric prior, residual planner, compliant controller,
and measured terminal events.

The manuscript is organized as:

1. Introduction
2. Problem Formulation and System Overview
3. Posterior-Guided Tactile Shortcut Search
4. Compliant Capture and Recovery
5. Simulation and Hardware Experiments
6. Conclusion

This order follows the argument rhythm used in the local Boyuan and Junyuan
reference papers: state the physical failure first, define the system and
intermediate variable, develop the controller around that variable, and give the
experiment section enough space to separate general evidence from case studies.

## Current evidence boundary

- In 24 paired M8 MuJoCo starts, Proposed and Proposed+ reach 20/24 and 19/24
  verified engagements.
- On 19 common completions, Proposed is shorter than Pure Outward in 19/19
  measured planar paths and faster in 18/19 trials. Median paired reductions are
  41.14 mm and 16.79 s.
- Proposed+ and Proposed are statistically comparable on the present batch; the
  data do not establish an entropy-grid advantage.
- Three recoveries produce 9/12 engagements versus 0/12 without recovery on the
  fixed post-capture stress set.
- The M10/M12 simulation transfer result is mixed; fine-tuning does not dominate
  zero-shot or calibration-only controls.
- All 78 scheduled real conical scans across M8/M10/M12 completed and passed the
  acquisition checks. This is an acquisition result, not a task success rate.
- Selected M12 Proposed+, Proposed, and Improved Direct executions reached the
  measured terminal event. Pure Outward exceeded the 130 s bound and engaged
  only in an unrestricted diagnostic run of about 395 s.
- Two independently logged M10 Proposed+ executions reached the terminal state
  at one fixed, in-sample-adjacent site. They are cross-size feasibility cases,
  not unseen-position or population-level evidence.
- Hardware force and torque are baseline-relative at the Bota measurement
  origin; complete gravity/tip-moment correction and certified absolute wrench
  metrology are not claimed.
- The task ends at geometric socket engagement. First-turn coupling, thread
  release, and extraction are outside the demonstrated scope.

## Project contents

- `main.tex`: double-anonymous manuscript.
- `references.bib`: cited references.
- `ieeeconf.cls`: local PaperCept class snapshot.
- `figures/`: static simulation and hardware figures used by the manuscript.

## Evidence provenance

Simulation claims come from the evaluation reports under
`mujoco_unscrewing_probe_new/outputs/current/evaluations/`. Hardware acquisition,
full-loop, and M10 pilot evidence comes from `real_device/data/`,
`real_device/REAL_DEVICE_TRACE.md`, and `real_device/CLAIM_BOUNDARY.md`. The
August 11 experimental deck is used only to select the latest paper-ready traces;
the manuscript follows the stricter claim boundaries in the run records.

## Build and validation

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Validated on 2026-08-15:

- 7 US-Letter pages;
- no LaTeX errors, unresolved references, unresolved citations, or overfull
  horizontal boxes;
- only non-fatal underfull spacing warnings remain;
- all seven pages visually inspected after rendering;
- the author line remains blank for double-anonymous review.

Before submission, add author metadata only when permitted by the portal and
repeat the final author-side read-through at 100% zoom.
