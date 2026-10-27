# Cross-size S4 development full-loop evaluation

All estimator variants use the same Proposed+ hybrid controller. Pure
outward remains the declared nonlearned baseline; straight-to-center is
a privileged execution diagnostic. Every arm shares the size-conditioned
commissioned admittance/R3 funnel, raw hard thresholds, 130 s budget, and
zero extension. Strict success includes an offline scorer unavailable to
the deployed non-oracle controller.

| size | method | strict bottom | hard | control >90 | median time |
|---|---|---:|---:|---:|---:|
| M10 | M8 zero-shot + P+ hybrid | 7/12 | 2 | 1 | 50.19 s |
| M10 | S4 calibration-only + P+ hybrid | 9/12 | 1 | 1 | 50.18 s |
| M10 | S4 state/direction-head FT + P+ hybrid | 9/12 | 1 | 1 | 50.23 s |
| M10 | pure outward | 7/12 | 3 | 0 | 33.74 s |
| M10 | privileged straight-to-center | 7/12 | 4 | 0 | 4.37 s |

- Frozen FT full-loop gate: **FAIL**
- Necessity supported: **NO**
- Interpretation: `development_diagnostic_only_no_confirmatory_claim`

| M12 | M8 zero-shot + P+ hybrid | 9/12 | 1 | 0 | 71.26 s |
| M12 | S4 calibration-only + P+ hybrid | 9/12 | 0 | 0 | 71.26 s |
| M12 | S4 state/direction-head FT + P+ hybrid | 9/12 | 0 | 0 | 71.26 s |
| M12 | pure outward | 8/12 | 3 | 0 | 55.12 s |
| M12 | privileged straight-to-center | 11/12 | 0 | 0 | 6.62 s |

- Frozen FT full-loop gate: **FAIL**
- Necessity supported: **NO**
- Interpretation: `development_diagnostic_only_no_confirmatory_claim`

## Claim boundary

MuJoCo-only cross-size S4 evaluation. Development results may guide tuning and can never be confirmatory. Untouched results are test-only and become permanently unavailable for later tuning after readout. A fine-tuned pass supports necessity only if both zero-shot and a legal weight-identical S4 calibration-only variant fail the same frozen gate. This does not establish a real-robot fine-tuning requirement. Strict success includes offline true-center/contact-tag scoring unavailable to the non-oracle controller; privileged straight-to-center is an execution diagnostic, not a deployable policy or guaranteed performance upper bound.
