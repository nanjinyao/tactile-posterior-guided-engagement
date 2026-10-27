# Cross-size S4b development full-loop evaluation

All estimator variants use the same Proposed+ hybrid controller. Pure
outward remains the declared nonlearned baseline; straight-to-center is
a privileged execution diagnostic. Every arm shares the size-conditioned
commissioned admittance/R3 funnel, raw hard thresholds, 130 s budget, and
zero extension. Strict success includes an offline scorer unavailable to
the deployed non-oracle controller.

| size | method | strict bottom | hard | control >90 | median time |
|---|---|---:|---:|---:|---:|
| M10 | M8 zero-shot + P+ hybrid | 12/12 | 0 | 1 | 11.75 s |
| M10 | S4b calibration-only + P+ hybrid | 12/12 | 0 | 1 | 11.84 s |
| M10 | S4b base-anchored residual FT + P+ hybrid | 11/12 | 1 | 1 | 11.54 s |
| M10 | pure outward | 9/12 | 1 | 0 | 33.42 s |
| M10 | privileged straight-to-center | 12/12 | 0 | 0 | 3.56 s |

- Frozen FT full-loop gate: **FAIL**
- Necessity supported: **NO**
- Interpretation: `development_diagnostic_only_no_confirmatory_claim`

| M12 | M8 zero-shot + P+ hybrid | 10/12 | 2 | 0 | 13.77 s |
| M12 | S4b calibration-only + P+ hybrid | 10/12 | 1 | 0 | 13.91 s |
| M12 | S4b base-anchored residual FT + P+ hybrid | 10/12 | 2 | 0 | 13.66 s |
| M12 | pure outward | 5/12 | 4 | 0 | 47.84 s |
| M12 | privileged straight-to-center | 11/12 | 1 | 0 | 4.86 s |

- Frozen FT full-loop gate: **FAIL**
- Necessity supported: **NO**
- Interpretation: `development_diagnostic_only_no_confirmatory_claim`

## Claim boundary

MuJoCo-only cross-size S4 evaluation. Development results may guide tuning and can never be confirmatory. Untouched results are test-only and become permanently unavailable for later tuning after readout. A fine-tuned pass supports necessity only if both zero-shot and a legal weight-identical S4 calibration-only variant fail the same frozen gate. This does not establish a real-robot fine-tuning requirement. Strict success includes offline true-center/contact-tag scoring unavailable to the non-oracle controller; privileged straight-to-center is an execution diagnostic, not a deployable policy or guaranteed performance upper bound.
