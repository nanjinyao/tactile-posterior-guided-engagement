# Cross-size S4b development full-loop evaluation

All estimator variants use the same Proposed+ hybrid controller. Pure
outward remains the declared nonlearned baseline; straight-to-center is
a privileged execution diagnostic. Every arm shares the size-conditioned
commissioned admittance/R3 funnel, raw hard thresholds, 130 s budget, and
zero extension. Strict success includes an offline scorer unavailable to
the deployed non-oracle controller.

| size | method | strict bottom | hard | executed >90 | latent >90 | median time |
|---|---|---:|---:|---:|---:|---:|
| M10 | M8 zero-shot + P+ hybrid | 12/12 | 0 | 0 | 1 | 11.93 s |
| M10 | S4b calibration-only + P+ hybrid | 12/12 | 0 | 0 | 1 | 12.12 s |
| M10 | S4b base-anchored residual FT + P+ hybrid | 12/12 | 0 | 0 | 1 | 12.02 s |
| M10 | pure outward | 9/12 | 1 | 0 | 0 | 33.68 s |
| M10 | privileged straight-to-center | 12/12 | 0 | 0 | 0 | 3.83 s |

- Frozen FT full-loop gate: **PASS**
- Necessity supported: **NO**
- Interpretation: `development_diagnostic_only_no_confirmatory_claim`

| M12 | M8 zero-shot + P+ hybrid | 10/12 | 1 | 0 | 0 | 13.37 s |
| M12 | S4b calibration-only + P+ hybrid | 10/12 | 0 | 0 | 0 | 13.73 s |
| M12 | S4b base-anchored residual FT + P+ hybrid | 10/12 | 0 | 0 | 0 | 13.46 s |
| M12 | pure outward | 4/12 | 1 | 0 | 0 | 36.66 s |
| M12 | privileged straight-to-center | 11/12 | 0 | 0 | 0 | 5.45 s |

- Frozen FT full-loop gate: **PASS**
- Necessity supported: **NO**
- Interpretation: `development_diagnostic_only_no_confirmatory_claim`

## Claim boundary

MuJoCo-only cross-size S4 evaluation. Development results may guide tuning and can never be confirmatory. Untouched results are test-only and become permanently unavailable for later tuning after readout. A fine-tuned pass supports necessity only if both zero-shot and a legal weight-identical S4 calibration-only variant fail the same frozen gate. This does not establish a real-robot fine-tuning requirement. Strict success includes offline true-center/contact-tag scoring unavailable to the non-oracle controller; privileged straight-to-center is an execution diagnostic, not a deployable policy or guaranteed performance upper bound.
