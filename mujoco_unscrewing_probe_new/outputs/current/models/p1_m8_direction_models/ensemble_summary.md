# M8 P1 Three-Seed Direction Ensemble

Updated: 2026-07-10T01:55:48.836501+00:00

Seeds: `[20260710, 20260711, 20260712]`; confidence gate: `0.270`; APS quantile: `0.992`

| split | state acc. | angle median / p90 | within 7.5 deg | APS coverage / size | gated accept / accuracy / >90 |
|---|---:|---:|---:|---:|---:|
| calibration | 0.958 | 0.00 / 0.00 deg | 0.993 | 1.000 / 3.24 | 1.000 / 0.993 / 0.000 |
| p0b_test | 0.931 | 0.00 / 7.50 deg | 1.000 | 0.958 / 3.21 | 1.000 / 1.000 / 0.000 |
| p0d_test | 0.958 | 0.00 / 0.00 deg | 0.972 | 1.000 / 3.01 | 1.000 / 0.972 / 0.000 |
| p0c_hard | 0.922 | 0.00 / 0.00 deg | 1.000 | 1.000 / 2.63 | 1.000 / 1.000 / 0.000 |
| p0a_full_angle | 1.000 | 3.75 / 3.75 deg | 0.958 | 0.986 / 3.51 | 1.000 / 0.958 / 0.000 |

## Boundary

Three-seed probability ensemble on simulation-only grouped splits. Calibration uses only the combined P0b/P0d validation split. It is not a real-robot posterior or closed-loop performance proof.
