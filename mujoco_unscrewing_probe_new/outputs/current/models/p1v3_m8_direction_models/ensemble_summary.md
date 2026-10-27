# M8 P1 Three-Seed Direction Ensemble

Updated: 2026-07-11T05:09:20.752369+00:00

Seeds: `[20260720, 20260721, 20260722]`; structured gate: `{'min_max_probability': 0.0, 'max_aps_set_size': 3, 'min_circular_concentration': 0.0, 'max_ensemble_top1_span_deg': 7.5}`; APS quantile: `0.961`

| split | state acc. | angle median / p90 | within 7.5 deg | APS coverage / size | gated accept / accuracy / >90 |
|---|---:|---:|---:|---:|---:|
| calibration | 1.000 | 1.18 / 3.66 deg | 0.995 | 1.000 / 2.66 | 1.000 / 0.995 / 0.000 |
| p0b_test | 0.986 | 0.89 / 2.69 deg | 1.000 | 1.000 / 2.86 | 1.000 / 1.000 / 0.000 |
| p0d_test | 0.889 | 1.48 / 3.96 deg | 0.972 | 1.000 / 2.88 | 0.958 / 1.000 / 0.000 |
| p0c_hard | 1.000 | 0.64 / 1.86 deg | 1.000 | 1.000 / 2.93 | 0.989 / 1.000 / 0.000 |
| p0a_full_angle | 1.000 | 1.24 / 3.55 deg | 0.986 | 1.000 / 2.54 | 0.972 / 1.000 / 0.000 |
| targeted_val | 1.000 | 1.03 / 2.90 deg | 0.979 | 1.000 / 2.65 | 1.000 / 0.979 / 0.000 |

## Boundary

Three-seed probability ensemble on simulation-only grouped splits. Calibration uses only the combined P0b/P0d validation plus optional grouped targeted calibration. P1v2 is never loaded. It is not a real-robot posterior or closed-loop performance proof.
