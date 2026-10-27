# M8 P1 Direction Model Baselines

Updated: 2026-07-10T01:53:34.868303+00:00

Device: `cuda`; best CNN epoch: 21; temperature: 0.873; confidence gate: 0.629

| model / split | state accuracy | angle median / p90 | within 7.5 deg | APS coverage / size | gated accept / accuracy |
|---|---:|---:|---:|---:|---:|
| fourier / p0b_val | 0.986 | 0.00 / 7.50 deg | 0.986 | n/a | n/a |
| fourier / p0b_test | 1.000 | 0.00 / 7.50 deg | 0.972 | n/a | n/a |
| fourier / p0d_val | 0.958 | 0.00 / 37.50 deg | 0.833 | n/a | n/a |
| fourier / p0d_test | 1.000 | 0.00 / 140.25 deg | 0.819 | n/a | n/a |
| fourier / p0c_hard | 1.000 | 0.00 / 7.50 deg | 1.000 | n/a | n/a |
| fourier / p0a_full_angle | 1.000 | 3.75 / 41.25 deg | 0.708 | n/a | n/a |
| cnn / p0b_val | 0.944 | 0.00 / 0.00 deg | 1.000 | 1.000 / 3.31 | 0.986 / 1.000 |
| cnn / p0b_test | 0.917 | 0.00 / 7.50 deg | 0.986 | 0.986 / 3.94 | 0.903 / 0.985 |
| cnn / p0d_val | 0.958 | 0.00 / 0.00 deg | 0.958 | 1.000 / 3.56 | 0.972 / 0.971 |
| cnn / p0d_test | 0.931 | 0.00 / 0.00 deg | 0.972 | 1.000 / 3.53 | 0.972 / 0.971 |
| cnn / p0c_hard | 0.767 | 0.00 / 0.00 deg | 1.000 | 1.000 / 3.07 | 0.933 / 1.000 |
| cnn / p0a_full_angle | 0.986 | 3.75 / 3.75 deg | 0.917 | 0.986 / 4.25 | 0.861 / 0.968 |

## Boundary

Small simulation-only grouped baseline; P0b C6 augmentation is supplemented by physically rolled-out P0d sector anchors split by spatial orbit before nuisance expansion. P0c and P0a remain independent diagnostics. Not hardware validation, calibrated real posterior, or closed-loop gain.
