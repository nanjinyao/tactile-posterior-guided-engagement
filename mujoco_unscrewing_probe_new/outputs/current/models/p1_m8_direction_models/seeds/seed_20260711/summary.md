# M8 P1 Direction Model Baselines

Updated: 2026-07-10T01:54:29.132662+00:00

Device: `cuda`; best CNN epoch: 19; temperature: 1.068; confidence gate: 0.586

| model / split | state accuracy | angle median / p90 | within 7.5 deg | APS coverage / size | gated accept / accuracy |
|---|---:|---:|---:|---:|---:|
| fourier / p0b_val | 0.986 | 0.00 / 7.50 deg | 0.986 | n/a | n/a |
| fourier / p0b_test | 1.000 | 0.00 / 7.50 deg | 0.972 | n/a | n/a |
| fourier / p0d_val | 0.958 | 0.00 / 37.50 deg | 0.833 | n/a | n/a |
| fourier / p0d_test | 1.000 | 0.00 / 140.25 deg | 0.819 | n/a | n/a |
| fourier / p0c_hard | 1.000 | 0.00 / 7.50 deg | 1.000 | n/a | n/a |
| fourier / p0a_full_angle | 1.000 | 3.75 / 41.25 deg | 0.708 | n/a | n/a |
| cnn / p0b_val | 0.917 | 0.00 / 7.50 deg | 0.986 | 1.000 / 2.96 | 0.875 / 1.000 |
| cnn / p0b_test | 0.903 | 0.00 / 7.50 deg | 0.986 | 0.958 / 3.07 | 0.861 / 0.984 |
| cnn / p0d_val | 0.861 | 0.00 / 7.50 deg | 0.944 | 0.986 / 3.89 | 0.889 / 0.969 |
| cnn / p0d_test | 0.972 | 0.00 / 7.50 deg | 0.917 | 1.000 / 3.32 | 0.875 / 0.968 |
| cnn / p0c_hard | 0.933 | 0.00 / 7.50 deg | 0.989 | 0.989 / 2.83 | 0.822 / 0.986 |
| cnn / p0a_full_angle | 0.972 | 3.75 / 11.25 deg | 0.847 | 0.944 / 3.42 | 0.722 / 0.923 |

## Boundary

Small simulation-only grouped baseline; P0b C6 augmentation is supplemented by physically rolled-out P0d sector anchors split by spatial orbit before nuisance expansion. P0c and P0a remain independent diagnostics. Not hardware validation, calibrated real posterior, or closed-loop gain.
