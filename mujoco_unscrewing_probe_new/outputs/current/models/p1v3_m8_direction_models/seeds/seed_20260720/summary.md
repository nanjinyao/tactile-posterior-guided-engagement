# M8 P1 Direction Model Baselines

Updated: 2026-07-11T05:04:24.591979+00:00

Device: `cuda`; best CNN epoch: 18; temperature: 0.250; confidence gate: 0.475

| model / split | state accuracy | angle median / p90 | within 7.5 deg | APS coverage / size | gated accept / accuracy |
|---|---:|---:|---:|---:|---:|
| fourier / p0b_val | 0.986 | 0.00 / 7.50 deg | 0.972 | n/a | n/a |
| fourier / p0b_test | 1.000 | 0.00 / 7.50 deg | 0.972 | n/a | n/a |
| fourier / p0d_val | 0.972 | 0.00 / 172.50 deg | 0.778 | n/a | n/a |
| fourier / p0d_test | 0.972 | 0.00 / 37.50 deg | 0.722 | n/a | n/a |
| fourier / p0c_hard | 1.000 | 0.00 / 0.75 deg | 0.978 | n/a | n/a |
| fourier / p0a_full_angle | 1.000 | 3.75 / 11.25 deg | 0.819 | n/a | n/a |
| cnn / p0b_val | 0.986 | 0.00 / 0.00 deg | 1.000 | 1.000 / 2.69 | 1.000 / 1.000 |
| cnn / p0b_test | 0.986 | 0.00 / 7.50 deg | 1.000 | 1.000 / 2.82 | 0.986 / 1.000 |
| cnn / p0d_val | 1.000 | 0.00 / 7.50 deg | 1.000 | 1.000 / 2.51 | 1.000 / 1.000 |
| cnn / p0d_test | 0.875 | 0.00 / 7.50 deg | 0.986 | 1.000 / 2.79 | 0.958 / 0.986 |
| cnn / p0c_hard | 1.000 | 0.00 / 0.00 deg | 1.000 | 1.000 / 2.84 | 1.000 / 1.000 |
| cnn / p0a_full_angle | 1.000 | 3.75 / 3.75 deg | 0.972 | 1.000 / 2.54 | 0.972 / 0.971 |
| cnn / targeted_val | 1.000 | 3.19 / 4.31 deg | 0.979 | 1.000 / 2.69 | 1.000 / 0.979 |

## Boundary

Simulation-only grouped direction model. P0b C6 augmentation is supplemented by physical P0d sector anchors and optional targeted off-grid training/calibration data split by base pose. P0c and P0a remain independent diagnostics. P1v2 is never loaded. Not hardware validation or closed-loop gain.
