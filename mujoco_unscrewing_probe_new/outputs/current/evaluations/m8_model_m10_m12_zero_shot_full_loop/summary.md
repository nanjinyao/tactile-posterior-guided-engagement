# Frozen-M8 M10/M12 Zero-Shot Full-Loop Audit

Updated: 2026-07-12T11:55:24.541584+00:00

M10/M12 data are evaluation-only. The M8 P1v3 model, normalization, APS
and learning thresholds remain frozen. CAD geometry, capture radius,
coverage pitch, visual support and bounded transport cap are size-conditioned.

| size | method | CAD-bottom | capture | hard stop | bottom time | path | force/torque p90 | control-dir med/p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M10 | proposed | 23/30 | 27/30 | 3 | 26.28 s | 39.10 mm | 9.96 N / 56.06 mNm | 2.58/21.78 deg |
| M10 | outward | 21/30 | 28/30 | 1 | 53.51 s | 110.57 mm | 9.39 N / 34.21 mNm | n/a/n/a deg |
| M10 | oracle | 27/30 | 30/30 | 0 | 3.81 s | 4.38 mm | 8.11 N / 28.00 mNm | n/a/n/a deg |

- M10 gate: **FAIL**
- M10 Proposed failure split: `{"bottom_verified": 23, "capture_without_bottom": 3, "hard_stop": 3, "no_sensor_capture": 1}`

| M12 | proposed | 21/30 | 27/30 | 4 | 39.53 s | 67.18 mm | 9.57 N / 68.83 mNm | 3.58/24.76 deg |
| M12 | outward | 19/30 | 28/30 | 6 | 62.44 s | 125.05 mm | 11.46 N / 89.49 mNm | n/a/n/a deg |
| M12 | oracle | 30/30 | 30/30 | 0 | 5.06 s | 5.83 mm | 8.28 N / 32.90 mNm | n/a/n/a deg |

- M12 gate: **FAIL**
- M12 Proposed failure split: `{"bottom_verified": 21, "capture_without_bottom": 3, "hard_stop": 4, "no_sensor_capture": 2}`

## Claim boundary

Simulation-only paired S2 audit inside size-specific synthetic visual 95% disks. It tests a frozen M8 direction learner with CAD-conditioned geometry and coverage, but retains simulation-initial control and terminal thresholds. A pass supports bounded zero-shot full-loop transfer in these MuJoCo fixtures; it does not establish probability calibration, real iiwa transfer, manufacturing tolerance, passivity, motor coupling, threaded unscrewing, or extraction. Oracle uses the true center only as an offline execution-feasibility upper bound.
