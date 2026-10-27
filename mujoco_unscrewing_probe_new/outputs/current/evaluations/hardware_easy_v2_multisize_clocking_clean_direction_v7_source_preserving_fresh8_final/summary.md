# Hardware-easy-v2 shared M8/M10/M12 audit

Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.

| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8_v5 / m8_source | 171 / 180 | 9 | 0 | 1.65 / 8.02 deg | 0.895 | 0.953 / 2.95 | 0.801 / 1.0 | 0 |
| m8_v5 / m8_easy | 167 / 180 | 13 | 0 | 1.21 / 2.76 deg | 1.000 | 1.000 / 2.23 | 1.000 / 1.0 | 0 |
| m8_v5 / m10_easy | 163 / 180 | 17 | 0 | 1.22 / 3.29 deg | 1.000 | 1.000 / 2.34 | 1.000 / 1.0 | 0 |
| m8_v5 / m12_easy | 167 / 180 | 13 | 0 | 1.75 / 4.04 deg | 0.994 | 1.000 / 2.26 | 0.994 / 0.9939759036144579 | 0 |
| joint_multisize / m8_source | 171 / 180 | 9 | 0 | 4.47 / 174.67 deg | 0.725 | 0.889 / 3.68 | 0.690 / 0.923728813559322 | 2 |
| joint_multisize / m8_easy | 167 / 180 | 13 | 0 | 0.84 / 3.05 deg | 0.994 | 1.000 / 2.61 | 0.964 / 1.0 | 0 |
| joint_multisize / m10_easy | 163 / 180 | 17 | 0 | 0.85 / 2.37 deg | 1.000 | 1.000 / 2.75 | 0.994 / 1.0 | 0 |
| joint_multisize / m12_easy | 167 / 180 | 13 | 0 | 0.93 / 3.29 deg | 0.994 | 1.000 / 2.65 | 0.988 / 0.9939393939393939 | 0 |

| size | independent gate |
|---|---:|
| M8 | FAIL |
| M10 | PASS |
| M12 | PASS |

Overall promotion: **FAIL**.

Boundary: Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 fingerprint model with frozen domain/size calibration. It is neither real-hardware fine-tuning nor a full-loop engagement result.
