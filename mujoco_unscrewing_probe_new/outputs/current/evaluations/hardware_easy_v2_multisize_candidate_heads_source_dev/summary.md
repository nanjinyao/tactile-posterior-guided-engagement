# Hardware-easy-v2 shared M8/M10/M12 audit

Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.

| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8_v5 / m8_source | 163 / 180 | 17 | 0 | 1.52 / 3.27 deg | 0.982 | 1.000 / 2.25 | 0.951 / 1.0 | 0 |
| m8_v5 / m8_easy | 157 / 180 | 23 | 0 | 1.36 / 3.37 deg | 1.000 | 1.000 / 2.11 | 0.994 / 1.0 | 0 |
| m8_v5 / m10_easy | 156 / 180 | 24 | 0 | 1.31 / 3.63 deg | 1.000 | 0.981 / 2.06 | 0.994 / 1.0 | 0 |
| m8_v5 / m12_easy | 156 / 180 | 24 | 0 | 1.99 / 4.65 deg | 0.994 | 0.949 / 2.22 | 0.968 / 0.9933774834437086 | 0 |
| joint_multisize / m8_source | 163 / 180 | 17 | 0 | 1.32 / 2.54 deg | 1.000 | 1.000 / 2.07 | 1.000 / 1.0 | 0 |
| joint_multisize / m8_easy | 157 / 180 | 23 | 0 | 1.65 / 3.33 deg | 1.000 | 1.000 / 2.07 | 1.000 / 1.0 | 0 |
| joint_multisize / m10_easy | 156 / 180 | 24 | 0 | 1.26 / 2.84 deg | 1.000 | 0.994 / 2.03 | 1.000 / 1.0 | 0 |
| joint_multisize / m12_easy | 156 / 180 | 24 | 0 | 1.81 / 4.06 deg | 1.000 | 0.955 / 2.08 | 1.000 / 1.0 | 0 |

| size | independent gate |
|---|---:|
| M8 | PASS |
| M10 | PASS |
| M12 | PASS |

Overall promotion: **PASS**.

Boundary: Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 fingerprint model with frozen domain/size calibration. It is neither real-hardware fine-tuning nor a full-loop engagement result.
