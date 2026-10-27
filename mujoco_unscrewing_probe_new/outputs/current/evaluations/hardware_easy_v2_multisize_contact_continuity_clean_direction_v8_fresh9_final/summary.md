# Hardware-easy-v2 shared M8/M10/M12 audit

Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.

| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8_v5 / m8_source | 154 / 180 | 26 | 0 | 1.75 / 4.97 deg | 0.929 | 0.981 / 2.45 | 0.883 / 0.9926470588235294 | 0 |
| m8_v5 / m8_easy | 161 / 180 | 19 | 0 | 1.49 / 3.17 deg | 1.000 | 1.000 / 2.11 | 0.994 / 1.0 | 0 |
| m8_v5 / m10_easy | 160 / 180 | 20 | 0 | 1.40 / 2.92 deg | 1.000 | 1.000 / 2.16 | 1.000 / 1.0 | 0 |
| m8_v5 / m12_easy | 158 / 180 | 22 | 0 | 1.85 / 4.64 deg | 0.981 | 1.000 / 2.20 | 0.987 / 0.9807692307692307 | 0 |
| joint_multisize / m8_source | 154 / 180 | 26 | 0 | 3.79 / 163.64 deg | 0.734 | 0.838 / 3.60 | 0.721 / 0.954954954954955 | 2 |
| joint_multisize / m8_easy | 161 / 180 | 19 | 0 | 2.03 / 4.03 deg | 1.000 | 1.000 / 2.40 | 0.988 / 1.0 | 0 |
| joint_multisize / m10_easy | 160 / 180 | 20 | 0 | 1.18 / 2.67 deg | 1.000 | 0.994 / 2.23 | 1.000 / 1.0 | 0 |
| joint_multisize / m12_easy | 158 / 180 | 22 | 0 | 1.93 / 3.74 deg | 0.975 | 1.000 / 2.40 | 0.994 / 0.9745222929936306 | 0 |

| size | independent gate |
|---|---:|
| M8 | FAIL |
| M10 | PASS |
| M12 | FAIL |

Overall promotion: **FAIL**.

Boundary: Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 fingerprint model with frozen domain/size calibration. It is neither real-hardware fine-tuning nor a full-loop engagement result.
