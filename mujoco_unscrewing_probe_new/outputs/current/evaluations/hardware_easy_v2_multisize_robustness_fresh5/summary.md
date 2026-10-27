# Hardware-easy-v2 shared M8/M10/M12 audit

Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.

| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8_v5 / m8_source | 174 / 180 | 6 | 0 | 2.14 / 5.65 deg | 0.925 | 0.971 / 2.74 | 0.868 / 1.0 | 0 |
| m8_v5 / m8_easy | 155 / 180 | 25 | 0 | 1.62 / 3.08 deg | 1.000 | 0.987 / 2.17 | 1.000 / 1.0 | 0 |
| m8_v5 / m10_easy | 159 / 180 | 21 | 0 | 1.48 / 3.66 deg | 0.994 | 1.000 / 2.26 | 0.994 / 1.0 | 0 |
| m8_v5 / m12_easy | 152 / 180 | 28 | 0 | 1.63 / 4.00 deg | 0.954 | 0.993 / 2.39 | 0.980 / 0.959731543624161 | 0 |
| joint_multisize / m8_source | 174 / 180 | 6 | 0 | 1.70 / 4.76 deg | 0.954 | 0.937 / 2.25 | 0.943 / 0.9939024390243902 | 0 |
| joint_multisize / m8_easy | 155 / 180 | 25 | 0 | 1.32 / 2.58 deg | 0.994 | 0.994 / 2.08 | 0.987 / 1.0 | 0 |
| joint_multisize / m10_easy | 159 / 180 | 21 | 0 | 1.15 / 2.94 deg | 1.000 | 1.000 / 2.03 | 1.000 / 1.0 | 0 |
| joint_multisize / m12_easy | 152 / 180 | 28 | 0 | 1.28 / 3.30 deg | 0.987 | 0.993 / 2.06 | 1.000 / 0.9868421052631579 | 0 |

| size | independent gate |
|---|---:|
| M8 | FAIL |
| M10 | PASS |
| M12 | PASS |

Overall promotion: **FAIL**.

Boundary: Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 fingerprint model with frozen domain/size calibration. It is neither real-hardware fine-tuning nor a full-loop engagement result.
