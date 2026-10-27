# Hardware-easy-v2 shared M8/M10/M12 audit

Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.

| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8_v5 / m8_source | 171 / 180 | 9 | 0 | 2.12 / 4.87 deg | 0.953 | 0.988 / 2.77 | 0.889 / 0.9868421052631579 | 0 |
| m8_v5 / m8_easy | 149 / 180 | 31 | 0 | 1.37 / 3.32 deg | 1.000 | 1.000 / 2.17 | 0.987 / 1.0 | 0 |
| m8_v5 / m10_easy | 149 / 180 | 31 | 0 | 1.58 / 3.30 deg | 1.000 | 1.000 / 2.15 | 1.000 / 1.0 | 0 |
| m8_v5 / m12_easy | 152 / 180 | 28 | 0 | 1.69 / 4.37 deg | 0.987 | 0.987 / 2.33 | 0.980 / 0.9865771812080537 | 0 |
| joint_multisize / m8_source | 171 / 180 | 9 | 0 | 1.95 / 4.05 deg | 0.959 | 0.994 / 2.33 | 0.953 / 0.9815950920245399 | 0 |
| joint_multisize / m8_easy | 149 / 180 | 31 | 0 | 1.29 / 2.40 deg | 1.000 | 1.000 / 2.05 | 0.987 / 1.0 | 0 |
| joint_multisize / m10_easy | 149 / 180 | 31 | 0 | 1.39 / 2.63 deg | 1.000 | 1.000 / 1.99 | 1.000 / 1.0 | 0 |
| joint_multisize / m12_easy | 152 / 180 | 28 | 0 | 1.33 / 3.03 deg | 0.993 | 0.993 / 2.05 | 0.987 / 0.9933333333333333 | 0 |

| size | independent gate |
|---|---:|
| M8 | FAIL |
| M10 | PASS |
| M12 | PASS |

Overall promotion: **FAIL**.

Boundary: Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 fingerprint model with frozen domain/size calibration. It is neither real-hardware fine-tuning nor a full-loop engagement result.
