# Hardware-easy-v2 shared M8/M10/M12 audit

Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.

| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8_v5 / m8_source | 170 / 180 | 10 | 0 | 1.91 / 5.28 deg | 0.906 | 0.976 / 2.50 | 0.871 / 0.9864864864864865 | 0 |
| m8_v5 / m8_easy | 162 / 180 | 18 | 0 | 1.30 / 3.28 deg | 0.988 | 1.000 / 2.18 | 1.000 / 0.9876543209876543 | 0 |
| m8_v5 / m10_easy | 162 / 180 | 18 | 0 | 1.42 / 3.16 deg | 1.000 | 1.000 / 2.25 | 1.000 / 1.0 | 0 |
| m8_v5 / m12_easy | 160 / 180 | 20 | 0 | 1.59 / 3.97 deg | 0.994 | 0.994 / 2.36 | 0.988 / 0.9936708860759493 | 0 |
| joint_multisize / m8_source | 170 / 180 | 10 | 0 | 2.17 / 5.00 deg | 0.947 | 0.994 / 2.90 | 0.876 / 0.9932885906040269 | 0 |
| joint_multisize / m8_easy | 162 / 180 | 18 | 0 | 1.39 / 3.80 deg | 0.994 | 1.000 / 2.41 | 0.988 / 0.99375 | 0 |
| joint_multisize / m10_easy | 162 / 180 | 18 | 0 | 1.22 / 2.55 deg | 1.000 | 1.000 / 2.46 | 1.000 / 1.0 | 0 |
| joint_multisize / m12_easy | 160 / 180 | 20 | 0 | 1.46 / 3.16 deg | 1.000 | 1.000 / 2.51 | 0.994 / 1.0 | 0 |

| size | independent gate |
|---|---:|
| M8 | PASS |
| M10 | PASS |
| M12 | PASS |

Overall promotion: **PASS**.

Boundary: Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 fingerprint model with frozen domain/size calibration. It is neither real-hardware fine-tuning nor a full-loop engagement result.
