# Hardware-easy-v2 shared M8/M10/M12 audit

Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.

| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8_v5 / m8_source | 170 / 180 | 10 | 0 | 1.29 / 5.79 deg | 0.918 | 0.994 / 2.49 | 0.894 / 0.993421052631579 | 1 |
| m8_v5 / m8_easy | 161 / 180 | 19 | 0 | 1.11 / 2.81 deg | 1.000 | 1.000 / 2.17 | 1.000 / 1.0 | 0 |
| m8_v5 / m10_easy | 162 / 180 | 18 | 0 | 0.79 / 2.97 deg | 0.994 | 1.000 / 2.30 | 0.994 / 0.9937888198757764 | 0 |
| m8_v5 / m12_easy | 154 / 180 | 26 | 0 | 1.27 / 4.65 deg | 0.974 | 0.994 / 2.36 | 0.981 / 0.9735099337748344 | 0 |
| joint_multisize / m8_source | 170 / 180 | 10 | 0 | 3.67 / 160.67 deg | 0.753 | 0.876 / 3.98 | 0.659 / 0.9285714285714286 | 1 |
| joint_multisize / m8_easy | 161 / 180 | 19 | 0 | 1.86 / 6.13 deg | 0.925 | 0.988 / 2.93 | 0.919 / 0.9594594594594594 | 0 |
| joint_multisize / m10_easy | 162 / 180 | 18 | 0 | 1.85 / 6.33 deg | 0.920 | 0.994 / 2.93 | 0.877 / 0.9577464788732394 | 0 |
| joint_multisize / m12_easy | 154 / 180 | 26 | 0 | 2.13 / 6.77 deg | 0.916 | 0.994 / 3.14 | 0.818 / 0.9603174603174603 | 0 |

| size | independent gate |
|---|---:|
| M8 | FAIL |
| M10 | FAIL |
| M12 | FAIL |

Overall promotion: **FAIL**.

Boundary: Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 fingerprint model with frozen domain/size calibration. It is neither real-hardware fine-tuning nor a full-loop engagement result.
