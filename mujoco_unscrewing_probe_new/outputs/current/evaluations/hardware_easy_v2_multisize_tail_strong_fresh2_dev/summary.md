# Hardware-easy-v2 shared M8/M10/M12 audit

Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.

| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8_v5 / m8_source | 167 / 180 | 13 | 0 | 1.76 / 5.93 deg | 0.934 | 0.982 / 2.49 | 0.934 / 0.9807692307692307 | 0 |
| m8_v5 / m8_easy | 146 / 180 | 34 | 0 | 1.40 / 3.36 deg | 0.986 | 0.993 / 2.19 | 1.000 / 0.9863013698630136 | 0 |
| m8_v5 / m10_easy | 146 / 180 | 34 | 0 | 1.50 / 4.54 deg | 0.993 | 1.000 / 2.28 | 0.979 / 1.0 | 0 |
| m8_v5 / m12_easy | 141 / 180 | 39 | 0 | 1.71 / 4.52 deg | 0.979 | 0.986 / 2.26 | 0.986 / 0.9856115107913669 | 0 |
| joint_multisize / m8_source | 167 / 180 | 13 | 0 | 2.41 / 6.96 deg | 0.910 | 0.994 / 2.40 | 0.904 / 0.9602649006622517 | 0 |
| joint_multisize / m8_easy | 146 / 180 | 34 | 0 | 1.36 / 2.82 deg | 0.993 | 0.986 / 2.03 | 0.993 / 0.993103448275862 | 0 |
| joint_multisize / m10_easy | 146 / 180 | 34 | 0 | 1.27 / 2.99 deg | 1.000 | 1.000 / 2.05 | 0.993 / 1.0 | 0 |
| joint_multisize / m12_easy | 141 / 180 | 39 | 0 | 1.23 / 3.07 deg | 0.993 | 0.993 / 2.04 | 1.000 / 0.9929078014184397 | 0 |

| size | independent gate |
|---|---:|
| M8 | FAIL |
| M10 | PASS |
| M12 | PASS |

Overall promotion: **FAIL**.

Boundary: Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 fingerprint model with frozen domain/size calibration. It is neither real-hardware fine-tuning nor a full-loop engagement result.
