# Hardware-easy-v2 shared M8/M10/M12 audit

Captured/incomplete scans are mechanics-routing outcomes and are not direction-labelled.

| model / domain | valid / total | captured | hard stops | median / p90 | within 7.5 deg | APS coverage / size | accept / accepted accuracy | accepted >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8_v5 / m8_source | 172 / 180 | 8 | 0 | 2.06 / 5.85 deg | 0.930 | 0.994 / 2.82 | 0.831 / 0.993006993006993 | 0 |
| m8_v5 / m8_easy | 156 / 180 | 24 | 0 | 0.88 / 2.40 deg | 0.981 | 0.987 / 2.13 | 1.000 / 0.9807692307692307 | 0 |
| m8_v5 / m10_easy | 159 / 180 | 21 | 0 | 1.52 / 3.94 deg | 0.975 | 1.000 / 2.20 | 0.987 / 0.9872611464968153 | 0 |
| m8_v5 / m12_easy | 157 / 180 | 23 | 0 | 1.89 / 4.88 deg | 0.981 | 0.987 / 2.26 | 1.000 / 0.9808917197452229 | 0 |
| joint_multisize / m8_source | 172 / 180 | 8 | 0 | 2.52 / 5.32 deg | 0.959 | 0.988 / 2.51 | 0.901 / 0.9870967741935484 | 0 |
| joint_multisize / m8_easy | 156 / 180 | 24 | 0 | 1.00 / 2.10 deg | 0.987 | 0.994 / 1.96 | 1.000 / 0.9871794871794872 | 0 |
| joint_multisize / m10_easy | 159 / 180 | 21 | 0 | 1.12 / 2.70 deg | 0.987 | 1.000 / 2.08 | 0.994 / 0.9936708860759493 | 0 |
| joint_multisize / m12_easy | 157 / 180 | 23 | 0 | 1.21 / 3.46 deg | 1.000 | 1.000 / 2.07 | 1.000 / 1.0 | 0 |

| size | independent gate |
|---|---:|
| M8 | FAIL |
| M10 | PASS |
| M12 | PASS |

Overall promotion: **FAIL**.

Boundary: Fresh MuJoCo contact-geometry evaluation of a shared M8/M10/M12 fingerprint model with frozen domain/size calibration. It is neither real-hardware fine-tuning nor a full-loop engagement result.
