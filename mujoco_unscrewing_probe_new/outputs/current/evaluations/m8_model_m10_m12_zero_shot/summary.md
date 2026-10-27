# Frozen M8 Direction Model: M8 Reference And M10/M12 Zero-Shot Transfer

Updated: 2026-07-12T08:57:27.610241+00:00

The M8 P1v3 ensemble, preprocessing, 15 deg probe, thresholds, and weights
were frozen. Only CAD geometry and the dimensionless bit-radius scale were
changed. No M10/M12 sample was used for training or calibration.

| size | scans | hard stop | top1 med/p90 | <=15 deg | APS coverage/size | state acc | context z p90 / >5 | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M8 | 72/72 | 0 | 1.75/9.25 deg | 1.000 | 1.000/2.62 | 1.000 | 2.37 / 0.014 | PASS |
| M10 | 72/72 | 0 | 1.75/9.25 deg | 0.986 | 0.986/2.71 | 1.000 | 3.57 / 0.014 | PASS |
| M12 | 72/72 | 0 | 1.75/9.25 deg | 0.972 | 0.972/2.94 | 0.986 | 5.26 / 0.153 | PASS |

## Claim boundary

A pass supports only zero-shot transfer of the normalized conical direction
interface in these MuJoCo contact bands. It does not validate the M8 state
head calibration, posterior distance, planner parameters, full M10/M12 insertion
or real hardware. A failed state head does not invalidate direction transfer
because state is a soft likelihood, but it forbids reusing M8 state calibration
without adaptation.
