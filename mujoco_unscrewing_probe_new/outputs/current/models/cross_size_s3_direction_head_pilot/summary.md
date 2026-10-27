# Cross-size S3 direction-head adaptation pilot

All weights except `direction_head` are bitwise frozen from the three M8 seeds.
Train and calibration are disjoint complete C6 orbit sets; no old S1/S2
evaluation artifact is loaded. Calibration-only changes temperature/APS only.

| size | variant | train med/p90 | calibration med/p90 | calibration <=15 | APS coverage/size |
|---|---|---:|---:|---:|---:|
| M10 | cal_only | 16.75/148.25 deg | 20.50/152.00 deg | 0.278 | 0.972/43.53 |
| M10 | fine_tuned | 1.75/1.75 deg | 1.75/28.00 deg | 0.861 | 1.000/5.64 |
| M12 | cal_only | 16.75/159.50 deg | 28.00/148.25 deg | 0.306 | 0.972/38.67 |
| M12 | fine_tuned | 1.75/1.75 deg | 1.75/31.75 deg | 0.806 | 0.972/5.39 |

These are fitting/calibration diagnostics only. The untouched full-loop pilot
is the sole S3 task-level evaluation.
