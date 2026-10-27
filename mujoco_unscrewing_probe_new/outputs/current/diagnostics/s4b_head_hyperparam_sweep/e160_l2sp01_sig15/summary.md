# Cross-size S4 state + direction head adaptation

The calibration-only variant preserves every base checkpoint byte and fits
only state/direction temperature plus APS on calibration. The fine-tuned
variant freezes the encoder/shared trunk; complete C6 orbits are split into
train, validation and calibration before sampling. Model selection uses
catastrophic circular-control error and worst-state p90, not a pooled top-1
median. These are model diagnostics, not full-loop evidence.

| size | variant | role | state acc | MAP med/p90 | control med/p90 | control >90 | worst-state control p90 |
|---|---|---|---:|---:|---:|---:|---:|
| M10 | cal_only | train | 0.528 | 11.25/40.50 deg | 12.39/36.88 deg | 3 | 45.48 deg |
| M10 | cal_only | validation | 0.611 | 18.75/26.25 deg | 15.79/25.93 deg | 1 | 40.47 deg |
| M10 | cal_only | calibration | 0.333 | 11.25/11.25 deg | 9.55/36.37 deg | 1 | 46.12 deg |
| M10 | fine_tuned | train | 0.847 | 3.75/11.25 deg | 4.64/11.74 deg | 0 | 15.00 deg |
| M10 | fine_tuned | validation | 1.000 | 3.75/11.25 deg | 2.67/14.91 deg | 1 | 19.63 deg |
| M10 | fine_tuned | calibration | 0.778 | 3.75/11.25 deg | 2.82/10.62 deg | 1 | 13.90 deg |
| M12 | cal_only | train | 0.444 | 18.75/129.75 deg | 22.10/109.94 deg | 10 | 132.69 deg |
| M12 | cal_only | validation | 0.611 | 18.75/142.50 deg | 18.72/84.08 deg | 4 | 127.47 deg |
| M12 | cal_only | calibration | 0.333 | 11.25/93.75 deg | 12.57/99.93 deg | 5 | 131.37 deg |
| M12 | fine_tuned | train | 0.931 | 3.75/3.75 deg | 3.48/12.03 deg | 1 | 17.46 deg |
| M12 | fine_tuned | validation | 0.861 | 3.75/30.00 deg | 4.24/17.18 deg | 0 | 23.72 deg |
| M12 | fine_tuned | calibration | 0.806 | 3.75/22.50 deg | 5.56/44.65 deg | 3 | 116.35 deg |
