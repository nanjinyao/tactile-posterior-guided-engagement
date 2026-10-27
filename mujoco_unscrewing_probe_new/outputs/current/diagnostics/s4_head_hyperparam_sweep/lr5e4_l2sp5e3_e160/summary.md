# Cross-size S4 state + direction head adaptation

The calibration-only variant preserves every base checkpoint byte and fits
only state/direction temperature plus APS on calibration. The fine-tuned
variant freezes the encoder/shared trunk; complete C6 orbits are split into
train, validation and calibration before sampling. Model selection uses
catastrophic circular-control error and worst-state p90, not a pooled top-1
median. These are model diagnostics, not full-loop evidence.

| size | variant | role | state acc | MAP med/p90 | control med/p90 | control >90 | worst-state control p90 |
|---|---|---|---:|---:|---:|---:|---:|
| M10 | cal_only | train | 0.533 | 11.25/33.75 deg | 16.92/49.89 deg | 2 | 55.65 deg |
| M10 | cal_only | validation | 0.633 | 18.75/26.25 deg | 14.00/23.25 deg | 1 | 23.24 deg |
| M10 | cal_only | calibration | 0.400 | 11.25/11.25 deg | 11.80/32.61 deg | 0 | 34.25 deg |
| M10 | fine_tuned | train | 0.933 | 3.75/3.75 deg | 1.53/8.18 deg | 0 | 9.72 deg |
| M10 | fine_tuned | validation | 0.900 | 11.25/12.00 deg | 3.93/13.44 deg | 0 | 25.00 deg |
| M10 | fine_tuned | calibration | 0.900 | 11.25/11.25 deg | 5.36/14.76 deg | 0 | 20.10 deg |
| M12 | cal_only | train | 0.567 | 18.75/97.50 deg | 21.46/82.45 deg | 5 | 104.92 deg |
| M12 | cal_only | validation | 0.667 | 18.75/37.50 deg | 18.93/38.97 deg | 3 | 102.22 deg |
| M12 | cal_only | calibration | 0.433 | 11.25/81.00 deg | 8.71/70.17 deg | 3 | 129.98 deg |
| M12 | fine_tuned | train | 0.933 | 3.75/3.75 deg | 4.47/18.03 deg | 0 | 26.32 deg |
| M12 | fine_tuned | validation | 0.900 | 11.25/18.75 deg | 4.95/29.72 deg | 1 | 31.22 deg |
| M12 | fine_tuned | calibration | 0.800 | 11.25/27.00 deg | 10.52/66.12 deg | 3 | 145.82 deg |
