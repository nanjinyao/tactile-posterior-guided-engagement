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
| M10 | fine_tuned | train | 0.833 | 3.75/3.75 deg | 3.82/20.60 deg | 0 | 24.09 deg |
| M10 | fine_tuned | validation | 0.800 | 11.25/11.25 deg | 7.20/12.37 deg | 0 | 12.82 deg |
| M10 | fine_tuned | calibration | 0.733 | 11.25/11.25 deg | 8.46/17.81 deg | 0 | 19.37 deg |
| M12 | cal_only | train | 0.567 | 18.75/97.50 deg | 21.46/82.45 deg | 5 | 104.92 deg |
| M12 | cal_only | validation | 0.667 | 18.75/37.50 deg | 18.93/38.97 deg | 3 | 102.22 deg |
| M12 | cal_only | calibration | 0.433 | 11.25/81.00 deg | 8.71/70.17 deg | 3 | 129.98 deg |
| M12 | fine_tuned | train | 0.900 | 3.75/3.75 deg | 7.74/39.74 deg | 0 | 52.32 deg |
| M12 | fine_tuned | validation | 0.833 | 11.25/12.00 deg | 7.38/37.87 deg | 1 | 47.62 deg |
| M12 | fine_tuned | calibration | 0.800 | 11.25/33.75 deg | 10.54/58.58 deg | 3 | 141.88 deg |
