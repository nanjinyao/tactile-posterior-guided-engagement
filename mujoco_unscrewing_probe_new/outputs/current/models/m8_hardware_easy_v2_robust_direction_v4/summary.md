# M8 hardware-easy-v2 strict-replay adaptation

The last convolution, shared block, and two heads were adapted at low learning rate.
Batch-normalization statistics and per-seed normalization remain frozen from P1v3.
Captured or incomplete traces were excluded from direction supervision.

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| strict_calibration | 248 | 0.97 / 3.18 deg | 0.992 | 1.000 / 2.70 | 0.952 / 1.0 |
| easy_calibration | 87 | 1.71 / 4.02 deg | 0.977 | 0.989 / 2.69 | 0.954 / 0.9759036144578314 |

Boundary: Simulation-only hardware-easy-v2 contact-geometry adaptation with frozen strict replay. It is not real-hardware fine-tuning, calibrated Bota sensing, FDM compliance, or full-loop evidence.
