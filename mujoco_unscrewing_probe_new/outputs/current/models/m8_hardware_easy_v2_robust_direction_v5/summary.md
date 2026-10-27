# M8 hardware-easy-v2 strict-replay adaptation

The last convolution, shared block, and two heads were adapted at low learning rate.
Batch-normalization statistics and per-seed normalization remain frozen from P1v3.
Captured or incomplete traces were excluded from direction supervision.

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| strict_calibration | 275 | 0.93 / 3.64 deg | 0.964 | 0.989 / 2.31 | 0.967 / 0.9887218045112782 |
| easy_calibration | 181 | 0.96 / 3.20 deg | 0.994 | 1.000 / 2.20 | 1.000 / 0.994475138121547 |

Boundary: Simulation-only hardware-easy-v2 contact-geometry adaptation with frozen strict replay. It is not real-hardware fine-tuning, calibrated Bota sensing, FDM compliance, or full-loop evidence.
