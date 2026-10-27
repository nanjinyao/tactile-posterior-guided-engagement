# M8 hardware-easy-v2 strict-replay adaptation

The last convolution, shared block, and two heads were adapted at low learning rate.
Batch-normalization statistics and per-seed normalization remain frozen from P1v3.
Captured or incomplete traces were excluded from direction supervision.

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| strict_calibration | 192 | 0.62 / 2.38 deg | 1.000 | 1.000 / 2.72 | 0.990 / 1.0 |
| easy_calibration | 44 | 1.70 / 4.85 deg | 0.977 | 0.977 / 2.73 | 0.932 / 0.975609756097561 |

Boundary: Simulation-only hardware-easy-v2 contact-geometry adaptation with frozen strict replay. It is not real-hardware fine-tuning, calibrated Bota sensing, FDM compliance, or full-loop evidence.
