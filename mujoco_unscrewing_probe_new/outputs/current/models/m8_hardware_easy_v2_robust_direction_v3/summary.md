# M8 hardware-easy-v2 strict-replay adaptation

The last convolution, shared block, and two heads were adapted at low learning rate.
Batch-normalization statistics and per-seed normalization remain frozen from P1v3.
Captured or incomplete traces were excluded from direction supervision.

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| strict_calibration | 218 | 0.86 / 3.59 deg | 0.982 | 1.000 / 2.76 | 0.761 / 1.0 |
| easy_calibration | 64 | 1.36 / 4.60 deg | 0.969 | 0.969 / 2.72 | 0.781 / 1.0 |

Boundary: Simulation-only hardware-easy-v2 contact-geometry adaptation with frozen strict replay. It is not real-hardware fine-tuning, calibrated Bota sensing, FDM compliance, or full-loop evidence.
