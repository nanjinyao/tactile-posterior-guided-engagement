# M8 hardware-easy-v2 strict-replay adaptation

Only the state and direction heads were adapted. Encoder, shared representation,
batch-normalization statistics, and per-seed normalization remain frozen from P1v3.
Captured or incomplete traces were excluded from direction supervision.

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| strict_calibration | 192 | 1.01 / 2.76 deg | 0.974 | 1.000 / 2.93 | 0.000 / n/a |
| easy_calibration | 44 | 3.03 / 8.61 deg | 0.841 | 0.955 / 3.23 | 0.000 / n/a |

Boundary: Simulation-only hardware-easy-v2 contact-geometry adaptation with frozen strict replay. It is not real-hardware fine-tuning, calibrated Bota sensing, FDM compliance, or full-loop evidence.
