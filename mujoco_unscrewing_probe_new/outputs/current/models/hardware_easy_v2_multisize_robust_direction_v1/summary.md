# Shared M8/M10/M12 hardware-easy-v2 adaptation

One three-seed weight ensemble was initialized from M8 easy-v2 v5.
M8 source replay and all three easy-v2 sizes are balanced by domain.
Captured/incomplete easy traces were excluded after physical-orbit splitting.
The global calibration and gate are deliberately conservative and runtime-compatible.

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| m10_easy | 128 | 0.57 / 1.91 deg | 1.000 | 1.000 / 2.01 | 1.000 / 1.0 |
| m12_easy | 140 | 0.73 / 3.22 deg | 0.993 | 1.000 / 2.01 | 1.000 / 0.9928571428571429 |
| m8_easy | 130 | 0.52 / 2.09 deg | 1.000 | 1.000 / 1.90 | 1.000 / 1.0 |
| m8_source | 192 | 0.60 / 1.76 deg | 1.000 | 1.000 / 2.02 | 1.000 / 1.0 |

This table is calibration/development evidence only; no fresh audit was run by this trainer.

Boundary: Simulation-only shared M8/M10/M12 contact-geometry adaptation. It does not align the Bota measurement origin, common 42 mm hardware sensor-to-tip lever, printed mass/COM/inertia, FDM compliance/friction, or measured sensor noise, and it is not full-loop evidence.
