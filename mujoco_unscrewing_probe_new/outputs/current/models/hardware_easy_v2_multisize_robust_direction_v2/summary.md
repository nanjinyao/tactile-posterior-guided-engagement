# Shared M8/M10/M12 hardware-easy-v2 adaptation

One three-seed weight ensemble was initialized from M8 easy-v2 v5.
M8 source replay and all three easy-v2 sizes are balanced by domain.
Adaptation mode: heads.
Captured/incomplete easy traces were excluded after physical-orbit splitting.
The global calibration and gate are deliberately conservative and runtime-compatible.

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| m10_easy | 128 | 0.77 / 2.93 deg | 0.992 | 1.000 / 2.09 | 1.000 / 0.9921875 |
| m12_easy | 140 | 1.27 / 4.05 deg | 0.993 | 0.993 / 2.24 | 0.979 / 0.9927007299270073 |
| m8_easy | 130 | 0.49 / 1.86 deg | 1.000 | 1.000 / 2.02 | 1.000 / 1.0 |
| m8_source | 223 | 0.56 / 1.90 deg | 1.000 | 1.000 / 2.06 | 1.000 / 1.0 |

This table is calibration/development evidence only; no fresh audit was run by this trainer.

Boundary: Simulation-only shared M8/M10/M12 contact-geometry adaptation. It does not align the Bota measurement origin, common 42 mm hardware sensor-to-tip lever, printed mass/COM/inertia, FDM compliance/friction, or measured sensor noise, and it is not full-loop evidence.
