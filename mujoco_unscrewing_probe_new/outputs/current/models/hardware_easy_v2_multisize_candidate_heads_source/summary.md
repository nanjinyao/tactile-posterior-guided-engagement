# Shared M8/M10/M12 hardware-easy-v2 adaptation

One three-seed weight ensemble was initialized from M8 easy-v2 v5.
M8 source replay and all three easy-v2 sizes are balanced by domain.
Adaptation mode: heads.
Captured/incomplete easy traces were excluded after physical-orbit splitting.
The global calibration and gate are deliberately conservative and runtime-compatible.

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| m10_easy | 128 | 0.80 / 2.95 deg | 0.992 | 1.000 / 2.08 | 1.000 / 0.9921875 |
| m12_easy | 140 | 1.40 / 4.16 deg | 0.986 | 0.993 / 2.25 | 0.979 / 0.9854014598540146 |
| m8_easy | 130 | 0.44 / 1.84 deg | 1.000 | 1.000 / 2.04 | 1.000 / 1.0 |
| m8_source | 223 | 0.51 / 1.86 deg | 1.000 | 1.000 / 2.04 | 1.000 / 1.0 |

This table is calibration/development evidence only; no fresh audit was run by this trainer.

Boundary: Simulation-only shared M8/M10/M12 contact-geometry adaptation. It does not align the Bota measurement origin, common 42 mm hardware sensor-to-tip lever, printed mass/COM/inertia, FDM compliance/friction, or measured sensor noise, and it is not full-loop evidence.
