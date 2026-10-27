# Shared M8/M10/M12 hardware-easy-v2 adaptation

One three-seed weight ensemble was initialized from M8 easy-v2 v5.
M8 source replay and all three easy-v2 sizes are balanced by domain.
Adaptation mode: tail.
Captured/incomplete easy traces were excluded after physical-orbit splitting.
The global calibration and gate are deliberately conservative and runtime-compatible.
APS calibration target: 95.0% coverage (alpha=0.050).

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| m10_easy | 128 | 0.64 / 2.64 deg | 1.000 | 1.000 / 2.80 | 1.000 / 1.0 |
| m12_easy | 140 | 1.02 / 3.68 deg | 0.993 | 1.000 / 2.85 | 0.943 / 0.9924242424242424 |
| m8_easy | 152 | 0.62 / 2.69 deg | 1.000 | 1.000 / 2.73 | 0.980 / 1.0 |
| m8_source | 254 | 0.60 / 2.01 deg | 1.000 | 1.000 / 2.62 | 0.992 / 1.0 |

This table is calibration/development evidence only; no fresh audit was run by this trainer.

Boundary: Simulation-only shared M8/M10/M12 contact-geometry adaptation. It does not align the Bota measurement origin, common 42 mm hardware sensor-to-tip lever, printed mass/COM/inertia, FDM compliance/friction, or measured sensor noise, and it is not full-loop evidence.
