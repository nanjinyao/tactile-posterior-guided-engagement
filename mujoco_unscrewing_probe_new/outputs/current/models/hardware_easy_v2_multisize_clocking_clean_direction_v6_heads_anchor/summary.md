# Shared M8/M10/M12 hardware-easy-v2 adaptation

One three-seed weight ensemble was initialized from the audited clean M8 base.
M8 source replay and all three easy-v2 sizes are balanced by domain.
Adaptation mode: heads.
Captured/incomplete easy traces were excluded after physical-orbit splitting.
The global calibration and gate are deliberately conservative and runtime-compatible.
APS calibration target: 95.0% coverage (alpha=0.050).

| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |
|---|---:|---:|---:|---:|---:|
| m10_easy | 128 | 1.70 / 5.50 deg | 0.984 | 1.000 / 3.12 | 0.000 / n/a |
| m12_easy | 140 | 2.33 / 7.06 deg | 0.907 | 0.993 / 3.46 | 0.000 / n/a |
| m8_easy | 130 | 2.27 / 6.62 deg | 0.931 | 0.992 / 3.16 | 0.000 / n/a |
| m8_source | 192 | 0.49 / 1.50 deg | 1.000 | 1.000 / 2.87 | 0.000 / n/a |

This table is calibration/development evidence only; no fresh audit was run by this trainer.

Boundary: Simulation-only shared M8/M10/M12 contact-geometry adaptation. It does not align the Bota measurement origin, common 42 mm hardware sensor-to-tip lever, printed mass/COM/inertia, FDM compliance/friction, or measured sensor noise, and it is not full-loop evidence.
