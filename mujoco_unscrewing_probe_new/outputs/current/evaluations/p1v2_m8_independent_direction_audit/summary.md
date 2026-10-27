# M8 P1v2 Frozen Independent Direction Audit

Updated: 2026-07-10T04:16:18.187499+00:00

| samples | angle median / p90 | within 7.5 deg | APS coverage / size | accept rate | ECE | state diagnostic |
|---:|---:|---:|---:|---:|---:|---:|
| 180 | 3.75 / 11.25 deg | 0.761 | 0.972 / 3.68 | 0.994 | 0.069 | 0.983 |

## Radial strata

| level | radius | state | within 7.5 deg | p90 | APS coverage |
|---:|---:|---|---:|---:|---:|
| 0 | 1.366 mm | supported_search | 0.639 | 15.00 deg | 0.944 |
| 1 | 2.277 mm | supported_search | 0.750 | 11.25 deg | 0.972 |
| 2 | 2.945 mm | supported_search | 0.889 | 7.50 deg | 1.000 |
| 3 | 3.486 mm | double_overhang | 0.722 | 11.25 deg | 1.000 |
| 4 | 4.550 mm | double_overhang | 0.806 | 112.50 deg | 0.944 |

## Boundary

Frozen unseen-angle and unseen-nuisance MuJoCo audit evaluated once with the pre-existing P1 ensemble. No fitting, calibration, threshold selection, or active-sampling update used these trials. Not real-robot posterior or closed-loop benefit evidence.
