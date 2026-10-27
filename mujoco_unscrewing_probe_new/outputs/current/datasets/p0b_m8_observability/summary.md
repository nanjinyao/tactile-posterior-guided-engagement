# M8 P0b Nuisance Observability Report

Updated: 2026-07-10T01:31:33.402756+00:00

Successful trials: 360

| feature family | state accuracy | median / p90 angle error | radial MAE | action collisions | Fisher sigma_min median / p05 |
|---|---:|---:|---:|---:|---:|
| shape | 0.911 | 0.00 / 7.50 deg | 0.209 mm | 23 / 120 | 36.40 / 22.48 |
| shape_plus_absolute | 0.936 | 0.00 / 7.50 deg | 0.157 mm | 12 / 120 | 43.37 / 25.43 |

## Boundary

Pilot feature/Fisher/collision diagnostics under four bounded nuisances; not a trained CNN, formal global identifiability proof, hardware result, or safe-transport validation.

The first decision should be based on direction error and action collisions. Radial outputs remain optional until their nuisance-conditioned benefit is confirmed in downstream posterior transport.
