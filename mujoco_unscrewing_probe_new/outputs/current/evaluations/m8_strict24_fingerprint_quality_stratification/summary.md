# M8 strict-24 fingerprint-quality post-hoc stratification

## Result

- `fuzzy_short`: 6 cases; direct-first strict CAD-bottom success in 2/6.
- `clear_long`: 5 cases; direct-first strict CAD-bottom success in 5/5.

Success is the existing full strict CAD-bottom contract, not offline capture alone.

## Locked post-hoc rules

- `fuzzy_short`: APS >= 5, ensemble span >= 15 deg, and max(Proposed, Proposed+) transport <= 1.2 mm.
- `clear_long`: APS <= 3, concentration >= 0.995, ensemble span <= 7.5 deg, and min(Proposed, Proposed+) transport >= 2.7 mm.

## fuzzy_short members

| pose | APS | concentration | span (deg) | Proposed transport (mm) | Proposed+ transport (mm) | direct-first strict bottom |
|---|---:|---:|---:|---:|---:|:---:|
| sobol_002 | 10 | 0.566096 | 127.5 | 0.000000 | 0.000000 | yes |
| sobol_008 | 5 | 0.986389 | 22.5 | 1.139900 | 0.260123 | no |
| sobol_009 | 7 | 0.747771 | 15.0 | 0.674063 | 0.000000 | no |
| sobol_015 | 9 | 0.954427 | 22.5 | 1.117690 | 0.000000 | yes |
| sobol_016 | 8 | 0.835303 | 37.5 | 0.903266 | 0.542486 | no |
| sobol_023 | 10 | 0.908293 | 60.0 | 1.000402 | 0.040493 | no |

## clear_long members

| pose | APS | concentration | span (deg) | Proposed transport (mm) | Proposed+ transport (mm) | direct-first strict bottom |
|---|---:|---:|---:|---:|---:|:---:|
| sobol_000 | 3 | 0.997400 | 0.0 | 2.872823 | 2.872823 | yes |
| sobol_005 | 3 | 0.996315 | 7.5 | 2.832506 | 2.832506 | yes |
| sobol_007 | 2 | 0.997484 | 7.5 | 2.730061 | 2.730061 | yes |
| sobol_010 | 3 | 0.998157 | 0.0 | 2.847814 | 4.082930 | yes |
| sobol_013 | 3 | 0.998094 | 0.0 | 2.887805 | 2.887805 | yes |

## Frozen source identity

- Strict manifest: `outputs/current/evaluations/m8_strict_24_complete_loop/manifest.json`
- Manifest SHA-256: `09f3d2f2fa542486209ead98a2521fc081bafc8e28f3a407c8b96caccdcb7b99`
- Strict report: `outputs/current/evaluations/m8_strict_24_complete_loop/report.json`
- Report SHA-256: `e4ffde67d5ac3e0643a51850701850d50ca246ff9f85d9200c72e12a8e40c563`

## Claim boundary

Post-hoc descriptive stratification of one frozen strict-24 M8 simulation audit. The thresholds were not preregistered, the two observed count ratios are not probability estimates or significance tests, and the strata do not establish generalization, causality, hardware transfer, or a guaranteed relationship between fingerprint quality and direct-first success.
