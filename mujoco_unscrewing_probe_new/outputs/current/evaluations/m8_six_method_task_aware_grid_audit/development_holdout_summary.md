# M8 Proposed+ Development/Holdout Split

The first 12 lexicographic Sobol poses were used while correcting the task-aware grid policy. Only the remaining 24 poses are an untouched holdout for this policy version.

## Untouched holdout

| method | success | capture time | total/contact path | wear | task contacts | global fallback |
|---|---:|---:|---:|---:|---:|---:|
| proposed_plus_hybrid | 21/24 | 9.72 s | 5.97/5.10 mm | 9.77 mJ | 3.0 | 9 |
| proposed_posterior | 23/24 | 15.60 s | 20.58/20.48 mm | 37.02 mJ | n/a | 3 |
| baseline_a | 21/24 | 35.66 s | 76.28/76.20 mm | 137.25 mJ | n/a | n/a |
| baseline_b | 19/24 | 7.47 s | 5.30/5.30 mm | 13.48 mJ | n/a | n/a |
| baseline_c_entropy | 23/24 | 20.01 s | 12.87/2.95 mm | 1.10 mJ | n/a | n/a |
| baseline_d_random | 13/24 | 11.60 s | 23.76/23.68 mm | 36.94 mJ | n/a | n/a |

## Development set

| method | success | capture time | total/contact path | wear | task contacts | global fallback |
|---|---:|---:|---:|---:|---:|---:|
| proposed_plus_hybrid | 11/12 | 9.97 s | 6.32/4.79 mm | 13.63 mJ | 4.5 | 2 |
| proposed_posterior | 11/12 | 12.36 s | 14.34/14.28 mm | 28.32 mJ | n/a | 1 |
| baseline_a | 9/12 | 36.47 s | 77.93/77.93 mm | 134.94 mJ | n/a | n/a |
| baseline_b | 9/12 | 7.79 s | 6.05/6.05 mm | 13.06 mJ | n/a | n/a |
| baseline_c_entropy | 11/12 | 17.43 s | 7.08/2.57 mm | 0.94 mJ | n/a | n/a |
| baseline_d_random | 6/12 | 5.53 s | 11.40/11.33 mm | 21.12 mJ | n/a | n/a |

## Boundary

The holdout is untouched by the final policy correction but shares the same synthetic MuJoCo nuisance and pre-hardware wrench prior. It supports a simulation mechanism audit only, not real iiwa, cross-size, wear, or universal superiority claims.
