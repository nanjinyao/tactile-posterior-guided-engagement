# M8 Fresh 24-Pose Task-Aware Validation

| method | success | capture time | total/contact path | wear | hard stop |
|---|---:|---:|---:|---:|---:|
| proposed_plus_hybrid | 21/24 | 18.35 s | 10.58/9.58 mm | 12.11 mJ | 0 |
| proposed_posterior | 23/24 | 12.58 s | 14.69/14.66 mm | 29.64 mJ | 0 |
| baseline_a | 23/24 | 33.28 s | 70.71/70.71 mm | 124.80 mJ | 0 |
| baseline_b | 22/24 | 7.93 s | 6.18/6.18 mm | 14.26 mJ | 0 |
| baseline_c_entropy | 23/24 | 12.14 s | 6.82/2.18 mm | 0.62 mJ | 0 |

## Boundary

Fresh simulation-only Sobol validation frozen after policy and gate design. It uses a synthetic visual prior and pre-hardware wrench nuisance, so it does not establish iiwa, real wear, M10/M12, or universal superiority claims.
