# M8 Fair Policy Benchmark

| method | eventual success | first attempt | capture time median | contact path median | wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_1p5 | 16/18 | 16/18 | 20.615249999997303 s | 38.65 mm | 66.63 mJ | 8.11 / 4.94 N | 21.96 mNm |
| proposed_posterior | 17/18 | 17/18 | 16.06000000000792 s | 26.64 mm | 42.35 mJ | 7.97 / 4.82 N | 20.42 mNm |
| baseline_a | 18/18 | 18/18 | 15.21150000000673 s | 31.59 mm | 49.17 mJ | 8.23 / 4.15 N | 21.08 mNm |
| baseline_b | 17/18 | 16/18 | 7.564999999998527 s | 5.56 mm | 12.68 mJ | 6.63 / 4.05 N | 22.83 mNm |

## Paired counts

- both_success: 17
- left_only_success: 0
- right_only_success: 0
- both_failed: 1
- mcnemar_exact_two_sided_p: 1.0

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed, and capture evaluation still uses privileged MuJoCo contact labels; this is not hardware evidence.
