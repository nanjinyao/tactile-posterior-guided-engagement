# M8 Fair Policy Benchmark

| method | eventual success | first attempt | capture time median | contact path median | wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_posterior | 28/30 | 28/30 | 15.554500000007518 s | 21.77 mm | 39.86 mJ | 8.07 / 4.82 N | 23.07 mNm |
| baseline_a | 29/30 | 29/30 | 18.465500000002315 s | 38.94 mm | 61.14 mJ | 8.42 / 4.72 N | 22.82 mNm |
| baseline_b | 29/30 | 29/30 | 7.3599999999986405 s | 5.27 mm | 12.85 mJ | 6.89 / 4.13 N | 23.73 mNm |

## Paired counts

- both_success: 28
- left_only_success: 0
- right_only_success: 1
- both_failed: 1
- mcnemar_exact_two_sided_p: 1.0

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed, and capture evaluation still uses privileged MuJoCo contact labels; this is not hardware evidence.
