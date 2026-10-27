# M8 Fair Policy Benchmark

| method | eventual success | first attempt | capture time median | contact path median | wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_posterior | 34/36 | 34/36 | 15.99550000000805 s | 21.72 mm | 45.72 mJ | 9.74 / 4.96 N | 26.20 mNm |
| baseline_a | 31/36 | 31/36 | 44.31900000002959 s | 82.85 mm | 151.37 mJ | 14.79 / 14.56 N | 65.22 mNm |
| baseline_b | 34/36 | 34/36 | 7.459499999998585 s | 5.52 mm | 13.50 mJ | 7.18 / 4.07 N | 23.68 mNm |

## Paired counts

- both_success: 32
- left_only_success: 2
- right_only_success: 2
- both_failed: 0
- mcnemar_exact_two_sided_p: 1.0

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed, and capture evaluation still uses privileged MuJoCo contact labels; this is not hardware evidence.
