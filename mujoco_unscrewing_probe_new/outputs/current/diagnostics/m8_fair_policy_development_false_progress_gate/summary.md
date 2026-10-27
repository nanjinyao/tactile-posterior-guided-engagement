# M8 Fair Policy Benchmark

| method | eventual success | first attempt | capture time median | contact path median | wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_posterior | 36/36 | 36/36 | 15.625500000007605 s | 21.20 mm | 39.79 mJ | 8.55 / 4.94 N | 22.39 mNm |
| baseline_a | 35/36 | 35/36 | 14.464500000006186 s | 29.48 mm | 51.13 mJ | 8.52 / 4.98 N | 22.23 mNm |
| baseline_b | 34/36 | 34/36 | 7.376249999998631 s | 5.38 mm | 13.40 mJ | 6.44 / 4.11 N | 23.29 mNm |

## Paired counts

- both_success: 34
- left_only_success: 2
- right_only_success: 0
- both_failed: 0
- mcnemar_exact_two_sided_p: 0.5

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed, and capture evaluation still uses privileged MuJoCo contact labels; this is not hardware evidence.
