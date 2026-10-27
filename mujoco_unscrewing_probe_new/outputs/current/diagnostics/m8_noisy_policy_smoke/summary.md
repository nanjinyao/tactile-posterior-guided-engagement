# M8 Fair Policy Benchmark

| method | eventual success | first attempt | capture time median | contact path median | wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_posterior | 2/2 | 2/2 | 13.704000000005003 s | 17.59 mm | 33.89 mJ | 8.40 / 4.75 N | 22.46 mNm |
| baseline_a | 2/2 | 2/2 | 28.88900000002144 s | 61.88 mm | 88.99 mJ | 8.38 / 5.07 N | 26.67 mNm |
| baseline_b | 2/2 | 2/2 | 7.186249999998736 s | 5.24 mm | 14.44 mJ | 6.35 / 4.06 N | 19.36 mNm |

## Paired counts

- both_success: 2
- left_only_success: 0
- right_only_success: 0
- both_failed: 0
- mcnemar_exact_two_sided_p: 1.0

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed, and capture evaluation still uses privileged MuJoCo contact labels; this is not hardware evidence.
