# M8 Fair Policy Benchmark

| method | eventual success | first attempt | capture time median | contact path median | wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_1p5 | 8/18 | 8/18 | 26.738999999983033 s | 38.85 mm | 66.63 mJ | 25.03 / 5.95 N | 158.47 mNm |
| proposed_posterior | 7/18 | 7/18 | 20.266499999998118 s | 26.64 mm | 42.37 mJ | 25.05 / 6.34 N | 152.86 mNm |
| baseline_a | 17/18 | 17/18 | 39.939000000008676 s | 88.01 mm | 136.66 mJ | 13.19 / 10.40 N | 44.19 mNm |
| baseline_b | 17/18 | 16/18 | 7.564999999998527 s | 5.63 mm | 13.04 mJ | 8.10 / 4.09 N | 22.83 mNm |

## Paired counts

- both_success: 7
- left_only_success: 0
- right_only_success: 10
- both_failed: 1
- mcnemar_exact_two_sided_p: 0.001953125

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed, and capture evaluation still uses privileged MuJoCo contact labels; this is not hardware evidence.
