# M8 Fair Policy Benchmark

| method | eventual success | first attempt | capture time median | contact path median | wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_posterior | 36/36 | 36/36 | 14.056500000005688 s | 17.72 mm | 34.01 mJ | 9.48 / 4.88 N | 22.33 mNm |
| baseline_a | 35/36 | 35/36 | 42.83800000002252 s | 84.09 mm | 136.87 mJ | 9.44 / 5.17 N | 23.44 mNm |
| baseline_b | 29/36 | 25/36 | 7.506999999998559 s | 6.02 mm | 14.19 mJ | 7.53 / 4.39 N | 21.34 mNm |

## Paired counts

- both_success: 29
- left_only_success: 7
- right_only_success: 0
- both_failed: 0
- mcnemar_exact_two_sided_p: 0.015625

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. When enabled, the F/T bias/noise/coupling/delay model is a pre-hardware engineering prior rather than a measured sensor distribution. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed, and capture evaluation still uses privileged MuJoCo contact labels; this is not hardware evidence.
