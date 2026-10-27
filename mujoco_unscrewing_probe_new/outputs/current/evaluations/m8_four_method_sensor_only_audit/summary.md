# M8 Fair Policy Benchmark

Efficiency medians are conditioned on successful offline-validated capture; force/torque p90 and hard stops use all trials.

| method | eventual success | first attempt | capture time median | successful contact path median | successful wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_posterior | 34/36 | 34/36 | 12.47100000000375 s | 14.61 mm | 29.87 mJ | 9.48 / 4.88 N | 22.33 mNm |
| baseline_a | 30/36 | 30/36 | 36.064499999990176 s | 77.07 mm | 135.71 mJ | 9.44 / 5.17 N | 23.03 mNm |
| baseline_b | 28/36 | 25/36 | 7.490999999998568 s | 5.36 mm | 13.27 mJ | 7.54 / 4.39 N | 21.34 mNm |
| baseline_c_random | 19/36 | 19/36 | 10.186000000000957 s | 20.82 mm | 32.90 mJ | 8.97 / 4.77 N | 21.86 mNm |

## Paired counts

- both_success: 27
- left_only_success: 7
- right_only_success: 1
- both_failed: 1
- mcnemar_exact_two_sided_p: 0.0703125

### Proposed vs random
- both_success: 18
- left_only_success: 16
- right_only_success: 1
- both_failed: 1
- mcnemar_exact_two_sided_p: 0.000274658203125

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. When enabled, the F/T bias/noise/coupling/delay model is a pre-hardware engineering prior rather than a measured sensor distribution. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed. Online handoff is sensor-only; MuJoCo geometry is used only for offline success scoring. This is not hardware evidence.
