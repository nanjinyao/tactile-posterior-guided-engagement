# M8 Fair Policy Benchmark

| method | eventual success | first attempt | capture time median | contact path median | wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_posterior | 7/8 | 7/8 | 12.047000000003232 s | 13.96 mm | 26.82 mJ | 9.53 / 5.07 N | 21.34 mNm |
| baseline_a | 5/8 | 5/8 | 33.96549999998015 s | 71.69 mm | 113.03 mJ | 11.99 / 5.20 N | 48.84 mNm |
| baseline_b | 8/8 | 6/8 | 7.817499999998387 s | 6.11 mm | 13.37 mJ | 8.24 / 4.32 N | 19.80 mNm |
| baseline_c_random | 4/8 | 4/8 | 5.527999999999656 s | 51.73 mm | 78.13 mJ | 8.74 / 4.57 N | 19.84 mNm |

## Paired counts

- both_success: 7
- left_only_success: 0
- right_only_success: 1
- both_failed: 0
- mcnemar_exact_two_sided_p: 1.0

### Proposed vs random
- both_success: 4
- left_only_success: 3
- right_only_success: 0
- both_failed: 1
- mcnemar_exact_two_sided_p: 0.25

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. When enabled, the F/T bias/noise/coupling/delay model is a pre-hardware engineering prior rather than a measured sensor distribution. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed. Online handoff is sensor-only; MuJoCo geometry is used only for offline success scoring. This is not hardware evidence.
