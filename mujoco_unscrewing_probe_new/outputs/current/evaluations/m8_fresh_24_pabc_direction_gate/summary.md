# M8 Fair Policy Benchmark

Efficiency medians are conditioned on successful offline-validated capture; force/torque p90 and hard stops use all trials.

| method | eventual success | first attempt | capture time median | successful total/contact path | successful wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_posterior | 23/24 | 23/24 | 12.581000000003884 s | 14.69 / 14.66 mm | 29.64 mJ | 8.63 / 5.13 N | 24.60 mNm |
| baseline_a | 23/24 | 23/24 | 33.27949999997688 s | 70.71 / 70.71 mm | 124.80 mJ | 8.25 / 4.83 N | 22.06 mNm |
| baseline_b | 22/24 | 18/24 | 7.931749999998324 s | 6.18 / 6.18 mm | 14.26 mJ | 7.36 / 4.39 N | 23.44 mNm |
| baseline_c_entropy | 23/24 | 23/24 | 12.144500000003351 s | 6.82 / 2.18 mm | 0.62 mJ | 8.02 / 3.44 N | 10.74 mNm |

## Paired counts

- both_success: 21
- left_only_success: 2
- right_only_success: 1
- both_failed: 0
- mcnemar_exact_two_sided_p: 1.0

### Proposed vs entropy-grid
- both_success: 22
- left_only_success: 1
- right_only_success: 1
- both_failed: 0
- mcnemar_exact_two_sided_p: 1.0

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. When enabled, the F/T bias/noise/coupling/delay model is a pre-hardware engineering prior rather than a measured sensor distribution. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed. Online handoff is sensor-only; MuJoCo geometry is used only for offline success scoring. This is not hardware evidence. Baseline C uses a simulation-calibrated CAD/F-T likelihood that must be recalibrated on iiwa hardware.
