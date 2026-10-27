# M8 Fair Policy Benchmark

Efficiency medians are conditioned on successful offline-validated capture; force/torque p90 and hard stops use all trials.

| method | eventual success | first attempt | capture time median | successful contact path median | successful wear median | raw / filtered force p90 | torque p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| proposed_posterior | 3/4 | 3/4 | 15.558000000007523 s | 21.66 mm | 36.25 mJ | 6.85 / 5.09 N | 21.74 mNm |
| baseline_a | 3/4 | 3/4 | 23.146999999991404 s | 50.27 mm | 56.36 mJ | 6.92 / 4.74 N | 23.79 mNm |
| baseline_b | 4/4 | 4/4 | 7.524999999998549 s | 5.58 mm | 13.39 mJ | 6.51 / 4.07 N | 19.82 mNm |
| baseline_c_entropy | 3/4 | 3/4 | 23.090499999991536 s | 5.57 mm | 1.50 mJ | 8.70 / 6.11 N | 47.15 mNm |

## Paired counts

- both_success: 3
- left_only_success: 0
- right_only_success: 1
- both_failed: 0
- mcnemar_exact_two_sided_p: 1.0

### Proposed vs entropy-grid
- both_success: 3
- left_only_success: 0
- right_only_success: 0
- both_failed: 1
- mcnemar_exact_two_sided_p: 1.0

## Boundary

Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact nuisances are unseen Sobol samples. When enabled, the F/T bias/noise/coupling/delay model is a pre-hardware engineering prior rather than a measured sensor distribution. Proposed and Baseline B share an identical first fingerprint and frozen learner. Baseline-B reset travel is not simulated or timed. Online handoff is sensor-only; MuJoCo geometry is used only for offline success scoring. This is not hardware evidence. Baseline C uses a simulation-calibrated CAD/F-T likelihood that must be recalibrated on iiwa hardware.
