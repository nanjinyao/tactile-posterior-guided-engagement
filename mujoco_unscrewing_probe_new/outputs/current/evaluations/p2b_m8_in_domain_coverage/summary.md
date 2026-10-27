# M8 P2b In-Domain Coverage Benchmark

Updated: 2026-07-10T03:26:53.519438+00:00

| method | success | success time median / p90 | contact path median | work proxy median | force / torque p90 |
|---|---:|---:|---:|---:|---:|
| learned | 21/30 (0.700) | 15.08 / 33.51 s | 31.29 mm | 59.77 mJ | 17.78 N / 77.04 mNm |
| baseline | 24/30 (0.800) | 20.48 / 40.61 s | 50.37 mm | 86.02 mJ | 17.39 N / 65.07 mNm |
| oracle | 30/30 (1.000) | 1.23 / 2.10 s | 2.68 mm | 4.31 mJ | 6.69 N / 23.47 mNm |

## Coverage Contract

- visual acceptance radius: 4.90 mm
- baseline coverage radius/path/time: 4.65 mm / 169.65 mm / 75.40 s
- derived program guard: 175.45 s; not a success threshold

## Paired Result

- complete_pose_triplets: 30
- both_learned_and_baseline_success: 21
- learned_only_success: 0
- baseline_only_success: 3
- both_failed: 6
- learned_minus_baseline_time_median_s_on_both_success: -3.656999999991477
- learned_time_median_s_on_both_success: 15.076000000006934
- baseline_time_median_s_on_both_success: 17.114500000005464
- learned_faster_rate_on_both_success: 0.6190476190476191
- learned_minus_baseline_path_median_mm_on_both_success: -19.088804475310006
- learned_path_median_mm_on_both_success: 19.75830978045606
- baseline_path_median_mm_on_both_success: 35.23759169342798
- learned_shorter_path_rate_on_both_success: 1.0
- learned_minus_baseline_work_median_mj_on_both_success: -26.415557488012084
- learned_work_median_mj_on_both_success: 38.56103737021374
- baseline_work_median_mj_on_both_success: 61.05175195736035
- learned_lower_work_rate_on_both_success: 0.9523809523809523

## Radius Stratification

| radius | learned | baseline | oracle |
|---|---:|---:|---:|
| 1.518_mm | 6/6 | 6/6 | 6/6 |
| 2.171_mm | 6/6 | 6/6 | 6/6 |
| 2.823_mm | 5/6 | 6/6 | 6/6 |
| 3.404_mm | 4/6 | 6/6 | 6/6 |
| 4.875_mm | 0/6 | 0/6 | 6/6 |

## Boundary

Nominal-dynamics M8 simulation inside a declared 4.90 mm visual acceptance set. Search failure means coverage exhausted, a safety stop, or the non-performance program guard. Capture still uses privileged MuJoCo tags, and no nuisance or hardware result is included.
