# M8 P2b In-Domain Coverage Benchmark

Updated: 2026-07-10T03:04:11.702940+00:00

| method | success | success time median / p90 | contact path median | work proxy median | force / torque p90 |
|---|---:|---:|---:|---:|---:|
| learned | 1/1 (1.000) | 33.51 / 33.51 s | 60.78 mm | 109.48 mJ | 7.20 N / 20.48 mNm |
| baseline | 1/1 (1.000) | 40.61 / 40.61 s | 87.39 mm | 147.89 mJ | 7.18 N / 19.24 mNm |
| oracle | 1/1 (1.000) | 1.47 / 1.47 s | 3.27 mm | 5.85 mJ | 6.45 N / 21.31 mNm |

## Coverage Contract

- visual acceptance radius: 4.90 mm
- baseline coverage radius/path/time: 4.65 mm / 169.65 mm / 75.40 s
- derived program guard: 118.90 s; not a success threshold

## Paired Result

- complete_pose_triplets: 1
- both_learned_and_baseline_success: 1
- learned_only_success: 0
- baseline_only_success: 0
- both_failed: 0
- learned_minus_baseline_time_median_s_on_both_success: -7.100000000033901
- learned_faster_rate_on_both_success: 1.0

## Boundary

Nominal-dynamics M8 simulation inside a declared 4.90 mm visual acceptance set. Search failure means coverage exhausted or a safety stop; the derived time budget is only a program guard. Capture still uses privileged MuJoCo tags, and no nuisance or hardware result is included.
