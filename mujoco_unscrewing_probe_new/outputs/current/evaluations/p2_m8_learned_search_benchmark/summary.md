# M8 P2 Learned Search Pilot

Updated: 2026-07-10T02:37:12.538554+00:00

| method | success | penalized time median / p90 | contact path median | wear proxy median | force / torque p90 |
|---|---:|---:|---:|---:|---:|
| learned | 7/18 (0.389) | 30.00 / 30.00 s | 30.51 mm | 55.270 mJ | 13.00 N / 64.13 mNm |
| baseline | 6/18 (0.333) | 30.00 / 30.00 s | 56.08 mm | 91.253 mJ | 16.64 N / 62.16 mNm |
| oracle | 18/18 (1.000) | 2.09 / 3.07 s | 4.71 mm | 7.851 mJ | 11.99 N / 60.29 mNm |

Learned direction within 7.5 deg: 1.000; median / p90 error: 0.00 / 0.00 deg.

## Paired Result

- learned - baseline penalized time median: +0.00 s
- learned faster than baseline: 0.333
- learned - oracle penalized time median: +26.93 s

## Visual-Prior Strata

Synthetic visual 95% radius: 4.90 mm.

| stratum | learned | baseline | oracle |
|---|---:|---:|---:|
| inside_synthetic_visual_95 | 7/9 | 6/9 | 9/9 |
| outside_synthetic_visual_95 | 0/9 | 0/9 | 9/9 |

## Boundary

Small held-out-pose nominal-dynamics M8 simulation pilot. Learned probe cost is included; oracle uses true center only as a lower bound. The capture/progress verifier still uses privileged MuJoCo contact tags, and the visual covariance is synthetic rather than measured.
