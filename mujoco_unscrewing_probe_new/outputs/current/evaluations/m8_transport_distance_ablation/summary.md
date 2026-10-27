# M8 Posterior Transport Distance Ablation

| policy | transport | local radius | bottom | time | XY path | wear proxy | raw / filtered peak force | peak torque |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cap_1p5_mm | 1.500 mm | 2.945 mm | True | 28.07 s | 45.56 mm | 83.76 mJ | 7.07 / 4.62 N | 20.89 mNm |
| cap_2p0_mm | 2.000 mm | 2.445 mm | True | 22.19 s | 32.34 mm | 61.54 mJ | 7.09 / 4.65 N | 21.56 mNm |
| posterior_limited | 2.257 mm | 2.188 mm | True | 17.26 s | 21.82 mm | 43.21 mJ | 7.01 / 4.52 N | 22.60 mNm |

The posterior-limited row removes only the fixed absolute cap; the lower-quantile transport rule and the residual 95% spiral coverage bound remain active.
