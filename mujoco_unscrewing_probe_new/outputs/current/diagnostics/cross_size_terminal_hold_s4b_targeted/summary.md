# M10 commissioning-v2 terminal-hold targeted diagnostic

Development-only known failures: `m10_s3test_002`, `006`, and `007`.
Methods: privileged straight-to-center and S4b fine-tuned Proposed+ hybrid.

| method | strict CAD bottom | hard stops | terminal rejections | peak raw F | peak raw torque |
|---|---:|---:|---:|---:|---:|
| oracle | 3/3 | 0 | 0 | 10.32 N | 30.53 mNm |
| S4b FT | 2/3 | 0 | 0 | 10.32 N | 41.59 mNm |

`S4b FT / 006` ended in `search_radius_exhausted` before terminal entry.  It is
therefore an estimator/search failure, not evidence against the terminal fix.

## Event evidence

| pose/method | released old lateral command error | continuous depth samples | low-wrench dwell | result |
|---|---:|---:|---:|---|
| 002/oracle | 0.0838 mm | 118 | 100/100 | CAD bottom |
| 002/S4b FT | 0.0856 mm | 100 | 100/100 | CAD bottom |
| 006/oracle | 0.0272 mm | 113 | 100/100 | CAD bottom |
| 007/oracle | 0.1240 mm | 268 | 100/100 | CAD bottom |
| 007/S4b FT | 0.0111 mm | 281 | 100/100 | CAD bottom |

Every success used `cad_depth_low_wrench_dwell`.  The raw hard-wrench vector and
all terminal-verifier thresholds are unchanged.  The controller only performs
a measured-tip bumpless rebase and disables the axial admittance integrator
during the verified terminal dwell; raw safety monitoring continues.

## Claim boundary

This targeted run reuses known development failures and is not an untouched
evaluation.  Results from successive controller versions cannot be combined
into a single 12-pose rate.  The evidence supports a full-development oracle
rerun as the next gate, not a claim that M10 already achieves at least 11/12.
