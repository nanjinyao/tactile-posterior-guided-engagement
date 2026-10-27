# M8 Jin-2021-Style Proxy vs Improved Fingerprint

This is a fresh paired MuJoCo comparison. `Jin-2021-style` means an M8 proxy with locked paper-inspired preprocessing/CNN structure, not a faithful reproduction.

## Structured fresh P1v6 direction stress (mixed result)

| estimator | n | error median / p90 | within 7.5 deg | within 15 deg | >90 deg | common 12-sector accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Improved fingerprint | 180 | 10.74 / 32.74 deg | 0.272 | 0.800 | 0.039 | 0.300 |
| Jin-2021-style proxy | 180 | 13.00 / 17.00 deg | 0.494 | 0.694 | 0.011 | 0.694 |

Paired base-pose cluster bootstrap for `median(|e_improved| - |e_proxy|)`: -0.805 deg, 95% percentile CI [-2.139, 0.663] deg; 60 `base_pose_id` clusters, 20000 resamples, seed 20260734. Negative favors the improved estimator. Each cluster is one state/radius/local-angle/physical-sector pose and retains all three nuisance repeats. The 95% interval includes zero; this paired-median P1v6 contrast does not establish a direction-error advantage.

Each P1v6 NPZ phase sequence was loaded once and passed unchanged to both frozen runtimes. This structured radial/sector stress result is mixed: the improved estimator has the lower median and higher within-15-deg rate, while the proxy has the better p90, within-7.5-deg, catastrophic-tail, and common-12-sector metrics. It is not evidence of uniformly better direction accuracy.

## Fresh strict-48 operational direction distribution

Operational errors are recomputed from each frozen `RunResult.control_direction_deg` against `RunResult.true_direction_deg` on the same 48 noisy poses used by the direct-first task comparison. `control_direction_deg` is the direction actually executed by the ray. This is distinct from the P1v6 structured stress set; no simulator rollout was rerun.

| estimator | n | executed-control error median / p90 | within 7.5 deg | within 15 deg | >90 deg |
|---|---:|---:|---:|---:|---:|
| Improved fingerprint | 48 | 3.34 / 20.55 deg | 0.792 | 0.875 | 0.000 |
| Jin-2021-style proxy | 48 | 8.18 / 17.36 deg | 0.458 | 0.833 | 0.062 |

### Top-class diagnostic (not the executed ray)

The frozen `RunResult.direction_error_deg` is retained only as a diagnostic of `predicted_direction_deg` (the ensemble top class / `top_direction_deg`). It must not be read as strict-48 controller performance.

| estimator | n | top-class error median / p90 | within 7.5 deg | within 15 deg | >90 deg |
|---|---:|---:|---:|---:|---:|
| Improved fingerprint | 48 | 3.87 / 12.16 deg | 0.729 | 0.938 | 0.021 |
| Jin-2021-style proxy | 48 | 8.18 / 17.36 deg | 0.458 | 0.833 | 0.062 |

## Fresh paired strict direct-first task layer

All rows use the same strict controller and accumulated 130 s active-time budget.

| estimator | first-ray capture | strict CAD-bottom | hard stop | safe termination | budget violations |
|---|---:|---:|---:|---:|---:|
| Improved fingerprint | 37/48 | 32/48 | 1/48 | 47/48 | 0 |
| Jin-2021-style proxy | 24/48 | 18/48 | 0/48 | 48/48 | 0 |

## Exact paired tests

- `first_ray_capture`: both=21, improved-only=16, Jin-proxy-only=3, neither=8, exact McNemar p=0.00442505.
- `strict_cad_bottom`: both=16, improved-only=16, Jin-proxy-only=2, neither=14, exact McNemar p=0.00131226.
- `hard_stop`: both=0, improved-only=1, Jin-proxy-only=0, neither=47, exact McNemar p=1.
- `safe_termination`: both=47, improved-only=0, Jin-proxy-only=1, neither=0, exact McNemar p=1.

## Claim boundary

Fresh paired MuJoCo M8 evidence only. The Jin-style arm is a frozen engineering proxy, not a faithful paper reproduction. Because the M8 geometry, labels, data distribution, and controller differ, these numbers cannot be compared directly with the paper's reported 97.4% square classification accuracy. No hardware or universal superiority claim is licensed.
