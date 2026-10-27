# M8 Five-Method Closed-Loop Audit: Direct x3 + Shared R3 Funnel

Updated: 2026-07-15T13:21:44.889944+00:00

Every method shares the same 130 s, zero-extension, sensor-only capture,
tip-RCC, maximum-three post-capture recovery, and CAD-bottom verifier.
The two direct methods additionally receive at most three same-rollout
rays; every miss pays for unload/lift/shift/full re-probe.

| method | CAD-bottom | capture | hard stop | median bottom time | median XY path | median wear | direct attempts | R3 recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed | 20/24 | 23/24 | 3 | 13.87 s | 14.52 mm | 28.48 mJ | n/a | 5 |
| Proposed+ | 19/24 | 23/24 | 4 | 13.74 s | 6.76 mm | 9.54 mJ | n/a | 12 |
| Pure Outward | 19/24 | 23/24 | 4 | 34.63 s | 71.27 mm | 120.38 mJ | n/a | 11 |
| Improved fingerprint + direct x3 | 17/24 | 19/24 | 3 | 9.22 s | 6.07 mm | 13.67 mJ | 1.0 | 4 |
| Jin-style 3-channel + direct x3 | 16/24 | 18/24 | 3 | 8.96 s | 5.67 mm | 13.22 mJ | 1.0 | 3 |

## Direct x3 estimator/interface comparison

| estimator | first-ray capture | any-ray capture | CAD-bottom | first-probe control error median / p90 | within 15 deg |
|---|---:|---:|---:|---:|---:|
| Improved fingerprint + direct x3 | 18/24 | 19/24 | 17/24 | 4.75 / 31.45 deg | 0.792 |
| Jin-style 3-channel + direct x3 | 16/24 | 19/24 | 16/24 | 7.41 / 13.79 deg | 0.917 |

## Paired improved-versus-Jin direct x3

- strict bottom: `{"both_failed": 4, "both_success": 13, "exact_mcnemar_p": 1.0, "left_only_success": 4, "right_only_success": 3}`
- first-ray capture: `{"both_false": 4, "both_true": 14, "exact_mcnemar_p": 0.6875, "improved_only_true": 4, "jin_only_true": 2, "metric": "first_ray_capture"}`
- any-ray capture: `{"both_false": 2, "both_true": 16, "exact_mcnemar_p": 1.0, "improved_only_true": 3, "jin_only_true": 3, "metric": "any_ray_capture"}`
- first-probe within 15 deg: `{"both_false": 1, "both_true": 18, "exact_mcnemar_p": 0.375, "improved_only_true": 1, "jin_only_true": 4, "metric": "first_probe_within_15_deg"}`

## Failure categories

- `proposed`: `{"cad_bottom_verified": 20, "hard_stop": 3, "search_radius_exhausted": 1}`
- `proposed_plus`: `{"cad_bottom_verified": 19, "hard_stop": 4, "search_radius_exhausted": 1}`
- `outward`: `{"cad_bottom_verified": 19, "funnel_safe_abort:terminal_dwell_rejected": 1, "hard_stop": 4}`
- `improved_direct_retry3`: `{"cad_bottom_verified": 17, "hard_stop": 3, "search_radius_exhausted": 4}`
- `jin_direct_retry3`: `{"cad_bottom_verified": 16, "hard_stop": 3, "search_radius_exhausted": 5}`

## Claim boundary

Fresh paired M8 MuJoCo evidence under one 130 s complete-loop contract. The Jin arm is an M8 paper-style engineering proxy, not a faithful reproduction. Three rays are a bounded same-rollout recovery policy, not three independent trials. The shared R3 funnel can improve some recoverable contacts but does not guarantee CAD-bottom completion. No hardware, passivity, motor-coupling, or unscrewing claim follows.
