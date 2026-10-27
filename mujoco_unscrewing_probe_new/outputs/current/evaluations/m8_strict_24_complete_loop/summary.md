# M8 Strict 24-Pose Complete-Loop Audit

Strict success = sensor handoff + offline capture validation + a declared CAD-bottom terminal dwell, no hard stop/extension, and active time <= 130 s.
The task set is conditioned on ambiguous top contact; it is not the upstream visual/contact router.

| method | role | strict CAD-bottom | offline capture | false handoff | hard stop | safe abort | budget violation | bottom time | path | wear |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proposed | paper-facing complete method | 19/24 | 23/24 | 0 | 0 | 0 | 0 | 13.81 s | 14.29 mm | 27.94 mJ |
| proposed_plus | adaptive research branch; no superiority claim | 19/24 | 23/24 | 0 | 1 | 0 | 0 | 13.74 s | 6.76 mm | 9.54 mJ |
| outward | geometry-only comparator | 19/24 | 23/24 | 1 | 0 | 0 | 0 | 34.63 s | 71.27 mm | 120.38 mJ |
| entropy_grid | strong transparent active-perception diagnostic | 22/24 | 23/24 | 0 | 0 | 0 | 0 | 17.53 s | 14.07 mm | 0.85 mJ |
| direct_first | mechanism diagnostic; no physical retry/reset claim | 15/24 | 18/24 | 0 | 0 | 0 | 0 | 9.11 s | 5.58 mm | 13.39 mJ |
| random | weak-search lower diagnostic | 10/24 | 12/24 | 1 | 1 | 0 | 0 | 11.46 s | 20.97 mm | 28.91 mJ |
| oracle | privileged upper-bound diagnostic | 22/24 | 24/24 | 0 | 0 | 0 | 0 | 3.16 s | 3.72 mm | 5.19 mJ |

## Failure categories

- `proposed`: `{"cad_bottom_verified": 19, "capture_without_bottom": 4, "search_radius_exhausted": 1}`
- `proposed_plus`: `{"cad_bottom_verified": 19, "capture_without_bottom": 3, "hard_stop": 1, "search_radius_exhausted": 1}`
- `outward`: `{"cad_bottom_verified": 19, "capture_without_bottom": 4, "sensor_capture_false_positive": 1}`
- `entropy_grid`: `{"cad_bottom_verified": 22, "capture_without_bottom": 1, "search_radius_exhausted": 1}`
- `direct_first`: `{"cad_bottom_verified": 15, "capture_without_bottom": 3, "search_radius_exhausted": 6}`
- `random`: `{"cad_bottom_verified": 10, "capture_without_bottom": 2, "hard_stop": 1, "search_radius_exhausted": 11}`
- `oracle`: `{"cad_bottom_verified": 22, "capture_without_bottom": 2}`

## Claim boundary

Frozen simulation-only M8 complete-loop audit conditioned on the declared ambiguous top-contact annulus. Strict success requires sensor-only handoff, offline-validated capture, CAD-bottom terminal verification, no hard stop, and completion within the shared 130 s active budget. It does not establish hardware transfer, passivity, motor coupling, threaded unscrewing, extraction, or universal superiority. Oracle is an upper-bound diagnostic; direct_first is not a physically reset retry baseline.
