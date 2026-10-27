# M8 Strict 24-Pose Complete-Loop Audit

Strict success = CAD-bottom verified, no hard stop, and event time <= 130 s.
The task set is conditioned on ambiguous top contact; it is not the upstream visual/contact router.

| method | CAD-bottom | offline capture | false handoff | hard stop | safe abort | budget violation | bottom time | path | wear |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proposed | 1/2 | 1/2 | 0 | 0 | 0 | 0 | 9.34 s | 5.90 mm | 12.62 mJ |
| proposed_plus | 1/2 | 1/2 | 0 | 0 | 0 | 0 | 9.61 s | 5.84 mm | 7.35 mJ |
| outward | 1/2 | 2/2 | 0 | 0 | 0 | 0 | 32.04 s | 65.46 mm | 114.74 mJ |
| entropy_grid | 1/2 | 1/2 | 0 | 0 | 0 | 0 | 9.21 s | 4.64 mm | 0.34 mJ |
| direct_first | 1/2 | 1/2 | 0 | 0 | 0 | 0 | 8.86 s | 5.58 mm | 11.71 mJ |
| random | 0/2 | 0/2 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |
| oracle | 2/2 | 2/2 | 0 | 0 | 0 | 0 | 4.02 s | 4.20 mm | 6.78 mJ |

## Failure categories

- `proposed`: `{"cad_bottom_verified": 1, "search_radius_exhausted": 1}`
- `proposed_plus`: `{"cad_bottom_verified": 1, "search_radius_exhausted": 1}`
- `outward`: `{"cad_bottom_verified": 1, "capture_without_bottom": 1}`
- `entropy_grid`: `{"cad_bottom_verified": 1, "search_radius_exhausted": 1}`
- `direct_first`: `{"cad_bottom_verified": 1, "search_radius_exhausted": 1}`
- `random`: `{"search_radius_exhausted": 2}`
- `oracle`: `{"cad_bottom_verified": 2}`

## Claim boundary

Frozen simulation-only M8 complete-loop audit conditioned on the declared ambiguous top-contact annulus. Strict success requires sensor-only handoff, offline-validated capture, CAD-bottom terminal verification, no hard stop, and completion within the shared 130 s active budget. It does not establish hardware transfer, passivity, motor coupling, threaded unscrewing, extraction, or universal superiority. Oracle is an upper-bound diagnostic; direct_first is not a physically reset retry baseline.
