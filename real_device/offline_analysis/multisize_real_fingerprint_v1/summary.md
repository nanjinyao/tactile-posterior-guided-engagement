# M8/M10/M12 real conical fingerprint offline QC v1

Generated: `2026-07-28T10:42:32.203134+00:00`

## Verdict

- All three explicitly allowlisted fresh runs pass immutable-data and acquisition QC.
- The frozen simulation feature contract does **not** transfer directly to this first real dataset.
- Real lateral-force traces contain useful circular direction structure, but the current posterior remains multimodal and has one catastrophic top-1 holdout alias.
- CAD can supply an interpretable mouth-containment distance after a center posterior is formed; these samples do not calibrate physical chamfer capture.
- The 12 holdout points were revealed once for this v1 development assessment. They are now consumed and must not be used for further feature selection, fitting, or threshold tuning.

## Data authority

| size | run | clean | holdout | quarantine | sentinel | hash checks |
|---|---|---:|---:|---:|---:|---:|
| M12 | `M12_conical_fingerprints_26pt_20260728_175502_318` | 16 | 4 | 4 | 2 | 104 |
| M10 | `M10_conical_fingerprints_26pt_20260728_152808_877` | 16 | 4 | 4 | 2 | 104 |
| M8 | `M8_conical_fingerprints_26pt_20260728_161249_458` | 16 | 4 | 4 | 2 | 104 |

## Direction observability

| model | median | p90 | <=30 deg | >90 deg |
|---|---:|---:|---:|---:|
| simulation-inspired Fz/Tx/Ty/z 48-D Fourier ridge | 90.61 | 124.12 | 0.167 | 0.500 |
| real Fx/Fy circular template top-1 | 19.56 | 42.32 | 0.833 | 0.083 |
| real Fx/Fy best-of-top-2 oracle posterior coverage | 15.09 | 21.00 | 1.000 | 0.000 |
| frozen simulation CNN zero-shot on real | 99.76 | 157.70 | 0.182 | 0.530 |

The real template is conditioned on the registered CAD contact state to isolate direction observability. A deployable model must replace that label with a calibrated state/CAD likelihood.

This is a development holdout reveal, not a final untouched audit. Freeze the next adapter before collecting a new physical audit set.
The best-of-top-2 row only asks whether either posterior mode covers the truth; it is not an executable online direction selector.

## CAD geometric distance envelope (posterior fusion pending)

| size | feasible center hex AF [mm] | oracle entry min/median/max [mm] | top-1 ray hit | top-2 any hit |
|---|---:|---:|---:|---:|
| M12 | 5.000 | 3.001 / 6.044 / 9.843 | 0.500 | 0.750 |
| M10 | 4.140 | 3.284 / 5.323 / 8.018 | 0.500 | 0.750 |
| M8 | 3.300 | 2.645 / 4.077 / 6.024 | 0.500 | 0.750 |

These are geometric mouth-containment distances. They are not validated chamfer-capture distances because no acquisition sample contains a capture/landing outcome.

## Next training step

1. Keep the simulation encoder frozen as a teacher/initialization only.
2. Fit a small real-domain common-mode, frame, and phase adapter from clean samples.
3. Train/calibrate state and circular-direction heads without using holdout, sentinel, or quarantine.
4. Preserve a multimodal direction posterior and fuse it with exact CAD plus an independently calibrated visual center prior.
5. Use short full-loop trials to calibrate physical capture margin; do not learn continuous distance from these 48 clean scans.
