# S4b base-anchored residual adapter: train + head-held-out-validation selection

Selected shared parameters: alpha=0.92, minimum FT concentration=0.60, maximum base-FT mean disagreement=90 deg.

Selection uses development train plus validation held out from FT gradients.
The adapter never reads calibration examples or any full-loop pose labels.
Its raw posterior can still be wrong; the gate only decides whether a
directed shortcut is admissible. A rejected shortcut falls back to the
existing finite-budget search. Full-loop benefit is not established here.

| size | role | raw state acc | raw control med/p90 | raw >90 | gate coverage | accepted p90 | accepted >90 |
|---|---|---:|---:|---:|---:|---:|---:|
| M10 | train | 0.889 | 4.61/13.79 deg | 0 | 0.931 | 11.27 deg | 0 |
| M10 | validation | 1.000 | 4.00/15.44 deg | 1 | 0.972 | 12.01 deg | 0 |
| M12 | train | 0.931 | 4.91/14.06 deg | 1 | 0.819 | 9.61 deg | 0 |
| M12 | validation | 0.861 | 3.59/19.58 deg | 0 | 0.861 | 15.37 deg | 0 |

M10 validation retains one raw >90 deg posterior outlier. It is not
magically corrected because base and FT agree on the wrong half-plane;
the deployable FT-concentration guard abstains on that sample, so the
accepted directed-shortcut outlier count is zero.
