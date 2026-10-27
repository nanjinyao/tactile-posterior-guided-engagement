# Hardware-easy-v2 geometry-conditioned simulation adaptation

Date: 2026-07-21

## Outcome

The M8 three-seed direction ensemble was adapted to the printed
`hardware-easy-v2` contact geometry while replaying the frozen strict domain.
The promoted artifact is:

`outputs/current/models/m8_hardware_easy_v2_robust_direction_v5/`

Promotion is conditional: the posterior may command only bounded transport
when the frozen APS/disagreement gate accepts it. Rejected fingerprints route
to a guarded local spiral or a new probe. The direction network is neither a
capture detector nor the terminal engagement verifier.

## Derived contact geometry

The source configs and XML scenes remain unchanged. Separate derived scenes
use a circularly bounded 72-sector socket shell, avoiding the collision wings
created by enlarged six-wedge approximations.

| size | wall AF | mouth AF | head diameter | peg tip chamfer | paired flat-normal guidance estimate |
|---|---:|---:|---:|---:|---:|
| M8 | 6.50 mm | 8.10 mm | 14.950 mm | 0.60 mm | 1.65 mm |
| M10 | 8.54 mm | 10.54 mm | 18.975 mm | 0.80 mm | 2.07 mm |
| M12 | 10.60 mm | 13.00 mm | 21.275 mm | 1.00 mm | 2.50 mm |

The last column is a yaw-aligned, one-dimensional geometric diagnostic. It is
not a guaranteed capture radius: friction, tilt, yaw, burrs, FDM ridges and
multi-point contact can prevent sliding. The 45-degree lead-in is also close
to self-locking at the upper simulated friction range.

## Data and supervision contract

The initial M8 easy-v2 set contained 288 scans. A failed unseen-angle audit
showed that four angles per fundamental 60-degree sector were insufficient,
especially near the stochastic capture boundary. The final dense set therefore
uses:

- 8 radius levels, including both sides of the capture transition;
- 8 local angles at 7.5-degree spacing over one 60-degree sector;
- all six C6 rotations;
- two nuisance repeats, with one ideal and one pre-hardware noise/delay prior.

This produced 768/768 valid simulation executions. Of these, 675 complete
fingerprints were eligible for direction learning; 93 captured or incomplete
traces were retained as mechanics-routing outcomes and excluded from direction
supervision.

The first three post-freeze audits failed and were then explicitly relabelled
as development replay. The final `fresh4` strict and easy sets were generated
after v5 fitting was fixed, contain 180 tasks per domain, and retain
`training_use=forbidden`.

## Adaptation

- Base: the existing P1v3 M8 three-seed ensemble.
- Trainable: last convolution, shared block, state head and direction head.
- Frozen: earlier encoder blocks, all batch-normalization running statistics,
  per-seed input normalization and the inherited APS maximum set-size cap.
- Domain objective: 0.40 strict replay + 0.60 hardware-easy-v2.
- Direction targets: circular soft labels; captured traces contribute no
  direction target.
- Final promotion audit samples seen during fitting: zero.

## Fresh4 promotion result

| model / domain | valid / total | median / p90 error | within 7.5 deg | gate accept | accepted within 7.5 deg | accepted >90 deg |
|---|---:|---:|---:|---:|---:|---:|
| base P1v3 / strict | 167 / 180 | 12.15 / 53.87 deg | 31.1% | 20.4% | 32.4% | 0 |
| base P1v3 / easy-v2 | 170 / 180 | 14.39 / 55.53 deg | 20.6% | 17.6% | 30.0% | 0 |
| v5 / strict | 167 / 180 | 1.85 / 4.79 deg | 95.8% | 90.4% | 100% | 0 |
| v5 / easy-v2 | 170 / 180 | 1.78 / 3.45 deg | 100% | 98.2% | 100% | 0 |

Both strict non-forgetting and easy-v2 promotion gates pass. Thirteen strict
and ten easy tasks were already captured or did not yield a complete
fingerprint; they are reported separately rather than counted as direction
predictions. Two unguarded strict predictions exceeded 90 degrees, but the
transport gate rejected both. This is why the guarded interface, not raw
posterior accuracy alone, is the promoted object.

## Claim boundary and next hardware step

This is geometry-conditioned MuJoCo adaptation, not real-hardware fine-tuning
and not full-loop robot evidence. The Bota measurement-origin transform,
sensor-to-tip lever, printed mass/COM/inertia, FDM stiffness/friction, latency
and real noise distribution remain unmeasured. Before command authority, log
shadow fingerprints on the printed M8 tool, compare channel conventions and
calibration, then fine-tune only from grouped real trials while keeping the
same capture/terminal separation.

No visualization, video, PPT or robot-control path was changed by this update.
