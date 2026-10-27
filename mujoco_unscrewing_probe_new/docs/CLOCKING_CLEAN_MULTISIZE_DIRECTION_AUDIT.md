# Clocking-clean multisize direction-model audit

Date: 2026-07-24 (Asia/Singapore)

## Decision

The clean shared M8/M10/M12 candidate
`hardware_easy_v2_multisize_clocking_clean_direction_v7_source_preserving`
is **rejected for global simulation promotion**.

The untouched `fresh8` audit passes the M8 hardware-easy-v2, M10
hardware-easy-v2, and M12 hardware-easy-v2 domains, but fails the M8 source
domain. In that source domain the model accepts two direction errors greater
than 90 degrees and does not preserve the previously frozen M8-v5 behavior.
No promotion-candidate file was created and the real-device deployment
manifest was not modified by this audit.

The narrower M12 result is suitable only for an explicitly labelled
`DEVELOPMENT_SHADOW_ONLY` real-trace diagnostic. It is not evidence for
hardware control, fine-tuning success, capture, insertion, terminal
verification, or a complete task loop.

## Clean lineage

The v7 trainer starts from the audited clean M8-v2 base and consumes only the
geometry-audited canonical simulation datasets. Its adaptation report records:

- `easy_development_datasets_consumed = []` for M8, M10, and M12;
- `m8_source_development_datasets_consumed = []`;
- `test_samples_seen = 0`;
- `final_audit_run = false`;
- M8-source domain weight `0.72`, each easy domain weight
  `0.09333333333333334`;
- tail-only adaptation, learning rate `2e-5`, L2-SP weight `1.0`.

The frozen lineage audit is `PASS`. The canonical calibration table is
development evidence only:

| domain | samples | median / p90 error | accepted / accepted within 7.5 deg | accepted >90 deg |
|---|---:|---:|---:|---:|
| M8 source | 192 | 0.43 / 1.37 deg | 1.000 / 1.000 | 0 |
| M8 easy-v2 | 130 | 1.22 / 4.04 deg | 0.992 / 1.000 | 0 |
| M10 easy-v2 | 128 | 0.66 / 2.87 deg | 0.984 / 1.000 | 0 |
| M12 easy-v2 | 140 | 1.19 / 4.34 deg | 0.950 / 0.992 | 0 |

`fresh7` is preserved as immutable failure/development evidence. Its labels
were not added to the training set, normalization, calibration, or gate fit.
Because redesign followed the fresh7 failure, fresh7 is not claimed as the
final independent test. The post-selection `fresh8` rings, phases, nuisance
seeds, trial identifiers, and orbit identifiers were frozen before sampling
and have zero overlap with the canonical training groups.

## Frozen fresh8 data

Each domain contains 180 completed task outcomes and zero sampler failures.
All four sampling contracts set `training_use` to `forbidden`.

| domain | frozen stage | sampling-contract SHA-256 |
|---|---|---|
| M8 source | `M8_source_multisize_joint_fresh8_untouched` | `14690edae5e166550dfc8f11e9a10cabfdc21c046cafdb3c1ff4bb37f3c912d7` |
| M8 easy-v2 | `M8_hardware_easy_v2_multisize_fresh8_untouched` | `0e208c79dc83e8557c408b2b154cd74ce51e9789bb6d4a8f3a14b7dc06e79867` |
| M10 easy-v2 | `M10_hardware_easy_v2_multisize_fresh8_untouched` | `93a1453bb9f94eb8448d7ecaf08f9e88bd392bfe89c2d09c33ea9b9c3f3dea9d` |
| M12 easy-v2 | `M12_hardware_easy_v2_multisize_fresh8_untouched` | `9c719e928b8a9ab8f8135e6c92a928d11104ccd29f0e786c15bfd007658825d2` |

Captured or incomplete conical scans are mechanics-routing outcomes. They are
not assigned a direction label or direction error.

## Untouched final result

The evaluator was run once after all four fresh8 datasets reached 180/180.
There was no fitting, calibration, gate selection, or hyperparameter change
after this result was observed.

| domain | valid / total | routed capture / incomplete | median / p90 error | within 7.5 deg | APS coverage / mean size | accepted / accepted within 7.5 deg | accepted >90 deg | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M8 source | 171 / 180 | 9 | 4.47 / 174.67 deg | 0.725 | 0.889 / 3.68 | 0.690 / 0.924 | 2 | **FAIL** |
| M8 easy-v2 | 167 / 180 | 13 | 0.84 / 3.05 deg | 0.994 | 1.000 / 2.61 | 0.964 / 1.000 | 0 | PASS |
| M10 easy-v2 | 163 / 180 | 17 | 0.85 / 2.37 deg | 1.000 | 1.000 / 2.75 | 0.994 / 1.000 | 0 | PASS |
| M12 easy-v2 | 167 / 180 | 13 | 0.93 / 3.29 deg | 0.994 | 1.000 / 2.65 | 0.988 / 0.994 | 0 | PASS |

All four domains have zero hard-stop outcomes in this simulation audit.

The M8-source radial screen localizes the failure:

- radial levels 0, 1, and 2 pass;
- radial level 3 accepts 19 samples, of which 7 exceed 7.5 degrees;
- radial level 4 accepts only 2 of 36 samples, and both accepted samples exceed
  7.5 degrees.

The already frozen M8-v5 reference on the same fresh8 M8-source samples has
median / p90 error `1.65 / 8.02 deg`, accepted rate `0.801`, accepted accuracy
`1.000`, and zero accepted errors over 90 degrees. The v7 source behavior is
therefore a genuine non-forgetting and transport-safety failure, not merely a
strict aggregate threshold miss.

## Runtime compatibility and boundary

The v7 bundle retains the runtime's four input channels
`[Fz, Tx, Ty, tip_z]`, 360 phase bins, 16-dimensional context,
three-state head, 48-bin circular direction head, three-seed mean-probability
combiner, and M8/M10/M12 size contract. The checked-in inference-only
`CircularDirectionCNN` successfully loaded all three v7 state dictionaries and
normalizers.

The v7 contract deliberately tightens
`max_ensemble_top1_span_deg` from the v4 value `180` to `7.5`; the current
runtime implements this field, so the expected effect is extra shadow
abstention rather than an architecture mismatch. A read-only replay of
`M12_event_trigger_insert_live_20260724_023025_306/conical_directservo_trace.csv`
completed with circular mean `275.8469 deg`, APS size `3`, ensemble top-1 span
`7.5 deg`, and the simulation transport gate set to true.

This compatibility check does **not** authorize replacing the live manifest
silently. Any M12 diagnostic-only replacement must atomically bind the v7
model identifier, root, contract, and every artifact hash, must remain
`SHADOW_ONLY`, and must record `REJECTED_GLOBAL_AUDIT` plus
`M12_DEVELOPMENT_SHADOW_ONLY`. Reusing the v4 manifest or enabling
`CAPPED_SHORTCUT` would be invalid.

At completion of this audit the real-device deployment manifest is unchanged:

- SHA-256:
  `5a0018c024fd379e46035967aa673944732b362b364fa59ae26241a5f374d04f`;
- modification time: `2026-07-24 04:31:12 +0800`.

## Frozen artifact hashes

Core v7 artifacts:

- `ensemble_interface_contract.json`:
  `0fbb33d7dda8c6eeb3f133ccea8ed181a13855eb1d46414240150f2285087fea`;
- `adaptation_report.json`:
  `93323cdf7ba4f09c2837fa72c471cf6db16d13ec620ffcb3bebe959cef93ddd9`;
- `split_manifest.json`:
  `2e43ca375f9f9b1d80e3f5d040fad642a9d67a47332f009e3d4f81dd5519d875`;
- `summary.md`:
  `353c598814538d7a7ffcbdfa3de3d477f2f062db9c2be24375a9412d6dc5fca0`.

Seed artifacts:

| seed | checkpoint SHA-256 | metrics SHA-256 | normalization SHA-256 |
|---|---|---|---|
| 20260720 | `b3a586966ef3b323b8b2e4bbbd3c123769389a60dbe96a9a721f5cf4e4715317` | `aae32fc5090d8e037d2f3e575001e163eb0bcc25e0996a9fe72aa2d04bf9d729` | `e520ad1e1b0539bbf9d6d46161a9cac86ca5310732f9bc750b44b9fdb924c31a` |
| 20260721 | `a32598674056d2cde238725ded154592e33cf7a3c4317c3dbc29d57c292825c7` | `d0eec6b3cd093eb390e7a747421e89c1a3b836e057e389afc44d3835dc3098b5` | `e520ad1e1b0539bbf9d6d46161a9cac86ca5310732f9bc750b44b9fdb924c31a` |
| 20260722 | `c16559ec6b2adbb0049a4fd0a8c3ea9167b29a40d1405970b42b4d64fb60e80b` | `948286c9e1e61b2a379c6c6792179a46ed2254f52fbd0c89cbbc2693ffd15b59` | `e520ad1e1b0539bbf9d6d46161a9cac86ca5310732f9bc750b44b9fdb924c31a` |

Audit artifacts:

- clean-data audit:
  `895f2e821b487c175370854de596210ade8e4a983d5ecf354820dff032782181`;
- v7 lineage audit:
  `fac4ede8705c5e07b44e97f0b255c1628185e5efb1a4d17671ee44e660e225c4`;
- fresh8 final `report.json`:
  `f12764904cd30b953e9ba627058a1f442fe1471870f3cc56f801e1b94fa4355d`;
- fresh8 final `predictions.csv`:
  `f6ca684237d8ad45a2d3ba5f771f17f16c7e2e4bbc00c06e5522482a1fd1a004`;
- fresh8 final `summary.md`:
  `cf83c541945e6d094ce0901477890a54686fc06926db857ed2056e8afb5d169f`.

## Reproduction

Final evaluation:

```bash
python3 -m src.evaluate_hardware_easy_v2_multisize \
  --joint-model outputs/current/models/hardware_easy_v2_multisize_clocking_clean_direction_v7_source_preserving \
  --m8-source-audit outputs/current/evaluations/hardware_easy_v2_multisize_fresh8/m8_source \
  --m8-easy-audit outputs/current/evaluations/hardware_easy_v2_multisize_fresh8/m8_easy \
  --m10-easy-audit outputs/current/evaluations/hardware_easy_v2_multisize_fresh8/m10_easy \
  --m12-easy-audit outputs/current/evaluations/hardware_easy_v2_multisize_fresh8/m12_easy \
  --output outputs/current/evaluations/hardware_easy_v2_multisize_clocking_clean_direction_v7_source_preserving_fresh8_final
```

The evaluator exits with status 2 because the global gate fails. This exit is
the expected fail-closed result and the report remains immutable evidence.

Verification:

- 14 targeted lineage, training-policy, holdout-overlap, and multisize tests
  pass;
- the audit, sampler, trainer, evaluator, and promotion-builder modules pass
  Python bytecode compilation;
- running the generic promotion builder against the failed fresh8 report
  raises `ValueError: final promotion gates did not all pass` and writes no
  output file.
