# Conical Contact Continuity and v8 Fresh9 Audit

Date: 2026-07-24 (Asia/Singapore)

## Physical question and closure

A peg supported only by the socket's inner chamfer can unload or descend when
the conical phase points into the aperture.  This is physically plausible in
both MuJoCo and hardware.  The descent is useful task progress, but the
resulting partial phase trace is not a valid circular direction fingerprint.

The implemented closure is:

```text
complete supported cone -> direction inference may be used

low support / safe downward progress
  -> interrupt cone
  -> return smoothly upright from the actual last orientation
  -> fresh physical capture requalification
       -> confirmed: shared compliant funnel
       -> rejected: direction-free global outward fallback or safe stop
```

The online simulation guard uses measured F/T, measured tip-Z trend, elapsed
time, and target force only.  It never reads MuJoCo contact count or contact
geometry.  A partial/incomplete cone cannot reach learned inference, classical
direction estimation, posterior transport, or direct-ray generation.

The real M12 controller uses the corresponding
`CONTACT_SCAN -> LOW_FORCE_PHASE_HOLD -> RECONTACT/APERTURE_CANDIDATE` machine.
Phase, XY, and orientation are held on the first low-normal-force sample.
After two persistent samples, the controller may reacquire at 0.20 mm/s for at
most 0.25 mm.  Recontact within 0.05 mm requires three hysteresis-clearing
samples before the scan resumes.  Otherwise it returns upright at fixed XYZ
and requires fresh Bota force/depth evidence.  Every real low-force-mode
sampling trace is quarantined from direction training.

## Offline simulation hygiene

Legacy canonical NPZ files are not rewritten.  Their immutable `raw_scan`
arrays are revalidated in memory:

- `contact_count == 0` for at least 6 ms: quarantine;
- full-chamfer-depth progress during the scan: quarantine;
- early/endpoint capture, incomplete scan, failure, or non-finite signal:
  quarantine;
- low force with continuous MuJoCo contact and no full-depth progress:
  diagnostic only, retained as the intended unloading fingerprint.

The frozen audit
`outputs/current/reports/hardware_easy_v2_multisize_contact_continuity_data_audit.json`
binds the ordered bytes of all 2,304 canonical NPZ files.

| size | total | admitted | quarantined |
|---|---:|---:|---:|
| M8 | 768 | 675 | 93 |
| M10 | 768 | 611 | 157 |
| M12 | 768 | 575 | 193 |

Contact-count truth is used only for offline data hygiene.  It is not a
deployable hardware signal and is not an insertion-success certificate.

## v8 lineage

The v8 development candidate was retrained from the audited clean M8 base with
M8-source weight 0.72, tail adaptation, learning rate `2e-5`, and L2-SP weight
1.0.  Its split/runtime/adaptation artifacts bind both:

1. the earlier clocking/contact-XML equivalence audit; and
2. the new exact-byte contact-continuity audit.

The separate lineage audit passes.  This means only that v8 used the declared
clean data and is eligible for an independent simulation audit; it is not a
promotion.

## Newly frozen fresh9 result

Fresh9 radii, phases, and random seeds were defined only after v8 was frozen.
All four domains completed 180/180 trials with zero sampler failures.

| domain | valid / total | median / p90 error | within 7.5 deg | APS coverage / mean size | accepted / accepted within 7.5 deg | accepted >90 deg | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M8 source | 154 / 180 | 3.79 / 163.64 deg | 0.734 | 0.838 / 3.60 | 0.721 / 0.955 | 2 | **FAIL** |
| M8 easy-v2 | 161 / 180 | 2.03 / 4.03 deg | 1.000 | 1.000 / 2.40 | 0.988 / 1.000 | 0 | PASS |
| M10 easy-v2 | 160 / 180 | 1.18 / 2.67 deg | 1.000 | 0.994 / 2.23 | 1.000 / 1.000 | 0 | PASS |
| M12 easy-v2 | 158 / 180 | 1.93 / 3.74 deg | 0.975 | 1.000 / 2.40 | 0.994 / 0.975 | 0 | **FAIL** |

M8 source fails because two outermost-ring predictions above 90 degrees were
accepted and that ring lacks required accepted coverage.  M12 has good
aggregate error but fails the strict radial screen because the innermost ring
contains 4/21 accepted errors above 7.5 degrees, exceeding the allowance of
two.

Therefore:

```text
v8 = REJECTED_GLOBAL_AUDIT
hardware deployment manifest = unchanged
v7 = M12 development shadow only, still no XY authority
```

No threshold is retuned on fresh9 and no promotion artifact is written.  The
control closure does not depend on model success: the observable transition
machine, direction-free fallback, shared physical funnel, and terminal
verifier remain classical and independent.

## Reproduction

```bash
python3 -m src.audit_contact_continuity_multisize_data

python3 -m src.finetune_hardware_easy_v2_multisize \
  --base-model outputs/current/models/m8_hardware_easy_v2_robust_direction_v2 \
  --geometry-equivalence-audit outputs/current/reports/hardware_easy_v2_clocking_clean_data_audit.json \
  --contact-continuity-audit outputs/current/reports/hardware_easy_v2_multisize_contact_continuity_data_audit.json \
  --m8-easy-data outputs/current/datasets/m8_hardware_easy_v2_direction_dense_training \
  --m10-easy-data outputs/current/datasets/hardware_easy_v2_multisize_dense/m10 \
  --m12-easy-data outputs/current/datasets/hardware_easy_v2_multisize_dense/m12 \
  --output outputs/current/models/hardware_easy_v2_multisize_contact_continuity_clean_direction_v8 \
  --epochs 120 --patience 20 --batch-size 64 \
  --learning-rate 2e-5 --l2sp 1.0 --source-weight 0.72 \
  --soft-sigma-deg 6 --aps-alpha 0.05 --split-seed 20260731 \
  --adaptation-mode tail

python3 -m src.audit_clocking_clean_multisize_model \
  --model outputs/current/models/hardware_easy_v2_multisize_contact_continuity_clean_direction_v8 \
  --geometry-equivalence-audit outputs/current/reports/hardware_easy_v2_clocking_clean_data_audit.json \
  --contact-continuity-audit outputs/current/reports/hardware_easy_v2_multisize_contact_continuity_data_audit.json \
  --output outputs/current/reports/hardware_easy_v2_multisize_contact_continuity_clean_direction_v8_lineage_audit.json

python3 -m src.evaluate_hardware_easy_v2_multisize \
  --joint-model outputs/current/models/hardware_easy_v2_multisize_contact_continuity_clean_direction_v8 \
  --m8-source-audit outputs/current/evaluations/hardware_easy_v2_multisize_fresh9/m8_source \
  --m8-easy-audit outputs/current/evaluations/hardware_easy_v2_multisize_fresh9/m8_easy \
  --m10-easy-audit outputs/current/evaluations/hardware_easy_v2_multisize_fresh9/m10_easy \
  --m12-easy-audit outputs/current/evaluations/hardware_easy_v2_multisize_fresh9/m12_easy \
  --output outputs/current/evaluations/hardware_easy_v2_multisize_contact_continuity_clean_direction_v8_fresh9_final
```

The evaluator exits with status 2 because the frozen global gate fails.  This
is the intended fail-closed behavior.

## Frozen hashes

- contact-continuity data audit:
  `980bf7c0e1743b4d4c4e00c2792c5f8ecc3ffc2520bacaf64af892f0f162d605`
- v8 lineage audit:
  `fbd120b634f63435a01b533da78a38bd652a24d5e444e5762e37cf9030cd277f`
- v8 runtime contract:
  `5d0448af00791de03c77a0b9534161dfe8a4c31922a28112cbfa2fd83717c4e4`
- fresh9 final report:
  `8c72ef74cbe1f7232d89b2fc3b9276f096d57b87b7759e189c53593bcf60a178`
- fresh9 predictions:
  `0aaebb94b99bf884bd09034219c3a4df45c4e5725fe78e1bd866762fa30bec6e`
- fresh9 summary:
  `41bc87f172f131aef6d99da0caf6394032f4a753fbb7909a01d049d07a296935`

No ROS, Bota, KST, MATLAB, or robot connection was opened during this work.
