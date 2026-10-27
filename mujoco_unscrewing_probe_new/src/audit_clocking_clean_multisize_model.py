"""Independently audit the lineage of a clocking-clean multisize model.

This audit is intentionally separate from the trainer.  It accepts only the
three canonical hardware-easy-v2 datasets admitted by the geometry-equivalence
audit, an audited clean base ensemble, grouped splits made before capture
filtering, and zero development/final-audit samples.  It does not evaluate or
promote a model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.audit_contact_continuity_multisize_data import (
    trial_manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    ROOT
    / "outputs/current/models/hardware_easy_v2_multisize_clocking_clean_direction_v5"
)
DEFAULT_GEOMETRY_AUDIT = (
    ROOT / "outputs/current/reports/hardware_easy_v2_clocking_clean_data_audit.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/reports/"
    "hardware_easy_v2_multisize_clocking_clean_direction_v5_lineage_audit.json"
)
SIZES = ("m8", "m10", "m12")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _empty_development(value: Any, label: str) -> None:
    if isinstance(value, dict):
        _require(
            all(not list(items) for items in value.values()),
            f"{label} is not empty",
        )
    else:
        _require(not list(value), f"{label} is not empty")


def build_report(
    model_root: Path,
    geometry_audit_path: Path,
    contact_continuity_audit_path: Path | None = None,
) -> dict[str, Any]:
    model_root = model_root.resolve()
    geometry_audit_path = geometry_audit_path.resolve()
    paths = {
        name: model_root / name
        for name in (
            "adaptation_report.json",
            "split_manifest.json",
            "ensemble_interface_contract.json",
        )
    }
    for name, path in paths.items():
        _require(path.is_file(), f"missing {name}: {path}")
    _require(
        geometry_audit_path.is_file(),
        f"missing geometry-equivalence audit: {geometry_audit_path}",
    )

    adaptation = json.loads(paths["adaptation_report.json"].read_text())
    split = json.loads(paths["split_manifest.json"].read_text())
    contract = json.loads(paths["ensemble_interface_contract.json"].read_text())
    geometry = json.loads(geometry_audit_path.read_text())
    geometry_hash = _sha256(geometry_audit_path)
    _require(geometry.get("status") == "PASS", "geometry audit did not pass")
    continuity = None
    continuity_hash = None
    if contact_continuity_audit_path is not None:
        contact_continuity_audit_path = (
            contact_continuity_audit_path.resolve()
        )
        _require(
            contact_continuity_audit_path.is_file(),
            "missing contact-continuity audit: "
            f"{contact_continuity_audit_path}",
        )
        continuity = json.loads(
            contact_continuity_audit_path.read_text()
        )
        continuity_hash = _sha256(contact_continuity_audit_path)
        _require(
            continuity.get("schema_version") == 1
            and continuity.get("status") == "PASS",
            "contact-continuity audit did not pass",
        )

    recorded_geometry = adaptation.get("geometry_equivalence_audit", {})
    _require(
        recorded_geometry.get("sha256") == geometry_hash,
        "trainer did not bind the current geometry audit hash",
    )
    _require(
        split.get("geometry_equivalence_audit", {}).get("sha256") == geometry_hash,
        "split manifest geometry audit hash mismatch",
    )
    adaptation_contract = contract.get("adaptation_contract", {})
    _require(
        adaptation_contract.get("geometry_equivalence_audit", {}).get("sha256")
        == geometry_hash,
        "runtime contract geometry audit hash mismatch",
    )
    if continuity is not None:
        for owner, record in (
            ("adaptation", adaptation.get("contact_continuity_audit", {})),
            ("split manifest", split.get("contact_continuity_audit", {})),
            (
                "runtime contract",
                adaptation_contract.get("contact_continuity_audit", {}),
            ),
        ):
            _require(
                record.get("sha256") == continuity_hash,
                f"{owner} contact-continuity audit hash mismatch",
            )

    split_hash = _sha256(paths["split_manifest.json"])
    _require(
        adaptation.get("split_manifest_sha256") == split_hash,
        "adaptation report split-manifest hash mismatch",
    )
    audited_base = geometry["clean_base_model"]
    expected_base = Path(audited_base["root"]).resolve()
    _require(
        Path(adaptation["base_model_root"]).resolve() == expected_base,
        "adaptation used a base other than the audited clean base",
    )
    _require(
        Path(adaptation_contract["base_model_root"]).resolve() == expected_base,
        "runtime contract base differs from the audited clean base",
    )
    expected_base_hash = audited_base["ensemble_interface_contract_sha256"]
    _require(
        adaptation["base_model_contract_sha256"] == expected_base_hash,
        "adaptation base-contract hash mismatch",
    )
    _require(
        adaptation_contract["base_model_contract_sha256"] == expected_base_hash,
        "runtime base-contract hash mismatch",
    )

    _empty_development(
        adaptation.get("easy_development_datasets_consumed", {}),
        "adaptation easy development",
    )
    _empty_development(
        adaptation.get("m8_source_development_datasets_consumed", []),
        "adaptation M8-source development",
    )
    _empty_development(split.get("easy_development", {}), "split easy development")
    _empty_development(
        split.get("m8_source_development", []),
        "split M8-source development",
    )
    _empty_development(
        adaptation_contract.get("easy_development_datasets_consumed", {}),
        "runtime easy development",
    )
    _empty_development(
        adaptation_contract.get("m8_source_development_datasets_consumed", []),
        "runtime M8-source development",
    )
    _require(
        int(adaptation.get("test_samples_seen", -1)) == 0,
        "final/untouched samples were seen during fitting",
    )
    _require(
        adaptation.get("final_audit_run") is False,
        "trainer reports that a final audit was run during fitting",
    )
    _require(
        int(
            adaptation_contract.get(
                "final_promotion_audit_samples_seen_during_fitting", -1
            )
        )
        == 0,
        "runtime contract reports final-audit leakage",
    )
    _require(
        int(
            adaptation_contract.get(
                "captured_or_incomplete_traces_used_for_direction_supervision", -1
            )
        )
        == 0,
        "runtime contract reports ambiguous direction supervision",
    )
    _require(
        int(
            split.get(
                "captured_or_incomplete_traces_used_for_direction_supervision", -1
            )
        )
        == 0,
        "split manifest reports ambiguous direction supervision",
    )
    if continuity is not None:
        for owner, record in (
            ("runtime contract", adaptation_contract),
            ("split manifest", split),
        ):
            _require(
                int(
                    record.get(
                        "contact_transition_or_capture_quarantined_traces_used_for_direction_supervision",
                        -1,
                    )
                )
                == 0,
                f"{owner} reports contact-transition direction leakage",
            )

    dataset_evidence: dict[str, Any] = {}
    for size in SIZES:
        trained = adaptation["dataset_contracts"][size]
        admitted = geometry["datasets"][size]
        domain = split["easy_domains"][size]
        _require(
            Path(trained["root"]).resolve() == Path(admitted["dataset_root"]).resolve(),
            f"{size}: trained dataset is not the admitted canonical dataset",
        )
        _require(
            trained["sampling_contract_sha256"]
            == admitted["sampling_contract_sha256"],
            f"{size}: sampling-contract hash mismatch",
        )
        _require(
            trained["model_sha256"] == admitted["current_model_sha256"],
            f"{size}: contact-model hash mismatch",
        )
        _require(
            domain.get("group_intersection") == [],
            f"{size}: train/calibration group leakage",
        )
        _require(
            domain.get("split_before_capture_filter") is True,
            f"{size}: split was not frozen before capture filtering",
        )
        _require(
            int(domain.get("development_datasets_merged", -1)) == 0,
            f"{size}: development data were merged",
        )
        if continuity is None:
            expected_valid = int(
                admitted["valid_complete_direction_fingerprints"]
            )
            expected_excluded = int(
                admitted[
                    "captured_or_incomplete_excluded_from_direction_supervision"
                ]
            )
        else:
            continuity_item = continuity["datasets"][size]
            _require(
                Path(continuity_item["dataset_root"]).resolve()
                == Path(trained["root"]).resolve(),
                f"{size}: contact-continuity dataset mismatch",
            )
            _require(
                continuity_item["sampling_contract_sha256"]
                == trained["sampling_contract_sha256"],
                f"{size}: contact-continuity sampling-contract mismatch",
            )
            _require(
                continuity_item["trial_manifest_sha256"]
                == trial_manifest_sha256(Path(trained["root"])),
                f"{size}: canonical trial bytes changed after continuity audit",
            )
            hygiene = domain.get("direction_hygiene", {})
            expected_hygiene = {
                "direction_supervision_admitted": continuity_item[
                    "direction_supervision_admitted"
                ],
                "direction_supervision_quarantined": continuity_item[
                    "direction_supervision_quarantined"
                ],
                "direction_quarantine_reason_counts": continuity_item[
                    "direction_quarantine_reason_counts"
                ],
                "direction_diagnostic_counts": continuity_item[
                    "direction_diagnostic_counts"
                ],
            }
            for key, expected in expected_hygiene.items():
                _require(
                    hygiene.get(key) == expected,
                    f"{size}: split {key} differs from continuity audit",
                )
            expected_valid = int(
                continuity_item["direction_supervision_admitted"]
            )
            expected_excluded = int(
                continuity_item["direction_supervision_quarantined"]
            )
        _require(
            int(domain["valid_full_fingerprints"]) == expected_valid,
            f"{size}: valid-fingerprint count differs from active audit",
        )
        _require(
            int(domain["capture_or_incomplete_excluded"])
            == expected_excluded,
            f"{size}: direction-exclusion count differs from active audit",
        )
        dataset_evidence[size] = {
            "root": trained["root"],
            "sampling_contract_sha256": trained["sampling_contract_sha256"],
            "contact_model_sha256": trained["model_sha256"],
            "valid_full_fingerprints": domain["valid_full_fingerprints"],
            "capture_or_incomplete_excluded": domain[
                "capture_or_incomplete_excluded"
            ],
            "direction_hygiene": domain.get("direction_hygiene"),
            "train_groups": len(domain["train_groups"]),
            "calibration_groups": len(domain["calibration_groups"]),
            "group_intersection": [],
            "development_datasets_merged": 0,
        }

    checkpoints = [Path(value).resolve() for value in contract["checkpoints"]]
    _require(len(checkpoints) == 3, "expected exactly three ensemble checkpoints")
    checkpoint_evidence = []
    for checkpoint in checkpoints:
        _require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
        normalization = checkpoint.parent / "normalization.npz"
        metrics = checkpoint.parent / "metrics.json"
        _require(normalization.is_file(), f"missing normalization: {normalization}")
        _require(metrics.is_file(), f"missing metrics: {metrics}")
        checkpoint_evidence.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "normalization_sha256": _sha256(normalization),
                "metrics_sha256": _sha256(metrics),
            }
        )

    return {
        "schema_version": 1,
        "stage": "clocking_clean_multisize_model_lineage_audit",
        "status": "PASS",
        "decision": (
            "The candidate is lineage-clean for an independent final simulation "
            "audit. This is not a promotion decision."
        ),
        "model_root": str(model_root),
        "model_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "geometry_equivalence_audit": {
            "path": str(geometry_audit_path),
            "sha256": geometry_hash,
            "status": "PASS",
        },
        "contact_continuity_audit": (
            None
            if continuity is None
            else {
                "path": str(contact_continuity_audit_path),
                "sha256": continuity_hash,
                "status": "PASS",
            }
        ),
        "audited_clean_base": {
            "root": str(expected_base),
            "ensemble_interface_contract_sha256": expected_base_hash,
            "failed_audit_development_sets_consumed": 0,
            "untouched_promotion_samples_seen": 0,
            "captured_or_incomplete_direction_labels_used": 0,
        },
        "training_data": dataset_evidence,
        "leakage_checks": {
            "development_datasets_consumed": 0,
            "final_audit_samples_seen_during_fitting": 0,
            "captured_or_incomplete_direction_labels_used": 0,
            "group_intersections": 0,
        },
        "checkpoints": checkpoint_evidence,
        "claim_boundary": (
            "Lineage audit only. It neither measures fresh generalization nor "
            "authorizes simulation promotion, runtime-manifest changes, or "
            "real-hardware control."
        ),
        "updated_utc": _utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--geometry-equivalence-audit", default=str(DEFAULT_GEOMETRY_AUDIT)
    )
    parser.add_argument("--contact-continuity-audit", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        Path(args.model),
        Path(args.geometry_equivalence_audit),
        (
            None
            if args.contact_continuity_audit is None
            else Path(args.contact_continuity_audit)
        ),
    )
    _atomic_json(Path(args.output).resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "model_root": report["model_root"],
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
