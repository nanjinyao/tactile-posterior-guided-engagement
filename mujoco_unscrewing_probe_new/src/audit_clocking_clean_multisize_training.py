"""Audit whether pre-v1.5 simulation data may be reused after CAD clocking.

The v1.5 ``-15 deg`` change rotates the printed peg *inside the flange*.  It
does not rotate the canonical MuJoCo peg relative to the socket after the
flange datum has been taught to fixture ``+X``.  The derived JSON specs changed
because they now carry that construction metadata, while the generated contact
XMLs remained byte-for-byte unchanged.

This script makes that distinction machine-checkable.  It does not edit a
dataset and it does not waive arbitrary spec drift: reuse is admitted only
when the current profile regenerates the exact model hash frozen by the
sampling contract, every completed trial carries an explicit direction-label
validity flag, and the selected base ensemble contains no failed-audit
development sets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.allen_socket_model import build_xml
from src.build_hardware_easy_v2_models import derive_spec
from src.hardware_tooling_profile import DEFAULT_PROFILE, HardwareToolingProfile
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
SIZES = ("m8", "m10", "m12")
DEFAULT_DATA = {
    "m8": ROOT / "outputs/current/datasets/m8_hardware_easy_v2_direction_dense_training",
    "m10": ROOT / "outputs/current/datasets/hardware_easy_v2_multisize_dense/m10",
    "m12": ROOT / "outputs/current/datasets/hardware_easy_v2_multisize_dense/m12",
}
DEFAULT_CLEAN_BASE = (
    ROOT / "outputs/current/models/m8_hardware_easy_v2_robust_direction_v2"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/reports/hardware_easy_v2_clocking_clean_data_audit.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _assert_clocking_contract(profile: HardwareToolingProfile) -> dict[str, Any]:
    frame = profile.data["frame_convention"]
    expected = {
        "integrated_peg_shape_rotation_from_legacy_deg": -15.0,
        "integrated_peg_selected_hex_flat_normal_deg": 45.0,
        "flange_reference_tangent_normal_deg": 45.0,
        "socket_selected_hex_flat_normal_deg": 0.0,
    }
    for key, value in expected.items():
        actual = float(frame[key])
        if abs(actual - value) > 1.0e-12:
            raise ValueError(
                f"tooling frame {key}={actual} does not match audited {value}"
            )
    return {
        **expected,
        "canonical_mujoco_peg_socket_relative_yaw_deg": 0.0,
        "training_label_rotation_applied_deg": 0.0,
        "physical_teaching_rule": (
            "Teach the flange 45-deg tangent-datum normal to fixture +X; "
            "then express wrench, conical phase, posterior and XY motion in "
            "that one canonical frame."
        ),
        "double_counting_guard": (
            "Do not add the CAD -15-deg construction rotation to MuJoCo or "
            "to historical direction labels."
        ),
    }


def _audit_base_model(path: Path) -> dict[str, Any]:
    contract_path = path / "ensemble_interface_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"clean base contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checkpoints = [_resolve(value) for value in contract.get("checkpoints", ())]
    if len(checkpoints) != 3 or any(not item.is_file() for item in checkpoints):
        raise ValueError("clean base must contain exactly three existing checkpoints")
    adaptation = contract.get("adaptation_contract", {})
    consumed = adaptation.get(
        "previously_untouched_development_datasets_consumed", ()
    )
    if consumed:
        raise ValueError(
            "selected base already consumed failed/untouched audit data"
        )
    if int(adaptation.get("untouched_strict_or_easy_samples_seen", 0)) != 0:
        raise ValueError("selected base saw an untouched promotion sample")
    if int(
        adaptation.get(
            "captured_or_incomplete_traces_used_for_direction_supervision", 0
        )
    ) != 0:
        raise ValueError("selected base used incomplete/captured direction labels")
    return {
        "root": str(path),
        "ensemble_interface_contract": str(contract_path),
        "ensemble_interface_contract_sha256": _sha256(contract_path),
        "checkpoint_sha256": {
            str(item): _sha256(item) for item in checkpoints
        },
        "failed_audit_development_sets_consumed": 0,
        "untouched_promotion_samples_seen": 0,
        "captured_or_incomplete_direction_labels_used": 0,
    }


def _audit_dataset(
    size: str,
    source: Path,
    profile: HardwareToolingProfile,
) -> dict[str, Any]:
    contract_path = source / "sampling_contract.json"
    manifest_path = source / "manifest.json"
    progress_path = source / "progress.json"
    for path in (contract_path, manifest_path, progress_path):
        if not path.is_file():
            raise FileNotFoundError(f"{size}: missing dataset artifact {path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if (
        contract.get("training_use")
        != "allowed_only_after_grouped_split_and_capture_filter"
    ):
        raise ValueError(f"{size}: canonical dataset is not training-authorized")
    geometry = contract.get("geometry", {})
    if geometry.get("profile") != "hardware_easy_v2":
        raise ValueError(f"{size}: wrong dataset geometry profile")
    if manifest.get("geometry_profile") != "hardware_easy_v2":
        raise ValueError(f"{size}: wrong manifest geometry profile")
    total = int(contract["total"])
    if int(progress.get("completed", -1)) != total or int(
        progress.get("failed", -1)
    ) != 0:
        raise ValueError(f"{size}: canonical dataset is incomplete")

    current_spec_path = ROOT / "configs" / f"allen_{size}_hardware_easy_v2.json"
    current_model_path = ROOT / "models" / f"allen_{size}_hardware_easy_v2.xml"
    current_spec = json.loads(current_spec_path.read_text(encoding="utf-8"))
    if current_spec.get("geometry_profile") != "hardware_easy_v2":
        raise ValueError(f"{size}: current spec is not hardware_easy_v2")
    if str(current_spec.get("size_name", "")).lower() != size:
        raise ValueError(f"{size}: current spec size mismatch")

    contract_model_path = _resolve(geometry["model_path"])
    if contract_model_path != current_model_path.resolve():
        raise ValueError(f"{size}: dataset points to a different model path")
    current_model_hash = _sha256(current_model_path)
    if current_model_hash != geometry.get("model_sha256"):
        raise ValueError(
            f"{size}: contact XML changed; clocking-only reuse is forbidden"
        )
    regenerated_xml = build_xml(derive_spec(size, profile)).encode("utf-8")
    regenerated_hash = _sha256_bytes(regenerated_xml)
    if regenerated_hash != current_model_hash:
        raise ValueError(
            f"{size}: current tooling profile does not regenerate frozen contact XML"
        )

    trials = training.load_trials(source)
    if len(trials) != total:
        raise ValueError(f"{size}: expected {total} trials, found {len(trials)}")
    validity = []
    for trial in trials:
        if "direction_supervision_valid" not in trial.metadata:
            raise ValueError(
                f"{size}/{trial.metadata.get('trial_id')}: missing explicit "
                "direction_supervision_valid"
            )
        validity.append(bool(trial.metadata["direction_supervision_valid"]))
    valid_count = sum(validity)

    legacy_spec_hash = str(geometry["spec_sha256"])
    current_spec_hash = _sha256(current_spec_path)
    clocking = current_spec.get("mechanical_clocking_contract", {})
    if float(
        clocking.get("integrated_peg_shape_rotation_from_legacy_deg", 999.0)
    ) != -15.0:
        raise ValueError(f"{size}: current spec lacks v1.5 clocking metadata")
    semantics = str(clocking.get("mujoco_relative_yaw_zero_semantics", ""))
    if "remain aligned at relative yaw zero" not in semantics:
        raise ValueError(f"{size}: current spec lacks the zero-yaw simulation rule")

    return {
        "dataset_root": str(source),
        "sampling_contract": str(contract_path),
        "sampling_contract_sha256": _sha256(contract_path),
        "training_use": contract["training_use"],
        "legacy_spec_sha256": legacy_spec_hash,
        "current_spec": str(current_spec_path),
        "current_spec_sha256": current_spec_hash,
        "spec_hash_changed_only_under_explicit_audit": (
            legacy_spec_hash != current_spec_hash
        ),
        "contact_model": str(current_model_path),
        "sampling_contract_model_sha256": geometry["model_sha256"],
        "current_model_sha256": current_model_hash,
        "regenerated_model_sha256": regenerated_hash,
        "contact_xml_byte_identical": True,
        "completed_trials": total,
        "valid_complete_direction_fingerprints": valid_count,
        "captured_or_incomplete_excluded_from_direction_supervision": (
            total - valid_count
        ),
        "trials_missing_explicit_direction_validity": 0,
        "bit_radius_m": float(current_spec["key_af_m"]) / (3.0**0.5),
        "label_definition": "atan2(-y_offset, -x_offset) in canonical task XY",
        "label_rotation_applied_deg": 0.0,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    profile = HardwareToolingProfile.load(args.profile)
    datasets = {
        "m8": Path(args.m8_easy_data).resolve(),
        "m10": Path(args.m10_easy_data).resolve(),
        "m12": Path(args.m12_easy_data).resolve(),
    }
    report = {
        "schema_version": 1,
        "stage": "hardware_easy_v2_clocking_clean_multisize_training_audit",
        "status": "PASS",
        "decision": (
            "Reuse the three canonical simulation datasets without label "
            "rotation, but retrain from the audited clean base and exclude "
            "every failed-audit development dataset."
        ),
        "tooling_profile": str(profile.path),
        "tooling_profile_sha256": _sha256(profile.path),
        "clocking_contract": _assert_clocking_contract(profile),
        "clean_base_model": _audit_base_model(Path(args.base_model).resolve()),
        "datasets": {
            size: _audit_dataset(size, datasets[size], profile) for size in SIZES
        },
        "training_policy": {
            "canonical_datasets_only": True,
            "failed_audit_development_datasets_allowed": False,
            "group_complete_physical_C6_orbits_before_filtering": True,
            "captured_or_incomplete_direction_supervision_allowed": False,
            "fresh_promotion_audit_must_remain_forbidden": True,
        },
        "claim_boundary": (
            "This audit proves contact-model and label-frame equivalence for "
            "simulation reuse. It does not calibrate the real sensor frame, "
            "fixture-to-world transform, FDM friction/compliance, or qualify "
            "learned commands on hardware."
        ),
        "created_utc": _utc_now(),
    }
    return report


def run(args: argparse.Namespace) -> int:
    report = build_report(args)
    output = Path(args.output).resolve()
    _atomic_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "output_sha256": _sha256(output),
                "base_model": report["clean_base_model"]["root"],
                "valid_direction_fingerprints": {
                    size: item["valid_complete_direction_fingerprints"]
                    for size, item in report["datasets"].items()
                },
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--base-model", default=str(DEFAULT_CLEAN_BASE))
    parser.add_argument("--m8-easy-data", default=str(DEFAULT_DATA["m8"]))
    parser.add_argument("--m10-easy-data", default=str(DEFAULT_DATA["m10"]))
    parser.add_argument("--m12-easy-data", default=str(DEFAULT_DATA["m12"]))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
