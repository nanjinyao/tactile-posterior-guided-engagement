"""Fine-tune one guarded direction ensemble on M8/M10/M12 easy-v2 data.

The shared network starts from the promoted M8 hardware-easy-v2 v5 ensemble.
It replays the original M8 source domain and balances that anchor against three
size-specific hardware-easy-v2 domains.  Each easy dataset is split by complete
physical C6 orbit *before* captured/incomplete traces are removed.  Network
inputs use the bit radius declared by the corresponding size config.

The emitted directory is compatible with ``M8DirectionEnsemble``.  To preserve
that runtime contract, temperature, APS and transport-gate calibration are
global conservative values selected from equally weighted per-domain
calibration sets.  Runtime callers must still pass the correct size-specific
bit radius to ``infer_phase_sequence``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch import nn

import src.finetune_m8_hardware_easy_v2 as m8_adaptation
from src.audit_contact_continuity_multisize_data import (
    trial_manifest_sha256,
)
from src.m8_direction_runtime import M8DirectionEnsemble
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
SIZES = ("m8", "m10", "m12")
BASE_ROOT = (
    ROOT
    / "outputs/current/models/m8_hardware_easy_v2_robust_direction_v5"
)
DEFAULT_DATA = {
    "m8": (
        ROOT
        / "outputs/current/datasets/m8_hardware_easy_v2_direction_dense_training"
    ),
    "m10": ROOT / "outputs/current/datasets/hardware_easy_v2_multisize_dense/m10",
    "m12": ROOT / "outputs/current/datasets/hardware_easy_v2_multisize_dense/m12",
}
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/models/hardware_easy_v2_multisize_robust_direction_v1"
)
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _valid_direction_trial(trial: training.LoadedTrial) -> bool:
    if "direction_supervision_valid" not in trial.metadata:
        raise ValueError(
            f"{trial.metadata.get('trial_id', '<unknown>')}: easy-v2 trial lacks "
            "direction_supervision_valid; legacy cross-size data are not legal input"
        )
    return bool(trial.metadata["direction_supervision_valid"])


def _direction_hygiene_summary(
    trials: list[training.LoadedTrial],
) -> dict:
    """Summarize the reason-coded filter actually applied at load time."""

    admitted = 0
    reason_counts: dict[str, int] = {}
    diagnostic_counts: dict[str, int] = {}
    for trial in trials:
        admitted += int(_valid_direction_trial(trial))
        for reason in trial.metadata.get("direction_quarantine_reasons", []):
            key = str(reason)
            reason_counts[key] = reason_counts.get(key, 0) + 1
        for diagnostic in trial.metadata.get(
            "direction_diagnostic_flags", []
        ):
            key = str(diagnostic)
            diagnostic_counts[key] = diagnostic_counts.get(key, 0) + 1
    return {
        "direction_supervision_admitted": admitted,
        "direction_supervision_quarantined": len(trials) - admitted,
        "direction_quarantine_reason_counts": dict(
            sorted(reason_counts.items())
        ),
        "direction_diagnostic_counts": dict(
            sorted(diagnostic_counts.items())
        ),
        "legacy_raw_revalidated_in_memory": True,
        "source_npz_rewritten": False,
    }


def _physical_orbit_id(trial: training.LoadedTrial) -> str:
    """Group both nuisance repeats and all six C6 sectors before filtering."""

    orbit = str(trial.metadata["symmetry_orbit_id"])
    return orbit.rsplit("_n", 1)[0]


def _load_geometry_equivalence_audit(
    args: argparse.Namespace,
) -> tuple[dict | None, dict | None]:
    value = getattr(args, "geometry_equivalence_audit", None)
    if not value:
        return None, None
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"geometry-equivalence audit is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("geometry-equivalence audit did not pass")
    if report.get("schema_version") != 1:
        raise ValueError("unsupported geometry-equivalence audit schema")
    policy = report.get("training_policy", {})
    if not policy.get("canonical_datasets_only", False):
        raise ValueError("equivalence audit does not require canonical-only data")
    if policy.get("failed_audit_development_datasets_allowed", True):
        raise ValueError("equivalence audit permits failed-audit development data")
    if getattr(args, "m8_source_development_data", ()) or getattr(
        args, "easy_development_data", ()
    ):
        raise ValueError(
            "failed-audit development data are forbidden under the "
            "clocking-equivalence clean-training contract"
        )

    audited_base = report.get("clean_base_model", {})
    base_root = Path(args.base_model).resolve()
    if Path(audited_base.get("root", "")).resolve() != base_root:
        raise ValueError("trainer base model does not match equivalence audit")
    base_contract = base_root / "ensemble_interface_contract.json"
    if (
        not base_contract.is_file()
        or audited_base.get("ensemble_interface_contract_sha256")
        != _sha256(base_contract)
    ):
        raise ValueError("audited clean-base contract hash drift")

    expected_data = {
        "m8": Path(args.m8_easy_data).resolve(),
        "m10": Path(args.m10_easy_data).resolve(),
        "m12": Path(args.m12_easy_data).resolve(),
    }
    audited_datasets = report.get("datasets", {})
    for size, source in expected_data.items():
        item = audited_datasets.get(size, {})
        if Path(item.get("dataset_root", "")).resolve() != source:
            raise ValueError(f"{size}: trainer dataset does not match audit")
        contract_path = source / "sampling_contract.json"
        if item.get("sampling_contract_sha256") != _sha256(contract_path):
            raise ValueError(f"{size}: audited sampling contract hash drift")
    return report, {
        "path": str(path),
        "sha256": _sha256(path),
        "status": "PASS",
        "decision": report.get("decision"),
    }


def _load_contact_continuity_audit(
    args: argparse.Namespace,
) -> tuple[dict | None, dict | None]:
    value = getattr(args, "contact_continuity_audit", None)
    if not value:
        return None, None
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"contact-continuity audit is missing: {path}"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 1
        or report.get("status") != "PASS"
    ):
        raise ValueError("contact-continuity audit did not pass")
    if (
        report.get("policy", {}).get("low_force_with_continuous_contact")
        != "diagnostic_only_not_quarantine"
    ):
        raise ValueError(
            "contact-continuity audit has an incompatible low-force policy"
        )
    return report, {
        "path": str(path),
        "sha256": _sha256(path),
        "status": "PASS",
    }


def _validate_equivalent_spec_drift(
    size: str,
    source: Path,
    geometry: dict,
    spec_path: Path,
    audit: dict | None,
) -> None:
    if audit is None:
        raise ValueError(f"{size}: easy-v2 spec hash drift")
    item = audit.get("datasets", {}).get(size, {})
    if Path(item.get("dataset_root", "")).resolve() != source.resolve():
        raise ValueError(f"{size}: geometry audit dataset mismatch")
    if item.get("legacy_spec_sha256") != geometry.get("spec_sha256"):
        raise ValueError(f"{size}: legacy spec hash does not match audit")
    if item.get("current_spec_sha256") != _sha256(spec_path):
        raise ValueError(f"{size}: current spec hash does not match audit")
    model_path = _resolve_from_root(geometry["model_path"])
    if (
        item.get("sampling_contract_model_sha256")
        != geometry.get("model_sha256")
        or item.get("current_model_sha256") != _sha256(model_path)
        or item.get("regenerated_model_sha256") != _sha256(model_path)
        or not item.get("contact_xml_byte_identical", False)
    ):
        raise ValueError(f"{size}: contact XML equivalence was not proven")
    if float(item.get("label_rotation_applied_deg", math.nan)) != 0.0:
        raise ValueError(f"{size}: historical labels must not be yaw-rotated")


def _load_easy_dataset(
    size: str,
    source: Path,
    *,
    geometry_equivalence_audit: dict | None = None,
) -> tuple[dict, dict, list[training.LoadedTrial]]:
    contract_path = source / "sampling_contract.json"
    manifest_path = source / "manifest.json"
    progress_path = source / "progress.json"
    for required in (contract_path, manifest_path, progress_path):
        if not required.is_file():
            raise FileNotFoundError(f"{size}: missing completed dataset artifact {required}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if contract.get("training_use") != "allowed_only_after_grouped_split_and_capture_filter":
        raise ValueError(f"{size}: dataset is not authorized for grouped training")
    geometry = contract.get("geometry", {})
    if geometry.get("profile") != "hardware_easy_v2":
        raise ValueError(f"{size}: expected hardware_easy_v2 geometry contract")
    spec_path = _resolve_from_root(geometry["spec_path"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if str(spec.get("size_name", spec.get("name", ""))).split("_")[0] != size:
        raise ValueError(f"{size}: dataset/spec size mismatch at {spec_path}")
    if spec.get("geometry_profile") != "hardware_easy_v2":
        raise ValueError(f"{size}: spec is not the derived easy-v2 profile")
    if geometry.get("spec_sha256") != _sha256(spec_path):
        _validate_equivalent_spec_drift(
            size,
            source,
            geometry,
            spec_path,
            geometry_equivalence_audit,
        )
    total = int(contract["total"])
    if int(progress.get("completed", -1)) != total or int(progress.get("failed", -1)) != 0:
        raise ValueError(f"{size}: sampling is incomplete or contains failed trials")
    trials = training.load_trials(source)
    if len(trials) != total:
        raise ValueError(f"{size}: expected {total} ok trials, found {len(trials)}")
    if manifest.get("geometry_profile") != "hardware_easy_v2":
        raise ValueError(f"{size}: manifest geometry profile mismatch")
    return contract, spec, trials


def _load_forbidden_m8_source_dataset(
    source: Path,
) -> tuple[dict, dict, list[training.LoadedTrial]]:
    """Load an explicitly supplied failed M8 source audit as development data."""

    contract_path = source / "sampling_contract.json"
    manifest_path = source / "manifest.json"
    progress_path = source / "progress.json"
    for required in (contract_path, manifest_path, progress_path):
        if not required.is_file():
            raise FileNotFoundError(f"M8 source development: missing {required}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if contract.get("training_use") != "forbidden":
        raise ValueError(
            f"{source}: M8 source development input must retain training_use=forbidden"
        )
    geometry = contract.get("geometry", {})
    if geometry.get("profile") != "frozen_source":
        raise ValueError(f"{source}: expected a frozen_source audit")
    stage = str(contract.get("stage", ""))
    if "untouched" not in stage.lower() and "fresh" not in stage.lower():
        raise ValueError(f"{source}: source development input is not a fresh audit")
    spec_path = _resolve_from_root(geometry["spec_path"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not str(spec.get("name", "")).startswith("m8"):
        raise ValueError(f"{source}: development audit is not M8")
    if geometry.get("spec_sha256") != _sha256(spec_path):
        raise ValueError(f"{source}: M8 source spec hash drift")
    total = int(contract["total"])
    if int(progress.get("completed", -1)) != total or int(progress.get("failed", -1)) != 0:
        raise ValueError(f"{source}: M8 source development sampling is incomplete")
    trials = training.load_trials(source)
    if len(trials) != total:
        raise ValueError(f"{source}: expected {total} ok trials, found {len(trials)}")
    if manifest.get("geometry_profile") != "frozen_source":
        raise ValueError(f"{source}: M8 source manifest profile mismatch")
    return contract, spec, trials


def _load_forbidden_easy_dataset(
    size: str, source: Path
) -> tuple[dict, dict, list[training.LoadedTrial]]:
    """Load an explicitly supplied failed easy-v2 audit as development data."""

    if size not in SIZES:
        raise ValueError(f"unsupported easy development size: {size}")
    contract_path = source / "sampling_contract.json"
    manifest_path = source / "manifest.json"
    progress_path = source / "progress.json"
    for required in (contract_path, manifest_path, progress_path):
        if not required.is_file():
            raise FileNotFoundError(f"{size} easy development: missing {required}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if contract.get("training_use") != "forbidden":
        raise ValueError(
            f"{source}: easy development input must retain training_use=forbidden"
        )
    geometry = contract.get("geometry", {})
    if geometry.get("profile") != "hardware_easy_v2":
        raise ValueError(f"{source}: expected a hardware_easy_v2 failed audit")
    stage = str(contract.get("stage", ""))
    if "untouched" not in stage.lower() and "fresh" not in stage.lower():
        raise ValueError(f"{source}: easy development input is not a fresh audit")
    spec_path = _resolve_from_root(geometry["spec_path"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    declared_size = str(spec.get("size_name", spec.get("name", ""))).split("_")[0]
    if declared_size != size:
        raise ValueError(
            f"{source}: requested size {size} does not match spec size {declared_size}"
        )
    if spec.get("geometry_profile") != "hardware_easy_v2":
        raise ValueError(f"{source}: easy development spec profile mismatch")
    if geometry.get("spec_sha256") != _sha256(spec_path):
        raise ValueError(f"{source}: easy development spec hash drift")
    total = int(contract["total"])
    if int(progress.get("completed", -1)) != total or int(progress.get("failed", -1)) != 0:
        raise ValueError(f"{source}: easy development sampling is incomplete")
    trials = training.load_trials(source)
    if len(trials) != total:
        raise ValueError(f"{source}: expected {total} ok trials, found {len(trials)}")
    if manifest.get("geometry_profile") != "hardware_easy_v2":
        raise ValueError(f"{source}: easy development manifest profile mismatch")
    return contract, spec, trials


def _parse_easy_development_values(values: list[str] | tuple[str, ...]) -> dict[str, list[Path]]:
    by_size: dict[str, list[Path]] = {size: [] for size in SIZES}
    seen: set[tuple[str, Path]] = set()
    for value in values:
        size, separator, path_value = str(value).partition("=")
        size = size.strip().lower()
        if separator != "=" or size not in SIZES or not path_value.strip():
            raise ValueError(
                f"invalid --easy-development-data {value!r}; expected SIZE=PATH "
                "with SIZE in m8,m10,m12"
            )
        path = Path(path_value.strip()).resolve()
        key = (size, path)
        if key in seen:
            raise ValueError(f"duplicate easy development dataset: {size}={path}")
        seen.add(key)
        by_size[size].append(path)
    return by_size


def _split_physical_orbits(
    trials: list[training.LoadedTrial], *, seed: int
) -> tuple[set[str], set[str], dict]:
    """Split complete physical orbits using every trial, before capture filtering."""

    groups_by_state: dict[str, list[str]] = {}
    states_by_group: dict[str, set[str]] = {}
    trial_counts: dict[str, int] = {}
    for trial in trials:
        group = _physical_orbit_id(trial)
        state = str(trial.metadata["search_state"])
        states_by_group.setdefault(group, set()).add(state)
        trial_counts[group] = trial_counts.get(group, 0) + 1
    for group, states in states_by_group.items():
        if len(states) != 1:
            raise ValueError(f"easy-v2 physical group {group} spans states {states}")
        state = next(iter(states))
        groups_by_state.setdefault(state, []).append(group)

    rng = np.random.default_rng(seed)
    train_groups: set[str] = set()
    calibration_groups: set[str] = set()
    state_counts = {}
    for state, groups in sorted(groups_by_state.items()):
        ordered = sorted(groups)
        rng.shuffle(ordered)
        if len(ordered) < 2:
            raise ValueError(f"easy-v2 state {state} has fewer than two physical groups")
        calibration_count = min(
            len(ordered) - 1, max(1, int(round(0.20 * len(ordered))))
        )
        calibration_groups.update(ordered[:calibration_count])
        train_groups.update(ordered[calibration_count:])
        state_counts[state] = {
            "physical_orbits": len(ordered),
            "train_orbits": len(ordered) - calibration_count,
            "calibration_orbits": calibration_count,
        }
    intersection = train_groups & calibration_groups
    if intersection:
        raise RuntimeError(f"easy-v2 train/calibration orbit leakage: {intersection}")
    return train_groups, calibration_groups, {
        "split_before_capture_filter": True,
        "group_unit": (
            "symmetry_orbit_id without nuisance suffix; contains every nuisance "
            "repeat and all six C6 sectors"
        ),
        "state_counts": state_counts,
        "train_groups": sorted(train_groups),
        "calibration_groups": sorted(calibration_groups),
        "group_intersection": [],
        "all_group_trial_counts": dict(sorted(trial_counts.items())),
    }


def _make_size_example(
    trial: training.LoadedTrial, *, bit_radius_m: float, size: str
) -> dict:
    indices = [trial.columns.index(name) for name in training.INPUT_CHANNELS]
    raw = np.vstack([trial.sequence[index] for index in indices])
    shape, context = M8DirectionEnsemble._network_inputs(
        raw,
        float(trial.metadata["target_down_force_n"]),
        float(bit_radius_m),
    )
    angle_deg = training._direction_angle_deg(trial.metadata, 0)
    return {
        "sequence": shape,
        "context": context,
        "state": training.STATE_TO_INDEX[str(trial.metadata["search_state"])],
        "direction_class": training._direction_class(angle_deg),
        "direction_angle_deg": angle_deg,
        "base_pose_id": str(trial.metadata["base_pose_id"]),
        "trial_id": str(trial.metadata["trial_id"]),
        "rotation_index": int(trial.metadata.get("rotation_index", 0)),
        "radial_level": int(trial.metadata.get("radial_level", -1)),
        "radius_m": float(trial.metadata.get("radius_m", float("nan"))),
        "observation_stratum": (
            "ideal"
            if trial.metadata.get("wrench_observation", "ideal") == "ideal"
            else "sensor_prior"
        ),
        "size": size,
        "physical_orbit_id": _physical_orbit_id(trial),
    }


def _easy_examples(
    trials: list[training.LoadedTrial],
    groups: set[str],
    *,
    bit_radius_m: float,
    size: str,
) -> list[dict]:
    return [
        _make_size_example(trial, bit_radius_m=bit_radius_m, size=size)
        for trial in trials
        if _physical_orbit_id(trial) in groups and _valid_direction_trial(trial)
    ]


def _prepare_m8_source(split_seed: int) -> tuple[list[dict], list[dict], dict]:
    p0b = training.load_trials(training.P0B)
    p0d = training.load_trials(training.P0D)
    targeted = training.load_trials(training.P1V3_TARGETED)
    p0b_split = training.grouped_split(p0b, split_seed)
    p0d_split = training.grouped_sector_split(p0d, split_seed)
    targeted_split = training.grouped_targeted_split(targeted, split_seed)
    train = (
        training.make_examples(p0b, set(p0b_split["train"]), tuple(range(6)))
        + training.make_sector_examples(p0d, set(p0d_split["train"]))
        + training.make_examples(targeted, set(targeted_split["train"]), (0,))
    )
    calibration = (
        training.make_examples(p0b, set(p0b_split["val"]), (0,))
        + training.make_sector_examples(p0d, set(p0d_split["val"]))
        + training.make_examples(targeted, set(targeted_split["val"]), (0,))
    )
    if not train or not calibration:
        raise ValueError("M8 source replay split is empty")
    provenance = {
        "domain": "m8_source",
        "split_seed": split_seed,
        "train_examples": len(train),
        "calibration_examples": len(calibration),
        "p0b_train_groups": sorted(p0b_split["train"]),
        "p0b_calibration_groups": sorted(p0b_split["val"]),
        "p0d_train_groups": sorted(p0d_split["train"]),
        "p0d_calibration_groups": sorted(p0d_split["val"]),
        "targeted_train_groups": sorted(targeted_split["train"]),
        "targeted_calibration_groups": sorted(targeted_split["val"]),
    }
    return train, calibration, provenance


def _prepare_domains(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    geometry_equivalence, geometry_equivalence_record = (
        _load_geometry_equivalence_audit(args)
    )
    contact_continuity, contact_continuity_record = (
        _load_contact_continuity_audit(args)
    )
    source_train, source_calibration, source_provenance = _prepare_m8_source(
        args.split_seed
    )
    source_development_records = []
    resolved_development_paths: set[Path] = set()
    for path_index, value in enumerate(
        getattr(args, "m8_source_development_data", ())
    ):
        source = Path(value).resolve()
        if source in resolved_development_paths:
            raise ValueError(f"duplicate M8 source development dataset: {source}")
        resolved_development_paths.add(source)
        contract, spec, trials = _load_forbidden_m8_source_dataset(source)
        train_groups, calibration_groups, split = _split_physical_orbits(
            trials, seed=args.split_seed + 2000 + 100 * path_index
        )
        bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
        development_train = _easy_examples(
            trials, train_groups, bit_radius_m=bit_radius, size="m8"
        )
        development_calibration = _easy_examples(
            trials, calibration_groups, bit_radius_m=bit_radius, size="m8"
        )
        if not development_train or not development_calibration:
            raise ValueError(
                f"{source}: no valid direction fingerprints after physical-orbit split"
            )
        hygiene = _direction_hygiene_summary(trials)
        valid_total = int(hygiene["direction_supervision_admitted"])
        if contact_continuity is not None:
            audited = contact_continuity.get("datasets", {}).get(size, {})
            if Path(audited.get("dataset_root", "")).resolve() != source:
                raise ValueError(
                    f"{size}: contact-continuity dataset root mismatch"
                )
            if audited.get("sampling_contract_sha256") != _sha256(
                source / "sampling_contract.json"
            ):
                raise ValueError(
                    f"{size}: contact-continuity sampling contract drift"
                )
            if audited.get(
                "trial_manifest_sha256"
            ) != trial_manifest_sha256(source):
                raise ValueError(
                    f"{size}: contact-continuity trial bytes drift"
                )
            expected_hygiene = {
                "direction_supervision_admitted": audited.get(
                    "direction_supervision_admitted"
                ),
                "direction_supervision_quarantined": audited.get(
                    "direction_supervision_quarantined"
                ),
                "direction_quarantine_reason_counts": audited.get(
                    "direction_quarantine_reason_counts", {}
                ),
                "direction_diagnostic_counts": audited.get(
                    "direction_diagnostic_counts", {}
                ),
            }
            for key, expected in expected_hygiene.items():
                if hygiene.get(key) != expected:
                    raise ValueError(
                        f"{size}: contact-continuity {key} mismatch"
                    )
        source_train.extend(development_train)
        source_calibration.extend(development_calibration)
        source_development_records.append(
            {
                "path": str(source),
                "sampling_contract_sha256": _sha256(
                    source / "sampling_contract.json"
                ),
                "original_training_use": "forbidden",
                "new_role": "development_after_failed_v1_evaluation",
                "explicit_failed_audit_opt_in": True,
                "geometry_profile": contract["geometry"]["profile"],
                "spec_path": str(
                    _resolve_from_root(contract["geometry"]["spec_path"])
                ),
                "spec_sha256": contract["geometry"]["spec_sha256"],
                "model_path": str(
                    _resolve_from_root(contract["geometry"]["model_path"])
                ),
                "model_sha256": contract["geometry"]["model_sha256"],
                "all_ok_trials": len(trials),
                "valid_full_fingerprints": valid_total,
                "capture_or_incomplete_excluded": len(trials) - valid_total,
                "direction_hygiene": hygiene,
                "train_examples_after_capture_filter": len(development_train),
                "calibration_examples_after_capture_filter": len(
                    development_calibration
                ),
                **split,
            }
        )
    train_examples = {"m8_source": source_train}
    calibration_examples = {"m8_source": source_calibration}
    easy_development_by_size = _parse_easy_development_values(
        tuple(getattr(args, "easy_development_data", ()))
    )
    provenance = {
        "m8_source": {
            **source_provenance,
            "train_examples_after_development_merge": len(source_train),
            "calibration_examples_after_development_merge": len(
                source_calibration
            ),
        },
        "m8_source_development": source_development_records,
        "easy_development": {size: [] for size in SIZES},
        "easy_domains": {},
    }
    dataset_records = {}
    data_paths = {
        "m8": Path(args.m8_easy_data).resolve(),
        "m10": Path(args.m10_easy_data).resolve(),
        "m12": Path(args.m12_easy_data).resolve(),
    }
    for offset, size in enumerate(SIZES):
        source = data_paths[size]
        contract, spec, trials = _load_easy_dataset(
            size,
            source,
            geometry_equivalence_audit=geometry_equivalence,
        )
        train_groups, calibration_groups, split = _split_physical_orbits(
            trials, seed=args.split_seed + 1000 + 100 * offset
        )
        bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
        domain = f"{size}_easy"
        train = _easy_examples(
            trials, train_groups, bit_radius_m=bit_radius, size=size
        )
        calibration = _easy_examples(
            trials, calibration_groups, bit_radius_m=bit_radius, size=size
        )
        if not train or not calibration:
            raise ValueError(f"{domain}: no valid full fingerprints after grouped split")
        hygiene = _direction_hygiene_summary(trials)
        valid_total = int(hygiene["direction_supervision_admitted"])
        development_records = []
        for development_index, development_source in enumerate(
            easy_development_by_size[size]
        ):
            development_contract, development_spec, development_trials = (
                _load_forbidden_easy_dataset(size, development_source)
            )
            development_train_groups, development_calibration_groups, development_split = (
                _split_physical_orbits(
                    development_trials,
                    seed=(
                        args.split_seed
                        + 3000
                        + 100 * offset
                        + 10 * development_index
                    ),
                )
            )
            development_bit_radius = float(
                development_spec["key_af_m"]
            ) / math.sqrt(3.0)
            if not math.isclose(
                development_bit_radius, bit_radius, rel_tol=0.0, abs_tol=1.0e-15
            ):
                raise ValueError(
                    f"{development_source}: easy development bit radius drift"
                )
            development_train = _easy_examples(
                development_trials,
                development_train_groups,
                bit_radius_m=development_bit_radius,
                size=size,
            )
            development_calibration = _easy_examples(
                development_trials,
                development_calibration_groups,
                bit_radius_m=development_bit_radius,
                size=size,
            )
            if not development_train or not development_calibration:
                raise ValueError(
                    f"{development_source}: no valid easy fingerprints after "
                    "physical-orbit split"
                )
            development_hygiene = _direction_hygiene_summary(
                development_trials
            )
            development_valid_total = int(
                development_hygiene["direction_supervision_admitted"]
            )
            train.extend(development_train)
            calibration.extend(development_calibration)
            development_records.append(
                {
                    "size": size,
                    "path": str(development_source),
                    "sampling_contract_sha256": _sha256(
                        development_source / "sampling_contract.json"
                    ),
                    "original_training_use": "forbidden",
                    "new_role": "development_after_failed_v2_evaluation",
                    "explicit_failed_audit_opt_in": True,
                    "geometry_profile": development_contract["geometry"][
                        "profile"
                    ],
                    "spec_path": str(
                        _resolve_from_root(
                            development_contract["geometry"]["spec_path"]
                        )
                    ),
                    "spec_sha256": development_contract["geometry"][
                        "spec_sha256"
                    ],
                    "model_path": str(
                        _resolve_from_root(
                            development_contract["geometry"]["model_path"]
                        )
                    ),
                    "model_sha256": development_contract["geometry"][
                        "model_sha256"
                    ],
                    "bit_radius_m": development_bit_radius,
                    "all_ok_trials": len(development_trials),
                    "valid_full_fingerprints": development_valid_total,
                    "capture_or_incomplete_excluded": (
                        len(development_trials) - development_valid_total
                    ),
                    "direction_hygiene": development_hygiene,
                    "train_examples_after_capture_filter": len(
                        development_train
                    ),
                    "calibration_examples_after_capture_filter": len(
                        development_calibration
                    ),
                    **development_split,
                }
            )
        train_examples[domain] = train
        calibration_examples[domain] = calibration
        provenance["easy_development"][size] = development_records
        provenance["easy_domains"][size] = {
            **split,
            "source": str(source),
            "sampling_contract_sha256": _sha256(source / "sampling_contract.json"),
            "spec_path": str(_resolve_from_root(contract["geometry"]["spec_path"])),
            "spec_sha256": contract["geometry"]["spec_sha256"],
            "bit_radius_m": bit_radius,
            "all_ok_trials": len(trials),
            "valid_full_fingerprints": valid_total,
            "capture_or_incomplete_excluded": len(trials) - valid_total,
            "direction_hygiene": hygiene,
            "canonical_train_examples_before_development_merge": (
                len(train) - sum(
                    record["train_examples_after_capture_filter"]
                    for record in development_records
                )
            ),
            "canonical_calibration_examples_before_development_merge": (
                len(calibration) - sum(
                    record["calibration_examples_after_capture_filter"]
                    for record in development_records
                )
            ),
            "development_datasets_merged": len(development_records),
            "train_examples_after_development_merge": len(train),
            "calibration_examples_after_development_merge": len(calibration),
        }
        dataset_records[size] = {
            "root": str(source),
            "sampling_contract": str(source / "sampling_contract.json"),
            "sampling_contract_sha256": _sha256(source / "sampling_contract.json"),
            "spec_path": str(_resolve_from_root(contract["geometry"]["spec_path"])),
            "spec_sha256": contract["geometry"]["spec_sha256"],
            "model_path": str(_resolve_from_root(contract["geometry"]["model_path"])),
            "model_sha256": contract["geometry"]["model_sha256"],
            "bit_radius_m": bit_radius,
        }
    easy_weight = (1.0 - args.source_weight) / len(SIZES)
    weights = {"m8_source": float(args.source_weight)}
    weights.update({f"{size}_easy": float(easy_weight) for size in SIZES})
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1.0e-12):
        raise RuntimeError("domain weights do not sum to one")
    provenance["domain_weights"] = weights
    provenance["split_seed"] = args.split_seed
    provenance["captured_or_incomplete_traces_used_for_direction_supervision"] = 0
    provenance[
        "contact_transition_or_capture_quarantined_traces_used_for_direction_supervision"
    ] = 0
    provenance["geometry_equivalence_audit"] = geometry_equivalence_record
    provenance["contact_continuity_audit"] = contact_continuity_record
    return train_examples, calibration_examples, weights, {
        "split": provenance,
        "datasets": dataset_records,
        "geometry_equivalence_audit": geometry_equivalence_record,
        "contact_continuity_audit": contact_continuity_record,
    }


def _domain_arrays(
    examples: dict[str, list[dict]],
    context_mean: np.ndarray,
    context_std: np.ndarray,
) -> dict[str, tuple[np.ndarray, ...]]:
    return {
        domain: training._arrays(items, context_mean, context_std)
        for domain, items in examples.items()
    }


def _configure_trainable(
    model: training.CircularDirectionCNN, adaptation_mode: str
) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules: list[nn.Module] = [model.state_head, model.direction_head]
    if adaptation_mode == "tail":
        modules.extend((model.encoder[8], model.shared))
    elif adaptation_mode != "heads":
        raise ValueError(f"unknown adaptation mode: {adaptation_mode}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    # Keep dropout and every batch-normalization layer in evaluation mode.
    model.eval()
    return tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def _fit_seed(
    base_checkpoint: Path,
    train_arrays: dict[str, tuple[np.ndarray, ...]],
    calibration_arrays: dict[str, tuple[np.ndarray, ...]],
    weights: dict[str, float],
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    l2sp: float,
    soft_sigma_deg: float,
    adaptation_mode: str,
) -> tuple[training.CircularDirectionCNN, list[dict], int, tuple[str, ...]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = training.CircularDirectionCNN().to(device)
    model.load_state_dict(
        torch.load(base_checkpoint, map_location=device, weights_only=True)
    )
    trainable_names = _configure_trainable(model, adaptation_mode)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    rng = np.random.default_rng(seed)
    steps = max(
        math.ceil(len(arrays[0]) / batch_size) for arrays in train_arrays.values()
    )
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            loss = torch.zeros((), dtype=torch.float32, device=device)
            for domain, arrays in train_arrays.items():
                count = len(arrays[0])
                indices = rng.choice(
                    count, size=min(batch_size, count), replace=True
                )
                loss = loss + weights[domain] * m8_adaptation._batch_loss(
                    model, arrays, indices, device, soft_sigma_deg
                )
            penalty = sum(
                torch.mean((parameter - initial[name]) ** 2)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
            loss = loss + l2sp * penalty
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation_losses = {
            domain: m8_adaptation._validation_loss(
                model, arrays, device, soft_sigma_deg
            )
            for domain, arrays in calibration_arrays.items()
        }
        score = max(validation_losses.values())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_loss_by_domain": validation_losses,
                "worst_domain_validation_loss": score,
            }
        )
        if score < best_score - 1.0e-4:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("multisize adaptation produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, trainable_names


def _balanced_domain_indices(counts: dict[str, int]) -> dict[str, np.ndarray]:
    target = max(counts.values())
    return {
        domain: np.resize(np.arange(count, dtype=int), target)
        for domain, count in counts.items()
    }


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    train_examples, calibration_examples, weights, provenance = _prepare_domains(args)
    split_manifest = {
        **provenance["split"],
        "created_utc": _utc_now(),
        "claim_boundary": (
            "Simulation-only grouped split. Every easy-v2 split is frozen at the "
            "complete physical C6-orbit level before captured traces are filtered."
        ),
    }
    _atomic_json(output / "split_manifest.json", split_manifest)

    base_root = Path(args.base_model).resolve()
    base_contract_path = base_root / "ensemble_interface_contract.json"
    base_contract = json.loads(base_contract_path.read_text(encoding="utf-8"))
    if len(base_contract.get("checkpoints", ())) != 3:
        raise ValueError("shared adaptation requires a three-seed base ensemble")

    per_domain_probabilities: dict[str, list[np.ndarray]] = {
        domain: [] for domain in calibration_examples
    }
    labels_by_domain: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    checkpoint_paths = []
    seed_records = []
    seen_seeds = set()
    for checkpoint_value in base_contract["checkpoints"]:
        base_checkpoint = _resolve_from_root(checkpoint_value)
        seed = int(base_checkpoint.parent.name.rsplit("_", 1)[-1])
        if seed in seen_seeds:
            raise ValueError(f"duplicate base seed {seed}")
        seen_seeds.add(seed)
        normalization_path = base_checkpoint.parent / "normalization.npz"
        normalization = np.load(normalization_path)
        context_mean = np.asarray(normalization["context_mean"], dtype=np.float32)
        context_std = np.asarray(normalization["context_std"], dtype=np.float32)
        train_arrays = _domain_arrays(train_examples, context_mean, context_std)
        calibration_arrays = _domain_arrays(
            calibration_examples, context_mean, context_std
        )
        model, history, best_epoch, trainable_names = _fit_seed(
            base_checkpoint,
            train_arrays,
            calibration_arrays,
            weights,
            seed=seed,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2sp=args.l2sp,
            soft_sigma_deg=args.soft_sigma_deg,
            adaptation_mode=args.adaptation_mode,
        )
        logits = {
            domain: training._collect_logits(model, arrays, device)
            for domain, arrays in calibration_arrays.items()
        }
        counts = {domain: len(value[3]) for domain, value in logits.items()}
        balanced = _balanced_domain_indices(counts)
        balanced_logits = np.concatenate(
            [logits[domain][1][balanced[domain]] for domain in sorted(logits)]
        )
        balanced_labels = np.concatenate(
            [logits[domain][3][balanced[domain]] for domain in sorted(logits)]
        )
        temperature = training._fit_temperature(
            balanced_logits, balanced_labels, device
        )
        for domain in sorted(logits):
            per_domain_probabilities[domain].append(
                m8_adaptation._softmax(logits[domain][1], temperature)
            )
            labels_by_domain[domain] = (logits[domain][3], logits[domain][4])

        seed_dir = output / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = seed_dir / "circular_cnn_best.pt"
        torch.save(model.state_dict(), checkpoint)
        shutil.copy2(normalization_path, seed_dir / "normalization.npz")
        metrics = {
            "stage": "hardware_easy_v2_shared_multisize_adaptation",
            "seed": seed,
            "temperature": temperature,
            "state_temperature": 1.0,
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": _sha256(base_checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "best_epoch": best_epoch,
            "history": history,
            "trainable_parameter_names": list(trainable_names),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "domain_weights": weights,
            "learning_rate": args.learning_rate,
            "l2sp": args.l2sp,
            "soft_direction_sigma_deg": args.soft_sigma_deg,
            "adaptation_mode": args.adaptation_mode,
            "temperature_calibration": "equal-count balance across all four domains",
            "batchnorm_running_statistics_frozen": True,
            "normalization_source": str(normalization_path),
            "updated_utc": _utc_now(),
        }
        _atomic_json(seed_dir / "metrics.json", metrics)
        checkpoint_paths.append(checkpoint)
        seed_records.append(metrics)

    mean_probabilities = {
        domain: np.mean(np.stack(values, axis=0), axis=0)
        for domain, values in per_domain_probabilities.items()
    }
    counts = {
        domain: len(labels_by_domain[domain][0]) for domain in mean_probabilities
    }
    balanced = _balanced_domain_indices(counts)
    balanced_probabilities = np.concatenate(
        [
            mean_probabilities[domain][balanced[domain]]
            for domain in sorted(mean_probabilities)
        ]
    )
    balanced_labels = np.concatenate(
        [
            labels_by_domain[domain][0][balanced[domain]]
            for domain in sorted(mean_probabilities)
        ]
    )
    aps_quantile = m8_adaptation._aps_quantile(
        balanced_probabilities,
        balanced_labels,
        alpha=args.aps_alpha,
    )
    features = {
        domain: m8_adaptation._gate_features(
            mean_probabilities[domain],
            np.stack(per_domain_probabilities[domain], axis=0),
            labels_by_domain[domain][1],
            aps_quantile,
        )
        for domain in mean_probabilities
    }
    inherited_aps_cap = int(
        base_contract.get("structured_transport_gate", {}).get(
            "max_aps_set_size", training.DIRECTION_CLASSES
        )
    )
    gate = m8_adaptation._select_worst_domain_gate(
        features, max_aps_size_cap=inherited_aps_cap
    )
    domain_metrics = {
        domain: m8_adaptation._metrics(
            features[domain],
            gate,
            mean_probabilities[domain],
            labels_by_domain[domain][0],
        )
        for domain in sorted(mean_probabilities)
    }

    contract = {
        "checkpoints": [str(path) for path in checkpoint_paths],
        "combiner": "mean globally calibrated class probabilities",
        "confidence_gate": gate["min_max_probability"],
        "aps_quantile": aps_quantile,
        "aps_miscoverage_target": args.aps_alpha,
        "direction_bins": training.DIRECTION_CLASSES,
        "direction_bin_width_deg": training.DIRECTION_BIN_DEG,
        "structured_transport_gate": gate,
        "state_role": base_contract["state_role"],
        "direction_role": base_contract["direction_role"],
        "fallback": base_contract["fallback"],
        "supported_sizes": list(SIZES),
        "runtime_input_contract": {
            "phase_channels": list(training.INPUT_CHANNELS),
            "phase_bins": 360,
            "size_argument": "caller must pass key_af_m/sqrt(3) as bit_radius_m",
            "size_bit_radius_m": {
                size: record["bit_radius_m"]
                for size, record in provenance["datasets"].items()
            },
        },
        "adaptation_contract": {
            "model_scope": "one shared M8/M10/M12 weight ensemble",
            "base_model_root": str(base_root),
            "base_model_contract_sha256": _sha256(base_contract_path),
            "geometry_profile": "hardware_easy_v2",
            "domain_weights": weights,
            "domain_balancing": "one independent batch per domain per optimizer step",
            "temperature_policy": "global equal-domain calibration for runtime compatibility",
            "aps_and_gate_policy": "single conservative gate selected on the worst of four calibration domains",
            "aps_miscoverage_target": args.aps_alpha,
            "adaptation_mode": args.adaptation_mode,
            "trainable_modules": (
                ["state_head", "direction_head"]
                if args.adaptation_mode == "heads"
                else [
                    "encoder.last_conv",
                    "shared",
                    "state_head",
                    "direction_head",
                ]
            ),
            "encoder_and_shared_frozen": args.adaptation_mode == "heads",
            "batchnorm_running_statistics_frozen": True,
            "per_seed_normalization_frozen_from_audited_base": True,
            "l2sp_anchored_to_audited_base": True,
            "easy_split_before_capture_filter": True,
            "captured_or_incomplete_traces_used_for_direction_supervision": 0,
            "contact_transition_or_capture_quarantined_traces_used_for_direction_supervision": 0,
            "final_promotion_audit_samples_seen_during_fitting": 0,
            "max_aps_set_size_cap_inherited_from_base": inherited_aps_cap,
            "dataset_contracts": provenance["datasets"],
            "geometry_equivalence_audit": provenance[
                "geometry_equivalence_audit"
            ],
            "contact_continuity_audit": provenance[
                "contact_continuity_audit"
            ],
            "m8_source_development_datasets_consumed": provenance["split"][
                "m8_source_development"
            ],
            "easy_development_datasets_consumed": provenance["split"][
                "easy_development"
            ],
        },
        "claim_boundary": (
            "Simulation-only shared M8/M10/M12 contact-geometry adaptation. "
            "It does not align the Bota measurement origin, common 42 mm hardware "
            "sensor-to-tip lever, printed mass/COM/inertia, FDM compliance/friction, "
            "or measured sensor noise, and it is not full-loop evidence."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "ensemble_interface_contract.json", contract)
    report = {
        "stage": "hardware_easy_v2_shared_multisize_adaptation",
        "device": str(device),
        "base_model_root": str(base_root),
        "base_model_contract_sha256": _sha256(base_contract_path),
        "dataset_contracts": provenance["datasets"],
        "geometry_equivalence_audit": provenance["geometry_equivalence_audit"],
        "contact_continuity_audit": provenance[
            "contact_continuity_audit"
        ],
        "adaptation_mode": args.adaptation_mode,
        "m8_source_development_datasets_consumed": provenance["split"][
            "m8_source_development"
        ],
        "easy_development_datasets_consumed": provenance["split"][
            "easy_development"
        ],
        "split_manifest_sha256": _sha256(output / "split_manifest.json"),
        "seeds": seed_records,
        "domain_weights": weights,
        "aps_quantile": aps_quantile,
        "aps_miscoverage_target": args.aps_alpha,
        "structured_transport_gate": gate,
        "calibration_metrics": domain_metrics,
        "test_samples_seen": 0,
        "final_audit_run": False,
        "claim_boundary": contract["claim_boundary"],
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "adaptation_report.json", report)
    lines = [
        "# Shared M8/M10/M12 hardware-easy-v2 adaptation",
        "",
        "One three-seed weight ensemble was initialized from the audited clean M8 base.",
        "M8 source replay and all three easy-v2 sizes are balanced by domain.",
        f"Adaptation mode: {args.adaptation_mode}.",
        "Captured/incomplete easy traces were excluded after physical-orbit splitting.",
        "The global calibration and gate are deliberately conservative and runtime-compatible.",
        f"APS calibration target: {1.0 - args.aps_alpha:.1%} coverage (alpha={args.aps_alpha:.3f}).",
        "",
        "| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, item in domain_metrics.items():
        accepted_accuracy = item["transport_accepted_within_7p5_rate"]
        lines.append(
            f"| {domain} | {item['samples']} | {item['angle_error_median_deg']:.2f} / "
            f"{item['angle_error_p90_deg']:.2f} deg | {item['angle_within_7p5_rate']:.3f} | "
            f"{item['aps_coverage']:.3f} / {item['aps_mean_set_size']:.2f} | "
            f"{item['transport_accept_rate']:.3f} / "
            f"{accepted_accuracy if accepted_accuracy is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "This table is calibration/development evidence only; no fresh audit was run by this trainer.",
            "",
            "Boundary: " + contract["claim_boundary"],
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "calibration": domain_metrics,
                "final_audit_run": False,
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(BASE_ROOT))
    parser.add_argument(
        "--geometry-equivalence-audit",
        default=None,
        help=(
            "explicit PASS report permitting metadata-only spec-hash drift; "
            "also locks the clean base and forbids failed-audit development data"
        ),
    )
    parser.add_argument(
        "--contact-continuity-audit",
        default=None,
        help=(
            "PASS report binding the exact canonical NPZ bytes to the "
            "reason-coded raw-contact direction-supervision filter"
        ),
    )
    parser.add_argument("--m8-easy-data", default=str(DEFAULT_DATA["m8"]))
    parser.add_argument("--m10-easy-data", default=str(DEFAULT_DATA["m10"]))
    parser.add_argument("--m12-easy-data", default=str(DEFAULT_DATA["m12"]))
    parser.add_argument(
        "--m8-source-development-data",
        action="append",
        default=[],
        help=(
            "failed M8 frozen-source audit to consume as development; may be "
            "repeated, and each contract must retain training_use=forbidden"
        ),
    )
    parser.add_argument(
        "--easy-development-data",
        action="append",
        default=[],
        metavar="SIZE=PATH",
        help=(
            "failed hardware-easy-v2 audit to consume as size-matched "
            "development; may be repeated, and every contract must retain "
            "training_use=forbidden"
        ),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--l2sp", type=float, default=1.0e-2)
    parser.add_argument("--source-weight", type=float, default=0.40)
    parser.add_argument("--soft-sigma-deg", type=float, default=6.0)
    parser.add_argument(
        "--aps-alpha",
        type=float,
        default=0.05,
        help="finite-sample APS miscoverage target; 0.05 aligns with the 95%% audit gate",
    )
    parser.add_argument("--split-seed", type=int, default=20260731)
    parser.add_argument(
        "--adaptation-mode", choices=("heads", "tail"), default="tail"
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("epochs, patience and batch size must be positive")
    if args.learning_rate <= 0.0 or args.l2sp < 0.0:
        parser.error("learning rate must be positive and l2sp nonnegative")
    if not 0.0 < args.source_weight < 1.0:
        parser.error("source weight must lie strictly between zero and one")
    if args.soft_sigma_deg < 0.0:
        parser.error("soft sigma must be nonnegative")
    if not 0.0 < args.aps_alpha < 1.0:
        parser.error("APS alpha must lie strictly between zero and one")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
