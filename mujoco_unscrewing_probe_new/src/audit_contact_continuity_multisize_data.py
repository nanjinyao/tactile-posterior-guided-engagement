"""Freeze a reason-coded contact-continuity audit for canonical easy-v2 data.

The report supplements, but never rewrites, the earlier geometry-equivalence
audit.  It binds the exact immutable NPZ bytes to the direction-supervision
decision recomputed from each raw conical scan.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from src.sample_m8_conical_pilot import (
    CONTACT_CONTINUITY_AUDIT_SCHEMA_VERSION,
    LOW_FORCE_ABSOLUTE_FLOOR_N,
    LOW_FORCE_TARGET_FRACTION,
    SUSTAINED_CONTACT_EVENT_MIN_S,
    WITHIN_SCAN_PROGRESS_CHAMFER_FRACTION,
    audit_dataset_contact_quarantine,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    "m8": ROOT
    / "outputs/current/datasets/m8_hardware_easy_v2_direction_dense_training",
    "m10": ROOT
    / "outputs/current/datasets/hardware_easy_v2_multisize_dense/m10",
    "m12": ROOT
    / "outputs/current/datasets/hardware_easy_v2_multisize_dense/m12",
}
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/reports/"
    "hardware_easy_v2_multisize_contact_continuity_data_audit.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trial_manifest_sha256(dataset_root: Path) -> str:
    """Hash ordered trial names and bytes without creating another manifest."""

    root = Path(dataset_root).resolve()
    digest = hashlib.sha256()
    paths = sorted((root / "trials").glob("*.npz"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_report(datasets: dict[str, Path]) -> dict:
    records = {}
    for size, root_value in sorted(datasets.items()):
        root = Path(root_value).resolve()
        sampling_contract = root / "sampling_contract.json"
        if not sampling_contract.is_file():
            raise FileNotFoundError(sampling_contract)
        audit = audit_dataset_contact_quarantine(root)
        records[size] = {
            "dataset_root": str(root),
            "sampling_contract": str(sampling_contract),
            "sampling_contract_sha256": _sha256(sampling_contract),
            "trial_manifest_sha256": trial_manifest_sha256(root),
            "trial_count": audit["trial_count"],
            "direction_supervision_admitted": audit["admitted_count"],
            "direction_supervision_quarantined": audit[
                "quarantined_count"
            ],
            "direction_quarantine_reason_counts": audit["reason_counts"],
            "direction_diagnostic_counts": audit["diagnostic_counts"],
            "source_trials_rewritten": False,
        }
    return {
        "schema_version": 1,
        "stage": "hardware_easy_v2_multisize_contact_continuity_data_audit",
        "status": "PASS",
        "policy": {
            "per_trial_schema_version": (
                CONTACT_CONTINUITY_AUDIT_SCHEMA_VERSION
            ),
            "sustained_event_min_s": SUSTAINED_CONTACT_EVENT_MIN_S,
            "low_force_target_fraction": LOW_FORCE_TARGET_FRACTION,
            "low_force_absolute_floor_n": LOW_FORCE_ABSOLUTE_FLOOR_N,
            "within_scan_progress_chamfer_fraction": (
                WITHIN_SCAN_PROGRESS_CHAMFER_FRACTION
            ),
            "low_force_with_continuous_contact": (
                "diagnostic_only_not_quarantine"
            ),
            "privileged_contact_count_role": (
                "offline_dataset_hygiene_only_never_online_control"
            ),
        },
        "datasets": records,
        "sampler_source": str(
            ROOT / "src/sample_m8_conical_pilot.py"
        ),
        "sampler_source_sha256": _sha256(
            ROOT / "src/sample_m8_conical_pilot.py"
        ),
        "claim_boundary": (
            "Read-only simulation direction-label hygiene. This report does "
            "not certify online capture, model accuracy, sim-to-real "
            "transfer, or hardware control."
        ),
        "created_utc": _utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m8-data", default=str(DEFAULT_DATASETS["m8"]))
    parser.add_argument("--m10-data", default=str(DEFAULT_DATASETS["m10"]))
    parser.add_argument("--m12-data", default=str(DEFAULT_DATASETS["m12"]))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        {
            "m8": Path(args.m8_data),
            "m10": Path(args.m10_data),
            "m12": Path(args.m12_data),
        }
    )
    output = Path(args.output).resolve()
    _atomic_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "admitted": {
                    size: record["direction_supervision_admitted"]
                    for size, record in report["datasets"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
