"""Build a simulation-only promotion candidate after an untouched fresh7 PASS.

The command is fail-closed: it writes nothing unless the lineage audit passes,
all three frozen size gates pass, all four evaluation datasets are fresh7 and
``training_use=forbidden``, and the evaluated model is byte-identical to the
lineage-audited candidate.  It never edits a deployment/runtime manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    ROOT
    / "outputs/current/models/hardware_easy_v2_multisize_clocking_clean_direction_v5"
)
DEFAULT_LINEAGE = (
    ROOT
    / "outputs/current/reports/"
    "hardware_easy_v2_multisize_clocking_clean_direction_v5_lineage_audit.json"
)
DEFAULT_EVALUATION = (
    ROOT
    / "outputs/current/evaluations/"
    "hardware_easy_v2_multisize_clocking_clean_direction_v5_fresh7_final/"
    "report.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/reports/"
    "hardware_easy_v2_multisize_clocking_clean_direction_v5_promotion_candidate.json"
)
DOMAINS = ("m8_source", "m8_easy", "m10_easy", "m12_easy")


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


def build_candidate(
    model_root: Path, lineage_path: Path, evaluation_path: Path
) -> dict[str, Any]:
    model_root = model_root.resolve()
    lineage_path = lineage_path.resolve()
    evaluation_path = evaluation_path.resolve()
    lineage = json.loads(lineage_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if lineage.get("status") != "PASS":
        raise ValueError("lineage audit did not pass")
    if Path(lineage["model_root"]).resolve() != model_root:
        raise ValueError("lineage audit refers to a different model")
    if not evaluation.get(
        "promotion_ready_for_geometry_conditioned_multisize_simulation", False
    ):
        raise ValueError("fresh7 promotion gates did not all pass")
    if not all(
        evaluation["size_promotion_gates"][size]["passed"]
        for size in ("m8", "m10", "m12")
    ):
        raise ValueError("at least one size gate failed")

    evaluated_model = evaluation["models"]["joint_multisize"]
    if Path(evaluated_model["root"]).resolve() != model_root:
        raise ValueError("fresh7 evaluated a different model")
    contract_path = model_root / "ensemble_interface_contract.json"
    if (
        evaluated_model["ensemble_interface_contract_sha256"]
        != _sha256(contract_path)
    ):
        raise ValueError("evaluated model contract hash drift")
    if (
        lineage["model_artifacts"]["ensemble_interface_contract.json"]["sha256"]
        != _sha256(contract_path)
    ):
        raise ValueError("lineage model contract hash drift")

    datasets = {}
    for domain in DOMAINS:
        item = evaluation["datasets"][domain]
        if item.get("training_use") != "forbidden":
            raise ValueError(f"{domain}: final dataset is not frozen/forbidden")
        if "fresh7" not in str(item.get("stage", "")).lower():
            raise ValueError(f"{domain}: final dataset is not fresh7")
        metrics = evaluation["results"]["joint_multisize"][domain]["metrics"]
        if int(metrics["total_task_outcomes"]) != 180:
            raise ValueError(f"{domain}: final audit is incomplete")
        datasets[domain] = {
            "source": item["source"],
            "sampling_contract_sha256": item["sampling_contract_sha256"],
            "training_use": "forbidden",
            "stage": item["stage"],
            "valid_full_fingerprint_samples": metrics[
                "valid_full_fingerprint_samples"
            ],
            "capture_or_incomplete_routed_samples": metrics[
                "capture_or_incomplete_routed_samples"
            ],
            "hard_stop_outcome_count": metrics["hard_stop_outcome_count"],
            "angle_error_median_deg": metrics["angle_error_median_deg"],
            "angle_error_p90_deg": metrics["angle_error_p90_deg"],
            "transport_accept_rate": metrics["transport_accept_rate"],
            "transport_accepted_within_7p5_rate": metrics[
                "transport_accepted_within_7p5_rate"
            ],
            "transport_accepted_catastrophic_gt90_count": metrics[
                "transport_accepted_catastrophic_gt90_count"
            ],
        }

    return {
        "schema_version": 1,
        "stage": "clocking_clean_multisize_simulation_promotion_candidate",
        "status": "CANDIDATE_PASS",
        "decision": (
            "Eligible to be considered for geometry-conditioned multisize "
            "simulation use. Runtime/deployment promotion remains a separate, "
            "explicit review."
        ),
        "model_root": str(model_root),
        "lineage_audit": {
            "path": str(lineage_path),
            "sha256": _sha256(lineage_path),
            "status": "PASS",
        },
        "fresh7_evaluation": {
            "path": str(evaluation_path),
            "sha256": _sha256(evaluation_path),
            "size_gates": evaluation["size_promotion_gates"],
        },
        "datasets": datasets,
        "runtime_manifest_modified": False,
        "claim_boundary": (
            "Simulation-only promotion candidate. This is not real-device "
            "fine-tuning, real full-loop evidence, or permission to give the "
            "learned posterior hardware motion authority."
        ),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--lineage-audit", default=str(DEFAULT_LINEAGE))
    parser.add_argument("--fresh7-evaluation", default=str(DEFAULT_EVALUATION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = build_candidate(
        Path(args.model), Path(args.lineage_audit), Path(args.fresh7_evaluation)
    )
    _atomic_json(Path(args.output).resolve(), candidate)
    print(json.dumps({"status": candidate["status"], "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
