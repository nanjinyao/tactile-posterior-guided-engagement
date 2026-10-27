from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.run_m8_strict_complete_loop_audit import _strict_bottom_success


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/current/evaluations/m8_strict_24_complete_loop"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/evaluations/m8_strict24_fingerprint_quality_stratification"
)

FINGERPRINT_METHODS = ("proposed", "proposed_plus", "direct_first")

# These thresholds are intentionally declared as constants.  They describe a
# post-hoc visualization stratum over the already frozen strict-24 audit; they
# are not a learned decision rule and must not be read as a calibrated quality
# classifier.
LOCKED_RULES: dict[str, dict[str, dict[str, float | int]]] = {
    "fuzzy_short": {
        "aps_set_size": {"operator": ">=", "value": 5},
        "ensemble_top1_span_deg": {"operator": ">=", "value": 15.0},
        "max_proposed_transport_mm": {"operator": "<=", "value": 1.2},
    },
    "clear_long": {
        "aps_set_size": {"operator": "<=", "value": 3},
        "circular_concentration": {"operator": ">=", "value": 0.995},
        "ensemble_top1_span_deg": {"operator": "<=", "value": 7.5},
        "min_proposed_transport_mm": {"operator": ">=", "value": 2.7},
    },
}

CLAIM_BOUNDARY = (
    "Post-hoc descriptive stratification of one frozen strict-24 M8 simulation "
    "audit. The thresholds were not preregistered, the two observed count ratios "
    "are not probability estimates or significance tests, and the strata do not "
    "establish generalization, causality, hardware transfer, or a guaranteed "
    "relationship between fingerprint quality and direct-first success."
)

STRICT_SUCCESS_CONTRACT = {
    "source_function": "src.run_m8_strict_complete_loop_audit._strict_bottom_success",
    "requires": [
        "inner_wall_confirmed",
        "offline_capture_validated",
        "bottom_reached",
        "result == cad_bottom_verified",
        "terminal_verification_mode in {bottom_force_dwell, cad_depth_low_wrench_dwell}",
        "hard_stop == false",
        "total_active_time_s <= 130.0",
        "fallback_budget_extension_s == 0.0",
    ],
    "capture_alone_is_not_success": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _same_fingerprint_contract(record: dict[str, Any], pose_id: str) -> None:
    methods = record["methods"]
    fingerprints = {
        methods[name].get("fingerprint_sha256") for name in FINGERPRINT_METHODS
    }
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError(f"{pose_id}: fingerprint SHA mismatch across shared methods")

    for field in (
        "aps_set_size",
        "circular_concentration",
        "ensemble_top1_span_deg",
    ):
        values = {methods[name].get(field) for name in FINGERPRINT_METHODS}
        if None in values or len(values) != 1:
            raise ValueError(f"{pose_id}: {field} mismatch across shared methods")


def _pose_row(record: dict[str, Any]) -> dict[str, Any]:
    pose_id = str(record["task"]["pose_id"])
    _same_fingerprint_contract(record, pose_id)
    methods = record["methods"]
    proposed = methods["proposed"]
    proposed_plus = methods["proposed_plus"]
    direct = methods["direct_first"]
    return {
        "pose_id": pose_id,
        "aps_set_size": int(proposed["aps_set_size"]),
        "circular_concentration": float(proposed["circular_concentration"]),
        "ensemble_top1_span_deg": float(proposed["ensemble_top1_span_deg"]),
        "proposed_transport_mm": float(proposed["transport_distance_mm"]),
        "proposed_plus_transport_mm": float(
            proposed_plus["transport_distance_mm"]
        ),
        "direct_first_strict_bottom_success": _strict_bottom_success(direct),
        "direct_first_offline_capture_validated": bool(
            direct.get("offline_capture_validated")
        ),
        "direct_first_result": str(direct.get("result")),
        "fingerprint_sha256": str(proposed["fingerprint_sha256"]),
    }


def _in_fuzzy_short(row: dict[str, Any]) -> bool:
    return bool(
        row["aps_set_size"] >= 5
        and row["ensemble_top1_span_deg"] >= 15.0
        and max(
            row["proposed_transport_mm"], row["proposed_plus_transport_mm"]
        )
        <= 1.2
    )


def _in_clear_long(row: dict[str, Any]) -> bool:
    return bool(
        row["aps_set_size"] <= 3
        and row["circular_concentration"] >= 0.995
        and row["ensemble_top1_span_deg"] <= 7.5
        and min(
            row["proposed_transport_mm"], row["proposed_plus_transport_mm"]
        )
        >= 2.7
    )


def _stratum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(
        bool(row["direct_first_strict_bottom_success"]) for row in rows
    )
    return {
        "member_count": len(rows),
        "direct_first_strict_bottom_success_count": successes,
        "direct_first_strict_bottom_failure_count": len(rows) - successes,
        "observed_count_ratio": f"{successes}/{len(rows)}",
        "members": rows,
    }


def build_report(source: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source = source.resolve()
    manifest_path = source / "manifest.json"
    source_report_path = source / "report.json"
    manifest = _read_json(manifest_path)
    source_report = _read_json(source_report_path)

    if int(manifest.get("completed", -1)) != 24:
        raise ValueError("strict source manifest is not the frozen 24-pose audit")
    if source_report.get("experiment") != "M8_strict_24_pose_complete_closed_loop":
        raise ValueError("strict source report has an unexpected experiment contract")
    if int(source_report.get("design", {}).get("poses", -1)) != 24:
        raise ValueError("strict source report does not declare 24 poses")

    expected_pose_ids = [str(task["pose_id"]) for task in manifest["tasks"]]
    if len(expected_pose_ids) != 24 or len(set(expected_pose_ids)) != 24:
        raise ValueError("strict source manifest does not contain 24 unique poses")

    trial_paths = sorted((source / "trials").glob("sobol_*.json"))
    records = [_read_json(path) for path in trial_paths]
    actual_pose_ids = [str(record["task"]["pose_id"]) for record in records]
    if sorted(actual_pose_ids) != sorted(expected_pose_ids):
        raise ValueError("strict trial set does not match the source manifest")

    all_rows = sorted((_pose_row(record) for record in records), key=lambda row: row["pose_id"])
    fuzzy_rows = [row for row in all_rows if _in_fuzzy_short(row)]
    clear_rows = [row for row in all_rows if _in_clear_long(row)]
    overlap = sorted(
        {row["pose_id"] for row in fuzzy_rows}
        & {row["pose_id"] for row in clear_rows}
    )
    if overlap:
        raise ValueError(f"strata unexpectedly overlap: {overlap}")

    return {
        "artifact": "M8_strict24_fingerprint_quality_post_hoc_stratification",
        "schema_version": 1,
        "source": {
            "strict_audit_directory": _relative(source),
            "manifest": {
                "path": _relative(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "report": {
                "path": _relative(source_report_path),
                "sha256": _sha256(source_report_path),
            },
            "trial_count": len(all_rows),
            "pose_ids": [row["pose_id"] for row in all_rows],
        },
        "analysis_design": {
            "type": "post_hoc_descriptive_stratification",
            "preregistered": False,
            "locked_rules": LOCKED_RULES,
            "quality_signals_are_shared_across": list(FINGERPRINT_METHODS),
            "strict_success_contract": STRICT_SUCCESS_CONTRACT,
            "no_probability_estimate": True,
            "no_significance_test": True,
        },
        "strata": {
            "fuzzy_short": _stratum(fuzzy_rows),
            "clear_long": _stratum(clear_rows),
        },
        "unassigned_pose_ids": [
            row["pose_id"]
            for row in all_rows
            if not _in_fuzzy_short(row) and not _in_clear_long(row)
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def render_summary(report: dict[str, Any]) -> str:
    fuzzy = report["strata"]["fuzzy_short"]
    clear = report["strata"]["clear_long"]
    lines = [
        "# M8 strict-24 fingerprint-quality post-hoc stratification",
        "",
        "## Result",
        "",
        (
            "- `fuzzy_short`: "
            f"{fuzzy['member_count']} cases; direct-first strict CAD-bottom "
            f"success in {fuzzy['observed_count_ratio']}."
        ),
        (
            "- `clear_long`: "
            f"{clear['member_count']} cases; direct-first strict CAD-bottom "
            f"success in {clear['observed_count_ratio']}."
        ),
        "",
        (
            "Success is the existing full strict CAD-bottom contract, not offline "
            "capture alone."
        ),
        "",
        "## Locked post-hoc rules",
        "",
        (
            "- `fuzzy_short`: APS >= 5, ensemble span >= 15 deg, and "
            "max(Proposed, Proposed+) transport <= 1.2 mm."
        ),
        (
            "- `clear_long`: APS <= 3, concentration >= 0.995, ensemble span "
            "<= 7.5 deg, and min(Proposed, Proposed+) transport >= 2.7 mm."
        ),
        "",
    ]
    for name in ("fuzzy_short", "clear_long"):
        lines.extend(
            [
                f"## {name} members",
                "",
                (
                    "| pose | APS | concentration | span (deg) | Proposed "
                    "transport (mm) | Proposed+ transport (mm) | direct-first "
                    "strict bottom |"
                ),
                "|---|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in report["strata"][name]["members"]:
            lines.append(
                "| {pose_id} | {aps_set_size} | {circular_concentration:.6f} | "
                "{ensemble_top1_span_deg:.1f} | {proposed_transport_mm:.6f} | "
                "{proposed_plus_transport_mm:.6f} | {success} |".format(
                    success=(
                        "yes"
                        if row["direct_first_strict_bottom_success"]
                        else "no"
                    ),
                    **row,
                )
            )
        lines.append("")

    manifest = report["source"]["manifest"]
    source_report = report["source"]["report"]
    lines.extend(
        [
            "## Frozen source identity",
            "",
            f"- Strict manifest: `{manifest['path']}`",
            f"- Manifest SHA-256: `{manifest['sha256']}`",
            f"- Strict report: `{source_report['path']}`",
            f"- Report SHA-256: `{source_report['sha256']}`",
            "",
            "## Claim boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def write_artifact(
    source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT
) -> dict[str, Any]:
    report = build_report(source)
    _atomic_text(
        output / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(output / "summary.md", render_summary(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen strict-24 fingerprint-quality stratification."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = write_artifact(args.source, args.output)
    for name in ("fuzzy_short", "clear_long"):
        stratum = report["strata"][name]
        print(
            f"{name}: {stratum['direct_first_strict_bottom_success_count']}/"
            f"{stratum['member_count']} strict CAD-bottom successes"
        )
    print(args.output / "report.json")
    print(args.output / "summary.md")


if __name__ == "__main__":
    main()
