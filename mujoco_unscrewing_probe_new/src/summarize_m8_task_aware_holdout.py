from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.run_m8_six_method_joint_funnel_audit import METHODS, _summary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "outputs/current/evaluations/m8_six_method_task_aware_grid_audit"
)


def _load_records(root: Path) -> list[dict]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "trials").glob("sobol_*.json"))
    ]
    errors = [record for record in records if "error" in record]
    if errors:
        raise RuntimeError(f"audit contains {len(errors)} failed workers")
    if not records:
        raise RuntimeError(f"no trial records under {root / 'trials'}")
    records.sort(key=lambda record: record["task"]["pose_id"])
    return records


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _method_table(report: dict) -> list[str]:
    lines = [
        "| method | success | capture time | total/contact path | wear | task contacts | global fallback |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = report["methods"][method]
        task_contacts = item.get(
            "entropy_probe_count_median", item.get("probe_count_median")
        )
        lines.append(
            f"| {method} | {item['successes']}/{item['trials']} | "
            f"{_fmt(item['time_success_median_s'])} s | "
            f"{_fmt(item['actual_path_median_mm'])}/{_fmt(item['contact_path_median_mm'])} mm | "
            f"{_fmt(item['wear_proxy_median_mj'])} mJ | "
            f"{_fmt(task_contacts, 1)} | "
            f"{item.get('global_fallback_count', 'n/a')} |"
        )
    return lines


def _write_markdown(output: Path, report: dict) -> None:
    development = report["development"]
    holdout = report["holdout"]
    lines = [
        "# M8 Proposed+ Development/Holdout Split",
        "",
        "The first 12 lexicographic Sobol poses were used while correcting the "
        "task-aware grid policy. Only the remaining 24 poses are an untouched "
        "holdout for this policy version.",
        "",
        "## Untouched holdout",
        "",
        *_method_table(holdout),
        "",
        "## Development set",
        "",
        *_method_table(development),
        "",
        "## Boundary",
        "",
        report["claim_boundary"],
        "",
    ]
    (output / "development_holdout_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(input_root: Path, development_poses: int) -> dict:
    records = _load_records(input_root)
    if development_poses <= 0 or development_poses >= len(records):
        raise ValueError("development_poses must leave a nonempty holdout")
    development_records = records[:development_poses]
    holdout_records = records[development_poses:]
    report = {
        "design": {
            "policy": "task-aware capture-probability plus information grid",
            "development_pose_ids": [
                record["task"]["pose_id"] for record in development_records
            ],
            "holdout_pose_ids": [
                record["task"]["pose_id"] for record in holdout_records
            ],
            "development_poses": len(development_records),
            "untouched_holdout_poses": len(holdout_records),
            "split_frozen_after_policy_development": True,
        },
        "development": _summary(development_records),
        "holdout": _summary(holdout_records),
        "all": _summary(records),
        "claim_boundary": (
            "The holdout is untouched by the final policy correction but shares "
            "the same synthetic MuJoCo nuisance and pre-hardware wrench prior. "
            "It supports a simulation mechanism audit only, not real iiwa, "
            "cross-size, wear, or universal superiority claims."
        ),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (input_root / "development_holdout_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(input_root, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a completed task-aware M8 audit into development and holdout reports."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--development-poses", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(Path(args.input).resolve(), args.development_poses)
    print(json.dumps(result, indent=2))
