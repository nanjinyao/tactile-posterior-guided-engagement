from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.funnel_guard import FunnelGuardConfig
from src.render_m8_search_comparison import ROOT, run_method


OUT = ROOT / "outputs/current/evaluations/m8_recoverable_funnel_r3_ablation"
TRIALS = OUT / "trials"
POSTER = OUT / "paired_recoverable_funnel_r3_poster.png"

ARM_ORDER = ("no_funnel", "rcc_shield_r0", "rcc_shield_r3")


@dataclass(frozen=True)
class RecoverableFunnelTask:
    task_id: str
    initial_xy_m: tuple[float, float]
    baseline_phase_rad: float
    friction_slide: float
    contact_timeconst_s: float
    target_down_force_n: float
    post_capture_tilt_deg: float
    post_capture_axis_deg: float
    post_capture_lateral_m: float


def build_tasks(limit: int = 12) -> list[RecoverableFunnelTask]:
    """Return a fixed, outcome-independent moderate post-capture stress set."""

    if not 1 <= int(limit) <= 12:
        raise ValueError("limit must be in [1, 12]")
    tilt_levels_deg = (1.00, 1.50, 2.00, 2.50)
    lateral_levels_m = (0.00015, 0.00020, 0.00025, 0.00030)
    tasks: list[RecoverableFunnelTask] = []
    for index in range(12):
        initial_angle_deg = 30.0 * index
        disturbance_axis_deg = (initial_angle_deg + (20.0 if index % 2 == 0 else -20.0)) % 360.0
        initial_angle = math.radians(initial_angle_deg)
        tasks.append(
            RecoverableFunnelTask(
                task_id=f"recoverable_{index:02d}",
                initial_xy_m=(
                    0.0010 * math.cos(initial_angle),
                    0.0010 * math.sin(initial_angle),
                ),
                baseline_phase_rad=math.radians((initial_angle_deg + 75.0) % 360.0),
                friction_slide=(0.65, 0.80, 0.95)[index % 3],
                contact_timeconst_s=(0.045, 0.060, 0.075)[(index // 2) % 3],
                target_down_force_n=(2.75, 3.00, 3.25)[(index // 4) % 3],
                post_capture_tilt_deg=tilt_levels_deg[index % 4],
                post_capture_axis_deg=disturbance_axis_deg,
                post_capture_lateral_m=lateral_levels_m[(index * 3) % 4],
            )
        )
    return tasks[: int(limit)]


def arm_config(arm: str) -> FunnelGuardConfig:
    if arm == "no_funnel":
        return FunnelGuardConfig(
            enabled=False,
            max_recoveries=0,
            retry_on_terminal_rejection=False,
        )
    if arm == "rcc_shield_r0":
        return FunnelGuardConfig(
            enabled=True,
            max_recoveries=0,
            retry_on_terminal_rejection=True,
        )
    if arm == "rcc_shield_r3":
        return FunnelGuardConfig(
            enabled=True,
            max_recoveries=3,
            retry_on_terminal_rejection=True,
        )
    raise ValueError(arm)


def _run_one(task: RecoverableFunnelTask, arm: str) -> tuple[str, str, dict]:
    result = run_method(
        "baseline",
        initial_xy_m=np.asarray(task.initial_xy_m, dtype=float),
        baseline_max_radius_m=0.0020,
        baseline_phase_rad=float(task.baseline_phase_rad),
        target_down_force_n=float(task.target_down_force_n),
        friction_slide=float(task.friction_slide),
        contact_timeconst_s=float(task.contact_timeconst_s),
        active_time_budget_s=60.0,
        continue_to_bottom=True,
        cad_conditioned_insertion=True,
        funnel_guard=arm_config(arm),
        insertion_command_bias_xy_m=np.zeros(2, dtype=float),
        post_capture_tilt_disturbance_deg=float(task.post_capture_tilt_deg),
        post_capture_tilt_axis_deg=float(task.post_capture_axis_deg),
        post_capture_lateral_disturbance_m=float(task.post_capture_lateral_m),
        post_capture_disturbance_persistent=True,
        post_capture_tilt_ramp_s=0.12,
        post_capture_tilt_hold_s=0.20,
    )
    value = asdict(result)
    if int(value["simulator_reset_count"]) != 0:
        raise RuntimeError(f"{task.task_id}/{arm}: simulator reset is forbidden")
    rebases = value.get("funnel_recovery_xy_rebases_mm") or []
    if len(rebases) != int(value["funnel_recovery_count"]):
        raise RuntimeError(f"{task.task_id}/{arm}: recovery/rebase count mismatch")
    if arm != "rcc_shield_r3" and rebases:
        raise RuntimeError(f"{task.task_id}/{arm}: unexpected measured-tip rebase")
    return task.task_id, arm, value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outcome(result: dict) -> str:
    if result["result"] == "cad_bottom_verified":
        return "bottom_verified"
    if bool(result["hard_stop"]):
        return "hard_stop"
    if result["result"] == "funnel_safe_abort":
        return "safe_abort"
    return str(result["result"])


def _summarize(results: list[dict]) -> dict:
    successes = [row for row in results if row["result"] == "cad_bottom_verified"]
    reasons = Counter(
        reason
        for row in results
        for reason in (row.get("funnel_recovery_reasons") or [])
    )
    abort_reasons = Counter(
        str(row["funnel_abort_reason"])
        for row in results
        if row.get("funnel_abort_reason") is not None
    )
    return {
        "trials": len(results),
        "bottom_verified": len(successes),
        "success_rate": len(successes) / len(results),
        "hard_stops": sum(bool(row["hard_stop"]) for row in results),
        "safe_aborts": sum(row["result"] == "funnel_safe_abort" for row in results),
        "bottom_not_verified": sum(row["result"] == "bottom_not_verified" for row in results),
        "recoveries": sum(int(row["funnel_recovery_count"]) for row in results),
        "recovery_reasons": dict(sorted(reasons.items())),
        "abort_reasons": dict(sorted(abort_reasons.items())),
        "terminal_attempts": sum(int(row["terminal_attempts"]) for row in results),
        "terminal_rejections": sum(int(row["terminal_rejection_count"]) for row in results),
        "simulator_resets": sum(int(row["simulator_reset_count"]) for row in results),
        "median_success_time_s": (
            None
            if not successes
            else float(median(float(row["time_to_bottom_s"]) for row in successes))
        ),
        "p90_max_force_n": float(np.quantile([row["max_force_n"] for row in results], 0.90)),
        "p90_max_torque_mnm": float(
            np.quantile([row["max_torque_mnm"] for row in results], 0.90)
        ),
    }


def _paired_counts(records: dict[str, dict], left: str, right: str) -> dict:
    counts = {
        "both_bottom": 0,
        f"{right}_only_bottom": 0,
        f"{left}_only_bottom": 0,
        "neither_bottom": 0,
    }
    for record in records.values():
        left_ok = record["methods"][left]["result"] == "cad_bottom_verified"
        right_ok = record["methods"][right]["result"] == "cad_bottom_verified"
        if left_ok and right_ok:
            counts["both_bottom"] += 1
        elif right_ok:
            counts[f"{right}_only_bottom"] += 1
        elif left_ok:
            counts[f"{left}_only_bottom"] += 1
        else:
            counts["neither_bottom"] += 1
    improvements = counts[f"{right}_only_bottom"]
    regressions = counts[f"{left}_only_bottom"]
    discordant = improvements + regressions
    if discordant == 0:
        exact_p = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(improvements, regressions) + 1)
        ) / (2**discordant)
        exact_p = min(1.0, 2.0 * tail)
    counts.update(
        {
            "discordant_improvements": improvements,
            "discordant_regressions": regressions,
            "exact_two_sided_sign_p": exact_p,
        }
    )
    return counts


def _write_csvs(records: dict[str, dict], summaries: dict[str, dict]) -> None:
    long_fields = [
        "task_id",
        "arm",
        "tilt_deg",
        "axis_deg",
        "lateral_mm",
        "friction_slide",
        "contact_timeconst_s",
        "result",
        "bottom_verified",
        "hard_stop",
        "time_to_bottom_s",
        "funnel_recovery_count",
        "funnel_recovery_reasons",
        "terminal_attempts",
        "terminal_rejection_count",
        "simulator_reset_count",
        "max_force_n",
        "max_torque_mnm",
        "funnel_abort_reason",
        "measured_tip_xy_rebases_mm",
    ]
    with (OUT / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=long_fields)
        writer.writeheader()
        for task_id in sorted(records):
            record = records[task_id]
            task = record["task"]
            for arm in ARM_ORDER:
                result = record["methods"][arm]
                writer.writerow(
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "tilt_deg": task["post_capture_tilt_deg"],
                        "axis_deg": task["post_capture_axis_deg"],
                        "lateral_mm": 1000.0 * task["post_capture_lateral_m"],
                        "friction_slide": task["friction_slide"],
                        "contact_timeconst_s": task["contact_timeconst_s"],
                        "result": result["result"],
                        "bottom_verified": result["result"] == "cad_bottom_verified",
                        "hard_stop": result["hard_stop"],
                        "time_to_bottom_s": result["time_to_bottom_s"],
                        "funnel_recovery_count": result["funnel_recovery_count"],
                        "funnel_recovery_reasons": json.dumps(
                            result.get("funnel_recovery_reasons") or []
                        ),
                        "terminal_attempts": result["terminal_attempts"],
                        "terminal_rejection_count": result["terminal_rejection_count"],
                        "simulator_reset_count": result["simulator_reset_count"],
                        "max_force_n": result["max_force_n"],
                        "max_torque_mnm": result["max_torque_mnm"],
                        "funnel_abort_reason": result["funnel_abort_reason"],
                        "measured_tip_xy_rebases_mm": json.dumps(
                            result.get("funnel_recovery_xy_rebases_mm") or []
                        ),
                    }
                )

    summary_fields = [
        "arm",
        "trials",
        "bottom_verified",
        "success_rate",
        "hard_stops",
        "safe_aborts",
        "recoveries",
        "terminal_attempts",
        "terminal_rejections",
        "simulator_resets",
        "median_success_time_s",
        "p90_max_force_n",
        "p90_max_torque_mnm",
    ]
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for arm in ARM_ORDER:
            writer.writerow({key: summaries[arm].get(key) for key in summary_fields})

    with (OUT / "paired_outcomes.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["task_id", "tilt_deg", "lateral_mm", *ARM_ORDER]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task_id in sorted(records):
            record = records[task_id]
            writer.writerow(
                {
                    "task_id": task_id,
                    "tilt_deg": record["task"]["post_capture_tilt_deg"],
                    "lateral_mm": 1000.0
                    * record["task"]["post_capture_lateral_m"],
                    **{
                        arm: _outcome(record["methods"][arm])
                        for arm in ARM_ORDER
                    },
                }
            )


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _color(outcome: str) -> tuple[int, int, int]:
    return {
        "bottom_verified": (48, 188, 116),
        "safe_abort": (242, 178, 72),
        "hard_stop": (225, 76, 85),
        "bottom_not_verified": (105, 127, 150),
    }.get(outcome, (105, 127, 150))


def _representative(records: dict[str, dict]) -> str:
    # Prefer a true completion-gain case, not merely a case where R0 exits.
    for task_id in sorted(records):
        methods = records[task_id]["methods"]
        if (
            methods["rcc_shield_r3"]["result"] == "cad_bottom_verified"
            and methods["rcc_shield_r3"]["funnel_recovery_count"] > 0
            and methods["no_funnel"]["result"] != "cad_bottom_verified"
            and methods["rcc_shield_r0"]["result"] != "cad_bottom_verified"
        ):
            return task_id
    for task_id in sorted(records):
        methods = records[task_id]["methods"]
        if (
            methods["rcc_shield_r3"]["result"] == "cad_bottom_verified"
            and methods["rcc_shield_r3"]["funnel_recovery_count"] > 0
            and (
                methods["no_funnel"]["result"] != "cad_bottom_verified"
                or methods["rcc_shield_r0"]["result"] != "cad_bottom_verified"
            )
        ):
            return task_id
    return sorted(records)[0]


def _poster(records: dict[str, dict], summaries: dict[str, dict]) -> str:
    selected_id = _representative(records)
    selected = records[selected_id]
    canvas = Image.new("RGB", (1920, 1080), (9, 14, 21))
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 45), "M8 post-capture physical-funnel ablation", fill=(240, 246, 252), font=_font(42, bold=True))
    draw.text(
        (66, 101),
        "same 12 tasks and sensor-only capture handoff; only post-capture funnel/retry policy changes",
        fill=(154, 177, 199),
        font=_font(22),
    )

    labels = {
        "no_funnel": "NO FUNNEL",
        "rcc_shield_r0": "TIP-RCC + SHIELD, R0",
        "rcc_shield_r3": "TIP-RCC + SHIELD, R3",
    }
    column_x = (70, 660, 1250)
    for arm, x in zip(ARM_ORDER, column_x):
        summary = summaries[arm]
        draw.rounded_rectangle((x, 160, x + 530, 325), radius=16, fill=(20, 30, 42), outline=(65, 88, 111), width=2)
        draw.text((x + 24, 178), labels[arm], fill=(235, 242, 249), font=_font(25, bold=True))
        draw.text(
            (x + 24, 224),
            f"bottom {summary['bottom_verified']}/{summary['trials']}   hard stop {summary['hard_stops']}   safe abort {summary['safe_aborts']}",
            fill=(188, 207, 225),
            font=_font(18),
        )
        draw.text(
            (x + 24, 264),
            f"recoveries {summary['recoveries']}   terminal attempts {summary['terminal_attempts']}   sim reset {summary['simulator_resets']}",
            fill=(188, 207, 225),
            font=_font(18),
        )

    draw.text((70, 365), "Paired outcomes", fill=(235, 242, 249), font=_font(25, bold=True))
    draw.text((70, 407), "task / stress", fill=(154, 177, 199), font=_font(17, bold=True))
    for arm, x in zip(ARM_ORDER, column_x):
        draw.text((x + 150, 407), labels[arm], fill=(154, 177, 199), font=_font(16, bold=True))
    row_y = 443
    for row_index, task_id in enumerate(sorted(records)):
        record = records[task_id]
        task = record["task"]
        y = row_y + 35 * row_index
        draw.text(
            (70, y + 4),
            f"{task_id[-2:]}  {task['post_capture_tilt_deg']:.1f} deg / {1000.0 * task['post_capture_lateral_m']:.2f} mm",
            fill=(203, 217, 230),
            font=_font(15),
        )
        for arm, x in zip(ARM_ORDER, column_x):
            outcome = _outcome(record["methods"][arm])
            draw.rounded_rectangle((x + 155, y, x + 505, y + 27), radius=7, fill=_color(outcome))
            draw.text((x + 170, y + 4), outcome.replace("_", " "), fill=(8, 14, 20), font=_font(14, bold=True))

    selected_y = 895
    task = selected["task"]
    draw.rounded_rectangle((70, selected_y, 1850, 1035), radius=14, fill=(16, 25, 35), outline=(73, 103, 130), width=2)
    draw.text(
        (92, selected_y + 16),
        f"Representative paired case {selected_id}: {task['post_capture_tilt_deg']:.1f} deg + {1000.0 * task['post_capture_lateral_m']:.2f} mm persistent side stress",
        fill=(235, 242, 249),
        font=_font(20, bold=True),
    )
    for index, arm in enumerate(ARM_ORDER):
        result = selected["methods"][arm]
        x = 94 + 568 * index
        draw.text(
            (x, selected_y + 58),
            f"{labels[arm]}: {_outcome(result)} | R={result['funnel_recovery_count']} | terminal={result['terminal_attempts']}",
            fill=_color(_outcome(result)),
            font=_font(16, bold=True),
        )
    rebases = selected["methods"]["rcc_shield_r3"].get("funnel_recovery_xy_rebases_mm") or []
    draw.text(
        (94, selected_y + 101),
        "R3 recovery: unload -> retract/upright -> measured-tip XY rebase -> low-force retouch -> new terminal dwell",
        fill=(113, 213, 246),
        font=_font(17, bold=True),
    )
    draw.text(
        (1195, selected_y + 101),
        f"measured rebases: {json.dumps(rebases)} | true center never supplied",
        fill=(154, 177, 199),
        font=_font(13),
    )
    canvas.save(POSTER)
    return selected_id


def _write_markdown(report: dict) -> None:
    lines = [
        "# M8 paired recoverable physical-funnel ablation",
        "",
        "All three arms use the same 12 task realizations, ordinary outward search, sensor-only capture handoff, nuisance parameters, 60 s active-time budget, CAD-conditioned insertion, and terminal verifier. Only the post-capture funnel/retry contract changes.",
        "",
        "| arm | bottom | hard stop | safe abort | recoveries | terminal attempts | median success time [s] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        row = report["methods"][arm]
        time_text = "-" if row["median_success_time_s"] is None else f"{row['median_success_time_s']:.3f}"
        lines.append(
            f"| {arm} | {row['bottom_verified']}/{row['trials']} | {row['hard_stops']} | {row['safe_aborts']} | {row['recoveries']} | {row['terminal_attempts']} | {time_text} |"
        )
    lines.extend(
        [
        "",
        report["observed_result"],
        "",
        "## Mechanism and boundary",
            "",
            "R0 keeps tip-frame RCC and the safety observer but permits no recovery. R3 permits at most three unload/retract/upright/retouch cycles. Every recovery rebases XY from the measured tip pose; the controller is never given the true socket center. Simulator resets are forbidden and counted explicitly.",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    tasks = build_tasks(args.tasks)
    OUT.mkdir(parents=True, exist_ok=True)
    TRIALS.mkdir(parents=True, exist_ok=True)

    contract = {
        "contract_version": "m8_recoverable_funnel_r3_v1",
        "selection": "fixed programmatic 12-case stress set; no outcome-based task filtering",
        "task_count": len(tasks),
        "common": {
            "search_policy": "ordinary outward spiral",
            "initial_radius_mm": 1.0,
            "baseline_max_radius_mm": 2.0,
            "sensor_only_capture_handoff": True,
            "active_time_budget_s": 60.0,
            "cad_conditioned_insertion": True,
            "persistent_post_capture_disturbance": True,
            "insertion_command_bias_mm": [0.0, 0.0],
            "simulator_resets_allowed": 0,
        },
        "stress_envelope": {
            "tilt_deg": [min(t.post_capture_tilt_deg for t in tasks), max(t.post_capture_tilt_deg for t in tasks)],
            "lateral_mm": [1000.0 * min(t.post_capture_lateral_m for t in tasks), 1000.0 * max(t.post_capture_lateral_m for t in tasks)],
            "friction_slide": [min(t.friction_slide for t in tasks), max(t.friction_slide for t in tasks)],
            "contact_timeconst_s": [min(t.contact_timeconst_s for t in tasks), max(t.contact_timeconst_s for t in tasks)],
        },
        "arms": {arm: asdict(arm_config(arm)) for arm in ARM_ORDER},
        "recovery_xy_rebase_source": "measured snapshot.tip_position[:2] after unload/retract; no true socket center input",
        "tasks": [asdict(task) for task in tasks],
    }
    (OUT / "task_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    records: dict[str, dict] = {
        task.task_id: {"task": asdict(task), "methods": {}} for task in tasks
    }
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for task in tasks:
            for arm in ARM_ORDER:
                cached = TRIALS / f"{task.task_id}__{arm}.json"
                if cached.is_file() and not args.no_resume:
                    records[task.task_id]["methods"][arm] = json.loads(
                        cached.read_text(encoding="utf-8")
                    )["result"]
                else:
                    futures.append(executor.submit(_run_one, task, arm))
        completed = sum(len(record["methods"]) for record in records.values())
        total = len(tasks) * len(ARM_ORDER)
        for future in as_completed(futures):
            task_id, arm, result = future.result()
            records[task_id]["methods"][arm] = result
            payload = {
                "task": records[task_id]["task"],
                "arm": arm,
                "funnel_config": asdict(arm_config(arm)),
                "result": result,
            }
            (TRIALS / f"{task_id}__{arm}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            completed += 1
            width = 30
            filled = round(width * completed / total)
            print(f"\r[{'#' * filled}{'-' * (width - filled)}] {completed:02d}/{total:02d}", end="", flush=True)
    print()

    for task_id, record in records.items():
        if set(record["methods"]) != set(ARM_ORDER):
            raise RuntimeError(f"{task_id}: incomplete paired arms")
        (TRIALS / f"{task_id}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )

    summaries = {
        arm: _summarize([records[task.task_id]["methods"][arm] for task in tasks])
        for arm in ARM_ORDER
    }
    paired = {
        "no_funnel_vs_r3": _paired_counts(records, "no_funnel", "rcc_shield_r3"),
        "r0_vs_r3": _paired_counts(records, "rcc_shield_r0", "rcc_shield_r3"),
    }
    selected_id = _poster(records, summaries)
    report = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "design": contract,
        "methods": summaries,
        "paired": paired,
        "observed_result": (
            f"On this fixed 12-case set, R3 verified {summaries['rcc_shield_r3']['bottom_verified']}/"
            f"{len(tasks)} bottoms versus {summaries['no_funnel']['bottom_verified']}/{len(tasks)} without "
            f"the funnel and {summaries['rcc_shield_r0']['bottom_verified']}/{len(tasks)} with R0. "
            f"Against no funnel, R3 uniquely recovered {paired['no_funnel_vs_r3']['discordant_improvements']} "
            f"paired cases and regressed {paired['no_funnel_vs_r3']['discordant_regressions']} "
            f"(exact two-sided paired sign p={paired['no_funnel_vs_r3']['exact_two_sided_sign_p']:.6f}). "
            f"It still produced {summaries['rcc_shield_r3']['hard_stops']} hard stops and "
            f"{summaries['rcc_shield_r3']['safe_aborts']} safe abort, so the closed loop is improved but not guaranteed."
        ),
        "representative_case": selected_id,
        "poster": str(POSTER.relative_to(ROOT)),
        "provenance": {
            "runner": "src/run_m8_recoverable_funnel_r3_ablation.py",
            "runner_sha256": _sha256(Path(__file__)),
            "controller": "src/render_m8_search_comparison.py",
            "controller_sha256": _sha256(ROOT / "src/render_m8_search_comparison.py"),
            "funnel_guard": "src/funnel_guard.py",
            "funnel_guard_sha256": _sha256(ROOT / "src/funnel_guard.py"),
        },
        "claim_boundary": (
            "This is a paired MuJoCo post-capture stress ablation, not a proof that a funnel must succeed. "
            "A bottom is counted only after the same CAD-depth/force/low-wrench dwell verifier. "
            "R3 improvement, if present, is causal only for this fixed moderate-stress set and current controller; "
            "it is not hardware evidence, a passivity proof, or a guarantee outside the tested envelope."
        ),
    }
    _write_csvs(records, summaries)
    _write_markdown(report)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"methods": summaries, "paired": paired, "poster": report["poster"]}, indent=2))


if __name__ == "__main__":
    main()
