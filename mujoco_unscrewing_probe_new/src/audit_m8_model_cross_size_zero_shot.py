from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from src.m8_direction_runtime import M8DirectionEnsemble
from src.sample_m8_conical_pilot import PilotTask, _sample_task, _worker_init
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/current/evaluations/m8_model_m10_m12_zero_shot"
MODEL_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
SIZES = ("m8", "m10", "m12")
PHASE_BINS = 360

# Frozen before rollout. These are direction-interface gates, not a state-head
# or complete insertion guarantee.
MIN_WITHIN_15_RATE = 0.90
MIN_APS_COVERAGE = 0.90
MAX_CATASTROPHIC_RATE = 0.0
MAX_HARD_STOPS = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _circular_error_deg(predicted_deg: float, target_deg: float) -> float:
    return abs((float(predicted_deg) - float(target_deg) + 180.0) % 360.0 - 180.0)


def build_cross_size_tasks(spec: dict, size: str, *, seed: int) -> list[PilotTask]:
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    full_support_boundary = head_radius - bit_radius
    inner_overlap_boundary = socket_top_apothem + bit_radius
    if not 0.0 < full_support_boundary < inner_overlap_boundary:
        raise ValueError(f"invalid support boundaries for {size}")

    radii = {
        "supported_search": (
            0.62 * full_support_boundary,
            0.88 * full_support_boundary,
        ),
        "double_overhang": (
            full_support_boundary
            + 0.25 * (inner_overlap_boundary - full_support_boundary),
            full_support_boundary
            + 0.75 * (inner_overlap_boundary - full_support_boundary),
        ),
        "outer_partial": (
            inner_overlap_boundary + 0.08 * bit_radius,
            inner_overlap_boundary + 0.22 * bit_radius,
        ),
    }
    local_angles_deg = (13.0, 43.0)
    base_count = len(radii) * 2 * len(local_angles_deg)
    power = int(math.ceil(math.log2(base_count)))
    nuisance = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(power)[:base_count]
    tasks: list[PilotTask] = []
    nuisance_index = 0
    for state, state_radii in radii.items():
        for radial_level, radius in enumerate(state_radii):
            outer_signed = head_radius - (float(radius) + bit_radius)
            inner_signed = (float(radius) - bit_radius) - socket_top_apothem
            for local_index, local_angle_deg in enumerate(local_angles_deg):
                sample = nuisance[nuisance_index]
                nuisance_index += 1
                friction = 0.58 + 0.34 * float(sample[0])
                contact_timeconst = 0.043 + 0.024 * float(sample[1])
                down_force = 2.6 + 0.8 * float(sample[2])
                yaw_rad = math.radians(-1.25 + 2.5 * float(sample[3]))
                orbit_id = f"{size}_{state}_r{radial_level}_l{local_index}"
                for sector_index in range(6):
                    angle_deg = local_angle_deg + 60.0 * sector_index
                    angle_rad = math.radians(angle_deg)
                    trial_id = f"{orbit_id}_s{sector_index}"
                    tasks.append(
                        PilotTask(
                            trial_id=trial_id,
                            base_pose_id=trial_id,
                            search_state=state,
                            radial_level=radial_level,
                            radius_m=float(radius),
                            base_angle_deg=local_angle_deg,
                            rotation_index=sector_index,
                            rotation_deg=60.0 * sector_index,
                            x_m=float(radius) * math.cos(angle_rad),
                            y_m=float(radius) * math.sin(angle_rad),
                            nominal_tip_z_m=0.00010,
                            outer_signed_distance_m=outer_signed,
                            inner_signed_distance_m=inner_signed,
                            nuisance_id=f"cross_size_sobol_{nuisance_index - 1}",
                            friction_slide=friction,
                            contact_timeconst_s=contact_timeconst,
                            target_down_force_n=down_force,
                            yaw_rad=yaw_rad,
                            nuisance_seed=seed * 100 + nuisance_index - 1,
                            symmetry_orbit_id=orbit_id,
                        )
                    )
    return tasks


def _sample_size(
    size: str,
    tasks: list[PilotTask],
    output_dir: Path,
    *,
    workers: int,
) -> list[dict]:
    spec_path = ROOT / "configs" / f"allen_{size}_source.json"
    model_path = ROOT / "models" / f"allen_{size}_source.xml"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    context = mp.get_context("spawn")
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)),
        mp_context=context,
        initializer=_worker_init,
        initargs=(
            str(output_dir),
            PHASE_BINS,
            str(model_path),
            str(spec_path),
        ),
    ) as pool:
        futures = {pool.submit(_sample_task, task): task for task in tasks}
        bar = tqdm(total=len(tasks), desc=f"{size.upper()} zero-shot", unit="scan", dynamic_ncols=True)
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "trial_id": task.trial_id,
                    "status": "failed",
                    "error": repr(exc),
                }
            results.append(result)
            bar.update(1)
            bar.set_postfix(
                ok=sum(item.get("status") in {"ok", "skipped"} for item in results),
                fail=sum(item.get("status") == "failed" for item in results),
                refresh=False,
            )
        bar.close()
    _atomic_json(
        output_dir / "sampling_manifest.json",
        {
            "size": size,
            "tasks": [asdict(task) for task in tasks],
            "results": sorted(results, key=lambda item: item["trial_id"]),
            "elapsed_s": time.perf_counter() - started,
            "updated_utc": _utc_now(),
        },
    )
    return results


def _evaluate_size(size: str, output_dir: Path, runtime: M8DirectionEnsemble) -> dict:
    spec = json.loads((ROOT / "configs" / f"allen_{size}_source.json").read_text(encoding="utf-8"))
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    records = []
    for path in sorted((output_dir / "trials").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            sequence = np.asarray(data["phase_sequence"], dtype=float)
            columns = tuple(map(str, data["phase_columns"].tolist()))
        if metadata.get("status") != "ok":
            records.append(
                {
                    "trial_id": metadata["trial_id"],
                    "status": metadata.get("status", "failed"),
                    "failure_reason": metadata.get("failure_reason", ""),
                    "max_force_n": metadata.get("max_force_n"),
                    "max_torque_nm": metadata.get("max_torque_nm"),
                }
            )
            continue
        selected = np.vstack(
            [
                sequence[columns.index("Fz_N")],
                sequence[columns.index("Tx_Nm")],
                sequence[columns.index("Ty_Nm")],
                sequence[columns.index("tip_z_m")],
            ]
        )
        estimate = runtime.infer_phase_sequence(
            selected,
            target_down_force_n=float(metadata["target_down_force_n"]),
            bit_radius_m=bit_radius,
        )
        true_direction_deg = (
            math.degrees(math.atan2(float(metadata["y_m"]), float(metadata["x_m"])))
            + 180.0
        ) % 360.0
        true_class = training._direction_class(true_direction_deg)
        _, context = runtime._network_inputs(
            selected,
            float(metadata["target_down_force_n"]),
            bit_radius,
        )
        context_z_max = max(
            float(np.max(np.abs((context - mean) / std)))
            for _, mean, std, _ in runtime.models
        )
        records.append(
            {
                "trial_id": metadata["trial_id"],
                "status": "ok",
                "search_state": metadata["search_state"],
                "radius_normalized_by_bit": float(metadata["radius_m"]) / bit_radius,
                "true_direction_deg": true_direction_deg,
                "top1_direction_deg": estimate.top_direction_deg,
                "circular_mean_direction_deg": estimate.circular_mean_direction_deg,
                "top1_error_deg": _circular_error_deg(estimate.top_direction_deg, true_direction_deg),
                "circular_mean_error_deg": _circular_error_deg(
                    estimate.circular_mean_direction_deg, true_direction_deg
                ),
                "aps_covered": true_class in estimate.aps_direction_classes,
                "aps_size": len(estimate.aps_direction_classes),
                "top_state": estimate.top_state,
                "state_correct": estimate.top_state == metadata["search_state"],
                "transport_allowed": estimate.transport_allowed,
                "confidence": estimate.confidence,
                "circular_concentration": estimate.circular_concentration,
                "ensemble_top1_span_deg": estimate.ensemble_top1_span_deg,
                "context_z_max": context_z_max,
                "max_force_n": metadata["max_force_n"],
                "max_torque_nm": metadata["max_torque_nm"],
                "final_xy_residual_m": metadata["final_xy_residual_m"],
            }
        )

    ok = [record for record in records if record["status"] == "ok"]
    top1_error = np.asarray([record["top1_error_deg"] for record in ok], dtype=float)
    mean_error = np.asarray([record["circular_mean_error_deg"] for record in ok], dtype=float)
    hard_stops = sum(record.get("failure_reason") == "hard_stop" for record in records)
    summary = {
        "size": size,
        "trials": len(records),
        "successful_scans": len(ok),
        "failed_scans": len(records) - len(ok),
        "hard_stops": hard_stops,
        "top1_error_median_deg": float(np.median(top1_error)) if len(ok) else None,
        "top1_error_p90_deg": float(np.quantile(top1_error, 0.90)) if len(ok) else None,
        "top1_within_7p5_rate": float(np.mean(top1_error <= 7.5 + 1e-9)) if len(ok) else 0.0,
        "top1_within_15_rate": float(np.mean(top1_error <= 15.0 + 1e-9)) if len(ok) else 0.0,
        "circular_mean_error_median_deg": float(np.median(mean_error)) if len(ok) else None,
        "circular_mean_error_p90_deg": float(np.quantile(mean_error, 0.90)) if len(ok) else None,
        "catastrophic_gt90_rate": float(np.mean(top1_error > 90.0)) if len(ok) else 1.0,
        "aps_coverage": float(np.mean([record["aps_covered"] for record in ok])) if len(ok) else 0.0,
        "aps_mean_size": float(np.mean([record["aps_size"] for record in ok])) if len(ok) else None,
        "state_accuracy": float(np.mean([record["state_correct"] for record in ok])) if len(ok) else 0.0,
        "transport_accept_rate": float(
            np.mean([record["transport_allowed"] for record in ok])
        ) if len(ok) else 0.0,
        "context_z_p90": float(
            np.quantile([record["context_z_max"] for record in ok], 0.90)
        ) if len(ok) else None,
        "context_z_gt5_rate": float(
            np.mean([record["context_z_max"] > 5.0 for record in ok])
        ) if len(ok) else 0.0,
        "max_force_p90_n": float(
            np.quantile([record["max_force_n"] for record in ok], 0.90)
        ) if len(ok) else None,
        "max_torque_p90_mnm": float(
            1000.0 * np.quantile([record["max_torque_nm"] for record in ok], 0.90)
        ) if len(ok) else None,
        "final_xy_residual_p90_mm": float(
            1000.0 * np.quantile([record["final_xy_residual_m"] for record in ok], 0.90)
        ) if len(ok) else None,
    }
    summary["by_state"] = {}
    for state in training.STATE_NAMES:
        state_records = [record for record in ok if record["search_state"] == state]
        if not state_records:
            continue
        summary["by_state"][state] = {
            "trials": len(state_records),
            "top1_within_15_rate": float(
                np.mean([record["top1_error_deg"] <= 15.0 + 1e-9 for record in state_records])
            ),
            "aps_coverage": float(
                np.mean([record["aps_covered"] for record in state_records])
            ),
            "state_accuracy": float(
                np.mean([record["state_correct"] for record in state_records])
            ),
        }
    summary["direction_interface_gate_pass"] = bool(
        summary["successful_scans"] == summary["trials"]
        and summary["hard_stops"] <= MAX_HARD_STOPS
        and summary["top1_within_15_rate"] >= MIN_WITHIN_15_RATE
        and summary["aps_coverage"] >= MIN_APS_COVERAGE
        and summary["catastrophic_gt90_rate"] <= MAX_CATASTROPHIC_RATE
    )
    return {"summary": summary, "records": records}


def _write_summary(path: Path, report: dict) -> None:
    def metric(value: float | None, digits: int = 2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Frozen M8 Direction Model: M8 Reference And M10/M12 Zero-Shot Transfer",
        "",
        f"Updated: {report['updated_utc']}",
        "",
        "The M8 P1v3 ensemble, preprocessing, 15 deg probe, thresholds, and weights",
        "were frozen. Only CAD geometry and the dimensionless bit-radius scale were",
        "changed. No M10/M12 sample was used for training or calibration.",
        "",
        "| size | scans | hard stop | top1 med/p90 | <=15 deg | APS coverage/size | state acc | context z p90 / >5 | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        item = report["sizes"][size]["summary"]
        lines.append(
            f"| {size.upper()} | {item['successful_scans']}/{item['trials']} | "
            f"{item['hard_stops']} | {metric(item['top1_error_median_deg'])}/"
            f"{metric(item['top1_error_p90_deg'])} deg | {item['top1_within_15_rate']:.3f} | "
            f"{item['aps_coverage']:.3f}/{metric(item['aps_mean_size'])} | "
            f"{item['state_accuracy']:.3f} | {metric(item['context_z_p90'])} / "
            f"{item['context_z_gt5_rate']:.3f} | "
            f"{'PASS' if item['direction_interface_gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "A pass supports only zero-shot transfer of the normalized conical direction",
            "interface in these MuJoCo contact bands. It does not validate the M8 state",
            "head calibration, posterior distance, planner parameters, full M10/M12 insertion",
            "or real hardware. A failed state head does not invalidate direction transfer",
            "because state is a soft likelihood, but it forbids reusing M8 state calibration",
            "without adaptation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--skip-sampling", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    task_counts = {}
    for index, size in enumerate(SIZES):
        spec = json.loads(
            (ROOT / "configs" / f"allen_{size}_source.json").read_text(encoding="utf-8")
        )
        tasks = build_cross_size_tasks(spec, size, seed=20260730 + index)
        task_counts[size] = len(tasks)
        if not args.skip_sampling:
            _sample_size(size, tasks, output / size, workers=args.workers)

    runtime = M8DirectionEnsemble(MODEL_ROOT)
    sizes = {
        size: _evaluate_size(size, output / size, runtime)
        for size in SIZES
    }
    report = {
        "stage": "M8_frozen_direction_model_M8_reference_M10_M12_zero_shot",
        "updated_utc": _utc_now(),
        "frozen_model_root": str(MODEL_ROOT.relative_to(ROOT)),
        "task_counts": task_counts,
        "gates_frozen_before_rollout": {
            "minimum_top1_within_15_rate": MIN_WITHIN_15_RATE,
            "minimum_aps_coverage": MIN_APS_COVERAGE,
            "maximum_catastrophic_gt90_rate": MAX_CATASTROPHIC_RATE,
            "maximum_hard_stops": MAX_HARD_STOPS,
        },
        "normalization": "F/Fd, tau/(Fd*bit_radius), z/bit_radius plus per-channel phase shape",
        "online_true_center_or_contact_tags": False,
        "sizes": sizes,
        "all_direction_gates_pass": all(
            sizes[size]["summary"]["direction_interface_gate_pass"] for size in SIZES
        ),
        "claim_boundary": (
            "Simulation-only zero-shot direction-interface audit. It does not prove "
            "state calibration, distance transfer, complete M10/M12 insertion, or hardware generalization."
        ),
    }
    _atomic_json(output / "report.json", report)
    _write_summary(output / "summary.md", report)
    print(json.dumps({size: sizes[size]["summary"] for size in SIZES}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
