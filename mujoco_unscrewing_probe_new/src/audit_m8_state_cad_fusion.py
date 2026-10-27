from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from src.m8_direction_runtime import M8DirectionEnsemble
from src.posterior_search_plan import plan_posterior_guided_search
from src.render_m8_search_comparison import (
    cad_radial_geometry_from_spec,
    center_posterior_from_direction,
)
from src.sample_m8_conical_pilot import SPEC_PATH, _load_trial
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/current/evaluations/p1v5_m8_final_untouched_direction_audit"
MODEL = ROOT / "outputs/current/models/p1v3_m8_direction_models"
OUT = ROOT / "outputs/current/evaluations/m8_state_cad_fusion_audit"
VISUAL_SIGMA_M = 0.0020
VISUAL_BOUND_M = math.sqrt(5.991) * VISUAL_SIGMA_M
CAPTURE_RADIUS_M = 0.00025


def _summary(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    return {
        "samples": len(values),
        "coverage_rate": float(np.mean([value["true_center_covered"] for value in values])),
        "reverse_move_rate": float(np.mean([value["reverse_move"] for value in values])),
        "transport_median_mm": float(np.median([value["transport_mm"] for value in values])),
        "local_radius_median_mm": float(np.median([value["local_radius_mm"] for value in values])),
        "post_transport_true_residual_median_mm": float(
            np.median([value["true_residual_mm"] for value in values])
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(SPEC_PATH).read_text(encoding="utf-8"))
    geometry = cad_radial_geometry_from_spec(spec)
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    runtime = M8DirectionEnsemble(MODEL)
    rows: list[dict] = []
    for index, path in enumerate(sorted((SOURCE / "trials").glob("*.npz"))):
        metadata, sequence, columns = _load_trial(path)
        indices = [columns.index(name) for name in training.INPUT_CHANNELS]
        estimate = runtime.infer_phase_sequence(
            sequence[indices],
            target_down_force_n=float(metadata["target_down_force_n"]),
            bit_radius_m=bit_radius_m,
        )
        current = np.array([metadata["x_m"], metadata["y_m"]], dtype=float)
        quality = float(
            estimate.circular_concentration
            * max(
                0.0,
                math.cos(
                    0.5
                    * math.radians(min(180.0, estimate.ensemble_top1_span_deg))
                ),
            )
        )
        variants = {}
        for name, use_state in (("direction_visual", False), ("state_cad_soft", True)):
            posterior = center_posterior_from_direction(
                current,
                estimate.direction_probabilities,
                VISUAL_SIGMA_M,
                np.random.default_rng(20260712000 + index),
                visual_bound_m=VISUAL_BOUND_M,
                state_probabilities=(estimate.state_probabilities if use_state else None),
                cad_geometry=(geometry if use_state else None),
            )
            plan = plan_posterior_guided_search(
                current,
                posterior,
                estimate.direction_xy,
                effective_capture_radius_m=CAPTURE_RADIUS_M,
                posterior_coverage=0.99,
                direction_quality=quality,
                max_transport_m=None,
                pitch_safety_factor=0.80,
            )
            true_residual = float(np.linalg.norm(plan.transport_target_xy))
            toward_center = float(
                np.dot(plan.transport_target_xy - current, -current)
                / max(np.linalg.norm(current), 1.0e-12)
            )
            variants[name] = {
                "transport_mm": 1000.0 * plan.transport_distance_m,
                "local_radius_mm": 1000.0 * plan.spiral_max_radius_m,
                "true_residual_mm": 1000.0 * true_residual,
                "true_center_covered": bool(
                    true_residual
                    <= plan.spiral_max_radius_m + CAPTURE_RADIUS_M + 1.0e-12
                ),
                "reverse_move": toward_center < -1.0e-9,
            }
        rows.append(
            {
                "trial_id": metadata["trial_id"],
                "radius_mm": 1000.0 * float(metadata["radius_m"]),
                "true_state": metadata["search_state"],
                "predicted_state": estimate.top_state,
                "state_probabilities": estimate.state_probabilities.tolist(),
                **variants,
            }
        )

    report = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "source": str(SOURCE.relative_to(ROOT)),
            "samples": len(rows),
            "same_frozen_direction_and_visual_prior": True,
            "only_difference": "soft support-state x approximate-CAD radial likelihood",
            "true_center_used_by_planner": False,
            "true_center_used_offline_for_coverage_scoring": True,
        },
        "methods": {
            "direction_visual": _summary(rows, "direction_visual"),
            "state_cad_soft": _summary(rows, "state_cad_soft"),
        },
        "claim_boundary": (
            "Offline action-interface audit on the frozen simulation P1v5 set. It tests whether the soft state-CAD likelihood "
            "changes posterior transport/coverage without using true center online. It is not dynamic insertion, hardware, or scale-transfer evidence."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# M8 Soft State-CAD Fusion Audit",
        "",
        "| interface | true-center covered | reverse move | transport median | local radius median | true residual median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["methods"].items():
        lines.append(
            f"| {name} | {value['coverage_rate']:.3f} | {value['reverse_move_rate']:.3f} | "
            f"{value['transport_median_mm']:.3f} mm | {value['local_radius_median_mm']:.3f} mm | "
            f"{value['post_transport_true_residual_median_mm']:.3f} mm |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
