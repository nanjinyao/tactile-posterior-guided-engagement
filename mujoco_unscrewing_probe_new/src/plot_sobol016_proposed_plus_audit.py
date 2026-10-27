#!/usr/bin/env python3
"""Build a bilingual, read-only audit of the sobol_016 Proposed+ video.

The script never reruns MuJoCo.  It reads the frozen selected-media manifest,
checks its link to the paired audit, extracts display frames from the frozen
MP4, and separates commanded grid endpoints from the measured/contact-limited
tip path.  True-center quantities are explicitly labelled offline diagnostics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "outputs/current/active_comparison/m8_retry3_selected_media/sobol_016"
    / "comparison_fuzzy_retry3_2x2_manifest.json"
)
OUTPUT = ROOT / "outputs/current/media/figures/paper"
AUDIT = ROOT / "outputs/current/diagnostics/m8_sobol016_proposed_plus_physics_audit"
FPS = 15.0
DISPLAY_TIMES_S = (7.0, 15.5, 22.0)
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def _configure_fonts() -> None:
    """Register one font that covers both the Chinese and English assets."""
    if FONT_REGULAR.exists():
        font_manager.fontManager.addfont(str(FONT_REGULAR))
        family = font_manager.FontProperties(fname=str(FONT_REGULAR)).get_name()
    else:
        family = "DejaVu Sans"
    if FONT_BOLD.exists():
        font_manager.fontManager.addfont(str(FONT_BOLD))
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _frame_panels(video: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(video)
    panels = []
    try:
        for time_s in DISPLAY_TIMES_S:
            frame = np.asarray(reader.get_data(int(round(time_s * FPS))))
            panels.append(frame[:400, :640, :3])
    finally:
        reader.close()
    return panels


def _audit_payload(manifest: dict) -> dict:
    result = manifest["results"]["proposed_plus"]
    task = manifest["task"]
    gates = result["hybrid_probe_gate_records"]
    first_gate = gates[0]
    capture_gate = gates[-1]
    initial = np.asarray(result["initial_xy_mm"], dtype=float)
    shortcut_length = float(first_gate["transport_distance_mm"])
    control_deg = float(result["control_direction_deg"])
    control_rad = math.radians(control_deg)
    shortcut_endpoint = initial + shortcut_length * np.asarray(
        [math.cos(control_rad), math.sin(control_rad)]
    )
    true_direction = float(result["true_direction_deg"])
    error = abs((control_deg - true_direction + 180.0) % 360.0 - 180.0)
    event = np.asarray(result["event_xy_mm"], dtype=float)
    capture_radius = float(result["entropy_grid_geometry_mm"]["capture_radius"])
    entropy_initial = float(result["entropy_grid_initial_nats"])
    entropy_final = float(result["entropy_grid_final_nats"])
    radius_initial = float(gates[1]["credible_radius_before_mm"])
    radius_final = float(result["entropy_grid_credible_radius_mm"])
    observations = result["entropy_grid_observations"]
    commanded_endpoints = [
        row["action_xy_mm"] for row in observations if row.get("action_xy_mm")
    ]

    assert manifest["contract"]["online_true_center_or_contact_tags"] is False
    assert manifest["verifications"]["proposed_plus"]["mismatch_count"] == 0
    assert first_gate["executed"] is True
    assert capture_gate["outcome"] == "capture_during_smooth_segment"
    assert float(result["time_to_inner_wall_s"]) < float(result["time_to_bottom_s"])
    assert result["result"] == "cad_bottom_verified"
    assert result["hard_stop"] is False
    assert entropy_final > entropy_initial
    assert radius_final > radius_initial

    source_files = [
        ROOT / "src/render_m8_search_comparison.py",
        ROOT / "src/entropy_grid_baseline.py",
        ROOT / "src/render_m8_retry3_selected_media.py",
    ]
    video = ROOT / manifest["artifacts"]["video"]["path"]
    return {
        "experiment": "sobol_016_proposed_plus_selected_video_physics_audit",
        "status": "read_only_frozen_artifact_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pose_id": manifest["pose_id"],
        "task": task,
        "artifact_provenance": {
            "manifest": {"path": _relative(MANIFEST), "sha256": _sha256(MANIFEST)},
            "video": {"path": _relative(video), "sha256": _sha256(video)},
            "source_trial": manifest["source_trial"],
            "source_audit_report": manifest["source_audit_report"],
            "render_replay_checked_substantive_leaves": int(
                manifest["verifications"]["proposed_plus"][
                    "checked_substantive_leaves"
                ]
            ),
            "render_replay_mismatch_count": 0,
            "current_source_diagnostic_only": [
                {"path": _relative(path), "sha256": _sha256(path)}
                for path in source_files
            ],
            "source_hash_boundary": (
                "The selected-media manifest freezes the trial/report/video hashes but "
                "does not freeze the generating source hashes. Current source hashes are "
                "diagnostic only because source files postdate the video."
            ),
        },
        "shortcut": {
            "initial_radius_mm": float(np.linalg.norm(initial)),
            "circular_mean_control_direction_deg": control_deg,
            "offline_true_direction_deg": true_direction,
            "offline_control_direction_error_deg": error,
            "top1_map_direction_deg": float(result["predicted_direction_deg"]),
            "aps_set_size": int(result["aps_set_size"]),
            "circular_concentration": float(result["circular_concentration"]),
            "first_executed_distance_mm": shortcut_length,
            "derived_command_endpoint_xy_mm": shortcut_endpoint.tolist(),
            "derived_endpoint_radius_mm": float(np.linalg.norm(shortcut_endpoint)),
            "captured_during_shortcut": False,
            "later_unexecuted_replan_distance_mm": float(
                result["transport_distance_mm"]
            ),
        },
        "task_aware_coverage": {
            "entropy_initial_nats": entropy_initial,
            "entropy_final_nats": entropy_final,
            "credible_radius_initial_mm": radius_initial,
            "credible_radius_final_mm": radius_final,
            "entropy_converged": False,
            "mode_switch": "capture_weighted_coverage_after_noncontracting_updates",
            "commanded_grid_endpoints_xy_mm": commanded_endpoints,
            "commanded_nominal_segment_length_sum_mm": float(
                shortcut_length
                + sum(float(row["travel_distance_mm"]) for row in gates[1:])
            ),
            "measured_tip_xy_path_mm": float(result["actual_xy_path_mm"]),
            "measured_contact_xy_path_mm": float(result["contact_xy_path_mm"]),
            "measured_path_used_for_capture_miss_updates": True,
            "full_measured_xy_polyline_persisted_in_frozen_trial": False,
        },
        "capture_and_terminal": {
            "capture_probe": int(capture_gate["probe"]),
            "capture_outcome": capture_gate["outcome"],
            "capture_event_xy_mm": event.tolist(),
            "capture_event_offline_radius_mm": float(np.linalg.norm(event)),
            "conservative_capture_radius_mm": capture_radius,
            "capture_mode": result["capture_mode"],
            "online_capture_sensor_only": bool(result["online_capture_sensor_only"]),
            "time_to_inner_wall_s": float(result["time_to_inner_wall_s"]),
            "time_to_bottom_s": float(result["time_to_bottom_s"]),
            "terminal_verification_mode": result["terminal_verification_mode"],
            "bottom_depth_error_mm": float(result["bottom_depth_error_mm"]),
            "funnel_recovery_count": int(result["funnel_recovery_count"]),
            "hard_stop": bool(result["hard_stop"]),
            "max_raw_physical_force_n": float(result["max_raw_physical_force_n"]),
            "max_raw_physical_torque_mnm": float(
                result["max_raw_physical_torque_mnm"]
            ),
        },
        "verdict": {
            "online_true_center_or_contact_tags_used": False,
            "physically_plausible_under_current_MuJoCo_contract": True,
            "shortcut_was_accurate": False,
            "entropy_grid_convergence_claim_supported": False,
            "supported_mechanism_claim": (
                "A short but directionally inaccurate shortcut reduces radial distance; "
                "contact-limited task-aware coverage, not entropy convergence, later "
                "triggers a persistent sensor capture cue and the shared physical funnel "
                "reaches the CAD-bottom terminal set."
            ),
        },
    }


TEXT = {
    "en": {
        "title": "sobol_016 audit: recovery from an inaccurate short shortcut",
        "offline": "offline truth for audit only",
        "start": "start",
        "shortcut": "first shortcut",
        "event": "sensor capture cue",
        "capture_disk": "conservative $r_{cap}$",
        "timeline": [
            "0.903 mm shortcut\n44.80 deg error\nno capture",
            "4 noncontracting updates\nH: 6.065 to 6.448 nat\ncoverage mode",
            "probe 5 smooth segment\nsensor cue at 15.45 s",
            "shared physical funnel + R3\nCAD bottom at 22.19 s",
        ],
        "frames": ["after shortcut", "coverage search", "verified bottom"],
        "footer": (
            "Yellow endpoints are commands, not measured-tip samples. True center and "
            "angles are drawn only for offline audit; online actions use posterior, F/T, "
            "robot pose, and CAD geometry."
        ),
    },
    "cn": {
        "title": "sobol_016 审计：错误但保守的短捷径如何被覆盖搜索恢复",
        "offline": "真值仅用于离线审计",
        "start": "起点",
        "shortcut": "首次捷径",
        "event": "传感器捕获提示",
        "capture_disk": "保守捕获半径 $r_{cap}$",
        "timeline": [
            "0.903 mm 捷径\n方向误差 44.80 度\n未捕获",
            "4 次后验不收缩\nH: 6.065 到 6.448 nat\n切换覆盖模式",
            "第 5 条平滑路径段\n15.45 s 触发传感器门",
            "共享物理漏斗 + R3\n22.19 s 到底",
        ],
        "frames": ["捷径后", "覆盖搜索", "到底验证"],
        "footer": (
            "黄色格点是命令端点，不是实测 tip 采样。真中心与角度仅为离线审计叠加；"
            "在线动作只使用 posterior、F/T、机器人位姿与 CAD 几何。"
        ),
    },
}


def _plot(report: dict, panels: list[np.ndarray], language: str, output: Path) -> None:
    text = TEXT[language]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11})
    fig = plt.figure(figsize=(15.6, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 6, height_ratios=(1.15, 0.72, 1.45))
    fig.suptitle(text["title"], fontsize=18, fontweight="bold")

    ax = fig.add_subplot(grid[:2, :2])
    shortcut = report["shortcut"]
    capture = report["capture_and_terminal"]
    start = np.asarray(
        [report["task"]["x_m"], report["task"]["y_m"]], dtype=float
    ) * 1000.0
    endpoint = np.asarray(shortcut["derived_command_endpoint_xy_mm"])
    event = np.asarray(capture["capture_event_xy_mm"])
    bound = 4.4
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            capture["conservative_capture_radius_mm"],
            fc="#dcfce7",
            ec="#16a34a",
            lw=2,
            label=text["capture_disk"],
        )
    )
    ax.scatter([0], [0], marker="*", s=120, c="#dc2626", label=text["offline"])
    ax.scatter(*start, s=75, c="#2563eb", zorder=5, label=text["start"])
    ax.annotate(
        "",
        xy=endpoint,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#7c3aed"),
    )
    ax.scatter(*endpoint, s=65, c="#7c3aed", zorder=5, label=text["shortcut"])
    ax.scatter(*event, s=75, c="#f59e0b", zorder=6, label=text["event"])
    ax.plot([endpoint[0], event[0]], [endpoint[1], event[1]], "--", c="#94a3b8")
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), xlabel="x [mm]", ylabel="y [mm]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.02,
        0.02,
        (
            f"r0={shortcut['initial_radius_mm']:.3f} mm\n"
            f"r1={shortcut['derived_endpoint_radius_mm']:.3f} mm\n"
            f"error={shortcut['offline_control_direction_error_deg']:.2f} deg"
        ),
        transform=ax.transAxes,
        va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="#cbd5e1", alpha=0.92),
    )

    timeline_ax = fig.add_subplot(grid[0, 2:])
    timeline_ax.axis("off")
    xs = np.linspace(0.08, 0.92, 4)
    timeline_ax.plot([xs[0], xs[-1]], [0.52, 0.52], lw=4, c="#cbd5e1")
    colors = ("#7c3aed", "#0ea5e9", "#f59e0b", "#16a34a")
    for index, (x, label, color) in enumerate(zip(xs, text["timeline"], colors), 1):
        timeline_ax.scatter([x], [0.52], s=260, c=color, zorder=4)
        timeline_ax.text(x, 0.52, str(index), ha="center", va="center", color="white", fontweight="bold")
        timeline_ax.text(x, 0.30 if index % 2 else 0.74, label, ha="center", va="center", fontsize=9)
    timeline_ax.set(xlim=(0, 1), ylim=(0.05, 0.95))

    table_ax = fig.add_subplot(grid[1, 2:])
    table_ax.axis("off")
    rows = [
        ["Entropy H" if language == "en" else "熵 H", "6.065", "6.448", "not converged" if language == "en" else "未收敛"],
        ["Credible radius" if language == "en" else "可信半径", "3.931 mm", "4.540 mm", "expanded" if language == "en" else "扩大"],
        ["XY path" if language == "en" else "XY 路径", "18.895 mm cmd", "8.052 mm measured", "contact-limited" if language == "en" else "接触限制"],
        ["Terminal", "15.45 s cue", "22.19 s bottom", "0 hard stop"],
    ]
    table = table_ax.table(
        cellText=rows,
        colLabels=("quantity", "before/command", "after/measured", "interpretation") if language == "en" else ("量", "之前/命令", "之后/实测", "解释"),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)

    for index, (panel, label, time_s) in enumerate(
        zip(panels, text["frames"], DISPLAY_TIMES_S)
    ):
        frame_ax = fig.add_subplot(grid[2, 2 * index : 2 * index + 2])
        frame_ax.imshow(panel)
        frame_ax.set_title(f"{label} | display t={time_s:.1f} s", fontweight="bold")
        frame_ax.axis("off")
    fig.text(0.5, 0.006, text["footer"], ha="center", va="bottom", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    _configure_fonts()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = _audit_payload(manifest)
    video = ROOT / manifest["artifacts"]["video"]["path"]
    panels = _frame_panels(video)
    outputs = {}
    for language in ("cn", "en"):
        path = OUTPUT / f"m8_sobol016_proposed_plus_audit_{language}.png"
        _plot(report, panels, language, path)
        outputs[language] = {
            "path": _relative(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    report["figures"] = outputs
    _atomic_json(AUDIT / "report.json", report)
    print(json.dumps({"status": report["status"], "figures": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
