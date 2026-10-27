from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import cKDTree

from src.entropy_grid_baseline import EntropyGridBelief, EntropyGridConfig


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/allen_m8_source.json"
CERTIFICATE = (
    ROOT
    / "outputs/current/evaluations/m8_entropy_grid_coverage_certificate/report.json"
)
OUTPUT = (
    ROOT / "outputs/current/media/figures/paper/m8_entropy_grid_coverage_certificate.png"
)
VISUAL_SIGMA_M = 0.0020
VISUAL_BOUND_M = math.sqrt(5.991) * VISUAL_SIGMA_M


def build_belief() -> EntropyGridBelief:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    r_eff = float(spec["derived"]["r_cap_top_m"])
    return EntropyGridBelief(
        np.zeros(2),
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        support_inner_radius_m=0.5 * float(spec["derived"]["socket_top_af_m"]),
        support_outer_radius_m=0.5 * float(spec["screw_head_diameter_m"]),
        probe_radius_m=float(spec["key_af_m"]) / math.sqrt(3.0),
        config=EntropyGridConfig(
            belief_spacing_m=min(0.00025, r_eff),
            action_spacing_m=min(0.00075, 0.95 * math.sqrt(2.0) * r_eff),
            credible_radius_stop_m=r_eff,
        ),
        effective_capture_radius_m=r_eff,
    )


def main() -> None:
    report = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    belief = build_belief()
    r_eff_mm = report["geometry"]["effective_capture_radius_mm"]
    support_mm = report["geometry"]["visual_support_radius_mm"]
    action_mm = 1000.0 * belief.action_xy_m
    radius = np.linalg.norm(action_mm, axis=1)
    padded = radius > support_mm + 1.0e-9

    angle = 2.0 * math.pi * np.arange(8192) / 8192
    boundary = support_mm * np.column_stack((np.cos(angle), np.sin(angle)))
    distances, indices = cKDTree(action_mm).query(boundary, k=1)
    worst_index = int(np.argmax(distances))
    worst = boundary[worst_index]
    nearest = action_mm[int(indices[worst_index])]
    worst_gap = float(distances[worst_index])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.0),
        gridspec_kw={"width_ratios": [1.25, 0.75]},
    )
    full, zoom = axes
    full.add_patch(
        Circle((0.0, 0.0), support_mm, fill=False, color="#17375e", lw=2.0)
    )
    full.scatter(
        action_mm[~padded, 0],
        action_mm[~padded, 1],
        s=9,
        color="#16a085",
        alpha=0.72,
        label="action nodes inside visual support",
    )
    full.scatter(
        action_mm[padded, 0],
        action_mm[padded, 1],
        s=20,
        facecolor="#f59e0b",
        edgecolor="#92400e",
        linewidth=0.35,
        label="boundary-padding nodes",
        zorder=3,
    )
    full.scatter([0.0], [0.0], marker="x", s=75, color="#2563eb", label="visual anchor")
    full.scatter([worst[0]], [worst[1]], marker="*", s=145, color="#dc2626", zorder=5, label="audited worst boundary point")
    full.set_aspect("equal")
    limit = support_mm + 0.65
    full.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x relative to visual anchor [mm]", ylabel="y [mm]")
    full.grid(alpha=0.18)
    full.legend(loc="lower left", fontsize=8.5, frameon=True)
    full.set_title("Finite visual disk + padded action lattice")

    zoom.add_patch(
        Circle((0.0, 0.0), support_mm, fill=False, color="#17375e", lw=2.0)
    )
    for point in action_mm[np.linalg.norm(action_mm - worst[None, :], axis=1) < 0.8]:
        zoom.add_patch(
            Circle(
                point,
                r_eff_mm,
                facecolor="#a7f3d0",
                edgecolor="#059669",
                lw=0.8,
                alpha=0.28,
            )
        )
        zoom.scatter(point[0], point[1], s=24, color="#047857", zorder=4)
    zoom.scatter([worst[0]], [worst[1]], marker="*", s=180, color="#dc2626", zorder=6)
    zoom.plot(
        [worst[0], nearest[0]],
        [worst[1], nearest[1]],
        color="#dc2626",
        lw=2.0,
        zorder=5,
    )
    zoom.annotate(
        f"nearest-node gap = {worst_gap:.4f} mm\n"
        f"r_eff = {r_eff_mm:.4f} mm",
        xy=(0.5 * (worst[0] + nearest[0]), 0.5 * (worst[1] + nearest[1])),
        xytext=(10, -50),
        textcoords="offset points",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#dc2626", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "#dc2626"},
    )
    zoom.set_aspect("equal")
    zoom.set_xlim(worst[0] - 0.65, worst[0] + 0.65)
    zoom.set_ylim(worst[1] - 0.65, worst[1] + 0.65)
    zoom.set_xlabel("x [mm]")
    zoom.set_ylabel("y [mm]")
    zoom.grid(alpha=0.18)
    zoom.set_title("Boundary node-spacing check (planner disks shown)")

    certificate = report["certificate"]
    figure.suptitle(
        "M8 entropy-grid finite-domain geometric coverage\n"
        f"h_grid = {report['geometry']['action_spacing_mm']:.4f} mm, "
        f"analytic/dense worst gap = {1000.0 * certificate['analytic_worst_case_gap_m']:.4f}/"
        f"{1000.0 * certificate['observed_worst_gap_m']:.4f} mm, "
        f"geometric condition satisfied = {certificate['certified']}",
        fontsize=16,
        fontweight="bold",
        color="#17375e",
    )
    figure.text(
        0.5,
        0.018,
        "The geometric condition assumes ideal tracking; contact execution, capture dwell, time budget, and engagement are evaluated separately.",
        ha="center",
        fontsize=9.5,
        color="#6b7280",
    )
    figure.tight_layout(rect=(0.0, 0.065, 1.0, 0.875))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
