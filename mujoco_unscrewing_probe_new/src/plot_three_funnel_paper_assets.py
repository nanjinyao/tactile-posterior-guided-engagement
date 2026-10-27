from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge
import numpy as np

from src.m8_direction_runtime import M8DirectionEnsemble


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current/media/figures/paper"
MODEL_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
P1V5 = ROOT / "outputs/current/evaluations/p1v5_m8_final_untouched_direction_audit"
M8_REPORT = ROOT / "outputs/current/evaluations/m8_four_method_sensor_only_audit/report.json"
FUNNEL_REPORT = ROOT / "outputs/current/evaluations/m8_funnel_guard_ablation/report.json"
S1_REPORT = ROOT / "outputs/current/evaluations/m8_model_m10_m12_zero_shot/report.json"
S2_REPORT = ROOT / "outputs/current/evaluations/m8_model_m10_m12_zero_shot_full_loop/report.json"


COLORS = {
    "fingerprint": "#1f5a96",
    "search": "#16847b",
    "physical": "#d28a20",
    "proposed": "#17375e",
    "outward": "#8c96a3",
    "direct": "#d97706",
    "random": "#c55a11",
    "oracle": "#7c3aed",
    "danger": "#b42318",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_three_funnel_overview() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0))
    panels = [
        (
            "1  Fingerprint funnel",
            COLORS["fingerprint"],
            "aliased static wrench",
            "15 deg phase probe",
            "posterior output",
        ),
        (
            "2  Search funnel",
            COLORS["search"],
            "visual/CAD center support",
            "bounded posterior transport",
            "residual tube",
        ),
        (
            "3  Physical funnel",
            COLORS["physical"],
            "chamfer / wall capture basin",
            "tip-RCC + bounded recovery",
            "verified / named exit",
        ),
    ]
    for ax, (title, color, top, middle, bottom) in zip(axes, panels):
        _clean_axis(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(
            Polygon(
                [[0.08, 0.86], [0.92, 0.86], [0.66, 0.14], [0.34, 0.14]],
                closed=True,
                facecolor=color,
                alpha=0.10,
                edgecolor=color,
                linewidth=2.4,
            )
        )
        for y, width, alpha in ((0.76, 0.72, 0.18), (0.51, 0.48, 0.35), (0.25, 0.25, 0.75)):
            ax.add_patch(
                Rectangle(
                    (0.5 - width / 2, y - 0.045),
                    width,
                    0.09,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="none",
                )
            )
        ax.add_patch(
            FancyArrowPatch(
                (0.5, 0.70),
                (0.5, 0.31),
                arrowstyle="-|>",
                mutation_scale=18,
                color=color,
                linewidth=2.0,
            )
        )
        ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=16, weight="bold", color=color)
        ax.text(0.5, 0.76, top, ha="center", va="center", fontsize=11)
        ax.text(0.5, 0.51, middle, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(0.5, 0.25, bottom, ha="center", va="center", fontsize=11, color="white", weight="bold")
    fig.suptitle(
        "One hidden center, three operational allocations of uncertainty and risk",
        fontsize=18,
        weight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "The stages need not contract monotonically: sensing allocates belief, search covers the residual set, and mechanics bounds consequences before terminal verification.",
        ha="center",
        fontsize=12,
        color="#334155",
    )
    fig.tight_layout()
    return _save(fig, "three_funnel_overview.png")


def plot_information_flow() -> Path:
    fig, ax = plt.subplots(figsize=(15.5, 5.0))
    _clean_axis(ax)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 5.0)
    boxes = [
        (0.4, 2.85, 2.1, 1.15, "Coarse visual prior\n$p_0(C)$", "#dbeafe", COLORS["fingerprint"]),
        (0.4, 1.25, 2.1, 1.15, "Approx. CAD\n$G, r_{eff}, z_b$", "#dcfce7", COLORS["search"]),
        (3.2, 2.05, 2.4, 1.3, "15 deg conical scan\n$y(\\phi)$", "#e0ecff", COLORS["fingerprint"]),
        (6.3, 2.05, 2.5, 1.3, "Diagnostic posterior\n$p(\\theta\\mid y),\ p(s\\mid y)$", "#e0ecff", COLORS["fingerprint"]),
        (9.5, 2.05, 2.5, 1.3, "Center belief\n$p(C\\mid y,v,G)$", "#dff5f0", COLORS["search"]),
        (12.7, 2.85, 2.4, 1.15, "Risk-bounded shortcut\n+ residual search", "#dff5f0", COLORS["search"]),
        (12.7, 1.25, 2.4, 1.15, "Tip-RCC + R3 /\nterminal verifier", "#fff1cf", COLORS["physical"]),
    ]
    for x, y, w, h, label, face, edge in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=13, weight="bold")
    arrows = [
        ((2.5, 3.42), (3.2, 2.90)),
        ((2.5, 1.82), (9.5, 2.35)),
        ((5.6, 2.70), (6.3, 2.70)),
        ((8.8, 2.70), (9.5, 2.70)),
        ((12.0, 2.70), (12.7, 3.42)),
        ((13.9, 2.85), (13.9, 2.40)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, color="#475569", linewidth=1.8))
    ax.text(7.75, 4.55, "Narrow learned diagnostic interface", ha="center", fontsize=15, weight="bold", color=COLORS["fingerprint"])
    ax.text(13.9, 0.65, "Physical control owns consequences and terminal decisions", ha="center", fontsize=12, weight="bold", color=COLORS["physical"])
    fig.tight_layout()
    return _save(fig, "three_funnel_information_flow.png")


def plot_related_work_gap() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8))
    panels = [
        (
            "Conical contact identification",
            COLORS["fingerprint"],
            ["designed phase probe", "contact-pattern classifier", "admittance slide"],
            "direction information",
            "no bounded center-belief search",
        ),
        (
            "Outward unscrewing search",
            COLORS["search"],
            ["vision rough pose", "outward spiral", "force/torque trigger"],
            "dense geometric search",
            "no learned tactile shortcut",
        ),
        (
            "Interactive / physical funnels",
            COLORS["physical"],
            ["belief contraction", "contact-rich interaction", "compliant insertion"],
            "uncertainty and safety funnels",
            "not tailored to Allen phase fingerprints",
        ),
    ]
    for ax, (title, color, steps, strength, gap) in zip(axes, panels):
        _clean_axis(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.94, title, ha="center", va="top", fontsize=14, weight="bold", color=color)
        for index, step in enumerate(steps):
            y = 0.73 - 0.18 * index
            ax.add_patch(Rectangle((0.14, y - 0.055), 0.72, 0.11, facecolor=color, alpha=0.13 + 0.10 * index, edgecolor=color, linewidth=1.4))
            ax.text(0.5, y, step, ha="center", va="center", fontsize=11, weight="bold" if index == 1 else None)
            if index < 2:
                ax.add_patch(FancyArrowPatch((0.5, y - 0.06), (0.5, y - 0.12), arrowstyle="-|>", mutation_scale=12, color="#64748b"))
        ax.text(0.5, 0.20, f"keeps: {strength}", ha="center", fontsize=10, color=color, weight="bold")
        ax.text(0.5, 0.08, f"missing: {gap}", ha="center", fontsize=9.5, color="#b42318")
    fig.suptitle("The gap is not another network; it is a bounded-search bridge between sensing and insertion", fontsize=17, weight="bold")
    fig.tight_layout()
    return _save(fig, "related_work_mechanism_gap.png")


def _representative_trial() -> Path:
    with (P1V5 / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(
        key=lambda row: (
            row["search_state"] != "double_overhang",
            abs(float(row["angle_error_deg"]) - 1.5),
        )
    )
    return P1V5 / "trials" / f"{rows[0]['trial_id']}.npz"


def plot_fingerprint_example() -> Path:
    trial = _representative_trial()
    with np.load(trial, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        sequence = np.asarray(data["phase_sequence"], dtype=float)
        columns = tuple(map(str, data["phase_columns"].tolist()))
    selected = np.vstack(
        [
            sequence[columns.index("Fz_N")],
            sequence[columns.index("Tx_Nm")],
            sequence[columns.index("Ty_Nm")],
            sequence[columns.index("tip_z_m")],
        ]
    )
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    estimate = runtime.infer_phase_sequence(
        selected,
        target_down_force_n=float(metadata["target_down_force_n"]),
        bit_radius_m=0.006 / math.sqrt(3.0),
    )
    phase_deg = np.linspace(0.0, 360.0, selected.shape[1], endpoint=False)
    fig = plt.figure(figsize=(15.6, 7.2))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.15), hspace=0.34, wspace=0.26)
    labels = (("$F_z$", selected[0], "N"), ("$\\tau_x$", 1000 * selected[1], "mNm"), ("$\\tau_y$", 1000 * selected[2], "mNm"), ("$z$", 1000 * selected[3], "mm"))
    for index, (label, values, unit) in enumerate(labels):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        ax.plot(phase_deg, values, color=COLORS["fingerprint"], linewidth=1.7)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.grid(alpha=0.22)
        ax.set_ylabel(f"{label} [{unit}]")
        if index >= 2:
            ax.set_xlabel("conical phase [deg]")
    axp = fig.add_subplot(grid[:, 2], projection="polar")
    angles = np.deg2rad(np.arange(48) * 7.5 + 3.75)
    probabilities = np.asarray(estimate.direction_probabilities, dtype=float)
    axp.bar(angles, probabilities, width=np.deg2rad(6.5), color=COLORS["fingerprint"], alpha=0.78)
    true_direction = (
        math.degrees(math.atan2(float(metadata["y_m"]), float(metadata["x_m"]))) + 180.0
    ) % 360.0
    axp.plot([math.radians(true_direction)] * 2, [0, probabilities.max() * 1.12], color="#b42318", linewidth=2.5, label="true")
    axp.plot([math.radians(estimate.circular_mean_direction_deg)] * 2, [0, probabilities.max() * 1.12], color="#16a34a", linewidth=2.5, label="posterior mean")
    axp.set_theta_zero_location("E")
    axp.set_theta_direction(1)
    axp.set_yticklabels([])
    axp.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    axp.set_title(
        f"direction posterior\nAPS={len(estimate.aps_direction_classes)} bins, error={abs((estimate.circular_mean_direction_deg-true_direction+180)%360-180):.2f} deg",
        pad=20,
        fontsize=14,
        weight="bold",
    )
    fig.suptitle(
        "A standardized conical action converts contact ambiguity into a phase fingerprint",
        fontsize=18,
        weight="bold",
    )
    return _save(fig, "fingerprint_phase_and_direction_posterior.png")


def plot_m8_benchmark() -> Path:
    report = _load_json(M8_REPORT)
    method_keys = ["proposed_posterior", "baseline_a", "baseline_b", "baseline_c_random"]
    labels = ["Proposed", "A: outward", "B: direct", "C: random"]
    colors = [COLORS["proposed"], COLORS["outward"], COLORS["direct"], COLORS["random"]]
    fields = [
        ("success_rate", "CAD capture success", "%", 100.0),
        ("time_success_median_s", "Time on successes", "s", 1.0),
        ("contact_path_median_mm", "Contact path on successes", "mm", 1.0),
        ("wear_proxy_median_mj", "Contact-work proxy", "mJ", 1.0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.8))
    for ax, (field, title, unit, scale) in zip(axes, fields):
        values = [scale * float(report["methods"][key][field]) for key in method_keys]
        bars = ax.bar(np.arange(4), values, color=colors, width=0.72)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_ylabel(unit)
        ax.set_xticks(np.arange(4), labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Frozen M8 paired benchmark: completion, speed and contact cost are different axes", fontsize=17, weight="bold")
    fig.tight_layout()
    return _save(fig, "m8_four_method_benchmark.png")


def plot_m8_direction_audit() -> Path:
    report = _load_json(P1V5 / "independent_direction_report.json")
    radial = [report["radial_metrics"][str(index)] for index in range(5)]
    sectors = [report["sector_metrics"][str(index)] for index in range(6)]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))
    radii = [item["radius_mm"] for item in radial]
    p90 = [item["angle_p90_deg"] for item in radial]
    aps = [100 * item["aps_coverage"] for item in radial]
    axes[0].plot(radii, p90, "o-", color=COLORS["fingerprint"], linewidth=2.2)
    axes[0].axhline(7.5, color="#b42318", linestyle="--", linewidth=1.3)
    axes[0].set_title("radial direction p90", weight="bold")
    axes[0].set_xlabel("initial offset [mm]")
    axes[0].set_ylabel("deg")
    axes[0].grid(alpha=0.22)
    axes[1].bar(np.arange(5), aps, color=COLORS["search"], alpha=0.85)
    axes[1].set_xticks(np.arange(5), [f"r{i}" for i in range(5)])
    axes[1].set_ylim(90, 101)
    axes[1].set_title("APS coverage by radial stratum", weight="bold")
    axes[1].set_ylabel("%")
    axes[1].grid(axis="y", alpha=0.22)
    sector_rates = [100 * item["within_7p5_rate"] for item in sectors]
    axes[2].bar(np.arange(6), sector_rates, color="#5b8fc2")
    axes[2].set_xticks(np.arange(6), [f"S{i}" for i in range(6)])
    axes[2].set_ylim(90, 101)
    axes[2].set_title("physical-sector consistency", weight="bold")
    axes[2].set_ylabel("within 7.5 deg [%]")
    axes[2].grid(axis="y", alpha=0.22)
    fig.suptitle(
        "Held-out M8 direction interface: 180 scans, no >90 deg errors",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout()
    return _save(fig, "m8_untouched_direction_audit.png")


def plot_direct_ray_failure_mechanism() -> Path:
    report = _load_json(M8_REPORT)
    diagnostic = report["direct_margin_diagnostic"]
    values = [
        100 * diagnostic["negative_margin_first_attempt_success_rate"],
        100 * diagnostic["nonnegative_margin_first_attempt_success_rate"],
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(
        [0, 1],
        values,
        color=[COLORS["danger"], COLORS["search"]],
        width=0.65,
    )
    ax.set_xticks([0, 1], ["ray misses capture tube\nmargin < 0", "ray intersects capture tube\nmargin >= 0"])
    ax.set_ylabel("first-attempt success [%]")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=13, weight="bold")
    ax.set_title("A correct direction is not the same as a guaranteed direct ray", fontsize=15, weight="bold")
    ax.text(0.5, -0.16, "13/36 paired poses had negative direct-ray geometric margin.", transform=ax.transAxes, ha="center", fontsize=11, color="#475569")
    fig.tight_layout()
    return _save(fig, "direct_ray_geometric_margin.png")


def plot_search_funnel_case() -> Path:
    path = M8_REPORT.parent / "trials" / "sobol_002.json"
    record = _load_json(path)
    task = record["task"]
    result = record["methods"]["proposed_posterior"]
    initial = np.array([task["x_m"], task["y_m"]], dtype=float) * 1000.0
    target = np.asarray(result["transport_target_xy_mm"], dtype=float)
    true_center = np.zeros(2)
    visual_radius = 4.895303872079853
    capture_radius = 0.25
    local_radius = float(result["spiral_max_radius_mm"])
    direction_deg = float(result["control_direction_deg"])
    aps = np.asarray(result["aps_direction_angles_deg"], dtype=float)
    differences = (aps - direction_deg + 180.0) % 360.0 - 180.0
    half_width = max(7.5, float(np.max(np.abs(differences))) + 3.75)

    fig, ax = plt.subplots(figsize=(8.4, 7.5))
    ax.set_aspect("equal")
    ax.add_patch(Circle(initial, visual_radius, facecolor="#dbeafe", edgecolor="#5b8fc2", alpha=0.34, linewidth=2, label="visual 95% support"))
    ax.add_patch(Circle(true_center, capture_radius, facecolor=COLORS["physical"], edgecolor="#9a6700", alpha=0.75, linewidth=1.5, label="$r_{eff}$ capture set"))
    ax.add_patch(Wedge(initial, visual_radius, direction_deg - half_width, direction_deg + half_width, width=visual_radius - 0.3, facecolor=COLORS["fingerprint"], alpha=0.14, edgecolor=COLORS["fingerprint"], linewidth=1.4, label="fingerprint APS sector"))
    ax.annotate("", xy=target, xytext=initial, arrowprops=dict(arrowstyle="-|>", color=COLORS["search"], linewidth=3, mutation_scale=18))
    ax.plot(*initial, "o", color="#0f172a", markersize=8)
    ax.annotate(
        "initial contact",
        initial,
        xytext=(-104, -18),
        textcoords="offset points",
        fontsize=10,
        weight="bold",
        arrowprops=dict(arrowstyle="-", color="#475569", linewidth=1.0),
    )
    ax.plot(*target, "o", color=COLORS["search"], markersize=8)
    ax.annotate(
        f"bounded move\n{result['transport_distance_mm']:.2f} mm",
        target,
        xytext=(52, 28),
        textcoords="offset points",
        fontsize=10,
        weight="bold",
        color=COLORS["search"],
        arrowprops=dict(arrowstyle="-", color=COLORS["search"], linewidth=1.0),
    )
    theta_max = 2.0 * math.pi * max(1.0, local_radius / 0.4)
    theta = np.linspace(0.0, theta_max, 500)
    radius = np.minimum(local_radius, 0.4 * theta / (2.0 * math.pi))
    phase = math.radians(direction_deg)
    spiral = target + np.column_stack((radius * np.cos(theta + phase), radius * np.sin(theta + phase)))
    ax.plot(spiral[:, 0], spiral[:, 1], color=COLORS["search"], linewidth=2.3, label="residual dense outward search")
    ax.plot(*true_center, marker="x", color=COLORS["danger"], markersize=11, mew=2.4, label="true center (offline only)")
    ax.set_xlim(-5.7, 6.0)
    ax.set_ylim(-6.0, 5.7)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.set_title(
        "Search funnel: consume reliable direction, retain residual search",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.5,
        -0.12,
        "Frozen sobol_002: bounded transport keeps a finite spiral; its outer boundary is not coverage-certified.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout()
    return _save(fig, "search_funnel_bounded_transport_case.png")


def plot_funnel_guard_ablation() -> Path:
    report = _load_json(FUNNEL_REPORT)
    legacy = report["methods"]["legacy_insertion"]
    guard = report["methods"]["funnel_guard"]
    labels = ["legacy", "tip-RCC + shield"]
    colors = ["#94a3b8", COLORS["physical"]]
    metrics = [
        ([legacy["hard_stops"], guard["hard_stops"]], "hard stops", "count"),
        ([legacy["max_cad_depth_overrun_mm"], guard["max_cad_depth_overrun_mm"]], "maximum CAD overrun", "mm"),
        ([legacy["bottom_verified"], guard["bottom_verified"]], "bottom verified", "/12"),
        ([legacy["p90_max_force_n"], guard["p90_max_force_n"]], "force p90", "N"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.6))
    for ax, (values, title, unit) in zip(axes, metrics):
        bars = ax.bar([0, 1], values, color=colors, width=0.65)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_ylabel(unit)
        ax.set_xticks([0, 1], labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("The physical funnel is a damage-limiting shield, not a completion guarantee", fontsize=17, weight="bold")
    fig.tight_layout()
    return _save(fig, "physical_funnel_guard_ablation.png")


def plot_cross_size_direction() -> Path:
    report = _load_json(S1_REPORT)
    sizes = ["m8", "m10", "m12"]
    labels = [size.upper() for size in sizes]
    summaries = [report["sizes"][size]["summary"] for size in sizes]
    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.6))
    items = [
        ([100 * item["top1_within_15_rate"] for item in summaries], "direction within 15 deg", "%"),
        ([100 * item["aps_coverage"] for item in summaries], "APS coverage", "%"),
        ([100 * item["context_z_gt5_rate"] for item in summaries], "context beyond 5 sigma", "%"),
        ([item["max_force_p90_n"] for item in summaries], "probe force p90", "N"),
    ]
    colors = ["#5b8fc2", "#2e7d78", "#d28a20"]
    for ax, (values, title, unit) in zip(axes, items):
        bars = ax.bar(np.arange(3), values, color=colors, width=0.68)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_ylabel(unit)
        ax.set_xticks(np.arange(3), labels)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Frozen M8 representation transfers direction better than control scale", fontsize=17, weight="bold")
    fig.tight_layout()
    return _save(fig, "cross_size_zero_shot_direction.png")


def plot_cross_size_full_loop() -> Path | None:
    if not S2_REPORT.exists():
        return None
    report = _load_json(S2_REPORT)
    sizes = [size for size in ("m10", "m12") if size in report["sizes"]]
    methods = ["proposed", "outward", "oracle"]
    method_labels = ["Proposed", "Outward", "Oracle"]
    colors = [COLORS["proposed"], COLORS["outward"], COLORS["oracle"]]
    fig, axes = plt.subplots(len(sizes), 3, figsize=(15.5, 4.4 * len(sizes)), squeeze=False)
    for row, size in enumerate(sizes):
        items = report["sizes"][size]["methods"]
        values_by_panel = [
            ([100 * items[m]["bottom_success_rate"] for m in methods], "CAD-bottom success", "%"),
            ([items[m]["time_to_bottom_success_median_s"] or 0.0 for m in methods], "time on successes", "s"),
            ([items[m]["total_path_success_median_mm"] or 0.0 for m in methods], "XY path on successes", "mm"),
        ]
        for ax, (values, title, unit) in zip(axes[row], values_by_panel):
            bars = ax.bar(np.arange(3), values, color=colors, width=0.68)
            ax.set_title(f"{size.upper()} | {title}", fontsize=13, weight="bold")
            ax.set_ylabel(unit)
            ax.set_xticks(np.arange(3), method_labels, rotation=15)
            ax.grid(axis="y", alpha=0.22)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("S2: frozen M8 learner, size-conditioned CAD support, shared terminal", fontsize=17, weight="bold")
    fig.tight_layout()
    return _save(fig, "cross_size_zero_shot_full_loop.png")


def plot_sim2real_budget() -> Path:
    fig, ax = plt.subplots(figsize=(15.5, 4.8))
    _clean_axis(ax)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 4.8)
    stages = [
        ("H0", "vision + F/T\nframe/bias/delay", "no contact", "#dbeafe"),
        ("H1", "12 safety scans\nshadow mode", "commission", "#e0ecff"),
        ("H2", "24 anchors\nsmall adapter only", "adapt", "#dff5f0"),
        ("H3", "12 untouched\nrandom tests", "freeze", "#fff1cf"),
    ]
    x_positions = [0.5, 4.3, 8.1, 11.9]
    for index, ((stage, body, footer, face), x) in enumerate(zip(stages, x_positions)):
        ax.add_patch(Rectangle((x, 1.25), 2.9, 2.2, facecolor=face, edgecolor="#334155", linewidth=1.8))
        ax.text(x + 0.25, 3.18, stage, fontsize=18, weight="bold", color="#17375e")
        ax.text(x + 1.45, 2.35, body, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(x + 1.45, 1.5, footer, ha="center", fontsize=11, color="#475569")
        if index < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 2.9, 2.35), (x_positions[index + 1], 2.35), arrowstyle="-|>", mutation_scale=18, linewidth=2, color="#64748b"))
    ax.text(7.75, 4.25, "Measure interfaces first; spend real labels only at the adaptation bottleneck", ha="center", fontsize=17, weight="bold")
    ax.text(7.75, 0.62, "M10/M12: independent commissioning and untouched tests are mandatory for every claimed size", ha="center", fontsize=12, color="#b45309", weight="bold")
    fig.tight_layout()
    return _save(fig, "sim2real_minimum_data_budget.png")


def run() -> list[Path]:
    paths = [
        plot_three_funnel_overview(),
        plot_information_flow(),
        plot_related_work_gap(),
        plot_fingerprint_example(),
        plot_search_funnel_case(),
        plot_m8_direction_audit(),
        plot_m8_benchmark(),
        plot_direct_ray_failure_mechanism(),
        plot_funnel_guard_ablation(),
        plot_cross_size_direction(),
        plot_sim2real_budget(),
    ]
    cross_size = plot_cross_size_full_loop()
    if cross_size is not None:
        paths.append(cross_size)
    manifest = {
        "assets": [str(path.relative_to(ROOT)) for path in paths],
        "source_reports": [
            str(path.relative_to(ROOT))
            for path in (M8_REPORT, FUNNEL_REPORT, S1_REPORT, S2_REPORT)
            if path.exists()
        ],
        "claim_boundary": "Paper figures summarize frozen local artifacts; selected examples do not replace paired evaluations.",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Generate three-funnel paper slide assets.").parse_args()


if __name__ == "__main__":
    parse_args()
    for generated in run():
        print(generated)
