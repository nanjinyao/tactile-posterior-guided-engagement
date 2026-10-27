from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.patches import Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "current"
FIGURE_OUT = OUT / "media" / "figures"


def hex_vertices(apothem_mm: float, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    radius = apothem_mm / math.cos(math.pi / 6.0)
    cx, cy = center
    return np.array(
        [
            [cx + radius * math.cos(math.pi / 6.0 + i * math.pi / 3.0),
             cy + radius * math.sin(math.pi / 6.0 + i * math.pi / 3.0)]
            for i in range(6)
        ]
    )


def add_hex(ax, apothem_mm: float, *, center=(0.0, 0.0), **kwargs) -> Polygon:
    patch = Polygon(hex_vertices(apothem_mm, center), closed=True, **kwargs)
    ax.add_patch(patch)
    return patch


def load_specs() -> list[dict]:
    specs = []
    for path in sorted((ROOT / "configs").glob("allen_*_source.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        specs.append(data)
    order = {"m8": 0, "m10": 1, "m12": 2}
    return sorted(specs, key=lambda x: order[x["name"]])


def mm(value_m: float) -> float:
    return 1000.0 * value_m


def draw_size_panel(ax, spec: dict) -> None:
    name = spec["name"].upper()
    bit_af = mm(spec["key_af_m"])
    bit_r = bit_af / math.sqrt(3.0)
    c = mm(spec["clearance_m"])
    chamfer = mm(spec["chamfer_width_m"])
    r_cap_wall = c
    r_cap_top = c + chamfer
    wall_af = bit_af + 2.0 * c
    top_af = wall_af + 2.0 * chamfer
    head_r = mm(spec["screw_head_diameter_m"]) / 2.0
    visual = [mm(v) for v in spec["visual_offset_sweep_m"]]
    full_support_r = max(head_r - bit_r, 0.0)
    partial_support_r = head_r + bit_r

    ax.set_aspect("equal", adjustable="box")
    lim = max(partial_support_r, max(visual)) + 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, lw=0.3, color="#d4d4d8")
    ax.axhline(0, color="#71717a", lw=0.45)
    ax.axvline(0, color="#71717a", lw=0.45)
    ax.set_title(f"{name}: center-offset regions", fontsize=12, weight="bold")
    ax.set_xlabel("x offset from true socket center [mm]", fontsize=8)
    ax.set_ylabel("y offset [mm]", fontsize=8)

    ax.add_patch(Circle((0, 0), partial_support_r, facecolor="#fee2e2", edgecolor="none", alpha=0.50, label="partial / unsafe support bound"))
    ax.add_patch(Circle((0, 0), head_r, facecolor="#f1f5f9", edgecolor="#64748b", lw=1.2, label="screw head top"))
    ax.add_patch(Circle((0, 0), full_support_r, facecolor="#dcfce7", edgecolor="#16a34a", lw=1.0, alpha=0.45, label="full bit support center region"))

    add_hex(ax, top_af / 2.0, facecolor="#bfdbfe", edgecolor="#2563eb", lw=1.2, alpha=0.35, label="socket top opening")
    add_hex(ax, wall_af / 2.0, facecolor="none", edgecolor="#1e40af", lw=1.0, linestyle="-", label="vertical wall opening")
    add_hex(ax, r_cap_top, facecolor="#fef08a", edgecolor="#ca8a04", lw=1.1, alpha=0.70, label="bit-center chamfer capture")
    add_hex(ax, r_cap_wall, facecolor="#facc15", edgecolor="#854d0e", lw=1.0, alpha=0.75, label="bit-center wall clearance")

    colors = ["#0f766e", "#0891b2", "#7c3aed"]
    for rv, color in zip(visual, colors):
        ax.add_patch(Circle((0, 0), rv, facecolor="none", edgecolor=color, lw=1.0, linestyle="--"))
        ax.text(rv / math.sqrt(2), rv / math.sqrt(2), f"{rv:g} mm", fontsize=7, color=color)

    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"bit R={bit_r:.2f} mm",
                f"head R={head_r:.2f} mm",
                f"full support if |e| <= {full_support_r:.2f} mm",
                f"capture top ~= {r_cap_top:.2f} mm",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.92},
    )


def draw_bit_outline(ax, spec: dict, center: tuple[float, float], *, color: str) -> None:
    bit_af = mm(spec["key_af_m"])
    add_hex(ax, bit_af / 2.0, center=center, facecolor=color, edgecolor="#111827", lw=0.9, alpha=0.16)
    ax.plot([center[0]], [center[1]], marker="o", ms=3.5, color="#111827")


def draw_case_panel(ax, spec: dict) -> None:
    name = spec["name"].upper()
    bit_af = mm(spec["key_af_m"])
    bit_r = bit_af / math.sqrt(3.0)
    c = mm(spec["clearance_m"])
    chamfer = mm(spec["chamfer_width_m"])
    top_af = bit_af + 2.0 * (c + chamfer)
    head_r = mm(spec["screw_head_diameter_m"]) / 2.0
    full_support_r = max(head_r - bit_r, 0.0)
    partial_support_r = head_r + bit_r

    ax.set_aspect("equal", adjustable="box")
    lim = partial_support_r + 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, lw=0.3, color="#d4d4d8")
    ax.set_title(f"{name}: top-view initial-offset sampling", fontsize=12, weight="bold")
    ax.set_xlabel("x offset [mm]", fontsize=8)
    ax.set_ylabel("y offset [mm]", fontsize=8)
    ax.add_patch(Circle((0, 0), head_r, facecolor="#f8fafc", edgecolor="#64748b", lw=1.3))
    ax.add_patch(Circle((0, 0), full_support_r, facecolor="#dcfce7", edgecolor="#22c55e", lw=1.0, alpha=0.35))
    ax.add_patch(Circle((0, 0), partial_support_r, facecolor="none", edgecolor="#ef4444", lw=1.0, linestyle="--"))
    sweep_colors = ["#0f766e", "#2563eb", "#7c3aed"]
    sweep_text_y = [-0.42 * lim, -0.52 * lim, -0.62 * lim]
    direction_angles = np.deg2rad(np.arange(0, 360, 45))
    representative_angles = np.deg2rad([30, 150, 285])
    for idx, (radius, color, text_y) in enumerate(zip([mm(v) for v in spec["visual_offset_sweep_m"]], sweep_colors, sweep_text_y)):
        if radius <= full_support_r * 0.75:
            state = "supported search"
        elif radius <= full_support_r + 0.75:
            state = "support boundary"
        elif radius <= head_r * 0.80:
            state = "partial-support hard"
        elif radius <= head_r:
            state = "severe partial / abort stress"
        else:
            state = "unsafe / re-detect"
        label = f"{radius:g} mm: {state}"
        text_xy = (-0.46 * lim, text_y)
        ax.add_patch(Circle((0, 0), radius, facecolor="none", edgecolor=color, lw=1.15, linestyle=":"))
        sample_xy = np.column_stack([radius * np.cos(direction_angles), radius * np.sin(direction_angles)])
        ax.scatter(sample_xy[:, 0], sample_xy[:, 1], s=18, color=color, edgecolors="white", linewidths=0.45, zorder=5)
        rep_angle = representative_angles[idx]
        draw_bit_outline(ax, spec, (radius * math.cos(rep_angle), radius * math.sin(rep_angle)), color=color)
        ax.annotate(
            label,
            xy=(radius / math.sqrt(2.0), -radius / math.sqrt(2.0)),
            xytext=text_xy,
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8,
            color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": color},
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#cbd5e1", "alpha": 0.88},
        )
    add_hex(ax, top_af / 2.0, facecolor="#bfdbfe", edgecolor="#2563eb", alpha=0.35, lw=1.2)
    add_hex(ax, c + chamfer, facecolor="#fef08a", edgecolor="#ca8a04", alpha=0.65, lw=1.0)

    ax.text(
        0.02,
        0.02,
        "Coarse initial-position view uses three cases:\n"
        "1) capture / inserted near center\n"
        "2) supported search on head/rim/chamfer\n"
        "3) unsafe / re-detect outside support\n"
        "M8 detailed state view splits supported contact further.\n"
        "Dots: sampled initial directions on each radius",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.92},
    )


def draw_m8_radial_state_panels(spec: dict) -> None:
    bit_af = mm(spec["key_af_m"])
    bit_r = bit_af / math.sqrt(3.0)
    c = mm(spec["clearance_m"])
    chamfer = mm(spec["chamfer_width_m"])
    top_af = bit_af + 2.0 * (c + chamfer)
    wall_af = bit_af + 2.0 * c
    head_r = mm(spec["screw_head_diameter_m"]) / 2.0
    full_support_r = max(head_r - bit_r, 0.0)
    no_overlap_r = head_r + bit_r
    double_overhang_r = 0.5 * (full_support_r + head_r)

    states = [
        {
            "title": "1. outer partial support",
            "offset": head_r + 0.7,
            "color": "#f97316",
            "note": "outer side partly unsupported; valid contact",
        },
        {
            "title": "2. double overhang",
            "offset": double_overhang_r,
            "color": "#a855f7",
            "note": "outer side and socket side both unsupported",
        },
        {
            "title": "3. supported search",
            "offset": full_support_r * 0.82,
            "color": "#22c55e",
            "note": "fully supported on screw-head top",
        },
        {
            "title": "4. rim / chamfer",
            "offset": c + chamfer * 1.4,
            "color": "#2563eb",
            "note": "stable chamfer contact; full fingerprint",
        },
        {
            "title": "5. inner-wall capture",
            "offset": c * 0.55,
            "color": "#0891b2",
            "note": "wall-constrained; verify not full cone",
        },
        {
            "title": "6. capture / inserted",
            "offset": 0.0,
            "color": "#eab308",
            "note": "centered enough to drop/insert",
        },
    ]

    fig, axes = plt.subplots(1, len(states), figsize=(18.6, 3.8), constrained_layout=True)
    lim = no_overlap_r + 1.5
    for ax, state in zip(axes, states):
        d = state["offset"]
        center = (d, 0.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 0.82, lim * 0.82)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, lw=0.25, color="#e5e7eb")
        ax.axhline(0, color="#cbd5e1", lw=0.7)

        ax.add_patch(Circle((0, 0), head_r, facecolor="#f8fafc", edgecolor="#64748b", lw=1.2))
        add_hex(ax, top_af / 2.0, facecolor="#dbeafe", edgecolor="#2563eb", alpha=0.45, lw=1.1)
        add_hex(ax, wall_af / 2.0, facecolor="none", edgecolor="#1e40af", alpha=0.95, lw=1.0)
        add_hex(ax, c + chamfer, facecolor="#fef08a", edgecolor="#ca8a04", alpha=0.80, lw=1.0)
        draw_bit_outline(ax, spec, center, color=state["color"])

        ax.plot([0, d], [0, 0], color=state["color"], lw=1.0, alpha=0.65)
        ax.text(0.02, 0.97, state["title"], transform=ax.transAxes, ha="left", va="top", fontsize=10, weight="bold")
        ax.text(0.02, 0.10, state["note"], transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
        ax.text(0.98, 0.97, f"|e|={d:.2f} mm", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#334155")

    fig.suptitle(
        "M8 top-view physical states from outside to inside after valid contact",
        fontsize=15,
        weight="bold",
    )
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUT / "top_view_m8_physical_states_outside_to_inside.png", dpi=220)
    plt.close(fig)


def write_note(specs: list[dict]) -> None:
    lines = [
        "# Top-View Initial Position Projection",
        "",
        "Generated by `src/plot_top_view_initial_positions.py`.",
        "",
        "This is a geometric projection of possible initial bit-center positions after coarse visual localization.",
        "It separates normal engagement trials from hard abort / re-detect cases.",
        "",
        "Interpretation:",
        "",
        "- The socket center is the origin.",
        "- Dashed circles are configured visual residual sweep radii.",
        "- The small yellow hexagons are bit-center capture regions, not physical hole size.",
        "- Normal trials should keep the bit footprint at least partially supported by the screw head top.",
        "- Positions outside head support are only unsafe / abort / re-detect cases.",
        "- The per-size initial-position plot uses three coarse cases: capture/inserted, supported search, unsafe/re-detect.",
        "- The detailed M8 physical-state plot starts after valid screw/head contact and splits the path into outer partial support, double-overhang bridge support, supported search, rim/chamfer, inner-wall capture, and capture/inserted.",
        "- Dots on each radius are directional initial samples; the translucent hexagon on each radius is only one representative bit footprint.",
        "",
        "| size | bit R circum [mm] | head R [mm] | full-support center bound [mm] | no-overlap bound [mm] | visual sweep [mm] | capture top [mm] |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for spec in specs:
        bit_af = mm(spec["key_af_m"])
        bit_r = bit_af / math.sqrt(3.0)
        head_r = mm(spec["screw_head_diameter_m"]) / 2.0
        full_support = max(head_r - bit_r, 0.0)
        no_overlap = head_r + bit_r
        capture_top = mm(spec["clearance_m"] + spec["chamfer_width_m"])
        sweep = " / ".join(f"{mm(v):g}" for v in spec["visual_offset_sweep_m"])
        lines.append(
            f"| {spec['name'].upper()} | {bit_r:.2f} | {head_r:.2f} | {full_support:.2f} | {no_overlap:.2f} | {sweep} | {capture_top:.2f} |"
        )
    lines.extend(
        [
            "",
            "Important boundary:",
            "",
            "```text",
            "A large initial offset can still be useful if the bit has interpretable contact with the screw head top, rim, or chamfer.",
            "Once the bit footprint is outside the screw-head support region, the trial should be treated as abort / re-detect rather than nominal engagement.",
            "```",
            "",
        ]
    )
    (ROOT / "docs" / "TOP_VIEW_INITIAL_POSITION_PROJECTION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = load_specs()

    for spec in specs:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.0), constrained_layout=True)
        draw_case_panel(ax, spec)
        fig.savefig(FIGURE_OUT / f"top_view_initial_position_cases_{spec['name']}.png", dpi=220)
        plt.close(fig)

    draw_m8_radial_state_panels(specs[0])
    write_note(specs)


if __name__ == "__main__":
    main()
