#!/usr/bin/env python3
"""Plot the presentation-sized summary of the frozen M8 strict audit.

The full audit figure intentionally preserves method roles and named failure
outcomes.  This compact companion is for the main talk: it keeps only the two
distribution-level views that are not repeated on adjacent slides and a large
contract callout derived from the same machine-readable report.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

from src.plot_m8_strict_complete_loop import (
    COLORS,
    DEFAULT_REPORT,
    METHODS,
    ROOT,
    SHORT,
    configure_fonts,
    load_report,
)


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/media/figures/paper/m8_strict_complete_loop_compact.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_compact_claims(report: dict) -> dict:
    """Fail closed if the frozen report no longer supports the callout."""

    paired = report.get("paired", {}).get("proposed_vs_proposed_plus")
    if not isinstance(paired, dict):
        raise ValueError("report lacks paired.proposed_vs_proposed_plus")
    required_counts = ("both_success", "left_only_success", "right_only_success", "both_failed")
    if any(int(paired.get(field, -1)) < 0 for field in required_counts):
        raise ValueError("invalid Proposed/Proposed+ paired counts")
    paired_total = sum(int(paired[field]) for field in required_counts)
    if paired_total != 24:
        raise ValueError(f"paired Proposed/Proposed+ counts total {paired_total}, not 24")
    p_value = float(paired.get("mcnemar_exact_two_sided_p", math.nan))
    if not math.isclose(p_value, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"compact frozen callout requires paired p=1.0, got {p_value}")

    nonzero_budget = {
        method: {
            "violations": int(report["methods"][method]["budget_contract_violations"]),
            "extensions": int(
                report["methods"][method]["fallback_budget_extension_nonzero"]
            ),
        }
        for method in METHODS
        if int(report["methods"][method]["budget_contract_violations"]) != 0
        or int(report["methods"][method]["fallback_budget_extension_nonzero"]) != 0
    }
    if nonzero_budget:
        raise ValueError(f"nonzero strict budget contract fields: {nonzero_budget}")
    return paired


def plot_completion(ax: plt.Axes, report: dict, bold: FontProperties) -> None:
    methods = report["methods"]
    x = np.arange(len(METHODS))
    capture = np.asarray(
        [methods[method]["offline_capture_successes"] for method in METHODS],
        dtype=float,
    )
    bottom = np.asarray(
        [methods[method]["strict_bottom_successes"] for method in METHODS],
        dtype=float,
    )
    colors = [COLORS[method] for method in METHODS]

    ax.bar(
        x,
        capture,
        width=0.78,
        color=colors,
        alpha=0.20,
        edgecolor=colors,
        linewidth=2.0,
        zorder=2,
    )
    ax.bar(x, bottom, width=0.46, color=colors, alpha=0.96, zorder=3)
    for index, (strict_count, capture_count) in enumerate(zip(bottom, capture)):
        ax.text(
            index,
            capture_count + 0.55,
            f"{int(strict_count)}/{int(capture_count)}",
            ha="center",
            va="bottom",
            fontsize=14.0,
            color="#0f172a",
            fontproperties=bold,
        )

    ax.set_ylim(0, 26.5)
    ax.set_yticks(np.arange(0, 25, 4))
    ax.tick_params(axis="y", labelsize=12.5)
    ax.set_ylabel("trials / 24", fontsize=14.0)
    ax.set_xticks(x, [SHORT[method] for method in METHODS])
    ax.tick_params(axis="x", labelsize=12.0, pad=7)
    ax.grid(axis="y", alpha=0.20, linewidth=1.0, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "A) Strict CAD-bottom vs offline capture",
        loc="left",
        fontsize=19.0,
        fontproperties=bold,
        color="#17375e",
        pad=13,
    )
    ax.legend(
        handles=(
            Patch(
                facecolor="#64748b",
                alpha=0.20,
                edgecolor="#475569",
                label="offline capture (score only)",
            ),
            Patch(facecolor="#475569", label="strict CAD-bottom"),
        ),
        loc="lower left",
        frameon=False,
        fontsize=12.0,
    )
    ax.text(
        0.99,
        0.025,
        "labels: strict / capture",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11.5,
        color="#64748b",
    )


def plot_efficiency_table(
    ax: plt.Axes,
    report: dict,
    regular: FontProperties,
    bold: FontProperties,
) -> None:
    methods = report["methods"]
    ax.axis("off")
    ax.set_title(
        "B) Success-subset medians",
        loc="left",
        fontsize=19.0,
        fontproperties=bold,
        color="#17375e",
        pad=13,
    )
    fields = (
        ("time_to_bottom_success_median_s", "bottom time\n[s]"),
        ("actual_path_success_median_mm", "actual XY path\n[mm]"),
        ("wear_success_median_mj", "contact work\n[mJ]"),
    )
    table = ax.table(
        cellText=[
            [f"{float(methods[method][field]):.2f}" for method in METHODS]
            for field, _ in fields
        ],
        rowLabels=[label for _, label in fields],
        colLabels=[
            f"{SHORT[method]}\nn={methods[method]['strict_bottom_successes']}"
            for method in METHODS
        ],
        cellLoc="center",
        rowLoc="right",
        bbox=(0.0, 0.21, 1.0, 0.63),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(1.0)
        cell.PAD = 0.045
        cell.get_text().set_fontproperties(regular)
        if row == 0 and column >= 0:
            method = METHODS[column]
            cell.set_facecolor(COLORS[method])
            cell.get_text().set_color("white")
            cell.get_text().set_fontproperties(bold)
            cell.get_text().set_fontsize(11.2)
        elif column == -1:
            cell.set_facecolor("#eef2f7")
            cell.get_text().set_color("#334155")
            cell.get_text().set_fontproperties(bold)
            cell.get_text().set_fontsize(11.7)
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
            cell.get_text().set_color("#0f172a")
    ax.text(
        0.0,
        0.10,
        "Each column uses that method's strict-success subset; n differs.",
        transform=ax.transAxes,
        fontsize=12.4,
        color="#92400e",
        fontproperties=regular,
    )
    ax.text(
        0.0,
        0.035,
        "Do not interpret these medians independently of panel A.",
        transform=ax.transAxes,
        fontsize=12.4,
        color="#92400e",
        fontproperties=bold,
    )


def add_contract_callout(
    figure: plt.Figure,
    paired: dict,
    regular: FontProperties,
    bold: FontProperties,
) -> None:
    box = FancyBboxPatch(
        (0.055, 0.058),
        0.89,
        0.145,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor="#ecfdf5",
        edgecolor="#0f766e",
        linewidth=2.0,
    )
    figure.add_artist(box)
    figure.text(
        0.078,
        0.155,
        "Paired P vs P+: "
        f"{int(paired['both_success'])} both success, "
        f"{int(paired['left_only_success'])} P-only, "
        f"{int(paired['right_only_success'])} P+-only, "
        f"{int(paired['both_failed'])} both fail; exact McNemar p = "
        f"{float(paired['mcnemar_exact_two_sided_p']):.1f}.",
        fontsize=15.8,
        color="#134e4a",
        fontproperties=bold,
    )
    figure.text(
        0.078,
        0.098,
        "P+ no superiority",
        fontsize=17.0,
        color="#9a3412",
        fontproperties=bold,
    )
    figure.text(
        0.38,
        0.099,
        "All 7 methods: zero budget violation / zero extension",
        fontsize=15.2,
        color="#134e4a",
        fontproperties=regular,
    )


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    output_path = args.output.resolve()
    report = load_report(report_path)
    paired = validate_compact_claims(report)
    regular, bold = configure_fonts()

    figure = plt.figure(figsize=(16.0, 9.0), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.12, 0.88),
        left=0.055,
        right=0.97,
        top=0.82,
        bottom=0.27,
        wspace=0.16,
    )
    plot_completion(figure.add_subplot(grid[0, 0]), report, bold)
    plot_efficiency_table(figure.add_subplot(grid[0, 1]), report, regular, bold)

    figure.text(
        0.055,
        0.947,
        "M8 strict complete-loop audit — compact main-deck result",
        fontsize=23.0,
        color="#17375e",
        fontproperties=bold,
    )
    figure.text(
        0.055,
        0.895,
        "24 frozen ambiguous-top-contact poses | simulation only | strict success includes CAD-bottom dwell and no hard stop | 130 s hard active budget",
        fontsize=12.8,
        color="#334155",
        fontproperties=regular,
    )
    add_contract_callout(figure, paired, regular, bold)
    figure.text(
        0.97,
        0.018,
        f"source: {report_path.relative_to(ROOT)}",
        ha="right",
        fontsize=8.2,
        color="#94a3b8",
        fontproperties=regular,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor="white")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
