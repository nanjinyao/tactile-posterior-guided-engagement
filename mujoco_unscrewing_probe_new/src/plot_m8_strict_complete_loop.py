#!/usr/bin/env python3
"""Plot the frozen M8 strict complete-loop audit without collapsing outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT / "outputs/current/evaluations/m8_strict_24_complete_loop/report.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/media/figures/paper/m8_strict_complete_loop.png"
)
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

METHODS = (
    "proposed",
    "proposed_plus",
    "outward",
    "entropy_grid",
    "direct_first",
    "random",
    "oracle",
)
DISPLAY = {
    "proposed": "Proposed",
    "proposed_plus": "Proposed+",
    "outward": "Outward",
    "entropy_grid": "Entropy-grid",
    "direct_first": "Direct-first",
    "random": "Random",
    "oracle": "Oracle",
}
SHORT = {
    "proposed": "Proposed",
    "proposed_plus": "P+",
    "outward": "Outward",
    "entropy_grid": "Entropy",
    "direct_first": "Direct-1",
    "random": "Random",
    "oracle": "Oracle",
}
COLORS = {
    "proposed": "#2563eb",
    "proposed_plus": "#0f766e",
    "outward": "#d97706",
    "entropy_grid": "#7c3aed",
    "direct_first": "#64748b",
    "random": "#9a3412",
    "oracle": "#15803d",
}
ROLES = {
    "proposed": "主方法：paper-facing complete method",
    "proposed_plus": "研究分支：adaptive policy；不声称 superiority",
    "outward": "几何比较：不使用 fingerprint 的 outward coverage",
    "entropy_grid": "强诊断：透明 active-perception / coverage policy",
    "direct_first": "机制诊断：只看第一次 direct ray；不是 retry baseline",
    "random": "弱下界：seed-fixed weak-search diagnostic",
    "oracle": "上界：使用 privileged center，不是可部署方法",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    design = report.get("design", {})
    if int(design.get("poses", -1)) != 24:
        raise ValueError("strict figure requires the frozen 24-pose report")
    if not bool(design.get("conditioned_on_ambiguous_top_contact", False)):
        raise ValueError("report is not conditioned on ambiguous top contact")
    if not bool(
        design.get("budget_is_hard_accumulated_simulated_active_controller_time", False)
    ):
        raise ValueError("report does not declare the hard active-time contract")
    if not math.isclose(float(design.get("active_time_budget_s", -1.0)), 130.0):
        raise ValueError("strict figure requires the frozen 130 s budget")
    methods = report.get("methods", {})
    missing = set(METHODS) - set(methods)
    if missing:
        raise ValueError(f"report is missing methods: {sorted(missing)}")
    for method in METHODS:
        item = methods[method]
        trials = int(item.get("trials", -1))
        capture = int(item.get("offline_capture_successes", -1))
        bottom = int(item.get("strict_bottom_successes", -1))
        if trials != 24 or not 0 <= bottom <= capture <= trials:
            raise ValueError(f"invalid capture/bottom counts for {method}")
        for field in (
            "time_to_bottom_success_median_s",
            "actual_path_success_median_mm",
            "wear_success_median_mj",
        ):
            value = item.get(field)
            if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"invalid {field} for {method}")
    return report


def configure_fonts() -> tuple[FontProperties, FontProperties]:
    for path in (FONT_REGULAR, FONT_BOLD):
        if not path.is_file():
            raise FileNotFoundError(path)
        fontManager.addfont(str(path))
    regular = FontProperties(fname=str(FONT_REGULAR))
    bold = FontProperties(fname=str(FONT_BOLD))
    plt.rcParams.update(
        {
            "font.family": regular.get_name(),
            "axes.unicode_minus": False,
            "axes.edgecolor": "#94a3b8",
            "axes.labelcolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )
    return regular, bold


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
        width=0.76,
        color=colors,
        alpha=0.22,
        edgecolor=colors,
        linewidth=1.4,
        zorder=2,
    )
    ax.bar(x, bottom, width=0.43, color=colors, alpha=0.94, zorder=3)
    for index, (strict_count, capture_count) in enumerate(zip(bottom, capture)):
        ax.text(
            index,
            capture_count + 0.55,
            f"{int(strict_count)}/{int(capture_count)}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#0f172a",
            fontproperties=bold,
        )
    ax.set_ylim(0, 26.2)
    ax.set_yticks(np.arange(0, 25, 4))
    ax.set_ylabel("trials out of 24")
    ax.set_xticks(x, [SHORT[method] for method in METHODS], rotation=24, ha="right")
    ax.grid(axis="y", alpha=0.20, zorder=0)
    ax.set_title(
        "A. CAD-bottom 与 offline capture 是不同结果",
        loc="left",
        fontsize=13.5,
        fontproperties=bold,
        color="#17375e",
        pad=9,
    )
    ax.text(
        0.0,
        0.985,
        "柱顶标注：strict bottom / offline capture",
        transform=ax.transAxes,
        va="top",
        fontsize=8.8,
        color="#64748b",
    )
    ax.legend(
        handles=(
            Patch(
                facecolor="#64748b",
                alpha=0.22,
                edgecolor="#475569",
                label="offline capture（MuJoCo 几何仅离线评分）",
            ),
            Patch(facecolor="#475569", label="strict CAD-bottom success"),
        ),
        loc="lower left",
        frameon=False,
        fontsize=8.2,
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
        "B. 效率只在各方法的 strict-success 子集上统计",
        loc="left",
        fontsize=13.5,
        fontproperties=bold,
        color="#17375e",
        pad=9,
    )
    fields = (
        ("time_to_bottom_success_median_s", "bottom time\nmedian [s]", 2),
        ("actual_path_success_median_mm", "actual XY path\nmedian [mm]", 2),
        ("wear_success_median_mj", "contact-work / wear proxy\nmedian [mJ]", 2),
    )
    cell_text = [
        [f"{float(methods[method][field]):.{digits}f}" for method in METHODS]
        for field, _, digits in fields
    ]
    row_labels = [label for _, label, _ in fields]
    column_labels = [
        f"{SHORT[method]}\nn={methods[method]['strict_bottom_successes']}"
        for method in METHODS
    ]
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=column_labels,
        cellLoc="center",
        rowLoc="right",
        bbox=(0.0, 0.16, 1.0, 0.73),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.8)
        cell.get_text().set_fontproperties(regular)
        if row == 0 and column >= 0:
            method = METHODS[column]
            cell.set_facecolor(COLORS[method])
            cell.get_text().set_color("white")
            cell.get_text().set_fontproperties(bold)
        elif column == -1:
            cell.set_facecolor("#f1f5f9")
            cell.get_text().set_color("#334155")
            cell.get_text().set_fontproperties(bold)
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
            cell.get_text().set_color("#0f172a")
    ax.text(
        0.0,
        0.04,
        "不同列的 n 不同；这些 median 不能脱离成功率单独解释，也不是全 24 点的期望成本。",
        transform=ax.transAxes,
        fontsize=8.6,
        color="#92400e",
        fontproperties=regular,
    )


def plot_failures(ax: plt.Axes, report: dict, bold: FontProperties) -> None:
    methods = report["methods"]
    x = np.arange(len(METHODS))
    width = 0.23
    hard = np.asarray([methods[method]["hard_stops"] for method in METHODS])
    false = np.asarray(
        [methods[method]["sensor_handoff_false_positive_count"] for method in METHODS]
    )
    budget = np.asarray(
        [
            methods[method].get("failure_categories", {}).get(
                "time_budget_exhausted", 0
            )
            for method in METHODS
        ]
    )
    series = (
        (hard, -width, "#dc2626", "hard stop"),
        (false, 0.0, "#f59e0b", "false sensor handoff"),
        (budget, width, "#7c3aed", "130 s hard-budget exhausted"),
    )
    for values, offset, color, label in series:
        bars = ax.bar(x + offset, values, width=width, color=color, label=label)
        for bar, value in zip(bars, values):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(value) + 0.08,
                    str(int(value)),
                    ha="center",
                    va="bottom",
                    fontsize=8.6,
                    color="#0f172a",
                    fontproperties=bold,
                )
    maximum = max(3, int(max(np.max(hard), np.max(false), np.max(budget))))
    ax.set_ylim(0, maximum + 0.8)
    ax.set_yticks(np.arange(0, maximum + 1, 1))
    ax.set_ylabel("trial count")
    ax.set_xticks(x, [SHORT[method] for method in METHODS], rotation=24, ha="right")
    ax.grid(axis="y", alpha=0.20)
    ax.set_title(
        "C. 安全、误交接与硬预算退出分别报告",
        loc="left",
        fontsize=13.5,
        fontproperties=bold,
        color="#17375e",
    )
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.1)
    total_contract_violations = sum(
        int(methods[method]["budget_contract_violations"]) for method in METHODS
    )
    ax.text(
        0.99,
        0.96,
        f"budget contract violations = {total_contract_violations}\n"
        "budget exhaustion 是受控失败，不是延期\n"
        "其他 failure 未堆叠；capture→bottom 缺口见 A",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#475569",
    )


def plot_roles(
    ax: plt.Axes,
    report: dict,
    regular: FontProperties,
    bold: FontProperties,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "D. 方法角色与允许的结论",
        loc="left",
        fontsize=13.5,
        fontproperties=bold,
        color="#17375e",
        pad=8,
    )
    y = 0.90
    for method in METHODS:
        ax.add_patch(
            Rectangle(
                (0.015, y - 0.026),
                0.028,
                0.052,
                transform=ax.transAxes,
                facecolor=COLORS[method],
                edgecolor="none",
            )
        )
        ax.text(
            0.055,
            y,
            f"{DISPLAY[method]} — {ROLES[method]}",
            transform=ax.transAxes,
            va="center",
            fontsize=8.8,
            color="#0f172a",
            fontproperties=regular,
        )
        y -= 0.105
    ax.add_patch(
        Rectangle(
            (0.015, 0.025),
            0.97,
            0.105,
            transform=ax.transAxes,
            facecolor="#fff7ed",
            edgecolor="#fdba74",
            linewidth=1.0,
        )
    )
    ax.text(
        0.035,
        0.078,
        "冻结配对结果只作描述：不把 selected median 或未校正的多重比较升级为普适 superiority。",
        transform=ax.transAxes,
        va="center",
        fontsize=8.6,
        color="#9a3412",
        fontproperties=bold,
    )


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    output_path = args.output.resolve()
    report = load_report(report_path)
    regular, bold = configure_fonts()

    figure = plt.figure(figsize=(16.0, 9.0), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.08, 0.92),
        height_ratios=(1.03, 0.97),
        left=0.06,
        right=0.985,
        top=0.865,
        bottom=0.115,
        hspace=0.44,
        wspace=0.25,
    )
    plot_completion(figure.add_subplot(grid[0, 0]), report, bold)
    plot_efficiency_table(figure.add_subplot(grid[0, 1]), report, regular, bold)
    plot_failures(figure.add_subplot(grid[1, 0]), report, bold)
    plot_roles(figure.add_subplot(grid[1, 1]), report, regular, bold)

    figure.text(
        0.06,
        0.955,
        "M8 strict complete-loop audit — 24 frozen poses, simulation only",
        fontsize=21,
        color="#17375e",
        fontproperties=bold,
    )
    figure.text(
        0.06,
        0.914,
        "条件：ambiguous top contact｜strict success = sensor handoff + offline capture + CAD-bottom dwell + no hard stop｜130 s hard active budget",
        fontsize=10.8,
        color="#334155",
        fontproperties=regular,
    )
    figure.text(
        0.06,
        0.040,
        "Simulation-only evidence. Offline capture uses privileged MuJoCo geometry only for scoring; online policy receives neither true center nor contact tags. "
        "No hardware, passivity, motor-coupling, threaded-unscrewing, extraction, or universal-ranking claim.",
        fontsize=8.5,
        color="#475569",
        fontproperties=regular,
    )
    figure.text(
        0.985,
        0.018,
        f"source: {report_path.relative_to(ROOT)}",
        ha="right",
        fontsize=7.4,
        color="#94a3b8",
        fontproperties=regular,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor="white")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
