from __future__ import annotations

"""Generate bilingual, provenance-bounded paper figures for the M8 study.

The figures deliberately separate three kinds of information:

* retained controller-side summaries from a completed M8 rollout;
* contracted MuJoCo adaptation samples for M10/M12; and
* a deterministic, explicitly *planned* M8 hardware sampling protocol.

No figure promotes offline center/contact tags into controller inputs.  The
cross-size dataset root is a command-line argument so a later S4 contract can
be rendered without changing this source file.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
)
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/current/media/figures/paper"
DEFAULT_SHORTCUT_TRIAL = (
    ROOT
    / "outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop"
    / "trials/sobol_000.json"
)
DEFAULT_PAIRED_TRIALS = (
    ROOT
    / "outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop"
    / "trials"
)
DEFAULT_R3_REPORT = (
    ROOT
    / "outputs/current/evaluations/m8_recoverable_funnel_r3_ablation"
    / "report.json"
)
DEFAULT_DATASET_ROOT = (
    ROOT / "outputs/current/datasets/cross_size_s4b_adaptation"
)
DEFAULT_CROSS_SIZE_TEST_REPORT = (
    ROOT
    / "outputs/current/evaluations/cross_size_s4b_full_loop_untouched_v1"
    / "report.json"
)
DEFAULT_M8_SPEC = ROOT / "configs/allen_m8_source.json"

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

COLORS = {
    "ink": "#172033",
    "muted": "#5f6b7a",
    "grid": "#d7dde7",
    "blue": "#2563eb",
    "blue_light": "#dbeafe",
    "cyan": "#0891b2",
    "green": "#15803d",
    "green_light": "#dcfce7",
    "orange": "#ea580c",
    "orange_light": "#ffedd5",
    "red": "#c62828",
    "red_light": "#fee2e2",
    "purple": "#7c3aed",
    "purple_light": "#ede9fe",
    "gray_light": "#f1f5f9",
    "supported_search": "#2563eb",
    "double_overhang": "#7c3aed",
    "outer_partial": "#ea580c",
}


TEXT: dict[str, dict[str, str]] = {
    "en": {
        "shortcut_title": "Risk-bounded tactile shortcut: posterior summary, projection, and residual geometry",
        "angular_title": "A  Retained angular posterior interface",
        "aps_only": "APS bins are retained; full 48-bin probabilities are not",
        "control_mean": "control mean",
        "true_direction": "true direction\n(evaluation reference)",
        "projection_title": "B  Projected-center quantiles and implemented shortcut",
        "shortcut": "executed shortcut",
        "distance_cap": "radial posterior cap $L_{rad}$",
        "capture_margin": "projection margin $0.25r_{eff}$",
        "geometry_title": "C  Shortcut endpoint still owns a residual search set",
        "initial": "initial contact",
        "endpoint": "shortcut endpoint",
        "center": "true center\n(evaluation reference)",
        "capture_set": "planner disk $B_{r_{eff}}$\n(evaluation overlay)",
        "residual_support": "99% residual radius",
        "residual_required": "endpoint residual > $r_{eff}$\nresidual search remains necessary",
        "not_guarantee": "SHORTCUT IS NOT A CAPTURE GUARANTEE",
        "shortcut_footer": "Representative M8 task; true geometry is shown only to interpret the posterior-guided command.",
        "paired_title": "Operational shortcut-policy comparison on paired common completions",
        "paired_subtitle": "Proposed minus Pure Outward under the same task, budget, physical funnel, and verifier",
        "delta_time": "active-time difference [s]",
        "delta_path": "measured XY-path difference [mm]",
        "delta_wear": "wear-proxy difference [mJ]",
        "negative_favors": "negative favors the shortcut arm",
        "paired_footer": "Proposed adds a conical probe, posterior shortcut, and local residual; Pure Outward runs global search. This is a system-level comparison; isolating shortcut length requires an L*=0 arm.",
        "median": "median",
        "lower": "lower",
        "exact_sign": "two-sided exact sign p",
        "r3_title": "Recovery restores engagement after rejected insertion",
        "r3_subtitle": "Same 12 post-capture stresses, Tip-RCC, and force guards",
        "no_funnel": "No compliant funnel",
        "r0": "Tip-RCC + guards\nno recovery",
        "r3_arm": "Tip-RCC + guards\nup to 3 recoveries",
        "bottom": "verified engagement",
        "safe_abort": "protected termination",
        "hard_stop": "force-limit termination",
        "not_verified": "not engaged",
        "r3_vs_no": "3 recoveries vs no funnel",
        "r3_vs_zero": "3 recoveries vs no recovery",
        "r3_footer": "Finite recovery changes reachable outcomes on this fixed MuJoCo set; two force-limit and one protected termination remain.",
        "sampling_title": "Symmetry-aware sampling for M10/M12 head adaptation",
        "train": "joint-head fine-tune / train",
        "val": "model selection / validation",
        "cal": "temperature + APS / calibration",
        "head_outline": "screw-head outline",
        "origin": "socket center (dataset coordinate)",
        "actual_sim": "MuJoCo contact scans",
        "orbits": "complete C6 orbits",
        "no_leak": "train / validation / calibration use disjoint C6 orbits",
        "state_supported": "supported search",
        "state_double": "double overhang",
        "state_outer": "outer partial",
        "sampling_footer": "Marker shape encodes train/validation/calibration; color encodes the contact regime.",
        "hardware_title": "Minimum M8 hardware adaptation plan — PLANNED, NO RESULTS",
        "hardware_map": "Top-view sampling plan",
        "ft_points": "fine-tune: 18 scans\n3 complete C6 orbits / 3 contact regimes",
        "cal_points": "calibration: 6 scans\n1 held-out complete C6 orbit",
        "test_points": "untouched conditioned-engagement starts: 12\nfixed before fine-tuning",
        "test_legend": "untouched conditioned engagement",
        "hardware_contract": "Protocol boundary",
        "hardware_lines": "• freeze the simulation encoder + shared trunk\n• update state + direction heads only\n• calibration cannot update weights\n• camera initializes a coarse visual prior only\n• independent center metrology/contact tags: offline scoring only\n• after initialization, feedback = wrench + robot pose\n• report success, timeout, hard stop and >90° error\n• every result cell remains — until executed",
        "planned": "PLANNED / NO MEASURED OUTCOME",
        "socket_hex": "socket-top geometry",
        "flow_title": "Three-layer closed-loop architecture and information boundary",
        "layer1": "LAYER 1  OBSERVABILITY / BELIEF",
        "layer2": "LAYER 2  TASK-AWARE SEARCH",
        "layer3": "LAYER 3  PHYSICAL CONTROL / SAFETY",
        "probe": "15° conical\nactive probe",
        "fingerprint": "four-channel fingerprint\nFz, τx, τy, tip-z",
        "posterior": "direction + contact-state\nposterior",
        "shortcut_box": "confidence-gated\nbounded shortcut",
        "residual_box": "task-aware residual\ngrid / outward reserve",
        "handoff": "sensor-only\ncapture handoff",
        "robot": "robot + M8\ncontact dynamics",
        "rcc": "tip-RCC +\nhealth/CAD shield",
        "r3": "bounded R3\nrecovery",
        "terminal": "independent\nterminal verifier",
        "outcomes": "success / safe abort\n/ hard stop",
        "recoverable": "recoverable rejection only",
        "sim_train": "simulation\npretraining",
        "head_ft": "diagnostic posterior\nheads only",
        "narrow_arrow": "posterior diagnostics only",
        "no_low_level": "no learned force / stiffness command",
        "camera_prior": "camera\ncoarse visual prior",
        "initialization_only": "initialization only",
        "offline": "OFFLINE EVALUATOR\nindependent center metrology + contact tags + CAD bottom\nscoring/provenance only — no controller return edge",
        "measurements": "after initialization: wrench + robot pose only",
        "probe_command": "probe command",
        "capture": "capture evidence",
        "controller_logs": "controller logs",
        "flow_footer": "Learning narrows uncertainty; search owns residual coverage; classical control and named exits own physical consequences.",
    },
    "cn": {
        "shortcut_title": "风险有界的触觉捷径：posterior 摘要、投影分位数与残余几何",
        "angular_title": "A  已留存的圆周 posterior 接口",
        "aps_only": "仅留存 APS bins；没有留存完整 48-bin 概率",
        "control_mean": "控制方向均值",
        "true_direction": "真实方向\n（评估参考）",
        "projection_title": "B  中心投影分位数与实际执行的捷径",
        "shortcut": "已执行捷径",
        "distance_cap": "径向 posterior 上限 $L_{rad}$",
        "capture_margin": "投影裕度 $0.25r_{eff}$",
        "geometry_title": "C  捷径端点仍对应有限残余搜索集合",
        "initial": "初始接触点",
        "endpoint": "捷径端点",
        "center": "真实中心\n（评估参考）",
        "capture_set": "规划圆盘 $B_{r_{eff}}$\n（评估叠加）",
        "residual_support": "99% 残余半径",
        "residual_required": "端点残差 > $r_{eff}$\n仍必须执行 residual search",
        "not_guarantee": "捷径不是 capture 成功保证",
        "shortcut_footer": "代表性 M8 任务；真实几何只用于解释 posterior-guided command。",
        "paired_title": "捷径策略组合的配对比较：只看共同完成任务",
        "paired_subtitle": "Proposed − Pure Outward；任务、预算、物理漏斗与 verifier 相同",
        "delta_time": "active time 差值 [s]",
        "delta_path": "measured XY path 差值 [mm]",
        "delta_wear": "wear proxy 差值 [mJ]",
        "negative_favors": "负值表示捷径分支更省",
        "paired_footer": "Proposed 增加锥形探测、posterior 捷径与局部 residual；Pure Outward 执行全局搜索。这里比较完整策略；隔离捷径长度仍需 L*=0 对照。",
        "median": "中位数",
        "lower": "更低",
        "exact_sign": "双侧精确 sign p",
        "r3_title": "有限恢复使被拒绝的插入重新可达",
        "r3_subtitle": "固定 12 个捕获后应力任务、Tip-RCC 与力限值",
        "no_funnel": "无柔顺物理漏斗",
        "r0": "Tip-RCC + 力保护\n不允许恢复",
        "r3_arm": "Tip-RCC + 力保护\n最多恢复 3 次",
        "bottom": "验证接合",
        "safe_abort": "保护性终止",
        "hard_stop": "力限值终止",
        "not_verified": "未接合",
        "r3_vs_no": "最多恢复 3 次 vs 无漏斗",
        "r3_vs_zero": "最多恢复 3 次 vs 不恢复",
        "r3_footer": "有限恢复改变该固定 MuJoCo 集合的可达结果；仍有 2 次力限值终止和 1 次保护性终止。",
        "sampling_title": "M10/M12 head adaptation 的对称性采样",
        "train": "state + direction heads / train",
        "val": "仅用于模型选择 / validation",
        "cal": "仅拟合 temperature + APS / calibration",
        "head_outline": "螺钉头外轮廓",
        "origin": "socket 中心（数据坐标）",
        "actual_sim": "MuJoCo 接触扫描",
        "orbits": "完整 C6 轨道",
        "no_leak": "train / validation / calibration 使用互不相交的 C6 轨道",
        "state_supported": "supported search",
        "state_double": "double overhang",
        "state_outer": "outer partial",
        "sampling_footer": "点形表示 train/validation/calibration，颜色表示接触状态。",
        "hardware_title": "最小 M8 真机 adaptation 计划——仅计划，尚无结果",
        "hardware_map": "俯视采样计划",
        "ft_points": "fine-tune：18 次扫描\n3 个完整 C6 轨道 / 3 种接触区间",
        "cal_points": "calibration：6 次扫描\n1 个留出的完整 C6 轨道",
        "test_points": "untouched 条件化 engagement 起点：12 个\nfine-tune 前冻结",
        "test_legend": "untouched 条件化 engagement",
        "hardware_contract": "实验合同边界",
        "hardware_lines": "• 冻结仿真 encoder + shared trunk\n• 仅更新 state + direction heads\n• calibration 禁止更新权重\n• camera 仅初始化粗粒度 visual prior\n• 独立中心测量/contact tags 仅离线评分\n• 初始化后 feedback 仅为 wrench + robot pose\n• 报告成功、超时、hard stop 与 >90° 错误\n• 真机执行前结果栏一律保持 —",
        "planned": "PLANNED / 尚无实测结果",
        "socket_hex": "socket-top 几何",
        "flow_title": "三层闭环架构与信息边界",
        "layer1": "第一层  可观测性 / BELIEF",
        "layer2": "第二层  任务感知搜索",
        "layer3": "第三层  物理控制 / 安全",
        "probe": "15° 锥形\n主动探测",
        "fingerprint": "四通道指纹\nFz、τx、τy、tip-z",
        "posterior": "方向 + 接触状态\nposterior",
        "shortcut_box": "置信门控的\n有界捷径",
        "residual_box": "任务感知 residual\ngrid / outward reserve",
        "handoff": "sensor-only\ncapture handoff",
        "robot": "机械臂 + M8\n接触动力学",
        "rcc": "tip-RCC +\nhealth/CAD shield",
        "r3": "有限 R3\n恢复",
        "terminal": "独立 terminal\nverifier",
        "outcomes": "success / safe abort\n/ hard stop",
        "recoverable": "仅 recoverable rejection",
        "sim_train": "仿真\n预训练",
        "head_ft": "仅诊断性 posterior\nheads",
        "narrow_arrow": "仅输出 posterior 诊断量",
        "no_low_level": "学习不输出 force / stiffness 指令",
        "camera_prior": "camera\n粗粒度 visual prior",
        "initialization_only": "仅用于初始化",
        "offline": "离线评估器\n独立中心测量 + contact tags + CAD bottom\n仅评分/追溯——没有返回控制器的边",
        "measurements": "初始化后：仅 wrench + robot pose",
        "probe_command": "探测指令",
        "capture": "capture 证据",
        "controller_logs": "控制器日志",
        "flow_footer": "学习只缩小不确定性；搜索负责残余覆盖；经典控制和具名失败出口负责物理后果。",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_fonts() -> None:
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
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
        }
    )


def _save(fig: plt.Figure, path: Path, description: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "src.plot_m8_bilingual_core_paper_assets",
            "Description": description,
        },
    )
    plt.close(fig)
    return path


def _wrap_deg(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def _plot_shortcut(trial_path: Path, output: Path, lang: str) -> Path:
    t = TEXT[lang]
    trial = _load_json(trial_path)
    result = trial["methods"]["proposed_plus"]
    task = trial["task"]

    initial = np.asarray(result["initial_xy_mm"], dtype=float)
    endpoint = np.asarray(result["transport_target_xy_mm"], dtype=float)
    center = np.zeros(2, dtype=float)
    r_eff = float(result["entropy_grid_geometry_mm"]["capture_radius"])
    residual_radius = float(result["spiral_max_radius_mm"])
    endpoint_residual = float(np.linalg.norm(endpoint - center))
    aps_deg = np.asarray(result["aps_direction_angles_deg"], dtype=float)
    control_deg = float(result["control_direction_deg"])
    true_deg = float(result["true_direction_deg"])
    q = result["transport_projected_quantiles_mm"]
    q10, q50, q90 = float(q["q10"]), float(q["q50"]), float(q["q90"])
    selected_q = float(result["transport_quantile"])
    quality = float(result["transport_direction_quality"])
    reverse_probability = float(result["transport_reverse_probability"])
    ensemble_span_deg = float(result["ensemble_top1_span_deg"])
    d_raw = float(result["transport_raw_distance_mm"])
    d_cap = float(result["transport_distance_cap_mm"])
    d_short = float(result["transport_distance_mm"])
    projection_margin = 0.25 * r_eff
    selected_bound = d_raw + projection_margin

    fig = plt.figure(figsize=(16.2, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.14, 1.45))

    ax0 = fig.add_subplot(grid[0, 0], projection="polar")
    ax0.set_theta_zero_location("E")
    ax0.set_theta_direction(1)
    ax0.set_ylim(0.0, 1.14)
    ax0.set_yticks([])
    ax0.grid(alpha=0.24)
    ax0.set_title(t["angular_title"], fontsize=13, pad=16)
    aps_rad = np.deg2rad(aps_deg)
    ax0.scatter(
        aps_rad,
        np.ones_like(aps_rad),
        s=110,
        color=COLORS["purple"],
        edgecolor="white",
        linewidth=1.1,
        zorder=4,
    )
    for angle in aps_rad:
        ax0.plot([angle, angle], [0.73, 1.0], color=COLORS["purple"], lw=2.0, alpha=0.72)
    ax0.annotate(
        "",
        xy=(math.radians(control_deg), 0.92),
        xytext=(math.radians(control_deg), 0.08),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["blue"], lw=3.0),
    )
    ax0.annotate(
        "",
        xy=(math.radians(true_deg), 1.02),
        xytext=(math.radians(true_deg), 0.42),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["red"], lw=2.0, linestyle="--"),
    )
    ax0.text(
        0.02,
        -0.16,
        (
            f"APS={int(result['aps_set_size'])}, ρ={float(result['circular_concentration']):.3f}\n"
            f"{t['control_mean']}: {control_deg:.2f}° | "
            f"{t['true_direction']}: {true_deg:.2f}°"
        ),
        transform=ax0.transAxes,
        fontsize=9.2,
        va="top",
    )
    ax0.text(
        0.5,
        0.02,
        t["aps_only"],
        transform=ax0.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=COLORS["muted"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["grid"], alpha=0.92),
    )

    ax1 = fig.add_subplot(grid[0, 1])
    ax1.set_title(t["projection_title"], fontsize=13, pad=12)
    xmax = max(q90, d_cap, selected_bound) + 0.55
    ax1.set_xlim(-0.1, xmax)
    ax1.set_ylim(-0.55, 1.12)
    ax1.set_yticks([])
    ax1.set_xlabel("projected center distance [mm]")
    ax1.grid(axis="x", alpha=0.22)
    ax1.axhline(0.24, color=COLORS["ink"], lw=1.4)
    for value, label, color, ytop in (
        (q10, "Q10", COLORS["cyan"], 0.76),
        (q50, "Q50", COLORS["purple"], 0.90),
        (q90, "Q90", COLORS["orange"], 0.76),
    ):
        ax1.vlines(value, 0.20, ytop, color=color, lw=2.2)
        ax1.text(value, ytop + 0.04, f"{label}\n{value:.2f}", ha="center", va="bottom", fontsize=9, color=color, weight="bold")
    ax1.annotate(
        "",
        xy=(d_short, 0.12),
        xytext=(0.0, 0.12),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["blue"], lw=4.0),
    )
    ax1.text(d_short / 2.0, -0.02, f"{t['shortcut']}\n{d_short:.3f} mm", ha="center", va="top", fontsize=9.5, color=COLORS["blue"], weight="bold")
    ax1.axvline(d_cap, color=COLORS["green"], lw=2.0, linestyle="--")
    ax1.text(d_cap, -0.43, f"{t['distance_cap']}\n{d_cap:.3f} mm", ha="center", va="top", fontsize=8.8, color=COLORS["green"])
    ax1.annotate(
        "",
        xy=(selected_bound, 0.47),
        xytext=(d_raw, 0.47),
        arrowprops=dict(arrowstyle="|-|", color=COLORS["red"], lw=1.8),
    )
    ax1.text((selected_bound + d_raw) / 2.0, 0.52, f"{t['capture_margin']}\n{projection_margin:.4f} mm", ha="center", va="bottom", fontsize=8.3, color=COLORS["red"])
    ax1.scatter([selected_bound], [0.24], s=65, marker="D", color=COLORS["blue"], zorder=5)
    ax1.text(selected_bound, 0.34, f"$Q^w_{{\\alpha}}(z)$={selected_bound:.3f}\n$\\alpha(q)$={selected_q:.3f}", ha="center", va="bottom", fontsize=8.8, color=COLORS["blue"])
    ax1.text(
        0.02,
        0.98,
        "$q=\\mathrm{clip}(\\kappa[\\cos(\\Delta_{ens}/2)]_+(1-2p_-),0,1)$\n"
        "$\\alpha(q)=.10+.15q,\\quad \\lambda(q)=.35+.30q$\n"
        "$L_{proj}=[Q^w_{\\alpha(q)}(z)-.25r_{eff}]_+$\n"
        "$L^*=\\min\\{L_{proj},L_{rad},L_{cmd,max}\\}$\n"
        f"$q={quality:.3f},\\ p_-={reverse_probability:.3f},\\ \\Delta_{{ens}}={ensemble_span_deg:.1f}^\\circ$",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        bbox=dict(boxstyle="round,pad=0.35", fc=COLORS["gray_light"], ec=COLORS["grid"]),
    )

    ax2 = fig.add_subplot(grid[0, 2])
    ax2.set_title(t["geometry_title"], fontsize=13, pad=12)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.22)
    ax2.add_patch(
        Circle(
            endpoint,
            residual_radius,
            facecolor=COLORS["blue_light"],
            edgecolor=COLORS["blue"],
            linestyle="--",
            linewidth=1.8,
            alpha=0.45,
        )
    )
    ax2.add_patch(
        Circle(
            center,
            r_eff,
            facecolor=COLORS["green_light"],
            edgecolor=COLORS["green"],
            linewidth=2.0,
            alpha=0.9,
        )
    )
    ax2.annotate(
        "",
        xy=endpoint,
        xytext=initial,
        arrowprops=dict(arrowstyle="-|>", color=COLORS["blue"], lw=3.4, mutation_scale=17),
    )
    ax2.plot(*initial, marker="o", ms=8, color=COLORS["ink"])
    ax2.plot(*endpoint, marker="o", ms=8, color=COLORS["blue"])
    ax2.plot(*center, marker="x", ms=11, mew=2.4, color=COLORS["red"])
    ax2.plot([endpoint[0], center[0]], [endpoint[1], center[1]], color=COLORS["red"], linestyle=":", lw=1.7)
    ax2.annotate(t["initial"], initial, xytext=(9, -18), textcoords="offset points", fontsize=8.8, weight="bold")
    ax2.annotate(t["endpoint"], endpoint, xytext=(10, 10), textcoords="offset points", fontsize=8.8, weight="bold", color=COLORS["blue"])
    ax2.annotate(t["center"], center, xytext=(-12, -24), textcoords="offset points", ha="right", fontsize=8.8, color=COLORS["red"])
    ax2.annotate(
        t["capture_set"],
        (center[0] + 0.70 * r_eff, center[1] + 0.70 * r_eff),
        xytext=(38, 36),
        textcoords="offset points",
        fontsize=7.9,
        color=COLORS["green"],
        arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.0),
    )
    ax2.text(
        endpoint[0] + 0.08,
        endpoint[1] - residual_radius - 0.16,
        f"{t['residual_support']}: {residual_radius:.3f} mm",
        fontsize=8.7,
        color=COLORS["blue"],
        ha="center",
        va="top",
    )
    ax2.text(
        0.98,
        0.04,
        f"{t['residual_required']}\n{endpoint_residual:.3f} mm > {r_eff:.3f} mm",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.1,
        color=COLORS["red"],
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=COLORS["red_light"], ec=COLORS["red"], alpha=0.94),
    )
    all_points = np.vstack((initial, endpoint, center))
    xlo = min(float(np.min(all_points[:, 0])) - 0.65, endpoint[0] - residual_radius - 0.35)
    xhi = max(float(np.max(all_points[:, 0])) + 0.65, endpoint[0] + residual_radius + 0.35)
    ylo = min(float(np.min(all_points[:, 1])) - 0.65, endpoint[1] - residual_radius - 0.35)
    yhi = max(float(np.max(all_points[:, 1])) + 0.65, endpoint[1] + residual_radius + 0.35)
    ax2.set_xlim(xlo, xhi)
    ax2.set_ylim(ylo, yhi)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.text(
        0.5,
        1.01,
        t["not_guarantee"],
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=COLORS["red"],
        weight="bold",
    )

    fig.suptitle(t["shortcut_title"], fontsize=17, weight="bold")
    fig.text(0.5, -0.02, t["shortcut_footer"], ha="center", va="top", fontsize=9.5, color=COLORS["muted"])
    return _save(
        fig,
        output / f"m8_shortcut_posterior_geometry_{lang}.png",
        "Retained posterior summary and implemented risk-bounded shortcut geometry",
    )


def _two_sided_sign_p(negative: int, positive: int) -> float:
    n = negative + positive
    if n == 0:
        return 1.0
    tail = min(negative, positive)
    return min(
        1.0,
        2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2.0**n),
    )


def _paired_shortcut_audit(trials_dir: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    source_trials: list[dict[str, str]] = []
    for path in sorted(trials_dir.glob("sobol_*.json")):
        trial = _load_json(path)
        source_trials.append(
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        )
        methods = trial["methods"]
        proposed = methods["proposed"]
        outward = methods["outward"]

        def strict_bottom(record: dict[str, Any]) -> bool:
            return bool(record["bottom_reached"]) and str(record["result"]) == "cad_bottom_verified"

        if not (strict_bottom(proposed) and strict_bottom(outward)):
            continue
        cases.append(
            {
                "pose_id": str(trial["task"]["pose_id"]),
                "delta_active_time_s": float(proposed["time_to_bottom_s"])
                - float(outward["time_to_bottom_s"]),
                "delta_measured_xy_path_mm": float(proposed["actual_xy_path_mm"])
                - float(outward["actual_xy_path_mm"]),
                "delta_wear_proxy_mj": float(proposed["wear_proxy_mj"])
                - float(outward["wear_proxy_mj"]),
            }
        )

    if len(source_trials) != 24 or len(cases) != 19:
        raise RuntimeError(
            f"Expected 24 paired trials and 19 common strict completions, found "
            f"{len(source_trials)} and {len(cases)}"
        )

    metrics: dict[str, dict[str, Any]] = {}
    for key in (
        "delta_active_time_s",
        "delta_measured_xy_path_mm",
        "delta_wear_proxy_mj",
    ):
        values = np.asarray([float(case[key]) for case in cases], dtype=float)
        negative = int(np.sum(values < 0.0))
        positive = int(np.sum(values > 0.0))
        metrics[key] = {
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "negative": negative,
            "positive": positive,
            "ties": int(np.sum(values == 0.0)),
            "two_sided_exact_sign_p": _two_sided_sign_p(negative, positive),
        }

    return {
        "comparison": "proposed_minus_pure_outward",
        "conditioning": "both methods satisfy strict CAD-bottom engagement",
        "shared_contract": (
            "same task, nuisance, sensor seed, outward-search primitive, physical "
            "funnel, 130 s budget, zero extension, zero simulator reset, and "
            "terminal verifier"
        ),
        "policy_difference": (
            "Proposed pays for a conical probe, executes a posterior-bounded "
            "straight transport, then searches a local residual set; Pure Outward "
            "uses no conical probe and searches globally from the initial point."
        ),
        "claim_boundary": (
            "Conditional paired efficiency diagnostic on common completions; not "
            "a single-variable shortcut ablation, a population causal effect, or "
            "a success-rate superiority test."
        ),
        "common_strict_completions": len(cases),
        "metrics": metrics,
        "cases": cases,
        "source_trials": source_trials,
    }


def _plot_paired_shortcut(audit: dict[str, Any], output: Path, lang: str) -> Path:
    t = TEXT[lang]
    cases = list(audit["cases"])
    pose_labels = [str(case["pose_id"]).replace("sobol_", "") for case in cases]
    x = np.arange(len(cases))
    plot_specs = (
        ("delta_active_time_s", t["delta_time"]),
        ("delta_measured_xy_path_mm", t["delta_path"]),
        ("delta_wear_proxy_mj", t["delta_wear"]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharex=False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.25, top=0.67, wspace=0.23)
    for ax, (key, ylabel) in zip(axes, plot_specs):
        values = np.asarray([float(case[key]) for case in cases], dtype=float)
        colors = np.where(values < 0.0, COLORS["green"], COLORS["red"])
        ax.axhline(0.0, color=COLORS["ink"], linewidth=1.0)
        ax.vlines(x, 0.0, values, colors=colors, linewidth=1.4, alpha=0.8)
        ax.scatter(x, values, c=colors, s=28, zorder=3, edgecolors="white", linewidths=0.5)
        ax.grid(axis="y", alpha=0.22)
        ax.set_title(ylabel, fontsize=10.0, pad=30)
        stats = audit["metrics"][key]
        ax.text(
            0.5,
            1.04,
            f"{t['median']} {stats['median']:.2f}; "
            f"{stats['negative']}/{len(cases)} {t['lower']}\n"
            f"{t['exact_sign']}={stats['two_sided_exact_sign_p']:.2g}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=COLORS["green"],
            weight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(pose_labels, rotation=55, ha="right", fontsize=6.7)
        ax.set_xlabel("paired Sobol pose id", fontsize=8.2)
    fig.suptitle(t["paired_title"], fontsize=15.2, weight="bold", y=0.98)
    fig.text(0.5, 0.875, t["paired_subtitle"], ha="center", va="center", fontsize=9.8)
    fig.text(0.5, 0.105, t["negative_favors"], ha="center", fontsize=8.6, color=COLORS["green"], weight="bold")
    fig.text(0.5, 0.012, t["paired_footer"], ha="center", va="bottom", fontsize=7.2, color=COLORS["muted"])
    return _save(
        fig,
        output / f"m8_paired_shortcut_ablation_{lang}.png",
        "Conditional paired shortcut-versus-pure-outward efficiency audit",
    )


def _plot_r3_ablation(report_path: Path, output: Path, lang: str) -> Path:
    t = TEXT[lang]
    report = _load_json(report_path)
    methods = report["methods"]
    arms = ("no_funnel", "rcc_shield_r0", "rcc_shield_r3")
    labels = (t["no_funnel"], t["r0"], t["r3_arm"])
    categories = (
        ("bottom_verified", t["bottom"], COLORS["green"]),
        ("safe_aborts", t["safe_abort"], "#f59e0b"),
        ("hard_stops", t["hard_stop"], COLORS["red"]),
        ("bottom_not_verified", t["not_verified"], "#94a3b8"),
    )
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    x = np.arange(len(arms))
    bottoms = np.zeros(len(arms), dtype=float)
    for key, label, color in categories:
        values = np.asarray([float(methods[arm][key]) for arm in arms])
        bars = ax.bar(x, values, bottom=bottoms, color=color, width=0.62, label=label)
        for bar, value, base in zip(bars, values, bottoms):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    base + value / 2.0,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    color="white" if color != "#f59e0b" else COLORS["ink"],
                    fontsize=10,
                    weight="bold",
                )
        bottoms += values
    ax.set_ylim(0, 14.0)
    ax.set_yticks(range(0, 13, 2))
    ax.set_ylabel("paired tasks / 12")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.4)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True, fontsize=8.3)
    paired = report["paired"]
    ax.text(
        0.02,
        0.89,
        f"{t['r3_vs_no']}: +7 / -0, "
        f"p={paired['no_funnel_vs_r3']['exact_two_sided_sign_p']:.4f}\n"
        f"{t['r3_vs_zero']}: +9 / -0, "
        f"p={paired['r0_vs_r3']['exact_two_sided_sign_p']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["grid"], alpha=0.94),
    )
    fig.suptitle(t["r3_title"], fontsize=14.8, weight="bold", y=0.98)
    ax.set_title(t["r3_subtitle"], fontsize=9.5, color=COLORS["muted"], pad=36)
    fig.text(0.5, 0.01, t["r3_footer"], ha="center", va="bottom", fontsize=7.8, color=COLORS["muted"])
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.16, top=0.78)
    return _save(
        fig,
        output / f"m8_r3_paired_ablation_{lang}.png",
        "Current paired no-funnel, R0, and R3 post-capture stress ablation",
    )


def _contract_roles(contract: dict[str, Any]) -> tuple[str, ...]:
    tasks = contract.get("tasks", {})
    train = next((key for key in ("train", "fine_tune", "finetune") if key in tasks), None)
    validation = next((key for key in ("validation", "val") if key in tasks), None)
    cal = next((key for key in ("calibration", "cal") if key in tasks), None)
    if train is None or (validation is None and cal is None):
        raise KeyError("dataset contract must expose train/fine_tune and calibration tasks")
    # Historical S3 contracts had only train/calibration.  S4 adds a disjoint
    # validation role, which must remain visually distinct from calibration.
    return tuple(role for role in (train, validation, cal) if role is not None)


def _resolve_contract_spec(contract: dict[str, Any]) -> dict[str, Any] | None:
    raw = contract.get("source_geometry", {}).get("spec")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return _load_json(path) if path.exists() else None


def _plot_cross_size(dataset_root: Path, sizes: list[str], output: Path, lang: str) -> tuple[Path, list[Path]]:
    t = TEXT[lang]
    contracts: list[tuple[str, Path, dict[str, Any]]] = []
    for size in sizes:
        path = dataset_root / size / "dataset_contract.json"
        if not path.exists():
            raise FileNotFoundError(f"missing dataset contract: {path}")
        contracts.append((size, path, _load_json(path)))

    fig, axes = plt.subplots(
        1,
        len(contracts),
        figsize=(8.0 * len(contracts), 7.0),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.19, top=0.78, wspace=0.24)
    state_label = {
        "supported_search": t["state_supported"],
        "double_overhang": t["state_double"],
        "outer_partial": t["state_outer"],
    }
    for ax, (size, _, contract) in zip(axes[0], contracts):
        roles = _contract_roles(contract)
        spec = _resolve_contract_spec(contract)
        role_style = {
            roles[0]: ("o", True, "-"),
        }
        if len(roles) == 2:
            role_style[roles[1]] = ("D", False, "--")
        else:
            role_style[roles[1]] = ("s", False, "-.")
            role_style[roles[2]] = ("D", False, "--")
        for role, (marker, fill, linestyle) in role_style.items():
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for task in contract["tasks"][role]:
                grouped[str(task["symmetry_orbit_id"])].append(task)
            for orbit in grouped.values():
                ordered = sorted(orbit, key=lambda item: int(item["rotation_index"]))
                xy = np.asarray([[1000.0 * float(item["x_m"]), 1000.0 * float(item["y_m"])] for item in ordered])
                closed = np.vstack((xy, xy[0]))
                state = str(ordered[0]["search_state"])
                color = COLORS.get(state, COLORS["muted"])
                ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.0, alpha=0.25, linestyle=linestyle)
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=45,
                    marker=marker,
                    facecolors=color if fill else "white",
                    edgecolors=color,
                    linewidths=1.45,
                    zorder=4,
                )
        if spec is not None:
            head_radius = 500.0 * float(spec["screw_head_diameter_m"])
            ax.add_patch(Circle((0.0, 0.0), head_radius, fill=False, ec=COLORS["muted"], lw=1.4, ls=":", alpha=0.8))
        ax.plot(0.0, 0.0, marker="+", ms=11, mew=2.0, color=COLORS["ink"])
        ax.set_aspect("equal")
        ax.grid(alpha=0.22)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        role_counts = [
            (
                role,
                len(contract["tasks"][role]),
                len({item["symmetry_orbit_id"] for item in contract["tasks"][role]}),
            )
            for role in roles
        ]
        count_line = " + ".join(
            f"{scans} {('val' if role.startswith('val') else 'cal' if role.startswith('cal') else 'train')}"
            for role, scans, _ in role_counts
        )
        orbit_line = "+".join(str(orbits) for _, _, orbits in role_counts)
        ax.set_title(
            f"{size.upper()}\n"
            f"{count_line} scans\n"
            f"{orbit_line} {t['orbits']}",
            fontsize=13,
        )
        ax.text(
            0.02,
            0.02,
            f"{t['no_leak']}\nstage: {contract.get('stage', 'unknown')}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.6,
            color=COLORS["muted"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["grid"], alpha=0.92),
        )

    has_validation = any("validation" in contract.get("tasks", {}) or "val" in contract.get("tasks", {}) for _, _, contract in contracts)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["ink"], markeredgecolor=COLORS["ink"], label=t["train"]),
    ]
    if has_validation:
        handles.append(
            Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=COLORS["ink"], label=t["val"])
        )
    handles.extend([
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=COLORS["ink"], label=t["cal"]),
        Line2D([0], [0], marker="o", color=COLORS["supported_search"], markerfacecolor=COLORS["supported_search"], label=t["state_supported"]),
        Line2D([0], [0], marker="o", color=COLORS["double_overhang"], markerfacecolor=COLORS["double_overhang"], label=t["state_double"]),
        Line2D([0], [0], marker="o", color=COLORS["outer_partial"], markerfacecolor=COLORS["outer_partial"], label=t["state_outer"]),
    ])
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(6, len(handles)),
        frameon=True,
        bbox_to_anchor=(0.5, 0.075),
        fontsize=9.2,
    )
    fig.suptitle(t["sampling_title"], fontsize=17, weight="bold", y=0.97)
    fig.text(
        0.5,
        0.018,
        f"{t['actual_sim']} | {t['sampling_footer']}",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color=COLORS["muted"],
    )
    path = _save(
        fig,
        output / f"cross_size_finetune_c6_sampling_{lang}.png",
        "Exact M10/M12 train, validation, and calibration tasks from dataset contracts",
    )
    return path, [item[1] for item in contracts]


def _c6_orbit(radius_mm: float, base_angle_deg: float, state: str, role: str) -> list[dict[str, Any]]:
    points = []
    for sector in range(6):
        angle_deg = base_angle_deg + 60.0 * sector
        angle = math.radians(angle_deg)
        points.append(
            {
                "role": role,
                "state": state,
                "orbit": f"planned_{role}_{state}",
                "sector": sector,
                "radius_mm": radius_mm,
                "angle_deg": angle_deg % 360.0,
                "x_mm": radius_mm * math.cos(angle),
                "y_mm": radius_mm * math.sin(angle),
            }
        )
    return points


def _hardware_plan(spec: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    bit_radius = 1000.0 * float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 500.0 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 500.0 * float(spec["derived"]["socket_top_af_m"])
    full_support = head_radius - bit_radius
    inner_overlap = socket_top_apothem + bit_radius
    radii = {
        "supported_search": 0.75 * full_support,
        "double_overhang": 0.50 * (full_support + inner_overlap),
        "outer_partial": inner_overlap + 0.15 * bit_radius,
    }
    fine_tune: list[dict[str, Any]] = []
    for state, base in (
        ("supported_search", 13.0),
        ("double_overhang", 23.0),
        ("outer_partial", 33.0),
    ):
        fine_tune.extend(_c6_orbit(radii[state], base, state, "fine_tune"))
    calibration = _c6_orbit(
        full_support + 0.70 * (inner_overlap - full_support),
        48.0,
        "double_overhang",
        "calibration",
    )
    max_radius = inner_overlap + 0.22 * bit_radius
    golden_angle = 137.50776405003785
    untouched: list[dict[str, Any]] = []
    for index in range(12):
        radius = max_radius * math.sqrt((index + 0.65) / 12.65)
        angle_deg = (7.0 + golden_angle * index) % 360.0
        angle = math.radians(angle_deg)
        untouched.append(
            {
                "role": "untouched_test",
                "state": "not_labeled_online",
                "orbit": None,
                "sector": None,
                "radius_mm": radius,
                "angle_deg": angle_deg,
                "x_mm": radius * math.cos(angle),
                "y_mm": radius * math.sin(angle),
            }
        )
    return {
        "fine_tune": fine_tune,
        "calibration": calibration,
        "untouched_test": untouched,
    }


def _regular_hex(apothem: float, rotation_deg: float = 30.0) -> np.ndarray:
    circumradius = apothem / math.cos(math.pi / 6.0)
    angles = np.deg2rad(rotation_deg + np.arange(6) * 60.0)
    return np.column_stack((circumradius * np.cos(angles), circumradius * np.sin(angles)))


def _plot_hardware(spec_path: Path, output: Path, lang: str) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    t = TEXT[lang]
    spec = _load_json(spec_path)
    plan = _hardware_plan(spec)
    head_radius = 500.0 * float(spec["screw_head_diameter_m"])
    socket_top_apothem = 500.0 * float(spec["derived"]["socket_top_af_m"])
    capture_radius = 1000.0 * float(spec["derived"]["r_cap_top_m"])

    fig = plt.figure(figsize=(15.8, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 0.85))
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title(t["hardware_map"], fontsize=14, pad=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.22)
    ax.add_patch(Circle((0.0, 0.0), head_radius, facecolor=COLORS["gray_light"], edgecolor=COLORS["muted"], lw=1.8, alpha=0.75))
    ax.add_patch(Polygon(_regular_hex(socket_top_apothem), closed=True, fill=False, edgecolor=COLORS["ink"], lw=2.0))
    ax.add_patch(Circle((0.0, 0.0), capture_radius, facecolor=COLORS["green_light"], edgecolor=COLORS["green"], lw=2.0, alpha=0.92))
    state_colors = {key: COLORS[key] for key in ("supported_search", "double_overhang", "outer_partial")}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in plan["fine_tune"]:
        grouped[str(item["orbit"])].append(item)
    for orbit in grouped.values():
        xy = np.asarray([[item["x_mm"], item["y_mm"]] for item in orbit])
        closed = np.vstack((xy, xy[0]))
        color = state_colors[str(orbit[0]["state"])]
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.0, alpha=0.30)
        ax.scatter(xy[:, 0], xy[:, 1], s=58, marker="o", facecolors=color, edgecolors="white", linewidths=0.9, zorder=5)
    cal_xy = np.asarray([[item["x_mm"], item["y_mm"]] for item in plan["calibration"]])
    ax.scatter(cal_xy[:, 0], cal_xy[:, 1], s=68, marker="D", facecolors="white", edgecolors=COLORS["orange"], linewidths=1.8, zorder=6)
    test_xy = np.asarray([[item["x_mm"], item["y_mm"]] for item in plan["untouched_test"]])
    ax.scatter(test_xy[:, 0], test_xy[:, 1], s=62, marker="x", color=COLORS["ink"], linewidths=1.8, zorder=6)
    limit = max(float(np.max(np.linalg.norm(test_xy, axis=1))), head_radius) + 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.text(
        0.5,
        0.50,
        t["planned"],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=25,
        color=COLORS["red"],
        alpha=0.16,
        weight="bold",
        rotation=20,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["ink"], markeredgecolor="white", label=f"{t['train']} (color = state)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=COLORS["orange"], label=t["cal"]),
        Line2D([0], [0], marker="x", color=COLORS["ink"], linestyle="none", label=t["test_legend"]),
        Line2D([0], [0], marker="o", color=COLORS["supported_search"], markerfacecolor=COLORS["supported_search"], label=t["state_supported"]),
        Line2D([0], [0], marker="o", color=COLORS["double_overhang"], markerfacecolor=COLORS["double_overhang"], label=t["state_double"]),
        Line2D([0], [0], marker="o", color=COLORS["outer_partial"], markerfacecolor=COLORS["outer_partial"], label=t["state_outer"]),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.0, frameon=True, ncol=2)

    side = fig.add_subplot(grid[0, 1])
    side.axis("off")
    cards = (
        (t["ft_points"], COLORS["blue_light"], COLORS["blue"]),
        (t["cal_points"], COLORS["orange_light"], COLORS["orange"]),
        (t["test_points"], COLORS["gray_light"], COLORS["ink"]),
    )
    y = 0.92
    for label, face, edge in cards:
        patch = FancyBboxPatch((0.04, y - 0.14), 0.92, 0.12, boxstyle="round,pad=0.018,rounding_size=0.02", transform=side.transAxes, fc=face, ec=edge, lw=1.6)
        side.add_patch(patch)
        side.text(0.08, y - 0.08, label, transform=side.transAxes, ha="left", va="center", fontsize=10.5, weight="bold", color=edge)
        y -= 0.17
    side.text(0.04, y - 0.01, t["hardware_contract"], transform=side.transAxes, ha="left", va="top", fontsize=13, weight="bold")
    side.text(
        0.04,
        y - 0.08,
        t["hardware_lines"],
        transform=side.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        linespacing=1.42,
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=COLORS["grid"]),
    )
    side.text(
        0.04,
        0.025,
        f"M8 CAD: head Ø {2.0 * head_radius:.2f} mm | socket-top AF {2.0 * socket_top_apothem:.2f} mm\n"
        f"offline top capture radius {capture_radius:.2f} mm | all coordinates are pre-experiment planned values",
        transform=side.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=COLORS["muted"],
    )
    fig.suptitle(t["hardware_title"], fontsize=17, weight="bold", color=COLORS["red"])
    path = _save(
        fig,
        output / f"m8_hardware_finetune_plan_{lang}.png",
        "Deterministic planned M8 hardware fine-tune/calibration/test sampling; no results",
    )
    return path, plan


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, face: str, edge: str, *, fontsize: float = 10.0, dashed: bool = False) -> tuple[float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        fc=face,
        ec=edge,
        lw=1.7,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize, weight="bold", color=COLORS["ink"])
    return x + w / 2.0, y + h / 2.0


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = COLORS["ink"], label: str | None = None, dashed: bool = False, rad: float = 0.0, label_offset: tuple[float, float] = (0.0, 0.018)) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=14,
        lw=1.8,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            0.5 * (start[0] + end[0]) + label_offset[0],
            0.5 * (start[1] + end[1]) + label_offset[1],
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            color=color,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88),
        )


def _plot_flow(output: Path, lang: str) -> Path:
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(16.0, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lanes = (
        (0.68, 0.22, COLORS["blue_light"], COLORS["blue"], t["layer1"]),
        (0.40, 0.22, COLORS["green_light"], COLORS["green"], t["layer2"]),
        (0.09, 0.27, COLORS["orange_light"], COLORS["orange"], t["layer3"]),
    )
    for y, h, face, edge, label in lanes:
        lane = FancyBboxPatch((0.02, y), 0.96, h, boxstyle="round,pad=0.012,rounding_size=0.018", transform=ax.transAxes, fc=face, ec=edge, lw=1.2, alpha=0.42)
        ax.add_patch(lane)
        ax.text(0.035, y + h - 0.028, label, transform=ax.transAxes, ha="left", va="top", fontsize=10.5, weight="bold", color=edge)

    probe = _box(ax, 0.08, 0.72, 0.15, 0.105, t["probe"], "white", COLORS["blue"])
    fingerprint = _box(ax, 0.31, 0.72, 0.18, 0.105, t["fingerprint"], "white", COLORS["blue"])
    posterior = _box(ax, 0.57, 0.72, 0.18, 0.105, t["posterior"], "white", COLORS["purple"])
    sim_train = _box(ax, 0.79, 0.795, 0.075, 0.065, t["sim_train"], COLORS["purple_light"], COLORS["purple"], fontsize=8.1, dashed=True)
    head_ft = _box(ax, 0.88, 0.795, 0.085, 0.065, t["head_ft"], COLORS["purple_light"], COLORS["purple"], fontsize=7.8, dashed=True)
    camera_prior = _box(ax, 0.80, 0.695, 0.155, 0.058, t["camera_prior"], COLORS["gray_light"], COLORS["muted"], fontsize=7.9, dashed=True)

    shortcut = _box(ax, 0.15, 0.445, 0.18, 0.105, t["shortcut_box"], "white", COLORS["green"])
    residual = _box(ax, 0.41, 0.445, 0.20, 0.105, t["residual_box"], "white", COLORS["green"])
    handoff = _box(ax, 0.70, 0.445, 0.16, 0.105, t["handoff"], "white", COLORS["green"])

    robot = _box(ax, 0.045, 0.20, 0.15, 0.095, t["robot"], "white", COLORS["orange"])
    rcc = _box(ax, 0.255, 0.20, 0.17, 0.095, t["rcc"], "white", COLORS["orange"])
    terminal = _box(ax, 0.49, 0.20, 0.16, 0.095, t["terminal"], "white", COLORS["red"])
    outcome = _box(ax, 0.735, 0.20, 0.20, 0.095, t["outcomes"], "white", COLORS["red"])
    r3 = _box(ax, 0.49, 0.105, 0.16, 0.055, t["r3"], COLORS["orange_light"], COLORS["orange"], fontsize=8.8)

    _arrow(ax, (0.23, 0.772), (0.31, 0.772), color=COLORS["blue"])
    _arrow(ax, (0.49, 0.772), (0.57, 0.772), color=COLORS["blue"])
    _arrow(ax, (0.865, 0.827), (0.88, 0.827), color=COLORS["purple"], dashed=True)
    _arrow(ax, (0.88, 0.795), (0.72, 0.825), color=COLORS["purple"], dashed=True, label=t["narrow_arrow"], rad=-0.10, label_offset=(0.0, 0.035))
    _arrow(ax, (0.80, 0.724), (0.75, 0.748), color=COLORS["muted"], dashed=True, label=t["initialization_only"], rad=0.08, label_offset=(0.015, -0.024))
    ax.text(0.875, 0.675, t["no_low_level"], transform=ax.transAxes, ha="center", va="top", fontsize=8.0, color=COLORS["red"], weight="bold")
    _arrow(ax, (posterior[0] - 0.04, 0.72), (shortcut[0] + 0.03, 0.55), color=COLORS["green"])
    _arrow(ax, (0.33, 0.497), (0.41, 0.497), color=COLORS["green"])
    _arrow(ax, (0.61, 0.497), (0.70, 0.497), color=COLORS["green"])
    _arrow(ax, (handoff[0], 0.445), (rcc[0], 0.295), color=COLORS["orange"], label=t["capture"], rad=0.10)
    _arrow(ax, (0.195, 0.247), (0.255, 0.247), color=COLORS["orange"])
    _arrow(ax, (0.425, 0.247), (0.49, 0.247), color=COLORS["red"])
    _arrow(ax, (0.65, 0.247), (0.735, 0.247), color=COLORS["red"])
    _arrow(ax, (terminal[0], 0.20), (r3[0], 0.16), color=COLORS["orange"], label=t["recoverable"], label_offset=(0.09, 0.0))
    _arrow(ax, (0.49, 0.132), (rcc[0] + 0.03, 0.20), color=COLORS["orange"], rad=-0.35)
    _arrow(ax, (robot[0] - 0.02, 0.295), (probe[0] - 0.02, 0.72), color=COLORS["blue"], label=t["measurements"], rad=0.07, label_offset=(-0.03, 0.0))
    _arrow(ax, (residual[0] - 0.02, 0.445), (robot[0] + 0.05, 0.295), color=COLORS["ink"], label=t["probe_command"], rad=-0.15, label_offset=(-0.02, -0.005))

    offline = FancyBboxPatch((0.31, 0.004), 0.55, 0.064, boxstyle="round,pad=0.012,rounding_size=0.014", transform=ax.transAxes, fc=COLORS["gray_light"], ec=COLORS["muted"], lw=1.5, linestyle="--")
    ax.add_patch(offline)
    ax.text(0.585, 0.036, t["offline"], transform=ax.transAxes, ha="center", va="center", fontsize=8.4, color=COLORS["muted"], weight="bold")
    _arrow(ax, (outcome[0], 0.20), (0.79, 0.068), color=COLORS["muted"], dashed=True, label=t["controller_logs"], label_offset=(0.04, 0.0))
    _arrow(ax, (robot[0], 0.20), (0.39, 0.068), color=COLORS["muted"], dashed=True, rad=-0.15)

    fig.suptitle(t["flow_title"], fontsize=18, weight="bold", y=0.98)
    fig.text(0.5, 0.005, t["flow_footer"], ha="center", va="bottom", fontsize=9.5, color=COLORS["muted"])
    return _save(
        fig,
        output / f"m8_three_layer_control_flow_{lang}.png",
        "Three-layer controller, search, adaptation, and offline-evaluation boundary",
    )


def _plot_cross_size_result(report_path: Path, output: Path, lang: str) -> Path:
    """Render the held-out transfer result without exposing internal run names."""

    report = _load_json(report_path)
    text = {
        "en": {
            "title": "Held-out cross-size engagement: adaptation does not dominate controls",
            "ylabel": "verified engagements (of 12)",
            "target": "80% task target",
            "methods": ["M8 zero-shot", "calibration", "adapted heads"],
            "footer": "White labels in bars: force-limit / protected. Same residual search, physical funnel, and 130 s budget.",
        },
        "cn": {
            "title": "跨尺寸 held-out 接合：适配 head 尚未优于对照",
            "ylabel": "验证接合数（共 12 次）",
            "target": "80% 任务目标",
            "methods": ["M8 zero-shot", "仅校准", "适配 head"],
            "footer": "柱内白字：力限值 / 保护性终止。所有接口共享 residual search、物理漏斗与 130 s 预算。",
        },
    }[lang]
    method_keys = ("zero_shot", "cal_only", "s4_fine_tuned")
    colors = (COLORS["muted"], COLORS["blue"], COLORS["green"])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), sharey=True)
    for axis, size in zip(axes, ("m10", "m12")):
        size_report = report["sizes"][size]
        methods = size_report["methods"]
        successes = [int(methods[key]["strict_bottom_successes"]) for key in method_keys]
        force_limits = [int(methods[key]["hard_stops"]) for key in method_keys]
        protected = [
            int(methods[key]["trials"])
            - int(methods[key]["strict_bottom_successes"])
            - int(methods[key]["hard_stops"])
            for key in method_keys
        ]
        x = np.arange(len(method_keys))
        bars = axis.bar(x, successes, color=colors, width=0.66, edgecolor="white", linewidth=1.0)
        axis.axhline(9.6, color=COLORS["red"], lw=1.4, ls="--", label=text["target"])
        for bar, success, force, safe in zip(bars, successes, force_limits, protected):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                success + 0.22,
                f"{success}/12",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                0.28,
                f"{force} / {safe}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="white" if success >= 2 else COLORS["ink"],
                weight="bold",
            )
        axis.set_title(size.upper(), fontsize=14)
        axis.set_xticks(x, text["methods"], rotation=12, ha="right")
        axis.set_ylim(0, 13.0)
        axis.grid(axis="y", color=COLORS["grid"], alpha=0.75)
        axis.set_axisbelow(True)
        axis.legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].set_ylabel(text["ylabel"])
    fig.suptitle(text["title"], fontsize=16, weight="bold", y=1.02)
    fig.text(0.5, -0.015, text["footer"], ha="center", fontsize=9.5, color=COLORS["muted"])
    fig.tight_layout()
    return _save(
        fig,
        output / f"cross_size_transfer_result_{lang}.png",
        "Held-out cross-size engagement comparison rendered from the frozen report",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shortcut-trial", type=Path, default=DEFAULT_SHORTCUT_TRIAL)
    parser.add_argument("--paired-trials", type=Path, default=DEFAULT_PAIRED_TRIALS)
    parser.add_argument("--r3-report", type=Path, default=DEFAULT_R3_REPORT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--cross-size-test-report", type=Path, default=DEFAULT_CROSS_SIZE_TEST_REPORT
    )
    parser.add_argument("--sizes", nargs="+", default=["m10", "m12"])
    parser.add_argument("--m8-spec", type=Path, default=DEFAULT_M8_SPEC)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _configure_fonts()
    output = args.output.resolve()
    shortcut_trial = args.shortcut_trial.resolve()
    paired_trials = args.paired_trials.resolve()
    r3_report = args.r3_report.resolve()
    dataset_root = args.dataset_root.resolve()
    cross_size_test_report = args.cross_size_test_report.resolve()
    m8_spec = args.m8_spec.resolve()
    output.mkdir(parents=True, exist_ok=True)

    paired_audit = _paired_shortcut_audit(paired_trials)
    paired_audit_path = output / "m8_paired_shortcut_ablation_audit.json"
    paired_audit_path.write_text(
        json.dumps(paired_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    generated: list[Path] = [paired_audit_path]
    contract_paths: list[Path] = []
    hardware_plan: dict[str, list[dict[str, Any]]] | None = None
    for lang in ("cn", "en"):
        generated.append(_plot_shortcut(shortcut_trial, output, lang))
        generated.append(_plot_paired_shortcut(paired_audit, output, lang))
        generated.append(_plot_r3_ablation(r3_report, output, lang))
        cross_path, used_contracts = _plot_cross_size(dataset_root, list(args.sizes), output, lang)
        generated.append(cross_path)
        contract_paths = used_contracts
        generated.append(_plot_cross_size_result(cross_size_test_report, output, lang))
        hardware_path, plan = _plot_hardware(m8_spec, output, lang)
        generated.append(hardware_path)
        hardware_plan = plan
        generated.append(_plot_flow(output, lang))

    manifest_path = output / "m8_bilingual_core_paper_assets_manifest.json"
    manifest = {
        "created_utc": _utc_now(),
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "claim_boundary": {
            "shortcut": "retained posterior summary and offline geometry; not a capture guarantee",
            "paired_shortcut": "conditional paired efficiency on 19 common strict completions; not a population causal effect or success-rate superiority claim",
            "r3": "fixed paired MuJoCo post-capture stress ablation; not hardware, passivity, or guaranteed recovery",
            "cross_size": "actual contracted MuJoCo adaptation scans; not hardware results",
            "cross_size_result": "held-out MuJoCo engagement comparison; adapted heads do not dominate zero-shot or calibration controls",
            "hardware": "deterministic sampling plan only; encoder/shared trunk frozen; state and direction heads only; no measured result",
            "flow": "camera initializes only a coarse visual prior; subsequent feedback is wrench plus robot pose; independent center/contact scorer has no controller return edge",
        },
        "inputs": {
            "shortcut_trial": {"path": str(shortcut_trial.relative_to(ROOT)), "sha256": _sha256(shortcut_trial)},
            "paired_shortcut_audit": {"path": str(paired_audit_path.relative_to(ROOT)), "sha256": _sha256(paired_audit_path)},
            "r3_report": {"path": str(r3_report.relative_to(ROOT)), "sha256": _sha256(r3_report)},
            "dataset_contracts": [
                {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
                for path in contract_paths
            ],
            "cross_size_test_report": {
                "path": str(cross_size_test_report.relative_to(ROOT)),
                "sha256": _sha256(cross_size_test_report),
            },
            "m8_spec": {"path": str(m8_spec.relative_to(ROOT)), "sha256": _sha256(m8_spec)},
        },
        "hardware_plan": hardware_plan,
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in generated
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": [str(path) for path in generated], "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
