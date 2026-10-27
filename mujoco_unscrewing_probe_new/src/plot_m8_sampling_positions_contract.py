"""Plot the provenance-bounded M8 conical-scan sampling inventory.

This script is intentionally a manifest-only renderer.  It does not launch
MuJoCo, load a learned model, or infer a capture basin.  Each dataset layer is
deduplicated by commanded initial bit-center XY and is then drawn at the exact
recorded coordinate.  Downstream grouped splitting and C6 tensor augmentation
do not create physical rollouts or new XY positions and are therefore not
drawn.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "outputs" / "current"
OUTPUT_DIR = CURRENT / "media" / "figures" / "paper"
OUTPUTS = {
    "en": OUTPUT_DIR / "m8_sampling_positions_contract_en.png",
    "cn": OUTPUT_DIR / "m8_sampling_positions_contract_cn.png",
}
MANIFEST_PATH = OUTPUT_DIR / "m8_sampling_positions_contract_manifest.json"

FULL_SUPPORT_RADIUS_MM = 3.036
VISUAL_95_RADIUS_MM = 4.895
SCREW_HEAD_RADIUS_MM = 6.500
XY_QUANTIZATION_NM = 1.0


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    relative_manifest: str
    short_label: str
    role_en: str
    role_cn: str
    color: str
    marker: str
    size: float
    zorder: int


DATASETS = (
    DatasetSpec(
        "p0a_m8_symmetry",
        "outputs/current/datasets/p0a_m8_symmetry/manifest.json",
        "P0a",
        "symmetry diagnostic",
        "对称性诊断",
        "#0f766e",
        "o",
        16.0,
        3,
    ),
    DatasetSpec(
        "p0b_m8_observability",
        "outputs/current/datasets/p0b_m8_observability/manifest.json",
        "P0b",
        "foundation pool",
        "基础数据池",
        "#0284c7",
        "o",
        17.0,
        2,
    ),
    DatasetSpec(
        "p0c_m8_collision_followup",
        "outputs/current/datasets/p0c_m8_collision_followup/manifest.json",
        "P0c",
        "hard-case diagnostic",
        "困难接触诊断",
        "#ea580c",
        "D",
        25.0,
        6,
    ),
    DatasetSpec(
        "p0d_m8_sector_anchors",
        "outputs/current/datasets/p0d_m8_sector_anchors/manifest.json",
        "P0d",
        "physical sector anchors",
        "物理扇区锚点",
        "#16a34a",
        "s",
        19.0,
        4,
    ),
    DatasetSpec(
        "p1v2_m8_independent_direction_audit",
        "outputs/current/evaluations/p1v2_m8_independent_direction_audit/manifest.json",
        "P1v2",
        "viewed design audit",
        "已查看设计审计",
        "#b45309",
        "P",
        29.0,
        7,
    ),
    DatasetSpec(
        "p1v3_m8_targeted_direction_training",
        "outputs/current/datasets/p1v3_m8_targeted_direction_training/manifest.json",
        "P1v3",
        "targeted train/calibration",
        "定向训练/校准",
        "#db2777",
        "^",
        27.0,
        8,
    ),
    DatasetSpec(
        "p1v4_m8_untouched_direction_audit",
        "outputs/current/evaluations/p1v4_m8_untouched_direction_audit/manifest.json",
        "P1v4",
        "viewed interface audit",
        "已查看接口审计",
        "#7c3aed",
        "X",
        29.0,
        9,
    ),
    DatasetSpec(
        "p1v5_m8_final_untouched_direction_audit",
        "outputs/current/evaluations/p1v5_m8_final_untouched_direction_audit/manifest.json",
        "P1v5",
        "final untouched test",
        "最终未触碰测试",
        "#ca8a04",
        "*",
        42.0,
        10,
    ),
)


TEXT = {
    "en": {
        "title": "M8 conical-fingerprint sampling positions",
        "subtitle": "Exact commanded initial bit-center XY from eight frozen manifests",
        "x": "Initial bit-center x relative to socket center [mm]",
        "y": "Initial bit-center y relative to socket center [mm]",
        "center": "true socket center",
        "support": "full support: r = 3.036 mm",
        "visual": "declared visual 95%: r = 4.895 mm",
        "head": "M8 screw-head CAD radius: r = 6.500 mm",
        "inventory": "Manifest inventory",
        "rollout_count": "1,674 MuJoCo physics scans",
        "not_hardware": "dynamic rollouts, not hardware trials",
        "layer_xy": "696 layer-wise unique XY entries",
        "layer_note": "nuisance repeats removed inside each dataset",
        "union_xy": "651 unique XY positions in the union",
        "union_note": "45 cross-dataset overlaps remain at exact coordinates",
        "augmentation": "0 synthetic XY points plotted",
        "augmentation_note": "group split / C6 augmentation is downstream only",
        "table_dataset": "set / role",
        "table_scans": "scans",
        "table_xy": "XY",
        "count_contract": "Count contract",
        "flow_1": "manifest records",
        "flow_2": "deduplicate nuisance repeats",
        "flow_3": "exact XY dots",
        "group_note": "Split groups and augmented tensors are not extra rollouts.",
        "claim": (
            "Inventory only: this figure does not establish deployment probability, radial "
            "identifiability, hardware coverage, capture, or task success. P1v5 remains "
            "untouched and is never used for fitting, calibration, thresholds, or interface changes."
        ),
    },
    "cn": {
        "title": "M8 锥形指纹采样位置",
        "subtitle": "来自 8 个冻结清单的指头中心初始指令 XY（精确坐标）",
        "x": "相对套筒中心的指头初始 x [mm]",
        "y": "相对套筒中心的指头初始 y [mm]",
        "center": "真实套筒中心",
        "support": "完整支撑：r = 3.036 mm",
        "visual": "声明的视觉 95%：r = 4.895 mm",
        "head": "M8 螺钉头 CAD 半径：r = 6.500 mm",
        "inventory": "清单统计",
        "rollout_count": "1,674 条 MuJoCo 物理仿真扫描",
        "not_hardware": "动力学 rollout，不是真机试验",
        "layer_xy": "696 个数据层内唯一 XY 条目",
        "layer_note": "每个数据集内部去除 nuisance 重复",
        "union_xy": "跨数据集合并后 651 个唯一 XY",
        "union_note": "45 个跨数据集重合点保留精确重合",
        "augmentation": "图中合成 XY 点为 0",
        "augmentation_note": "分组切分 / C6 增强只发生在下游",
        "table_dataset": "数据集 / 角色",
        "table_scans": "扫描",
        "table_xy": "XY",
        "count_contract": "计数口径",
        "flow_1": "清单记录",
        "flow_2": "去除 nuisance 重复",
        "flow_3": "精确 XY 点",
        "group_note": "切分组和增强张量都不是新增物理 rollout。",
        "claim": (
            "仅为采样清单：本图不证明部署概率、径向可辨识性、真机覆盖、倒角捕获或任务成功。"
            "P1v5 始终保持未触碰，禁止用于拟合、校准、阈值选择或接口修改。"
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xy_key(record: dict[str, Any]) -> tuple[int, int]:
    scale = 1.0e9 / XY_QUANTIZATION_NM
    return round(float(record["x_m"]) * scale), round(float(record["y_m"]) * scale)


def _load_layers() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    union_xy: set[tuple[int, int]] = set()
    total_rollouts = 0
    layer_unique_total = 0

    for spec in DATASETS:
        manifest_path = ROOT / spec.relative_manifest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [
            record
            for record in manifest["records"]
            if str(record.get("status", "ok")).lower() not in {"failed", "error"}
        ]
        by_xy: dict[tuple[int, int], dict[str, Any]] = {}
        for record in records:
            by_xy.setdefault(_xy_key(record), record)
        positions = [
            (1000.0 * float(record["x_m"]), 1000.0 * float(record["y_m"]))
            for record in by_xy.values()
        ]
        base_pose_groups = {str(record["base_pose_id"]) for record in records}
        symmetry_orbits = {
            str(record["symmetry_orbit_id"])
            for record in records
            if record.get("symmetry_orbit_id") is not None
        }
        radial_mm = [1000.0 * float(record["radius_m"]) for record in by_xy.values()]
        states = sorted({str(record["search_state"]) for record in records})

        total_rollouts += len(records)
        layer_unique_total += len(by_xy)
        union_xy.update(by_xy)
        layers.append(
            {
                "spec": spec,
                "manifest_path": manifest_path,
                "manifest_sha256": _sha256(manifest_path),
                "rollouts": len(records),
                "unique_xy": len(by_xy),
                "positions_mm": positions,
                "base_pose_groups": len(base_pose_groups),
                "symmetry_orbits": len(symmetry_orbits),
                "states": states,
                "radius_min_mm": min(radial_mm),
                "radius_max_mm": max(radial_mm),
            }
        )

    totals = {
        "physics_rollouts": total_rollouts,
        "layer_unique_xy_entries": layer_unique_total,
        "union_unique_xy_positions": len(union_xy),
        "cross_dataset_overlap_entries": layer_unique_total - len(union_xy),
    }
    expected = {
        "physics_rollouts": 1674,
        "layer_unique_xy_entries": 696,
        "union_unique_xy_positions": 651,
        "cross_dataset_overlap_entries": 45,
    }
    if totals != expected:
        raise RuntimeError(f"sampling inventory drifted: expected {expected}, got {totals}")
    return layers, totals


def _add_count_box(
    ax: plt.Axes,
    y: float,
    headline: str,
    detail: str,
    *,
    accent: str,
) -> None:
    box = FancyBboxPatch(
        (0.02, y - 0.055),
        0.96,
        0.105,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        linewidth=0.8,
    )
    ax.add_patch(box)
    ax.add_patch(
        FancyBboxPatch(
            (0.02, y - 0.055),
            0.015,
            0.105,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            transform=ax.transAxes,
            facecolor=accent,
            edgecolor=accent,
            linewidth=0,
        )
    )
    ax.text(0.055, y + 0.010, headline, transform=ax.transAxes, fontsize=9.2, weight="bold", va="center")
    ax.text(0.055, y - 0.025, detail, transform=ax.transAxes, fontsize=7.0, color="#475569", va="center")


def _plot_language(language: str, layers: list[dict[str, Any]], totals: dict[str, int]) -> None:
    text = TEXT[language]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # The installed TTC is registered by Matplotlib under its first
            # face (JP), while the glyph set covers the Chinese deck text.
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(12.6, 7.0), dpi=220, facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.58, 0.92),
        left=0.055,
        right=0.985,
        top=0.875,
        bottom=0.145,
        wspace=0.12,
    )
    ax = fig.add_subplot(grid[0, 0])
    info = fig.add_subplot(grid[0, 1])
    info.axis("off")

    fig.text(0.055, 0.945, text["title"], fontsize=18.0, weight="bold", color="#0f172a", va="top")
    fig.text(0.055, 0.905, text["subtitle"], fontsize=9.4, color="#475569", va="top")

    # CAD and declared operating radii are context boundaries, not capture sets.
    circle_specs = (
        (SCREW_HEAD_RADIUS_MM, "#64748b", ":", text["head"]),
        (VISUAL_95_RADIUS_MM, "#0891b2", "--", text["visual"]),
        (FULL_SUPPORT_RADIUS_MM, "#16a34a", "-", text["support"]),
    )
    for radius, color, linestyle, _ in circle_specs:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                fill=False,
                edgecolor=color,
                linewidth=1.25,
                linestyle=linestyle,
                zorder=1,
            )
        )

    for layer in layers:
        spec: DatasetSpec = layer["spec"]
        xs = [point[0] for point in layer["positions_mm"]]
        ys = [point[1] for point in layer["positions_mm"]]
        ax.scatter(
            xs,
            ys,
            s=spec.size,
            c=spec.color,
            marker=spec.marker,
            alpha=0.72,
            linewidths=0.35,
            edgecolors="white" if spec.marker not in {"X"} else None,
            zorder=spec.zorder,
        )

    ax.scatter([0.0], [0.0], marker="+", s=90, linewidths=1.8, color="#111827", zorder=20)
    ax.annotate(
        text["center"],
        xy=(0.0, 0.0),
        xytext=(0.35, -0.55),
        fontsize=7.2,
        color="#334155",
        arrowprops={"arrowstyle": "-", "color": "#64748b", "linewidth": 0.7},
    )

    ax.set_xlim(-8.35, 8.35)
    ax.set_ylim(-8.35, 8.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(text["x"], fontsize=9.0)
    ax.set_ylabel(text["y"], fontsize=9.0)
    ax.set_xticks(range(-8, 9, 2))
    ax.set_yticks(range(-8, 9, 2))
    ax.tick_params(labelsize=7.7)
    ax.grid(True, color="#e2e8f0", linewidth=0.65, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(0.8)

    radius_handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5, label=label)
        for _, color, linestyle, label in circle_specs[::-1]
    ]
    radius_legend = ax.legend(
        handles=radius_handles,
        loc="lower left",
        fontsize=7.0,
        frameon=True,
        framealpha=0.93,
        edgecolor="#cbd5e1",
        handlelength=2.7,
    )
    ax.add_artist(radius_legend)

    info.text(
        0.02,
        0.995,
        text["inventory"],
        transform=info.transAxes,
        fontsize=12.0,
        weight="bold",
        color="#0f172a",
        va="top",
    )
    _add_count_box(info, 0.905, text["rollout_count"], text["not_hardware"], accent="#0f766e")
    _add_count_box(info, 0.775, text["layer_xy"], text["layer_note"], accent="#0284c7")
    _add_count_box(info, 0.645, text["union_xy"], text["union_note"], accent="#7c3aed")
    _add_count_box(info, 0.515, text["augmentation"], text["augmentation_note"], accent="#ca8a04")

    info.text(0.02, 0.425, text["table_dataset"], transform=info.transAxes, fontsize=7.2, weight="bold", color="#334155")
    info.text(0.80, 0.425, text["table_scans"], transform=info.transAxes, fontsize=7.2, weight="bold", color="#334155", ha="right")
    info.text(0.97, 0.425, text["table_xy"], transform=info.transAxes, fontsize=7.2, weight="bold", color="#334155", ha="right")
    info.plot([0.02, 0.98], [0.410, 0.410], color="#cbd5e1", linewidth=0.8, transform=info.transAxes)
    y = 0.382
    for layer in layers:
        spec = layer["spec"]
        role = spec.role_en if language == "en" else spec.role_cn
        info.scatter([0.038], [y + 0.004], s=23, c=spec.color, marker=spec.marker, transform=info.transAxes, clip_on=False)
        info.text(0.072, y, f"{spec.short_label}  {role}", transform=info.transAxes, fontsize=6.75, va="center")
        info.text(0.80, y, f"{layer['rollouts']}", transform=info.transAxes, fontsize=6.75, ha="right", va="center")
        info.text(0.97, y, f"{layer['unique_xy']}", transform=info.transAxes, fontsize=6.75, ha="right", va="center")
        y -= 0.038

    info.text(0.02, 0.076, text["count_contract"], transform=info.transAxes, fontsize=7.1, weight="bold", color="#334155", va="bottom")
    info.text(
        0.02,
        0.045,
        f"{text['flow_1']}  →  {text['flow_2']}  →  {text['flow_3']}",
        transform=info.transAxes,
        fontsize=6.6,
        color="#334155",
        va="bottom",
    )
    info.text(0.02, 0.014, text["group_note"], transform=info.transAxes, fontsize=6.5, color="#64748b", va="bottom")

    fig.text(
        0.055,
        0.055,
        text["claim"],
        fontsize=7.0,
        color="#475569",
        va="bottom",
        ha="left",
        wrap=True,
    )
    output = OUTPUTS[language]
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    # Matplotlib writes RGBA PNGs by default.  Some original-resolution image
    # viewers incorrectly composite large RGBA figures with CJK glyph masks as
    # black rectangles.  The paper/deck asset is opaque, so store an explicit
    # RGB PNG and remove that decoder-dependent ambiguity.
    with Image.open(output) as rendered:
        rgb = rendered.convert("RGB")
    rgb.save(output, format="PNG", optimize=True)


def _write_manifest(layers: list[dict[str, Any]], totals: dict[str, int]) -> None:
    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "m8_sampling_positions_contract",
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "point_definition": "commanded initial bit-center XY relative to the true socket center",
        "status_filter": "records whose status is neither failed nor error",
        "deduplication": {
            "within_dataset": "x_m and y_m quantized to 1 nm; nuisance repeats are drawn once",
            "across_datasets": "not removed from drawing; exact overlaps remain coincident",
            "layer_unique_xy_entries": totals["layer_unique_xy_entries"],
            "union_unique_xy_positions": totals["union_unique_xy_positions"],
            "cross_dataset_overlap_entries": totals["cross_dataset_overlap_entries"],
        },
        "counts": totals,
        "augmentation_contract": {
            "synthetic_xy_points_plotted": 0,
            "group_split_is_physical_rollout": False,
            "c6_tensor_augmentation_is_physical_rollout": False,
            "c6_tensor_augmentation_creates_new_xy": False,
            "note": (
                "These eight manifests inventory physically simulated dynamic conical scans. "
                "Grouped splitting and optional C6 signal rotation happen downstream and are "
                "not additional rollouts or spatial samples."
            ),
        },
        "reference_radii_mm": {
            "full_support_center_radius": FULL_SUPPORT_RADIUS_MM,
            "declared_visual_95_radius": VISUAL_95_RADIUS_MM,
            "screw_head_cad_radius": SCREW_HEAD_RADIUS_MM,
            "boundary_note": "reference circles are not capture or success sets",
        },
        "datasets": [
            {
                "key": layer["spec"].key,
                "short_label": layer["spec"].short_label,
                "role_en": layer["spec"].role_en,
                "role_cn": layer["spec"].role_cn,
                "manifest": str(layer["manifest_path"].relative_to(ROOT)),
                "manifest_sha256": layer["manifest_sha256"],
                "physics_rollouts": layer["rollouts"],
                "unique_xy_positions_within_dataset": layer["unique_xy"],
                "base_pose_groups": layer["base_pose_groups"],
                "symmetry_orbits_when_declared": layer["symmetry_orbits"],
                "search_states": layer["states"],
                "radius_min_mm": layer["radius_min_mm"],
                "radius_max_mm": layer["radius_max_mm"],
            }
            for layer in layers
        ],
        "outputs": {
            language: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "language_contract": "English only" if language == "en" else "Chinese with standard technical abbreviations",
            }
            for language, path in OUTPUTS.items()
        },
        "claim_boundary": (
            "This is an exact spatial inventory of eight frozen simulation manifests. It is not "
            "evidence of deployment probability, radial identifiability, real-robot coverage, "
            "chamfer capture, or task success. P1v5 remains forbidden for fitting, calibration, "
            "threshold selection, or interface changes."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layers, totals = _load_layers()
    for language in ("en", "cn"):
        _plot_language(language, layers, totals)
    _write_manifest(layers, totals)
    for path in (*OUTPUTS.values(), MANIFEST_PATH):
        print(f"wrote {path}")
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()
