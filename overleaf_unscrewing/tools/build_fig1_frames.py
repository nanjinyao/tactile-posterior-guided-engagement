"""Rebuild Fig. 1 annotations around the documented hardware photograph.

The underlying robot/fixture photograph is unchanged.  Only the white side
panels are redrawn so that all displayed frames use the same right-handed task
convention: x left, z up, and y out of the page.  Insertion depth is therefore
measured along -z_p.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
SOURCE = FIG_DIR / "fig1_hardware_coordinate_frames_source.png"
OUTPUT = FIG_DIR / "fig1_hardware_coordinate_frames.png"


def arrow(ax, start, end, color, width=2.5, mutation=16, linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=width,
            linestyle=linestyle,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )


def out_of_page(ax, xy, color, label, label_xy):
    ax.add_patch(Circle(xy, radius=8.2, facecolor="white", edgecolor=color, linewidth=1.8, zorder=6))
    ax.add_patch(Circle(xy, radius=2.2, facecolor=color, edgecolor="none", zorder=7))
    ax.text(*label_xy, label, color=color, fontsize=17, ha="left", va="center", zorder=7)


def frame(ax, origin, name, color, connector, depth=False):
    x, y = origin
    arrow(ax, (x, y), (x - 118, y), color)
    arrow(ax, (x, y), (x, y - 112), color)
    ax.scatter([x], [y], s=28, color=color, zorder=5)
    ax.text(x - 132, y + 3, rf"$x_{{{name}}}$", color=color, fontsize=18, ha="right", va="center")
    ax.text(x + 2, y - 126, rf"$z_{{{name}}}$", color=color, fontsize=18, ha="center", va="bottom")
    ax.text(x + 15, y + 17, rf"$O_{{{name}}}$", color="#171717", fontsize=18, ha="left", va="center")
    out_of_page(ax, (x + 33, y - 25), color, rf"$y_{{{name}}}$", (x + 45, y - 25))
    ax.plot([x + 50, connector[0]], [y + 9, connector[1]], color="#171717", lw=1.35)
    ax.scatter([connector[0]], [connector[1]], s=22, facecolor="#171717", edgecolor="white", linewidth=0.6)
    if depth:
        arrow(ax, (x, y + 20), (x, y + 103), color, width=1.45, mutation=12, linestyle="--")
        ax.text(x + 14, y + 88, rf"depth $-z_{{{name}}}$", color=color, fontsize=12, ha="left", va="center")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing preserved source image: {SOURCE}")
    image = Image.open(SOURCE).convert("RGB")
    width, height = image.size
    fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    # Remove only the legacy annotations in the white margins; the central
    # hardware photograph and its documentary labels are preserved verbatim.
    ax.add_patch(plt.Rectangle((0, 0), 335, height, facecolor="white", edgecolor="none", zorder=2))
    ax.add_patch(plt.Rectangle((1107, 0), width - 1107, height, facecolor="white", edgecolor="none", zorder=2))

    red = "#ef2929"
    blue = "#087dcc"
    grey = "#666a70"

    ax.text(52, 126, "Interaction frame", color=red, fontsize=15.0, ha="left", va="center", zorder=5)
    ax.text(52, 153, r"$\{f\}$", color=red, fontsize=16.0, ha="left", va="center", zorder=5)
    frame(ax, (223, 331), "f", red, (531, 417))

    ax.text(44, 556, "Fixture frame", color=blue, fontsize=15.0, ha="left", va="center", zorder=5)
    ax.text(44, 582, r"$\{p\}$", color=blue, fontsize=16.0, ha="left", va="center", zorder=5)
    frame(ax, (212, 637), "p", blue, (566, 651), depth=True)

    ax.text(1138, 520, r"Robot base frame $\{b\}$", color=grey, fontsize=15.0, ha="left", va="center", zorder=5)
    frame(ax, (1381, 708), "b", grey, (1038, 778))

    fig.savefig(OUTPUT, dpi=150, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
