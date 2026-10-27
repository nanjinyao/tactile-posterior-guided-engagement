from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "outputs" / "current"
MANIFEST_PATH = CURRENT / "reports" / "m8_sampling_video_manifest.json"
KEYFRAME_DIR = CURRENT / "media" / "keyframes" / "stages"
CONTACT_SHEET_PATH = CURRENT / "media" / "figures" / "m8_sampling_video_contact_sheet.png"
INDEX_PATH = KEYFRAME_DIR / "index.json"

KEYFRAME_SPECS = {
    "V1": (0.55, "v1_outer_partial_conical_keyframe.png"),
    "V2": (0.55, "v2_double_overhang_conical_keyframe.png"),
    "V3": (0.55, "v3_supported_search_conical_keyframe.png"),
    "V4": (0.99, "v4_rim_chamfer_capture_keyframe.png"),
    "V5": (0.05, "v5_inner_wall_gate_keyframe.png"),
    "V6a": (0.62, "v6a_outer_partial_spiral_keyframe.png"),
    "V6b": (0.62, "v6b_double_overhang_spiral_keyframe.png"),
    "V6c": (0.62, "v6c_supported_search_spiral_keyframe.png"),
}


def read_video(path: Path) -> list:
    reader = imageio.get_reader(path)
    try:
        return [frame for frame in reader]
    finally:
        reader.close()


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = {record["key"]: record for record in manifest["videos"]}
    missing = set(KEYFRAME_SPECS) - set(records)
    if missing:
        raise RuntimeError(f"Missing video records: {sorted(missing)}")

    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    for path in KEYFRAME_DIR.glob("*.png"):
        path.unlink()

    selected: list[tuple[str, Image.Image]] = []
    index: list[dict] = []
    for key, (fraction, filename) in KEYFRAME_SPECS.items():
        record = records[key]
        video_path = ROOT / record["path"]
        frames = read_video(video_path)
        if not frames:
            raise RuntimeError(f"Video has no frames: {video_path}")
        frame_index = min(len(frames) - 1, int(round((len(frames) - 1) * fraction)))
        image = Image.fromarray(frames[frame_index]).convert("RGB")
        image.save(KEYFRAME_DIR / filename)
        selected.append((key, image))
        index.append(
            {
                "key": key,
                "video": record["path"],
                "keyframe": str((KEYFRAME_DIR / filename).relative_to(ROOT)),
                "fraction": fraction,
                "frame_index": frame_index,
                "frame_count": len(frames),
                "result": record["result"],
            }
        )

    canvas = Image.new("RGB", (1920, 720), (8, 10, 13))
    font = load_font()
    for i, (key, image) in enumerate(selected):
        thumb = image.resize((480, 360), Image.Resampling.LANCZOS)
        x = (i % 4) * 480
        y = (i // 4) * 360
        canvas.paste(thumb, (x, y))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rounded_rectangle(
            (x + 8, y + 8, x + 64, y + 35),
            radius=4,
            fill=(0, 0, 0, 185),
            outline=(255, 255, 255, 75),
        )
        draw.text((x + 17, y + 11), key, font=font, fill=(255, 235, 170, 255))
    canvas.save(CONTACT_SHEET_PATH)
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
