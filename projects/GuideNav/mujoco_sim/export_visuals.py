#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import GUIDENAV_ROOT


WORKSPACE_ROOT = GUIDENAV_ROOT.parents[1]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "local" / "guidenav_mujoco" / "latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export teaching images and selected keyframes to a local review folder."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--topomap-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-raw-thumbs", type=int, default=80)
    parser.add_argument("--thumb-width", type=int, default=180)
    parser.add_argument("--columns", type=int, default=5)
    return parser.parse_args()


def sorted_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda p: numeric_stem(p))


def numeric_stem(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError:
        try:
            return float(int(path.stem))
        except ValueError:
            return 0.0


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_images(paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in paths:
        dst = output_dir / path.name
        shutil.copy2(path, dst)
        copied.append(dst)
    return copied


def sampled(paths: list[Path], max_count: int) -> list[Path]:
    if len(paths) <= max_count:
        return paths
    if max_count <= 1:
        return [paths[0]]
    indexes = [round(i * (len(paths) - 1) / (max_count - 1)) for i in range(max_count)]
    return [paths[i] for i in indexes]


def make_contact_sheet(
    images: list[Path],
    output_path: Path,
    title: str,
    thumb_width: int,
    columns: int,
) -> None:
    if not images:
        return

    thumbs = []
    label_height = 28
    padding = 12
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        scale = thumb_width / float(image.width)
        thumb_height = max(1, int(round(image.height * scale)))
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        thumbs.append((image_path, image))

    thumb_height = max(image.height for _, image in thumbs)
    rows = (len(thumbs) + columns - 1) // columns
    title_height = 42
    sheet_width = columns * (thumb_width + padding) + padding
    sheet_height = title_height + rows * (thumb_height + label_height + padding) + padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((padding, 14), title, fill=(20, 24, 32), font=font)

    for idx, (image_path, image) in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        x = padding + col * (thumb_width + padding)
        y = title_height + row * (thumb_height + label_height + padding)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + thumb_width - 1, y + image.height - 1), outline=(120, 130, 145))
        draw.text((x, y + thumb_height + 6), image_path.name, fill=(35, 40, 50), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_index(
    output_dir: Path,
    raw_images: list[Path],
    keyframes: list[Path],
    raw_sheet: Path,
    keyframe_sheet: Path,
    raw_dir: Path,
    topomap_dir: Path,
) -> None:
    rows = []
    rows.append("<!doctype html><html><head><meta charset='utf-8'>")
    rows.append("<title>GuideNav MuJoCo Review</title>")
    rows.append(
        "<style>"
        "body{font-family:sans-serif;margin:24px;background:#f6f7f9;color:#1f2430}"
        "h1,h2{margin-bottom:8px}.meta{color:#5a6475}"
        ".sheet{max-width:100%;border:1px solid #ccd2dc;background:white}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}"
        ".card{background:white;border:1px solid #d8dde6;padding:8px}"
        ".card img{width:100%;height:auto;display:block}.cap{font-size:12px;color:#566070;margin-top:6px}"
        "code{background:#e8ebf0;padding:2px 4px;border-radius:3px}"
        "</style></head><body>"
    )
    rows.append("<h1>GuideNav MuJoCo Review</h1>")
    rows.append(f"<p class='meta'>Raw run: <code>{html.escape(str(raw_dir))}</code></p>")
    rows.append(f"<p class='meta'>Topomap: <code>{html.escape(str(topomap_dir))}</code></p>")
    rows.append(f"<p>撮影画像: <b>{len(raw_images)}</b> 枚 / キーフレーム: <b>{len(keyframes)}</b> 枚</p>")
    rows.append("<h2>撮影画像の一覧</h2>")
    rows.append(f"<p><img class='sheet' src='{raw_sheet.name}'></p>")
    rows.append("<h2>選ばれたキーフレーム</h2>")
    rows.append(f"<p><img class='sheet' src='{keyframe_sheet.name}'></p>")
    rows.append("<div class='grid'>")
    for image in keyframes:
        rel = image.relative_to(output_dir).as_posix()
        rows.append(
            "<div class='card'>"
            f"<img src='{html.escape(rel)}'>"
            f"<div class='cap'>{html.escape(image.name)}</div>"
            "</div>"
        )
    rows.append("</div></body></html>")
    (output_dir / "index.html").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_color_dir = args.raw_dir / "color"
    keyframe_dir = args.topomap_dir / "topo"
    if not raw_color_dir.exists():
        raise FileNotFoundError(f"Raw color directory not found: {raw_color_dir}")
    if not keyframe_dir.exists():
        raise FileNotFoundError(f"Topomap keyframe directory not found: {keyframe_dir}")

    raw_paths = sorted_pngs(raw_color_dir)
    keyframe_paths = sorted_pngs(keyframe_dir)
    if not raw_paths:
        raise FileNotFoundError(f"No recorded images found in: {raw_color_dir}")
    if not keyframe_paths:
        raise FileNotFoundError(f"No keyframes found in: {keyframe_dir}")

    reset_dir(args.output_dir)
    copied_raw = copy_images(raw_paths, args.output_dir / "recorded_images")
    copied_keyframes = copy_images(keyframe_paths, args.output_dir / "keyframes")

    raw_sheet = args.output_dir / "recorded_images_contact_sheet.jpg"
    keyframe_sheet = args.output_dir / "keyframes_contact_sheet.jpg"
    make_contact_sheet(
        sampled(copied_raw, args.max_raw_thumbs),
        raw_sheet,
        f"Recorded images sampled from {len(copied_raw)} frames",
        args.thumb_width,
        args.columns,
    )
    make_contact_sheet(
        copied_keyframes,
        keyframe_sheet,
        f"Selected keyframes ({len(copied_keyframes)} frames)",
        args.thumb_width,
        args.columns,
    )
    write_index(args.output_dir, copied_raw, copied_keyframes, raw_sheet, keyframe_sheet, args.raw_dir, args.topomap_dir)

    print(f"Review folder ready: {args.output_dir}")
    print(f"Open this file in a browser: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
