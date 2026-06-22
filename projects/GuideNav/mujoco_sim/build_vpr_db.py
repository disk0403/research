#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from common import GUIDENAV_ROOT
from guidenav.place_recognition import extract_database
from guidenav.utils import get_image_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the official GuideNav place-recognition HDF5 database from "
            "a MuJoCo topomap image directory."
        )
    )
    parser.add_argument("--topomap-dir", type=Path, default=GUIDENAV_ROOT / "data" / "mujoco_teach" / "topomap")
    parser.add_argument("--topomap-image-subdir", default="topo")
    parser.add_argument("--model-config-path", type=Path, default=GUIDENAV_ROOT / "config" / "models.yaml")
    parser.add_argument("--model-weight-dir", type=Path, default=GUIDENAV_ROOT / "model_weights")
    parser.add_argument("--pr-model", default="cosplace_hub")
    parser.add_argument("--img-size", type=int, nargs=2, default=(85, 64), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--half", action="store_true", help="store float descriptors as float16")
    return parser.parse_args()


def load_model_conf(args: argparse.Namespace) -> dict:
    with args.model_config_path.open("r", encoding="utf-8") as f:
        confs = yaml.safe_load(f)
    if args.pr_model not in confs:
        raise KeyError(f"Unknown place-recognition model '{args.pr_model}' in {args.model_config_path}")

    conf = copy.deepcopy(confs[args.pr_model])
    model_conf = conf.setdefault("model", {})

    checkpoint_path = model_conf.get("checkpoint_path")
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = args.model_weight_dir / checkpoint_path
        model_conf["checkpoint_path"] = checkpoint_path

    hub_dir = model_conf.get("torch_hub_dir")
    if hub_dir is not None:
        hub_dir = Path(hub_dir).expanduser()
        if not hub_dir.is_absolute():
            hub_dir = GUIDENAV_ROOT / hub_dir
        model_conf["torch_hub_dir"] = str(hub_dir)

    return conf


def main() -> None:
    args = parse_args()
    topomap_img_dir = args.topomap_dir / args.topomap_image_subdir
    if not topomap_img_dir.exists():
        raise FileNotFoundError(f"Topomap image directory not found: {topomap_img_dir}")

    image_paths = sorted(topomap_img_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not image_paths:
        raise FileNotFoundError(f"No topomap PNG images found in: {topomap_img_dir}")

    conf = load_model_conf(args)
    transform = get_image_transform(list(args.img_size))
    feature_path = topomap_img_dir / f"{conf['output']}.h5"

    print(f"Building VPR database with {args.pr_model}")
    print(f"Topomap images : {topomap_img_dir}")
    print(f"Feature output : {feature_path}")
    print("The first run may download CosPlace weights into model_weights/torch_hub_checkpoints.")

    output_path = extract_database.main(
        conf,
        topomap_img_dir,
        transform,
        export_dir=topomap_img_dir,
        as_half=bool(args.half),
        feature_path=feature_path,
        overwrite=bool(args.overwrite),
    )

    print(f"VPR database ready: {output_path}")


if __name__ == "__main__":
    main()
