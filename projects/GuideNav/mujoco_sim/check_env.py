#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

from common import GO2_MUJOCO_ROOT, GUIDENAV_ROOT, MODEL_DIR, POLICY_DIR


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    checks = [
        ("MuJoCo Python", has_module("mujoco")),
        ("NumPy", has_module("numpy")),
        ("ONNX Runtime", has_module("onnxruntime")),
        ("GLFW", has_module("glfw")),
        ("PyYAML", has_module("yaml")),
        ("OpenCV cv2", has_module("cv2")),
        ("pandas", has_module("pandas")),
        ("h5py", has_module("h5py")),
        ("PyTorch", has_module("torch")),
        ("torchvision", has_module("torchvision")),
        ("kornia", has_module("kornia")),
        ("timm", has_module("timm")),
        ("huggingface_hub", has_module("huggingface_hub")),
        ("romatch", has_module("romatch")),
        ("ROS2 rclpy", has_module("rclpy")),
        ("cv_bridge", has_module("cv_bridge")),
    ]
    paths = [
        ("go2-mujoco project", GO2_MUJOCO_ROOT),
        ("Go2 MuJoCo model", MODEL_DIR / "go2.xml"),
        ("Go2 assets", MODEL_DIR / "assets"),
        ("Go2 ONNX policy", POLICY_DIR / "policy.onnx"),
        ("Go2 policy config", POLICY_DIR / "params" / "deploy.yaml"),
        ("Official GuideNav CosPlace weight", GUIDENAV_ROOT / "model_weights" / "efficientnet_85x85.pth"),
        ("CosPlace torch.hub cache", GUIDENAV_ROOT / "model_weights" / "torch_hub_checkpoints"),
        ("LoFTR external method", GUIDENAV_ROOT / "guidenav" / "match_to_control" / "methods" / "LoFTR"),
        ("LiftFeat external method", GUIDENAV_ROOT / "guidenav" / "match_to_control" / "methods" / "LiftFeat"),
        ("MASt3R external method", GUIDENAV_ROOT / "guidenav" / "match_to_control" / "methods" / "mast3r"),
        ("Reloc3r external method", GUIDENAV_ROOT / "guidenav" / "match_to_control" / "methods" / "reloc3r"),
        ("Reloc3r Hugging Face cache", GUIDENAV_ROOT / "model_weights" / "huggingface"),
        ("MuJoCo topomap", GUIDENAV_ROOT / "data" / "mujoco_teach" / "topomap" / "topo"),
        ("CosPlace VPR DB", GUIDENAV_ROOT / "data" / "mujoco_teach" / "topomap" / "topo" / "global-feats-cosplace_hub.h5"),
    ]

    print("Python modules:")
    for label, ok in checks:
        mark = "OK" if ok else "MISSING"
        print(f"  {mark:7s} {label}")

    print("\nFiles:")
    for label, path in paths:
        mark = "OK" if Path(path).exists() else "MISSING"
        print(f"  {mark:7s} {label}: {path}")

    print("\nNotes:")
    print("  - Recording teaching data needs the MuJoCo/Go2 items.")
    print("  - replay_cosplace.py needs OpenCV, h5py, PyTorch/torchvision, CosPlace, Reloc3r, and the VPR DB.")
    print("  - Running official guidenav/navigate.py also needs ROS2, cv_bridge, and feature-matching model weights.")


if __name__ == "__main__":
    main()
