"""
CosPlace model loaded from the official gmberton/CosPlace torch.hub entrypoint.

The GuideNav repository references an EfficientNet checkpoint named
efficientnet_85x85.pth, but that weight file is not included in the repository.
This adapter keeps the GuideNav/HLoc-style model interface while using the
public CosPlace trained models distributed by the original CosPlace authors.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from .base_model import BaseModel


class CosPlaceHub(BaseModel):
    default_conf = {
        "source": "gmberton/CosPlace",
        "backbone": "ResNet50",
        "fc_output_dim": 512,
        "torch_hub_dir": "model_weights/torch_hub_checkpoints",
        "force_reload": False,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        hub_dir = Path(conf["torch_hub_dir"]).expanduser()
        if not hub_dir.is_absolute():
            hub_dir = Path.cwd() / hub_dir
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))

        self.net = torch.hub.load(
            conf["source"],
            "get_trained_model",
            backbone=conf["backbone"],
            fc_output_dim=int(conf["fc_output_dim"]),
            force_reload=bool(conf.get("force_reload", False)),
            trust_repo=True,
        )
        self.net = self.net.eval()

    def _forward(self, data):
        image = data["image"]
        desc = self.net(image)
        desc = F.normalize(desc, p=2, dim=1)
        return {"global_descriptor": desc}
