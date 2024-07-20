import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hulc.models.perceptual_encoders.clip_adapter_cls import build_model, load_clip


class VisionClip(nn.Module):
    def __init__(
        self, freeze_backbone: bool = False, model_name: str = "ViT-B/32"
    ):
        super(VisionClip, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load CLIP model
        print(f"loading vision CLIP model with backbone: {model_name}")
        self._load_clip(model_name)
        # for name, param in self.clip_rn50.named_parameters():
        #     print(name, param.shape)
        # if freeze_backbone:
        #     for param in self.clip_rn50.parameters():
        #         param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:

        model_path = 'runs/2022-10-08/13-04-12/saved_models/epoch=499.ckpt'
        model_clip = torch.load(model_path, map_location="cpu")
        state_dict = model_clip['state_dict']
        state_dict_new = {}
        for param in state_dict:
          if 'clip_vision_encoder.clip_rn50.' in param:
              param_new = param.replace('clip_vision_encoder.clip_rn50.', '')
              state_dict_new[param_new] = state_dict[param]
        self.clip_rn50 = build_model(state_dict_new).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.clip_rn50.eval()
        image_features = self.clip_rn50.encode_image(x)  # type:ignore
        return image_features
