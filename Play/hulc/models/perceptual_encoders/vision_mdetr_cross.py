import torch
import torch.nn as nn
import torch.nn.functional as F
from hulc.models.perceptual_encoders.backbone_full import Backbone
from hulc.models.perceptual_encoders.misc import NestedTensor
from hulc.models.perceptual_encoders.position_encoding import build_position_encoding
import math


class VisionMdetr(nn.Module):
    def __init__(
        self, device: torch.device, mdetr_out: int, visual_features: int, freeze_backbone: bool = True, model_name: str = "resnet101"
    ):
        super(VisionMdetr, self).__init__()
        # Load CLIP model
        print(f"loading vision Mdetr model with backbone: {model_name}")
        self.backbone = Backbone('resnet101', True, True, False)
        self.position_embedding = build_position_encoding()
        self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.input_proj.parameters():
                param.requires_grad = False
        mdter_checkpoint = torch.load('hulc/ckpts/mdetr_pretrained_resnet101_checkpoint.pth', map_location="cpu")['model']
        checkpoint_new = {}
        for param in mdter_checkpoint:
            if 'backbone.0.body' in param:
                param_new = param.replace('backbone.0.body', 'backbone.body')
                checkpoint_new[param_new] = mdter_checkpoint[param]
            elif 'input_proj' in param:
                param_new = param.replace('transformer.','')
                checkpoint_new[param_new] = mdter_checkpoint[param]
        self.load_state_dict(checkpoint_new, True)
        size = int(math.sqrt(mdetr_out))
        self.pool = nn.AvgPool2d(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        img = NestedTensor.from_tensor_list(x)
        with torch.no_grad():
            xs = self.backbone(img)
            out = []
            pos = []
            for name, x in xs.items():
                out.append(x)
                pos.append(self.position_embedding(x).to(x.tensors.dtype))
            x, mask = out[-1].decompose()
            pos_embed = pos[-1]
            x = self.input_proj(x)
            src = x.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return x, src, pos_embed, mask
