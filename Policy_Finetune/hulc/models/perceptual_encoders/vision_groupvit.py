import torch
import torch.nn as nn
import torch.nn.functional as F
from hulc.models.perceptual_encoders.groupvit.group_vit import GroupViT
from timm.models.layers import DropPath
from einops import rearrange


class VisionGroupVit(nn.Module):
    def __init__(
        self, device: torch.device, vit_out: int, visual_features: int, freeze_backbone: bool = True
    ):
        super(VisionGroupVit, self).__init__()
        # Load CLIP model
        print("loading vision GroupVit model")
        self.img_encoder = GroupViT()
        if freeze_backbone:
            for param in self.img_encoder.parameters():
                param.requires_grad = False
        checkpoint = torch.load('hulc/ckpts/group_vit_gcc_redcap_30e-3dd09a76.pth', map_location="cpu")['model']
        checkpoint_new = {}
        for param in checkpoint:
            if 'img_encoder' in param:
                param_new = param
                checkpoint_new[param_new] = checkpoint[param]
        self.load_state_dict(checkpoint_new, True)

        self.fc1 = nn.Linear(vit_out, 512)
        self.fc2 = nn.Linear(512, visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            fea_all, group_tokens = self.img_encoder(x)
        x = torch.mean(group_tokens, 1)
        output = F.relu(self.fc1(x))  # batch, 512
        output = self.fc2(output)  # batch, 64
        return output