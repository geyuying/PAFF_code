import torch
import torch.nn as nn
import torch.nn.functional as F

from hulc.models.perceptual_encoders.clip import build_model, load_clip


class VisionClip(nn.Module):
    def __init__(
        self, device: torch.device, clip_out: int, visual_features: int, freeze_backbone: bool = True, model_name: str = "ViT-B/32"
    ):
        super(VisionClip, self).__init__()
        # Load CLIP model
        self.device = device
        print(f"loading vision CLIP model with backbone: {model_name}")
        model_path = 'runs/2022-07-30/21-13-03/epoch899.ckpt'
        model_clip = torch.load(model_path, map_location="cpu")
        state_dict = model_clip['state_dict']
        state_dict_new = {}
        for param in state_dict:
            if 'clip_vision_encoder.clip_rn50.' in param:
                param_new = param.replace('clip_vision_encoder.clip_rn50.', '')
                state_dict_new[param_new] = state_dict[param]
        self.clip_model = build_model(state_dict_new).to(self.device)
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        if "RN50" in model_name:
            self.fc1 = nn.Linear(2048*clip_out, 512)
            self.fc2 = nn.Linear(512, visual_features)
        elif "ViT-B/32" in model_name:
            self.fc1 = nn.Linear(768*clip_out, 256)
            self.fc2 = nn.Linear(256, visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.clip_model.eval()
        with torch.no_grad():
            x = self.clip_model.encode_image(x)  # type:ignore

        x = x.reshape(x.shape[0], -1)
        output = F.relu(self.fc1(x.float()))  # batch, 512
        output = self.fc2(output)  # batch, 64
        return output
