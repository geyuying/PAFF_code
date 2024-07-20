import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hulc.models.perceptual_encoders.clip import build_model, load_clip


class VisionClip(nn.Module):
    def __init__(
        self, freeze_backbone: bool = True, model_name: str = "ViT-B/32"
    ):
        super(VisionClip, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load CLIP model
        print(f"loading vision CLIP model with backbone: {model_name}")
        self._load_clip(model_name)
        if freeze_backbone:
            for param in self.clip_rn50.parameters():
                param.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _load_clip(self, model_name: str) -> None:
        # model, _ = load_clip(model_name, device=self.device)
        # self.clip_rn50 = build_model(model.state_dict()).to(self.device)
        model_path = 'runs/2022-07-28/03-47-47/saved_models/epoch7.ckpt'
        model_clip = torch.load(model_path, map_location="cpu")
        state_dict = model_clip['state_dict']
        state_dict_new = {}
        for param in state_dict:
            if 'clip_vision_encoder.clip_rn50.' in param:
                param_new = param.replace('clip_vision_encoder.clip_rn50.', '')
                state_dict_new[param_new] = state_dict[param]
        self.clip_rn50 = build_model(state_dict_new).to(self.device)

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        self.clip_rn50.eval()
        with torch.no_grad():
            image_features = self.clip_rn50.encode_image(x)  # type:ignore

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            #print(text_features.shape, image_features.shape, logits_per_image.shape, logits_per_text.shape)
            # shape = [global_batch_size, global_batch_size]
            return logits_per_image

