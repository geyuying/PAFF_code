import torch
import torch.nn as nn
import torch.nn.functional as F
from hulc.models.perceptual_encoders.backbone_full import Backbone
from hulc.models.perceptual_encoders.misc import NestedTensor
import math

class VisionMdetr(nn.Module):
    def __init__(
        self, device: torch.device, mdetr_out: int, visual_features: int, freeze_backbone: bool = True, model_name: str = "resnet101"
    ):
        super(VisionMdetr, self).__init__()
        # Load CLIP model
        print(f"loading vision Mdetr model with backbone: {model_name}")
        self.backbone = Backbone('resnet101', True, True, False)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        mdter_checkpoint = torch.load('hulc/ckpts/mdetr_pretrained_resnet101_checkpoint.pth', map_location="cpu")['model']
        checkpoint_new = {}
        for param in mdter_checkpoint:
            if 'backbone.0.body' in param:
                param_new = param.replace('backbone.0.body', 'backbone.body')
                checkpoint_new[param_new] = mdter_checkpoint[param]
        self.load_state_dict(checkpoint_new, True)
        #size = int(math.sqrt(mdetr_out))
        #self.pool = nn.AvgPool2d(size)
        self.fc1 = nn.Linear(2048*mdetr_out, 512)
        self.fc2 = nn.Linear(512, visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        img = NestedTensor.from_tensor_list(x)
        with torch.no_grad():
            xs = self.backbone(img)
            out = []
            for name, x in xs.items():
                out.append(x)
            x, _ = out[-1].decompose()
        #x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        output = F.relu(self.fc1(x))  # batch, 512
        output = self.fc2(output)  # batch, 64
        return output
