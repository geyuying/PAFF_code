from typing import List
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast

class Mdetr(nn.Module):
    def __init__(self):
        #  choose model from https://www.sbert.net/docs/pretrained_models.html
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        mdter_checkpoint = torch.load('hulc/ckpts/mdetr_pretrained_resnet101_checkpoint.pth', map_location="cpu")['model']
        checkpoint_new = {}
        for param in mdter_checkpoint:
            if 'transformer.text_encoder' in param:
                param_new = param.replace('transformer.','')
                checkpoint_new[param_new] = mdter_checkpoint[param]
        self.load_state_dict(checkpoint_new, True)

    def forward(self, x: List) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            tokenized = self.tokenizer.batch_encode_plus(x, padding="longest", return_tensors="pt")
            encoded_text = self.text_encoder(**tokenized)

            # Transpose memory because pytorch's attention expects sequence first
            text_memory = encoded_text.last_hidden_state.transpose(0, 1)
            text_memory_mean = torch.mean(text_memory, 0)
        #emb = self.model.encode(x, convert_to_tensor=True)
        return torch.unsqueeze(text_memory_mean, 1)
