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
            tokenized = self.tokenizer.batch_encode_plus(x, padding="longest", return_tensors="pt").to("cuda")
            text_attention_mask = tokenized.attention_mask.float()
            encoded_text = self.text_encoder(**tokenized)
            # Transpose memory because pytorch's attention expects sequence first
            text_memory = encoded_text.last_hidden_state
            text_memory = text_memory * text_attention_mask.unsqueeze(-1).repeat(1, 1, text_memory.shape[-1])
            text_memory_mean = torch.sum(text_memory, 1) / torch.sum(text_attention_mask, 1).unsqueeze(-1).repeat(1, text_memory.shape[-1])

        return torch.unsqueeze(text_memory_mean, 1)
