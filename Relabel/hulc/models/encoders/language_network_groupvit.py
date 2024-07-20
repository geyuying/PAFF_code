from typing import List
import torch
import torch.nn as nn
from hulc.models.perceptual_encoders.groupvit.tokenizer import SimpleTokenizer
from hulc.models.perceptual_encoders.groupvit.transformer import TextTransformer

class Tokenize:

    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[:self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f'Input {texts[i]} is too long for context length {self.max_seq_len}')
            result[i, :len(tokens)] = torch.tensor(tokens)

        if expanded_dim:
            return result[0]

        return result

class GroupVit(nn.Module):
    def __init__(self):
        #  choose model from https://www.sbert.net/docs/pretrained_models.html
        super().__init__()
        self.tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=77)
        self.text_encoder = TextTransformer()
        checkpoint = torch.load('hulc/ckpts/group_vit_gcc_redcap_30e-3dd09a76.pth', map_location="cpu")['model']
        checkpoint_new = {}
        for param in checkpoint:
            if 'text_encoder' in param:
                param_new = param
                checkpoint_new[param_new] = checkpoint[param]
        self.load_state_dict(checkpoint_new, True)

    def forward(self, x: List) -> torch.Tensor:
        with torch.no_grad():
            tokenized = self.tokenizer(x)
            encoded_text = self.text_encoder(tokenized)
            print(encoded_text.shape)
        #emb = self.model.encode(x, convert_to_tensor=True)
        return torch.unsqueeze(encoded_text, 1)
