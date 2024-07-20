from typing import Dict
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizerFast, RobertaConfig
from torch import Tensor, nn
import copy


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):

        output = src
        output_all = []
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            output_all.append(output)
        if self.norm is not None:
            output = self.norm(output)

        return output, output_all


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        print(self.normalize_before)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class Mdetr(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base').to("cuda")
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=256,
            dropout=0.1,
        )
        encoder_layer = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, 6, None)
        mdter_checkpoint = torch.load('hulc/ckpts/mdetr_pretrained_resnet101_checkpoint.pth', map_location="cpu")['model']
        checkpoint_new = {}
        for param in mdter_checkpoint:
            if 'transformer.text_encoder' in param or 'transformer.encoder.' in param or 'resizer' in param:
                param_new = param.replace('transformer.','')
                checkpoint_new[param_new] = mdter_checkpoint[param]
        self.load_state_dict(checkpoint_new, True)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, lang, raw, src, pos_embed, mask):
        self.eval()
        with torch.no_grad():
            # tokenized = self.tokenizer.batch_encode_plus(raw, padding="longest", return_tensors="pt").to("cuda")
            # encoded_text = self.text_encoder(**tokenized)
            # # Transpose memory because pytorch's attention expects sequence first
            # text_memory = encoded_text.last_hidden_state
            # text_attention_mask = tokenized.attention_mask.float()
            # text_memory = text_memory * text_attention_mask.unsqueeze(-1).repeat(1, 1, text_memory.shape[-1])
            # text_memory_mean = torch.sum(text_memory, 1) / torch.sum(text_attention_mask, 1).unsqueeze(-1).repeat(1, text_memory.shape[-1])
            # print(lang[:3,0:10],text_memory_mean[:3, :10])
            # visual_memory = torch.zeros((49, 12, 256)).to("cuda")

            tokenized = self.tokenizer.batch_encode_plus(raw, padding="longest", return_tensors="pt").to("cuda")
            encoded_text = self.text_encoder(**tokenized)
            # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
            text_memory = encoded_text.last_hidden_state.transpose(0, 1)
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # Resize the encoder hidden states to be of the same d_model as the decoder
            text_memory_resized = self.resizer(text_memory)
            num = src.shape[0]
            #print(src.shape, text_memory_resized.shape, mask.shape, text_attention_mask.shape, pos_embed.shape)
            src = torch.cat([src, text_memory_resized], dim=0)
            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_attention_mask], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)
            img_memory, img_memory_all  = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
            visual_mean = torch.mean(img_memory[:num], 0)
            lang_cat = torch.cat((lang, visual_mean), 1)
            #print(lang.shape, visual_mean.shape, lang_cat.shape)

        return lang_cat
