import torch
import torch.nn as nn
from transformers import Dinov2Config, Dinov2Model

DINO_IMG_SIZE = 518
NUM_REGISTER_TOKENS = 4


class HFRegDinov2(nn.Module):
    """dinov2_vitl14_reg: transformers Dinov2Model + manually inserted register tokens."""

    def __init__(self):
        super().__init__()
        config = Dinov2Config(
            image_size=DINO_IMG_SIZE,
            patch_size=14,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )
        self.model = Dinov2Model(config)
        self.model.embeddings.register_tokens = nn.Parameter(
            torch.zeros(1, NUM_REGISTER_TOKENS, config.hidden_size)
        )

    def forward(self, x, is_training=True):
        emb = self.model.embeddings
        target_dtype = emb.patch_embeddings.projection.weight.dtype
        patch_emb = emb.patch_embeddings(x.to(dtype=target_dtype))

        batch_size = x.shape[0]
        cls_tokens = emb.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, patch_emb), dim=1)
        embeddings = embeddings + emb.interpolate_pos_encoding(embeddings, DINO_IMG_SIZE, DINO_IMG_SIZE)
        register_tokens = emb.register_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat((embeddings[:, :1], register_tokens, embeddings[:, 1:]), dim=1)
        embeddings = emb.dropout(embeddings)

        sequence_output = self.model.encoder(embeddings)[0]
        sequence_output = self.model.layernorm(sequence_output)
        return sequence_output[:, 1 + NUM_REGISTER_TOKENS:, :]


def map_dinov2_state_dict(state_dict, prefix=""):
    """Map dinov2 weights with official Meta naming to HFRegDinov2 naming.

    Args:
        state_dict: weights with Meta naming (or prefix already stripped)
        prefix: key prefix (e.g. "dino_model.")
    Returns:
        dict: state_dict with HFRegDinov2 naming
    """
    out = {}
    for key, value in state_dict.items():
        if prefix and not key.startswith(prefix):
            continue
        k = key[len(prefix):]
        if k == "cls_token":
            out["model.embeddings.cls_token"] = value
        elif k == "mask_token":
            out["model.embeddings.mask_token"] = value
        elif k == "pos_embed":
            out["model.embeddings.position_embeddings"] = value
        elif k == "register_tokens":
            out["model.embeddings.register_tokens"] = value
        elif k.startswith("patch_embed.proj."):
            out["model.embeddings.patch_embeddings.projection." + k.split(".")[-1]] = value
        elif k.startswith("norm."):
            out["model.layernorm." + k.split(".")[-1]] = value
        elif k.startswith("blocks."):
            parts = k.split(".")
            n = parts[1]
            rest = ".".join(parts[2:])
            if rest.startswith("attn.qkv."):
                typ = rest.split(".")[-1]
                q, kk, vv = value.chunk(3, dim=0)
                for name, val in [("query", q), ("key", kk), ("value", vv)]:
                    out[f"model.encoder.layer.{n}.attention.attention.{name}.{typ}"] = val
            elif rest.startswith("attn.proj."):
                out[f"model.encoder.layer.{n}.attention.output.dense.{rest.split('.')[-1]}"] = value
            elif rest == "ls1.gamma":
                out[f"model.encoder.layer.{n}.layer_scale1.lambda1"] = value
            elif rest == "ls2.gamma":
                out[f"model.encoder.layer.{n}.layer_scale2.lambda1"] = value
            elif rest.startswith(("norm1.", "norm2.", "mlp.")):
                out[f"model.encoder.layer.{n}.{rest}"] = value
            else:
                raise ValueError(f"unknown dinov2 key: {key}")
        else:
            raise ValueError(f"unknown dinov2 key: {key}")
    return out
