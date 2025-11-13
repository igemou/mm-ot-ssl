import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional

class TextEncoder(nn.Module):
    """
    Wrapper around a Hugging Face text model.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = False,
        output_hidden_states: bool = False,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = output_hidden_states
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = self.model.config.hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_feats = out.last_hidden_state  # (B, L, D)
        pooled = token_feats[:, 0, :]        # CLS token as sentence embedding
        return pooled, token_feats
