"""
q_former.py  –  Q-Former for Stage 1 (Vision-Language Alignment)

Architecture
────────────
1. **Frozen ViT encoder** (`google/vit-base-patch16-224`, hidden=768)
   Extracts patch embeddings from the input image.

2. **Learnable query tokens** (32 × 768)
   A small set of trainable vectors that are fed into the cross-attention
   transformer to distil the most relevant visual information.

3. **Cross-attention transformer**
   Built from a `BertModel` configured with `is_decoder=True` and
   `add_cross_attention=True` so each layer can attend both to the query
   tokens (self-attention) and to the ViT patch embeddings (cross-attention).

4. **Contrastive projection heads**
   Two linear projections mapping image and text features into a shared
   embedding space for image-text contrastive (ITC) learning.

Forward behaviour
─────────────────
* If *text* inputs are provided  →  returns `(image_proj, text_proj)`
  for contrastive loss.
* If only an image is provided   →  returns the raw query-token hidden
  states that can be handed to downstream modules (e.g., the MLP
  adapter in Stage 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertConfig,
    BertModel,
    ViTModel,
)


class QFormer(nn.Module):
    """Q-Former: learnable queries + cross-attention over a frozen ViT."""

    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",
        num_query_tokens: int = 32,
        cross_attention_layers: int = 6,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()

        # ── 1. Frozen ViT encoder ────────────────────────────────────
        self.vit = ViTModel.from_pretrained(vit_model_name)
        for param in self.vit.parameters():
            param.requires_grad = False

        vit_hidden = self.vit.config.hidden_size  # 768

        # ── 2. Learnable query tokens ────────────────────────────────
        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_size) * 0.02
        )

        # ── 3. Cross-attention transformer (BertModel as decoder) ───
        bert_cfg = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=cross_attention_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            is_decoder=True,
            add_cross_attention=True,
        )
        self.cross_attention = BertModel(bert_cfg)

        # Optional linear if ViT hidden ≠ Q-Former hidden
        self.vit_proj: nn.Module
        if vit_hidden != hidden_size:
            self.vit_proj = nn.Linear(vit_hidden, hidden_size)
        else:
            self.vit_proj = nn.Identity()

        # ── 4. Contrastive projection heads ──────────────────────────
        self.image_proj = nn.Linear(hidden_size, projection_dim)
        self.text_proj = nn.Linear(hidden_size, projection_dim)

    # -----------------------------------------------------------------
    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the frozen ViT and cross-attention to produce query outputs.

        Args:
            pixel_values: (B, 3, 224, 224)

        Returns:
            query_output: (B, num_query_tokens, hidden_size)
        """
        with torch.no_grad():
            vit_out = self.vit(pixel_values=pixel_values)
        encoder_hidden = self.vit_proj(vit_out.last_hidden_state)  # (B, P, H)

        batch_size = pixel_values.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)     # (B, Q, H)

        cross_out = self.cross_attention(
            inputs_embeds=queries,
            encoder_hidden_states=encoder_hidden,
        )
        return cross_out.last_hidden_state  # (B, Q, H)

    # -----------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values:  (B, 3, 224, 224)
            input_ids:     (B, L)  – optional text token ids
            attention_mask: (B, L) – optional

        Returns:
            If *text* is provided:
                (image_embeds, text_embeds) – both (B, projection_dim),
                L2-normalised and ready for contrastive loss.
            Otherwise:
                query_output – (B, num_query_tokens, hidden_size)
        """
        query_output = self._encode_image(pixel_values)  # (B, Q, H)

        if input_ids is None:
            return query_output

        # -- text branch: encode via the same cross-attention transformer
        # but *without* cross-attention (no encoder_hidden_states)
        text_out = self.cross_attention(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_hidden = text_out.last_hidden_state  # (B, L, H)

        # Pool: take the mean over query tokens / text tokens
        image_feat = query_output.mean(dim=1)  # (B, H)
        text_feat = text_hidden.mean(dim=1)     # (B, H)

        # Project + L2-normalise
        image_embeds = F.normalize(self.image_proj(image_feat), dim=-1)
        text_embeds = F.normalize(self.text_proj(text_feat), dim=-1)

        return image_embeds, text_embeds
