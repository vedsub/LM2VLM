"""
lm_to_vlm.py  –  Stage 2: Full Vision-Language Model

Architecture (see project diagram)
───────────────────────────────────
ViT → Q-Former → MLP Adapter → ┐
                                ├─ Concat → LLM (LoRA) → logits / loss
         text prompt tokens   → ┘

Components
──────────
1. **QFormer**       – pre-trained in Stage 1 (frozen or fine-tuned).
2. **MLP Adapter**   – Linear → GELU → Linear, projects Q-Former hidden
                       (768) to the LLM hidden size (576 for SmolLM-135M).
3. **Causal LLM**    – `HuggingFaceTB/SmolLM-135M-Instruct` with LoRA
                       adapters on `q_proj` and `v_proj`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlm_train.networks.q_former import QFormer


LLM_MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"


# ------------------------------------------------------------------
# MLP Adapter
# ------------------------------------------------------------------
class MLPAdapter(nn.Module):
    """Linear → GELU → Linear projection from Q-Former to LLM space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ------------------------------------------------------------------
# Full VLM
# ------------------------------------------------------------------
class VLM(nn.Module):
    """Vision-Language Model: Q-Former + MLP Adapter + LoRA-tuned LLM."""

    def __init__(
        self,
        qformer_kwargs: dict[str, Any] | None = None,
        llm_model_name: str = LLM_MODEL_NAME,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        freeze_qformer: bool = True,
    ) -> None:
        super().__init__()

        # ── 1. Q-Former (optionally frozen after Stage 1) ────────────
        self.q_former = QFormer(**(qformer_kwargs or {}))
        if freeze_qformer:
            for param in self.q_former.parameters():
                param.requires_grad = False

        qformer_hidden = self.q_former.query_tokens.size(-1)  # 768

        # ── 2. Causal LLM + LoRA ────────────────────────────────────
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float32,
        )

        llm_hidden: int = self.llm.config.hidden_size  # 576 for SmolLM-135M

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

        # ── 3. MLP Adapter (Q-Former dim → LLM dim) ─────────────────
        self.adapter = MLPAdapter(qformer_hidden, llm_hidden)

        # ── 4. Tokenizer (kept for convenience) ─────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings from the LLM's embedding layer."""
        return self.llm.get_input_embeddings()(input_ids)  # (B, L, D_llm)

    # -----------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pixel_values:   (B, 3, 224, 224)
            input_ids:      (B, L)
            attention_mask: (B, L)
            labels:         (B, L) – optional; prompt tokens should be -100

        Returns:
            dict with ``"logits"`` and optionally ``"loss"``.
        """
        batch_size = pixel_values.size(0)
        num_queries = self.q_former.num_query_tokens

        # ── image branch ─────────────────────────────────────────────
        query_output = self.q_former(pixel_values)          # (B, Q, 768)
        image_embeds = self.adapter(query_output)            # (B, Q, D_llm)

        # ── text branch ──────────────────────────────────────────────
        text_embeds = self._get_text_embeddings(input_ids)   # (B, L, D_llm)

        # ── concatenate: [image tokens | text tokens] ────────────────
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # (B, Q+L, D)

        # Extend attention mask to cover image tokens (always attended)
        image_mask = torch.ones(
            batch_size, num_queries,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_mask = torch.cat([image_mask, attention_mask], dim=1)  # (B, Q+L)

        # ── labels: prepend -100 for image positions ─────────────────
        combined_labels = None
        if labels is not None:
            image_labels = torch.full(
                (batch_size, num_queries),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([image_labels, labels], dim=1)  # (B, Q+L)

        # ── LLM forward ─────────────────────────────────────────────
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        result: dict[str, torch.Tensor] = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss
        return result
