"""
lm_dataloader.py  –  Stage 2 (Causal Language Modelling) DataLoader

Inherits from the Stage-1 CCAlignmentDataset but swaps in a causal LM
tokenizer (SmolLM-135M-Instruct) and formats text as:

    "Describe this image: {caption}"

Labels are constructed for causal LM loss: the prompt prefix
("Describe this image: ") is masked with -100 so only the caption tokens
contribute to the cross-entropy loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vlm_train.datasets.cc_dataloader import (
    CCAlignmentDataset,
    skip_none_collate,
)


DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parent / "cc_subset"
)

LM_MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
PROMPT_PREFIX = "Describe this image: "


class CCCausalLMDataset(CCAlignmentDataset):
    """Conceptual Captions dataset for Stage-2 causal language modelling.

    Differences from the parent (Stage-1) class:
      * Uses a causal LM tokenizer (SmolLM-135M-Instruct).
      * Formats text as ``"Describe this image: {caption}"``.
      * Returns **labels** with the prompt prefix masked to -100.

    Each sample returns:
        pixel_values   – (3, 224, 224) float tensor
        input_ids      – (max_length,)  long tensor
        attention_mask – (max_length,)  long tensor
        labels         – (max_length,)  long tensor  (prompt tokens = -100)
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        dataset_dir: str | Path = DEFAULT_DATASET_DIR,
        vit_model_name: str | None = None,
        lm_model_name: str = LM_MODEL_NAME,
        max_length: int = 256,
        timeout: int = 5,
    ) -> None:
        # Initialise the parent with the LM tokenizer instead of DistilBERT.
        # We pass text_model_name=lm_model_name so the parent stores it as
        # self.tokenizer.
        super().__init__(
            dataset_dir=dataset_dir,
            vit_model_name=vit_model_name or "google/vit-base-patch16-224",
            text_model_name=lm_model_name,
            max_length=max_length,
            timeout=timeout,
        )

        # Causal LM tokenizers often lack a pad token – fall back to eos.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache the token length of the prompt prefix (without special tokens)
        # so we can mask it cheaply in __getitem__.
        self._prompt_prefix_len: int = len(
            self.tokenizer(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
        )

    # ------------------------------------------------------------------
    # Override __getitem__ to add prompt formatting + label masking
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor] | None:
        sample = self.dataset[idx]
        image_url: str = sample["image_url"]
        caption: str = sample["caption"]

        # --- image ---
        image = self._download_image(image_url)
        if image is None:
            return None

        pixel_values = self._process_image(image)

        # --- text: "Describe this image: {caption}" ---
        full_text = f"{PROMPT_PREFIX}{caption}"

        tok = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tok["input_ids"].squeeze(0)            # (max_length,)
        attention_mask = tok["attention_mask"].squeeze(0)   # (max_length,)

        # --- labels: mask prompt prefix + padding with IGNORE_INDEX ---
        labels = input_ids.clone()

        # Mask the prompt prefix tokens (including any leading special token
        # such as BOS that the tokenizer might prepend).
        num_special = int(
            self.tokenizer(
                "", add_special_tokens=True, return_tensors="pt"
            )["input_ids"].numel()
        )
        prefix_end = num_special + self._prompt_prefix_len
        labels[:prefix_end] = self.IGNORE_INDEX

        # Mask padding tokens
        labels[attention_mask == 0] = self.IGNORE_INDEX

        return {
            "pixel_values": pixel_values,     # (3, 224, 224)
            "input_ids": input_ids,           # (max_length,)
            "attention_mask": attention_mask,  # (max_length,)
            "labels": labels,                 # (max_length,)
        }


# ------------------------------------------------------------------
# Convenience builder
# ------------------------------------------------------------------
def get_cc_causal_lm_dataloader(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    batch_size: int = 16,
    num_workers: int = 4,
    max_length: int = 256,
    shuffle: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Build a ready-to-use DataLoader for Stage-2 causal LM fine-tuning."""
    ds = CCCausalLMDataset(
        dataset_dir=dataset_dir,
        max_length=max_length,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=skip_none_collate,
        pin_memory=True,
        **kwargs,
    )
