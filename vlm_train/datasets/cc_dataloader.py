"""
cc_dataloader.py  –  Stage 1 (Alignment) DataLoader

Loads the filtered Conceptual Captions subset from disk and returns batches of
(pixel_values, input_ids, attention_mask) for vision-language alignment
pre-training.

Image processing  : ViTImageProcessor  ('google/vit-base-patch16-224')
Text tokenisation : AutoTokenizer      ('distilbert-base-uncased')
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import requests
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ViTImageProcessor


DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parent / "cc_subset"
)

VIT_MODEL_NAME = "google/vit-base-patch16-224"
TEXT_MODEL_NAME = "distilbert-base-uncased"


class CCAlignmentDataset(Dataset):
    """Conceptual Captions dataset for Stage-1 alignment.

    Each sample returns:
        pixel_values   – (3, 224, 224) float tensor
        input_ids      – (max_length,) long tensor
        attention_mask – (max_length,) long tensor
    """

    def __init__(
        self,
        dataset_dir: str | Path = DEFAULT_DATASET_DIR,
        vit_model_name: str = VIT_MODEL_NAME,
        text_model_name: str = TEXT_MODEL_NAME,
        max_length: int = 128,
        timeout: int = 5,
    ) -> None:
        super().__init__()
        self.dataset = load_from_disk(str(dataset_dir))

        self.image_processor = ViTImageProcessor.from_pretrained(vit_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        self.max_length = max_length
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _download_image(self, url: str) -> Image.Image | None:
        """Download an image from *url*; return ``None`` on failure."""
        try:
            resp = requests.get(url, timeout=self.timeout, stream=True)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """Run ViTImageProcessor and return pixel_values tensor."""
        encoding = self.image_processor(images=image, return_tensors="pt")
        return encoding["pixel_values"].squeeze(0)  # (3, 224, 224)

    def _tokenize_caption(self, caption: str) -> dict[str, torch.Tensor]:
        """Tokenize a caption string and return input_ids + attention_mask."""
        return self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor] | None:
        sample = self.dataset[idx]
        image_url: str = sample["image_url"]
        caption: str = sample["caption"]

        image = self._download_image(image_url)
        if image is None:
            return None  # handled by collate_fn

        pixel_values = self._process_image(image)
        tok = self._tokenize_caption(caption)

        return {
            "pixel_values": pixel_values,                # (3, 224, 224)
            "input_ids": tok["input_ids"].squeeze(0),    # (max_length,)
            "attention_mask": tok["attention_mask"].squeeze(0),  # (max_length,)
        }


# ------------------------------------------------------------------
# Collate – skip samples where image download failed (returned None)
# ------------------------------------------------------------------
def skip_none_collate(batch: list[dict[str, Any] | None]) -> dict[str, torch.Tensor]:
    """Stack only the non-``None`` samples in a batch."""
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return {}
    return {
        key: torch.stack([s[key] for s in batch])
        for key in batch[0]
    }


# ------------------------------------------------------------------
# Convenience builder
# ------------------------------------------------------------------
def get_cc_alignment_dataloader(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = 128,
    shuffle: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Build a ready-to-use DataLoader for Stage-1 alignment training."""
    ds = CCAlignmentDataset(
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
