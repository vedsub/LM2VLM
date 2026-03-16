"""
q_former_train.py  –  Stage 1: Contrastive Alignment Training

Trains the Q-Former using an image-text contrastive (ITC) objective on the
Conceptual Captions dataset.

Loss
────
The Q-Former forward pass returns L2-normalised (image_embeds, text_embeds),
each of shape (B, projection_dim).  We compute a symmetric cross-entropy loss
over the cosine-similarity matrix (CLIP-style):

    sim = image_embeds @ text_embeds.T * temperature
    loss = (CE(sim, targets) + CE(sim.T, targets)) / 2

Usage:
    python -m vlm_train.q_former_train
    python -m vlm_train.q_former_train --epochs 5 --batch_size 64 --lr 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from vlm_train.datasets.cc_dataloader import get_cc_alignment_dataloader
from vlm_train.networks.q_former import QFormer


DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "cc_subset"
DEFAULT_SAVE_PATH = Path(__file__).resolve().parent / "inference_results" / "qformer_stage1.pt"


# ------------------------------------------------------------------
# Contrastive loss (CLIP-style symmetric cross-entropy)
# ------------------------------------------------------------------
def contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE / CLIP loss.

    Args:
        image_embeds: (B, D) – L2-normalised image features.
        text_embeds:  (B, D) – L2-normalised text features.
        temperature:  learnable or fixed temperature scalar.

    Returns:
        Scalar loss.
    """
    # Cosine similarity matrix scaled by temperature
    logits = (image_embeds @ text_embeds.T) / temperature  # (B, B)

    # Ground-truth: the diagonal (i-th image matches i-th text)
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)        # image → text
    loss_t2i = F.cross_entropy(logits.T, targets)      # text  → image
    return (loss_i2t + loss_t2i) / 2.0


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train(
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
    temperature: float = 0.07,
    num_workers: int = 4,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    save_path: str | Path = DEFAULT_SAVE_PATH,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    dataloader = get_cc_alignment_dataloader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"Dataset loaded from {dataset_dir}  ({len(dataloader.dataset):,} samples)")

    # ── Model ─────────────────────────────────────────────────────────
    model = QFormer().to(device)
    model.train()

    # Only optimise the trainable parameters (ViT is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in trainable_params):,} / "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # ── Training ──────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader, 1):
            if not batch:  # empty batch (all images failed)
                continue

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward – returns (image_embeds, text_embeds)
            image_embeds, text_embeds = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = contrastive_loss(image_embeds, text_embeds, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if step % 50 == 0:
                avg = running_loss / num_batches
                print(f"  [Epoch {epoch}/{epochs}  Step {step}]  loss = {avg:.4f}")

        epoch_loss = running_loss / max(num_batches, 1)
        print(f"Epoch {epoch}/{epochs}  –  avg loss = {epoch_loss:.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Q-Former weights saved to {save_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 Q-Former contrastive training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        num_workers=args.num_workers,
        dataset_dir=args.dataset_dir,
        save_path=args.save_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
