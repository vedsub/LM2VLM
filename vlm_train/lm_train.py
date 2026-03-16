"""
lm_train.py  –  Stage 2: Causal Language Modelling Training

Loads the full VLM (Q-Former + MLP Adapter + LoRA-tuned SmolLM), restores
Stage-1 Q-Former weights, and trains with a causal LM objective on
Conceptual Captions.

Only the **MLP adapter** and **LoRA** parameters are updated; everything
else (ViT, Q-Former, base LLM weights) is frozen.

Usage:
    python -m vlm_train.lm_train
    python -m vlm_train.lm_train --epochs 3 --batch_size 16 --lr 2e-5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from vlm_train.datasets.lm_dataloader import get_cc_causal_lm_dataloader
from vlm_train.networks.lm_to_vlm import VLM


DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "cc_subset"
DEFAULT_QFORMER_WEIGHTS = (
    Path(__file__).resolve().parent / "inference_results" / "qformer_stage1.pt"
)
DEFAULT_SAVE_PATH = (
    Path(__file__).resolve().parent / "inference_results" / "vlm_stage2.pt"
)


# ------------------------------------------------------------------
# Collect trainable parameter groups
# ------------------------------------------------------------------
def _get_trainable_params(model: VLM) -> list[dict]:
    """Return only the MLP adapter and LoRA parameters for the optimiser.

    The Q-Former is already frozen inside the VLM constructor, and the
    base LLM weights are frozen by PEFT.  This function double-checks
    that by filtering explicitly.
    """
    adapter_params = list(model.adapter.parameters())
    lora_params = [
        p for name, p in model.llm.named_parameters() if p.requires_grad
    ]

    return [
        {"params": adapter_params, "lr_key": "adapter"},
        {"params": lora_params, "lr_key": "lora"},
    ]


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    num_workers: int = 4,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    qformer_weights: str | Path = DEFAULT_QFORMER_WEIGHTS,
    save_path: str | Path = DEFAULT_SAVE_PATH,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    dataloader = get_cc_causal_lm_dataloader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"Dataset loaded from {dataset_dir}  ({len(dataloader.dataset):,} samples)")

    # ── Model ─────────────────────────────────────────────────────────
    model = VLM(freeze_qformer=True).to(device)

    # Load Stage-1 Q-Former weights
    qformer_weights = Path(qformer_weights)
    if qformer_weights.exists():
        state = torch.load(qformer_weights, map_location=device, weights_only=True)
        # The checkpoint contains the full QFormer state dict; load into
        # the sub-module with strict=False to ignore missing keys from
        # ViT (which are re-loaded from HF anyway).
        model.q_former.load_state_dict(state, strict=False)
        print(f"Loaded Stage-1 Q-Former weights from {qformer_weights}")
    else:
        print(f"WARNING: Q-Former weights not found at {qformer_weights}; training from scratch.")

    # ── Optimizer (adapter + LoRA only) ───────────────────────────────
    param_groups = _get_trainable_params(model)
    all_trainable = [p for g in param_groups for p in g["params"]]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in all_trainable)
    print(
        f"Trainable parameters: {trainable_count:,} / {total_params:,} "
        f"({100 * trainable_count / total_params:.2f}%)"
    )

    optimizer = AdamW(all_trainable, lr=lr, weight_decay=0.01)

    # ── Training ──────────────────────────────────────────────────────
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader, 1):
            if not batch:
                continue

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]

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

    # Save only the trainable components (adapter + LoRA)
    save_dict = {
        "adapter": model.adapter.state_dict(),
        "lora": {
            name: param.cpu()
            for name, param in model.llm.named_parameters()
            if param.requires_grad
        },
    }
    torch.save(save_dict, save_path)
    print(f"VLM Stage-2 weights (adapter + LoRA) saved to {save_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 causal LM training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--qformer_weights", type=str, default=str(DEFAULT_QFORMER_WEIGHTS))
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        dataset_dir=args.dataset_dir,
        qformer_weights=args.qformer_weights,
        save_path=args.save_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
