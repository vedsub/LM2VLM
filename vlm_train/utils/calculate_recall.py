"""
calculate_recall.py  –  Recall@K evaluation for Stage-1 Q-Former

Evaluates the Q-Former's contrastive alignment by computing retrieval
recall (R@1, R@5, R@10) in both directions:

* **I2T** – given an image, retrieve the correct caption.
* **T2I** – given a caption, retrieve the correct image.

Usage:
    python -m vlm_train.utils.calculate_recall
    python -m vlm_train.utils.calculate_recall \\
        --model_path  vlm_train/inference_results/qformer_stage1.pt \\
        --dataset_dir vlm_train/datasets/cc_subset \\
        --batch_size  64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vlm_train.datasets.cc_dataloader import get_cc_alignment_dataloader
from vlm_train.networks.q_former import QFormer


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "inference_results" / "qformer_stage1.pt"
)
DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parent.parent / "datasets" / "cc_subset"
)

K_VALUES = (1, 5, 10)


# ------------------------------------------------------------------
# Recall helpers
# ------------------------------------------------------------------
def _recall_at_k(
    similarity: torch.Tensor,
    k: int,
) -> float:
    """Compute Recall@K from a (N, N) similarity matrix.

    Row *i* corresponds to query *i*; the ground-truth match is column *i*
    (the diagonal).
    """
    n = similarity.size(0)

    # Indices of the top-k most similar targets for each query
    _, topk_indices = similarity.topk(k, dim=1)  # (N, K)

    # Ground-truth: column index == row index
    targets = torch.arange(n, device=similarity.device).unsqueeze(1)  # (N, 1)

    # A query is a hit if the target appears anywhere in its top-k
    hits = (topk_indices == targets).any(dim=1).float()
    return hits.mean().item()


# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_recall(
    val_dataloader: DataLoader,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str | None = None,
) -> dict[str, float]:
    """Evaluate retrieval Recall@K for a trained Q-Former.

    Args:
        val_dataloader: DataLoader yielding batches with
                        ``pixel_values``, ``input_ids``, ``attention_mask``.
        model_path:     Path to the saved Q-Former state dict.
        device:         Compute device (auto-detected if *None*).

    Returns:
        Dict of metric names → values, e.g.
        ``{"I2T_R@1": 0.32, "T2I_R@5": 0.61, …}``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────
    model = QFormer()
    model_path = Path(model_path)
    if model_path.exists():
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"Loaded Q-Former weights from {model_path}")
    else:
        print(f"WARNING: weights not found at {model_path}; evaluating untrained model.")

    model.to(device)
    model.eval()

    # ── Collect features ──────────────────────────────────────────────
    all_image_feats: list[torch.Tensor] = []
    all_text_feats: list[torch.Tensor] = []

    print("Extracting features …")
    for batch in val_dataloader:
        if not batch:
            continue

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_embeds, text_embeds = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        all_image_feats.append(image_embeds.cpu())
        all_text_feats.append(text_embeds.cpu())

    # ── Aggregate & normalise ─────────────────────────────────────────
    image_features = torch.cat(all_image_feats, dim=0)  # (N, D)
    text_features = torch.cat(all_text_feats, dim=0)    # (N, D)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    n = image_features.size(0)
    print(f"Collected {n:,} image-text pairs.")

    # ── Cosine similarity matrix ──────────────────────────────────────
    sim = image_features @ text_features.T  # (N, N)

    # ── Recall@K ──────────────────────────────────────────────────────
    results: dict[str, float] = {}

    for k in K_VALUES:
        if k > n:
            break
        results[f"I2T_R@{k}"] = _recall_at_k(sim, k)      # image → text
        results[f"T2I_R@{k}"] = _recall_at_k(sim.T, k)    # text  → image

    # ── Print ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 45}")
    print(f"  Retrieval Recall  ({n:,} pairs)")
    print(f"{'═' * 45}")
    print(f"  {'Metric':<12} {'I2T':>10} {'T2I':>10}")
    print(f"  {'─' * 32}")
    for k in K_VALUES:
        i2t_key = f"I2T_R@{k}"
        t2i_key = f"T2I_R@{k}"
        if i2t_key in results:
            print(f"  R@{k:<9} {results[i2t_key]:>9.4f}  {results[t2i_key]:>9.4f}")
    print(f"{'═' * 45}\n")

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Q-Former retrieval recall (R@1, R@5, R@10)."
    )
    parser.add_argument(
        "--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Path to saved Q-Former weights.",
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=str(DEFAULT_DATASET_DIR),
        help="Path to the CC subset saved on disk.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    val_dataloader = get_cc_alignment_dataloader(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    evaluate_recall(
        val_dataloader=val_dataloader,
        model_path=args.model_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
