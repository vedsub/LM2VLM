"""
filter_dataset.py

Downloads the 'conceptual_captions' dataset from Hugging Face and saves a
small subset (default: 50 000 samples) to disk for efficient local training.

Usage:
    python -m vlm_train.utils.filter_dataset                  # defaults
    python -m vlm_train.utils.filter_dataset --num_samples 10000 --output_dir ./my_data
"""

import argparse
from pathlib import Path

from datasets import load_dataset


DEFAULT_NUM_SAMPLES = 50_000
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cc_subset"


def download_and_filter(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
) -> Path:
    """Download Conceptual Captions and save a filtered subset to disk.

    Args:
        num_samples: Number of samples to keep.
        output_dir:  Directory where the Arrow dataset will be saved.
        seed:        Random seed for reproducible shuffling.

    Returns:
        The resolved output path.
    """
    output_dir = Path(output_dir)

    print(f"Loading 'conceptual_captions' (train split) from Hugging Face …")
    dataset = load_dataset("conceptual_captions", split="train", trust_remote_code=True)

    total = len(dataset)
    num_samples = min(num_samples, total)
    print(f"Total samples available: {total:,}")
    print(f"Selecting {num_samples:,} samples (seed={seed}) …")

    subset = dataset.shuffle(seed=seed).select(range(num_samples))

    output_dir.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(output_dir))
    print(f"Subset saved to {output_dir}")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and filter a subset of Conceptual Captions.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of samples to keep (default: {DEFAULT_NUM_SAMPLES:,}).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save the filtered subset (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42).",
    )

    args = parser.parse_args()
    download_and_filter(
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
