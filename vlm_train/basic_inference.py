"""
basic_inference.py  –  Generate captions from a trained VLM

Loads the full VLM (Q-Former + MLP Adapter + LoRA-tuned SmolLM), restores
the trained adapter / LoRA weights, and autoregressively generates a
caption for a given image.

Usage:
    python -m vlm_train.basic_inference \\
        --image_path  path/to/photo.jpg \\
        --prompt      "Describe this image: " \\
        --model_path  vlm_train/inference_results/vlm_stage2.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor

from vlm_train.networks.lm_to_vlm import VLM


VIT_MODEL_NAME = "google/vit-base-patch16-224"
LLM_MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "inference_results" / "vlm_stage2.pt"
)


# ------------------------------------------------------------------
# Core inference function
# ------------------------------------------------------------------
def generate_caption(
    image_path: str | Path,
    text_prompt: str = "Describe this image: ",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a caption for *image_path* using the trained VLM.

    Args:
        image_path:     Path to a local image file.
        text_prompt:    Prompt prefix fed to the LLM (e.g. "Describe this image: ").
        model_path:     Path to the saved Stage-2 checkpoint
                        (dict with ``adapter`` and ``lora`` keys).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature:    Sampling temperature.
        top_p:          Nucleus-sampling probability mass.

    Returns:
        The decoded caption string.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Build model & load weights ────────────────────────────────
    model = VLM(freeze_qformer=True)

    model_path = Path(model_path)
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

        # Restore MLP adapter weights
        model.adapter.load_state_dict(checkpoint["adapter"])

        # Restore LoRA weights into the PEFT-wrapped LLM
        lora_state = checkpoint.get("lora", {})
        if lora_state:
            current_state = model.llm.state_dict()
            current_state.update(lora_state)
            model.llm.load_state_dict(current_state, strict=False)

        print(f"Loaded Stage-2 weights from {model_path}")
    else:
        print(f"WARNING: checkpoint not found at {model_path}; using untrained model.")

    model.to(device)
    model.eval()

    # ── 2. Image processing ──────────────────────────────────────────
    image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)

    image = Image.open(image_path).convert("RGB")
    pixel_inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = pixel_inputs["pixel_values"].to(device)  # (1, 3, 224, 224)

    # ── 3. Tokenize prompt ───────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_inputs = tokenizer(
        text_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = text_inputs["input_ids"].to(device)            # (1, L)
    attention_mask = text_inputs["attention_mask"].to(device)   # (1, L)

    # ── 4. Generate ──────────────────────────────────────────────────
    with torch.no_grad():
        # Build the combined [image | text] embeddings and mask the same
        # way the VLM forward pass does, then call llm.generate.
        num_queries = model.q_former.num_query_tokens

        query_output = model.q_former(pixel_values)      # (1, Q, 768)
        image_embeds = model.adapter(query_output)        # (1, Q, D_llm)

        text_embeds = model._get_text_embeddings(input_ids)  # (1, L, D_llm)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # (1, Q+L, D)

        image_mask = torch.ones(
            1, num_queries,
            dtype=attention_mask.dtype,
            device=device,
        )
        combined_mask = torch.cat([image_mask, attention_mask], dim=1)  # (1, Q+L)

        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # ── 5. Decode ────────────────────────────────────────────────────
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a caption for an image")
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe this image: ",
        help="Text prompt prefix.",
    )
    parser.add_argument(
        "--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Path to saved Stage-2 checkpoint.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    caption = generate_caption(
        image_path=args.image_path,
        text_prompt=args.prompt,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"\n{'─' * 60}")
    print(f"Image   : {args.image_path}")
    print(f"Prompt  : {args.prompt}")
    print(f"Caption : {caption}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
