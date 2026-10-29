"""Zero-shot vision-language model prompting for species classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from trailcam.models.cnn import get_device


def _prompt_for_species(name: str) -> str:
    """Turn species labels into natural-language CLIP prompts."""
    readable = name.replace("_", " ")
    return f"a camera trap photo of a {readable}"


@torch.no_grad()
def classify_zero_shot(
    image_paths: list[str] | list[Path],
    class_names: list[str],
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    """Classify crops with pretrained CLIP (no ENA24 fine-tuning).

    Returns integer class indices aligned with ``class_names``.
    """
    device = device or get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        # Match OpenAI CLIP checkpoints (avoids QuickGELU mismatch warning).
        force_quick_gelu=(pretrained == "openai"),
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    prompts = [_prompt_for_species(name) for name in class_names]
    text_tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    preds: list[int] = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="CLIP zero-shot"):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as img:
                images.append(preprocess(img.convert("RGB")))
        image_batch = torch.stack(images).to(device)
        image_features = model.encode_image(image_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())

    return np.asarray(preds, dtype=np.int64)
