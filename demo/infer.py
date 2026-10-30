"""Load the TrailCam CNN and run inference on OpenCV BGR frames."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from trailcam.models.cnn import build_cnn, build_transforms, get_device


def center_square_crop_bgr(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return center square crop and (x1, y1, x2, y2) box on the full frame."""
    h, w = frame.shape[:2]
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


class SpeciesPredictor:
    """ResNet-18 species classifier for demo overlays."""

    def __init__(
        self,
        checkpoint_path: Path,
        labels_path: Path,
        image_size: int = 224,
        device: torch.device | None = None,
    ) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {checkpoint_path}. Run: python scripts/train_cnn.py"
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Missing labels file {labels_path}. Run: python scripts/train_cnn.py"
            )

        report = json.loads(labels_path.read_text(encoding="utf-8"))
        self.label_names: list[str] = list(report["label_names"])
        self.device = device or get_device()
        self.transform = build_transforms(image_size=image_size, train=False)

        self.model = build_cnn(num_classes=len(self.label_names), pretrained=False)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_crop(
        self,
        crop_bgr: np.ndarray,
        top_k: int = 3,
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Classify an already-cropped BGR region."""
        if crop_bgr.size == 0:
            raise ValueError("Empty crop")
        crop_rgb = crop_bgr[:, :, ::-1]
        image = Image.fromarray(crop_rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        k = max(1, min(top_k, len(self.label_names)))
        idxs = np.argsort(probs)[::-1][:k]
        topk = [(self.label_names[int(i)], float(probs[int(i)])) for i in idxs]
        top_label, top_conf = topk[0]
        return top_label, top_conf, topk

    @torch.no_grad()
    def predict_bgr(
        self,
        frame_bgr: np.ndarray,
        top_k: int = 3,
        box: tuple[int, int, int, int] | None = None,
    ) -> tuple[str, float, list[tuple[str, float]], tuple[int, int, int, int]]:
        """Classify ``box`` on the frame, or the center square if ``box`` is None."""
        if box is None:
            crop_bgr, box = center_square_crop_bgr(frame_bgr)
        else:
            x1, y1, x2, y2 = box
            crop_bgr = frame_bgr[y1:y2, x1:x2]
        top_label, top_conf, topk = self.predict_crop(crop_bgr, top_k=top_k)
        return top_label, top_conf, topk, box

    @torch.no_grad()
    def predict_best_scale(
        self,
        frame_bgr: np.ndarray,
        box: tuple[int, int, int, int],
        top_k: int = 3,
        scales: tuple[float, ...] = (0.7, 1.0, 1.35),
    ) -> tuple[str, float, list[tuple[str, float]], tuple[int, int, int, int]]:
        """Try a few zoom levels around ``box`` and keep the most confident crop."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        base = max(x2 - x1, y2 - y1)

        best = None
        for scale in scales:
            side = max(32, int(round(base * scale)))
            side = min(side, min(w, h))
            half = side / 2.0
            nx1 = int(round(cx - half))
            ny1 = int(round(cy - half))
            nx2 = nx1 + side
            ny2 = ny1 + side
            if nx1 < 0:
                nx2 -= nx1
                nx1 = 0
            if ny1 < 0:
                ny2 -= ny1
                ny1 = 0
            if nx2 > w:
                nx1 -= nx2 - w
                nx2 = w
            if ny2 > h:
                ny1 -= ny2 - h
                ny2 = h
            nx1, ny1 = max(0, nx1), max(0, ny1)
            crop = frame_bgr[ny1:ny2, nx1:nx2]
            if crop.size == 0:
                continue
            label, conf, topk = self.predict_crop(crop, top_k=top_k)
            cand = (conf, label, topk, (nx1, ny1, nx2, ny2))
            if best is None or cand[0] > best[0]:
                best = cand

        if best is None:
            return self.predict_bgr(frame_bgr, top_k=top_k, box=box)
        conf, label, topk, best_box = best
        return label, conf, topk, best_box
