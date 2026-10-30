#!/usr/bin/env python3
"""TrailCam CNN demo: full scenes with animal boxes, video, webcam, or crops."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer import SpeciesPredictor, center_square_crop_bgr
from motion import MotionCropper
from scenes import load_scene_items


def repo_root() -> Path:
    return _ROOT


def default_sample(samples_dir: Path) -> Path | None:
    videos = sorted(samples_dir.glob("*.mp4")) + sorted(samples_dir.glob("*.avi"))
    return videos[0] if videos else None


def list_crop_paths(crops_dir: Path, limit: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(
        p for p in crops_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No crop images found under {crops_dir}")
    return paths


def parse_source(source: str) -> tuple[str, str | int]:
    """Return (kind, value) where kind is scenes|webcam|video|crops."""
    if source in {"crops", "scenes"}:
        return source, source
    if source.isdigit():
        return "webcam", int(source)
    return "video", source


def draw_overlay(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
    display_label: str,
    topk: list[tuple[str, float]],
    fps: float,
) -> np.ndarray:
    out = frame.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)

    header = display_label
    cv2.rectangle(out, (8, 8), (8 + 12 + 9 * max(len(header), 8), 42), (0, 0, 0), -1)
    cv2.putText(
        out,
        header,
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 180),
        2,
        cv2.LINE_AA,
    )

    if topk:
        y = 70
        cv2.rectangle(out, (8, 50), (320, 50 + 28 * len(topk) + 10), (0, 0, 0), -1)
        for name, conf in topk:
            line = f"{name}: {conf:.2f}"
            cv2.putText(
                out,
                line,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            y += 28

    cv2.putText(
        out,
        f"{fps:.1f} FPS",
        (out.shape[1] - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def run_demo(args: argparse.Namespace) -> None:
    root = repo_root()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    labels = Path(args.labels)
    if not labels.is_absolute():
        labels = root / labels

    predictor = SpeciesPredictor(checkpoint, labels, image_size=args.image_size)
    print(f"Loaded CNN on {predictor.device} with {len(predictor.label_names)} classes")

    source_arg = args.source
    if source_arg is None:
        manifest = root / "data" / "interim" / "crops_manifest.csv"
        raw_images = root / "data" / "raw" / "images"
        if manifest.exists() and raw_images.exists():
            source_arg = "scenes"
            print("Using default source: scenes (ENA24 full frames + animal boxes)")
        else:
            sample = default_sample(root / "demo" / "samples")
            if sample is None:
                raise SystemExit(
                    "No demo source found. Prepare data, or run:\n"
                    "  python demo/fetch_samples.py\n"
                    "Or pass --source scenes|crops|path/to/video.mp4|0"
                )
            source_arg = str(sample)
            print(f"Using default sample: {sample}")

    kind, value = parse_source(str(source_arg))
    cap: cv2.VideoCapture | None = None
    crop_paths: list[Path] = []
    scene_items: list[dict] = []
    item_idx = 0
    hold_ms = max(1, int(args.hold_ms))
    use_motion = kind in {"webcam", "video"} and not args.no_motion
    motion = (
        MotionCropper(
            min_area_frac=args.min_motion_area,
            pad_frac=args.motion_pad,
            hold_frames=args.motion_hold,
        )
        if use_motion
        else None
    )
    if use_motion:
        print("Motion ROI enabled (best-effort). Prefer --source scenes for accuracy.")

    if kind == "webcam":
        cap = cv2.VideoCapture(int(value))
        if not cap.isOpened():
            raise SystemExit(f"Could not open webcam index {value}")
    elif kind == "video":
        video_path = Path(str(value))
        if not video_path.is_absolute():
            video_path = root / video_path
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {video_path}")
    elif kind == "scenes":
        species = None
        if args.species:
            species = [s.strip() for s in args.species.split(",") if s.strip()]
        scene_items = load_scene_items(
            raw_dir=root / "data" / "raw",
            manifest_path=root / "data" / "interim" / "crops_manifest.csv",
            processed_dir=root / "data" / "processed",
            limit=args.limit,
            species=species,
            seed=args.seed,
            split=args.split,
        )
        print(
            f"Scenes: {len(scene_items)} full-frame animals from ENA24 "
            f"(split={args.split})"
        )
    else:
        crops_dir = Path(args.crops_dir)
        if not crops_dir.is_absolute():
            crops_dir = root / crops_dir
        crop_paths = list_crop_paths(crops_dir, args.limit)
        print(f"Slideshow: {len(crop_paths)} crops from {crops_dir}")

    window = "TrailCam CNN Demo"
    prev_t = time.perf_counter()
    fps = 0.0
    writer: cv2.VideoWriter | None = None
    frames_done = 0
    show_window = args.max_frames <= 0
    if show_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            if kind == "scenes":
                item = scene_items[item_idx % len(scene_items)]
                item_idx += 1
                frame = cv2.imread(str(item["image_path"]))
                if frame is None:
                    continue
                box = item["box"]
                x1, y1, x2, y2 = box
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                box = (x1, y1, x2, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                label, conf, topk = predictor.predict_crop(crop, top_k=args.top_k)
            elif kind == "crops":
                path = crop_paths[item_idx % len(crop_paths)]
                item_idx += 1
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                box = (0, 0, w, h)
                label, conf, topk = predictor.predict_crop(frame, top_k=args.top_k)
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok or frame is None:
                    if kind == "video":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if motion is not None:
                            motion.reset()
                        continue
                    break

                if motion is not None:
                    box = motion.find_box(frame)
                    if box is None:
                        _, box = center_square_crop_bgr(frame)
                        label, conf, topk = "Waiting for motion", 0.0, []
                    else:
                        label, conf, topk, box = predictor.predict_best_scale(
                            frame, box, top_k=args.top_k
                        )
                else:
                    label, conf, topk, box = predictor.predict_bgr(
                        frame, top_k=args.top_k
                    )

            if label == "Waiting for motion":
                display = label
            elif conf < args.confidence:
                display = "Unknown / low confidence"
            else:
                display = f"{label} ({conf:.2f})"

            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            annotated = draw_overlay(frame, box, display, topk, fps)
            # Normalize resolution so mixed ENA24 image sizes record cleanly.
            out_w, out_h = args.out_width, args.out_height
            if annotated.shape[1] != out_w or annotated.shape[0] != out_h:
                display_frame = cv2.resize(annotated, (out_w, out_h))
            else:
                display_frame = annotated

            if args.record:
                if writer is None:
                    out_path = Path(args.record)
                    if not out_path.is_absolute():
                        out_path = root / out_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_fps = 1000.0 / hold_ms if kind in {"scenes", "crops"} else 15.0
                    out_fps = max(2.0, min(out_fps, 30.0))
                    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
                    if not writer.isOpened():
                        raise SystemExit(f"Could not open VideoWriter for {out_path}")
                    print(f"Recording to {out_path}")
                repeats = max(1, int(round(hold_ms / 1000.0 * 10))) if kind == "scenes" else 1
                for _ in range(repeats):
                    writer.write(display_frame)

            if show_window:
                cv2.imshow(window, display_frame)
                delay = hold_ms if kind in {"crops", "scenes"} else 1
                key = cv2.waitKey(delay) & 0xFF
                if key == ord("q"):
                    break
            elif kind in {"crops", "scenes"}:
                time.sleep(hold_ms / 1000.0)

            frames_done += 1
            if args.max_frames > 0 and frames_done >= args.max_frames:
                print(f"Stopped after {frames_done} frames (--max-frames)")
                break
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="TrailCam CNN live demo")
    parser.add_argument(
        "--source",
        default=None,
        help="scenes | crops | video path | webcam index. Default: scenes if data ready",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(root / "results" / "models" / "cnn_resnet18.pt"),
    )
    parser.add_argument(
        "--labels",
        default=str(root / "results" / "cnn_metrics.json"),
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--crops-dir",
        default=str(root / "data" / "interim" / "crops"),
    )
    parser.add_argument("--limit", type=int, default=200, help="Max scenes/crops")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train", "all"],
        help="Which data split to show for --source scenes (default: test)",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional comma-separated species filter for scenes "
        "(e.g. 'White_Tailed_Deer,Northern Raccoon')",
    )
    parser.add_argument(
        "--hold-ms",
        type=int,
        default=900,
        help="Milliseconds to hold each scene/crop frame",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Optional path to save annotated demo MP4 while running",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until q). Useful for smoke tests.",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Disable motion ROI on video/webcam (use center crop)",
    )
    parser.add_argument(
        "--min-motion-area",
        type=float,
        default=0.002,
        help="Min moving blob area as a fraction of the frame",
    )
    parser.add_argument(
        "--motion-pad",
        type=float,
        default=0.35,
        help="Padding around the motion box before classifying",
    )
    parser.add_argument(
        "--motion-hold",
        type=int,
        default=20,
        help="Keep last motion box for N frames when the animal pauses",
    )
    parser.add_argument("--out-width", type=int, default=1280)
    parser.add_argument("--out-height", type=int, default=720)
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
