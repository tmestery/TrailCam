# Data

Local ENA24 working files for TrailCam. Contents under `raw/`, `interim/`, and `processed/` are **gitignored** (only each folder’s `README.md` is tracked). Rebuild anytime with:

```bash
source .venv/bin/activate
python scripts/prepare_data.py
```

## Layout

```text
data/
├── raw/                      # Full-frame ENA24 download
│   ├── images/               # {image_id}.jpg
│   ├── annotations.jsonl     # One JSON record per image (bboxes, categories)
│   └── categories.json       # Category id → species name
├── interim/                  # Cropped animals
│   ├── crops/{species}/      # {image_id}_{box_idx}.jpg
│   └── crops_manifest.csv    # Path, image_id, species, bbox
└── processed/                # Image-level splits (seed 42)
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── split_summary.json    # Image and crop counts per split
```

## Pipeline

1. **raw** – Download from Hugging Face (`davanstrien/ena24-detection`), save JPEG frames and a COCO-style annotation index.
2. **interim** – Crop each bounding box into a labeled animal image; skip invalid boxes and the empty Human class.
3. **processed** – Assign **original images** (not individual crops) to train / val / test at 70% / 15% / 15% so near-duplicate crops from the same photo stay in one split.

Ratios and seed live in [`configs/default.yaml`](../configs/default.yaml).

## Source and license

- [ENA24 on Hugging Face](https://huggingface.co/datasets/davanstrien/ena24-detection)
- [ENA24 on LILA BC](https://lila.science/datasets/ena24detection/)
- Dataset license: **CDLA Permissive 1.0** (separate from this repo’s MIT license)
