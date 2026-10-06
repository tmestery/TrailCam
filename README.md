# TrailCam

Wildlife species classification from camera-trap imagery, comparing classical ML, a CNN, and a zero-shot vision-language model to evaluate what's practical for offline field deployment.

## Dataset

[ENA24](https://huggingface.co/datasets/davanstrien/ena24-detection) — 8,789 camera-trap images, 9,772 bounding-box annotations, 23 species. Source: [LILA BC](https://lila.science/datasets/ena24detection/).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Layout

```text
TrailCam/
├── configs/                 # Experiment defaults (splits, paths, seed)
├── data/
│   ├── raw/                 # ENA24 download (gitignored)
│   ├── interim/             # Cropped animals (gitignored)
│   └── processed/           # Train/val/test splits (gitignored)
├── docs/
│   ├── deliverables/        # Proposal and course writeups
│   └── related-work/        # Literature notes
├── notebooks/               # Exploration
├── results/                 # Metrics and figures (gitignored)
├── scripts/                 # CLI entrypoints
└── src/trailcam/
    ├── data/                # Download, crop, split
    ├── features/            # Classical color/texture features
    ├── models/              # Perceptron, LR, SVM, CNN, LLM
    ├── eval/                # Accuracy, macro-F1
    └── utils/               # Path helpers
```

## License

MIT. Dataset is licensed separately under CDLA Permissive 1.0.
