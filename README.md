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

Optional notebook deps: `pip install -e ".[dev]"`

## Layout

```text
configs/          Experiment defaults (splits, paths, seed)
data/             Raw / interim crops / processed splits (gitignored)
docs/             Deliverables (proposal, etc.)
notebooks/        Exploration notebooks
results/          Metrics and figures (gitignored)
scripts/          CLI entrypoints (prepare_data, train, evaluate)
src/trailcam/     Package: data, features, models, eval, utils
```

## License

MIT. Dataset is licensed separately under CDLA Permissive 1.0.
