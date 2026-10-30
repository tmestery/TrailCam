# TrailCam CNN Demo

Show the trained ResNet-18 on **full camera-trap scenes**, with the CNN classifying **only the animal box** (highlighted on screen).

## Why `scenes` (not random YouTube clips)

The CNN was trained on **tight ENA24 animal crops**. Random online trail-cam videos are a different domain, and motion detection is not a real animal detector — so labels look wrong.

`--source scenes` uses your ENA24 full images plus the dataset’s annotated boxes (same crop style as training). That is the polished demo path: full frame + box on the animal + accurate species. It defaults to the **held-out test split** so you are not mostly replaying training images (those often hit 1.00 confidence from memorization).

In a deployed system you would swap those boxes for a camera-trap detector such as **MegaDetector**; classification stays the same.

## Prerequisites

```bash
pip install -e .
python scripts/prepare_data.py   # raw images + crops_manifest.csv
python scripts/train_cnn.py      # results/models/cnn_resnet18.pt
```

## Quick start (recommended)

```bash
python demo/live_demo.py
# default is --source scenes when data/ is ready
# press q to quit
```

Filter to flashy species and record a presentation MP4:

```bash
python demo/live_demo.py \
  --source scenes \
  --species "White_Tailed_Deer,Northern Raccoon,American Black Bear,Bobcat" \
  --limit 80 \
  --record demo/output/demo_run.mp4
```

## Other sources

```bash
# already-cropped animals only
python demo/live_demo.py --source crops --limit 200

# external video (motion ROI; accuracy not guaranteed)
python demo/fetch_samples.py
python demo/live_demo.py --source demo/samples/deer_archive.mp4

# webcam
python demo/live_demo.py --source 0
```

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--source` | `scenes` if data ready | `scenes` / `crops` / video path / webcam index |
| `--species` | all | Comma-separated filter for scenes |
| `--confidence` | `0.35` | Below this → `Unknown / low confidence` |
| `--record` | off | Save annotated MP4 while running |

## Presentation tip

Run `--source scenes` with a species filter and `--record` before class, then play `demo/output/demo_run.mp4`. You get full trail-cam frames, a box only on the animal, and labels that match the model.
