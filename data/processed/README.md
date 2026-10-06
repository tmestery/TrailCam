# processed/

Train / validation / test manifests after the image-level split. Created by `python scripts/prepare_data.py` (split step). Files here are gitignored.

## Expected contents

| Path | Description |
|------|-------------|
| `train.csv` | Crop rows assigned to training (~70%) |
| `val.csv` | Crop rows assigned to validation (~15%) |
| `test.csv` | Crop rows assigned to testing (~15%) |
| `split_summary.json` | Image and crop counts per split |

Splits are assigned by **original `image_id`** (seed 42 in `configs/default.yaml`) so all crops from the same photo stay in one fold.
