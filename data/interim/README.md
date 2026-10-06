# interim/

Cropped animal patches from ENA24 bounding boxes. Created by `python scripts/prepare_data.py` (crop step). Files here are gitignored.

## Expected contents

| Path | Description |
|------|-------------|
| `crops/{species_slug}/` | One JPEG crop per box, named `{image_id}_{box_idx}.jpg` |
| `crops_manifest.csv` | Index of all crops: path, `image_id`, species, category id, bbox |

Each crop is a labeled species image used for classical features, CNN training, and zero-shot VLM evaluation.
