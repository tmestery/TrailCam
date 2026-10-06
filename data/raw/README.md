# raw/

Full-frame ENA24 download. Created by `python scripts/prepare_data.py` (download step). Files here are gitignored.

## Expected contents

| Path | Description |
|------|-------------|
| `images/{image_id}.jpg` | Original camera-trap frames (~8,789 images) |
| `annotations.jsonl` | One JSON line per image: `image_id`, size, bbox list, category ids |
| `categories.json` | Category id → species name |

Source: [ENA24 on Hugging Face](https://huggingface.co/datasets/davanstrien/ena24-detection) (CDLA Permissive 1.0).
