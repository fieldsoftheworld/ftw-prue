# Pretrained Models

This directory contains implementations and utilities for using various pretrained satellite image encoders for feature extraction on the FTW dataset.

## Available Models

- `clay`, `terrafm`, `dinov3`, `terramind`, `croma`, `decur`, `dofa`, `prithvi`, `satlas`, `softcon`, `galileo`

## Path Configuration

Set environment variables to customize paths:

```bash
export FTW_CKPT_BASE_DIR="/path/to/checkpoints"      # Base directory for model checkpoints
export FTW_DATA_ROOT="/path/to/data"                  # Root directory for FTW dataset
export FTW_METADATA_PATH="/path/to/metadata.yaml"     # Path to metadata YAML file
```

**Defaults**:
- Checkpoint directory: `$WORK_DIR/gfm_ckpts/encoders` or `./gfm_ckpts/encoders`
- Data root: `./data/ftw`
- Metadata: `./configs/metadata.yaml`

Model checkpoints are expected at:
- `{FTW_CKPT_BASE_DIR}/{model_name}.{ext}` for `clay`, `terrafm`, `dinov3`
- `{FTW_CKPT_BASE_DIR}/GALILEO/{model_name}/` for Galileo benchmark models

## Usage

### Loading an Encoder

```python
import torch
from pretrained.pretrained_factory import get_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = get_encoder(model_name="clay", device=device)
```

### Feature Extraction

```python
from pretrained.models.model_utils import (
    get_model_and_preprocess,
    load_image,
    prepare_clay_sample,
    prepare_general_sample
)

device = torch.device("cuda")
encoder, preprocess, gsd, waves = get_model_and_preprocess("clay", device=device)

# For CLAY: requires temporal/spatial metadata
if model_name == "clay":
    sample = prepare_clay_sample(
        image_path="path/to/image.tif",
        preprocess=preprocess,
        gsd=gsd,
        waves=waves
    )
    with torch.no_grad():
        embeddings = encoder(sample)  # [1, L, D]

# For other models: simple tensor input
else:
    image, _, _ = load_image("path/to/image.tif")
    sample = {"image": image}
    sample = preprocess(sample)
    image_tensor = sample["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        embeddings = encoder(image_tensor)  # [1, L, D]
```

### Batch Feature Extraction

```python
from pretrained.models.model_utils import prepare_clay_batch

# For CLAY
batch = prepare_clay_batch(
    image_paths=["path1.tif", "path2.tif"],
    device=device,
    preprocess=preprocess,
    gsd=gsd,
    waves=waves
)
with torch.no_grad():
    embeddings = encoder(batch)  # [B, L, D]

# For other models
images = []
for img_path in batch_paths:
    image, _, _ = load_image(img_path)
    sample = {"image": image}
    sample = preprocess(sample)
    images.append(sample["image"])

batch_tensor = torch.stack(images).to(device)
with torch.no_grad():
    embeddings = encoder(batch_tensor)  # [B, L, D]
```

### Computing Features for Entire Dataset

```bash
python -m pretrained.models.compute_feats \
    --model clay \
    --data_path /path/to/ftw/data \
    --metadata /path/to/metadata.yaml \
    --output_dir /path/to/output \
    --batch_size 32
```

All arguments are optional and use defaults from `path_config` if not provided.

## Input/Output Formats

**Input**: `[B, C, H, W]` tensor
- CLAY: 4-channel `[B, 4, H, W]` + temporal/spatial metadata
- TerraFM/TeraMind/Galileo models: 4-channel `[B, 4, H, W]`
- DINOv3: RGB `[B, 3, H, W]`
- Dual-window: 8-channel `[B, 8, H, W]` (concatenated windows)

**Output**: `[B, L, D]` patch embeddings
- `B`: Batch size
- `L`: Number of patches
- `D`: Embedding dimension (model-specific)

## Model-Specific Notes

- **CLAY**: Requires `metadata.yaml` with GSD and wavelength information
- **DINOv3**: Automatically extracts RGB from 4-channel images
- **Galileo models**: Automatically handle 4-band to 13-band conversion and normalization

## Example Commands

### Compute Features for Dataset

Extract embeddings for all images using a pretrained model:

```bash
python -m pretrained.models.compute_feats --model clay --batch_size 32
```

With custom paths:

```bash
python -m pretrained.models.compute_feats \
    --model terrafm \
    --data_path /path/to/ftw/data \
    --metadata /path/to/metadata.yaml \
    --output_dir /path/to/output \
    --batch_size 32
```

Available models: `clay`, `terrafm`, `dinov3`, `terramind`, `croma`, `decur`, `dofa`, `prithvi`, `satlas`, `softcon`, `galileo`

### Using Python API

```python
import torch
from pretrained.pretrained_factory import get_encoder
from pretrained.models.model_utils import get_model_and_preprocess, load_image, preprocess_general

device = torch.device("cuda")
encoder, preprocess, _, _ = get_model_and_preprocess("terrafm", device=device)

image, _, _ = load_image("path/to/image.tif")
sample = preprocess({"image": image})
image_tensor = sample["image"].unsqueeze(0).to(device)

with torch.no_grad():
    embeddings = encoder(image_tensor)
```
