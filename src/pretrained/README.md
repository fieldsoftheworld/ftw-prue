# Pretrained Models and Feature Encoders

This directory contains pretrained models and feature encoders (backbones) that are used **by** segmentation models, but are not segmenters themselves.

## Structure

- **`gfms/`** - GeoFoundation Models (GFMs) and embedding extraction utilities
  - Scripts for extracting embeddings from various foundation models (Galileo, CROMA, DeCur, DOFA, Prithvi, Satlas, SoftCon)
  - Wrappers for loading and using GFM backbones
  - These are feature encoders that can be used as backbones for segmentation models

- **`models/`** - Other pretrained backbones (if needed)
  - DINOv3, TerraFM, TerraMind, Clay, etc.
  - These are optional backbones that segmentation models might use

## Usage

GFMs and other pretrained encoders are typically used **inside** Segmenter adapters:

```python
# Example: A segmentation model that uses Galileo embeddings
from pretrained.gfms import load_galileo_encoder

class MySegmenter:
    def __init__(self):
        self.backbone = load_galileo_encoder(...)
        self.decoder = MyDecoder(...)
    
    def predict(self, batch):
        features = self.backbone(batch)
        logits = self.decoder(features)
        return SemanticOutput(logits=logits)
```

## Migration from `ftw-prue-main/pretrained`

The extraction scripts and GFM integration code should be copied from:
- `ftw-prue-main/pretrained/models/galileo_benchmark/` → `src/pretrained/gfms/`
- `ftw-prue-main/pretrained/pretrained_factory.py` → `src/pretrained/pretrained_factory.py`

**Note:** `ftw-prue-main/pretrained/` is preferred over `prue-galileo-decode/GFMs/` because it includes:
- The full Galileo repository code
- A factory pattern (`pretrained_factory.py`) for loading encoders
- Other pretrained models (Clay, DINOv3, TerraFM, TerraMind)

## Key Distinction

- **`src/models/`** = Segmentation models (Segmenters) that output `SemanticOutput`, `InstanceOutput`, or `PanopticOutput`
- **`src/pretrained/`** = Feature encoders/backbones that are used **by** segmenters, but don't segment directly

