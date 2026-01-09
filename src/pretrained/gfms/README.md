# GeoFoundation Models (GFMs)

This directory contains utilities for working with GeoFoundation Models - pretrained feature encoders for geospatial data.

## Purpose

GFMs are **feature encoders**, not segmentation models. They extract rich embeddings from satellite imagery that can then be used as inputs to segmentation heads (like DECODE or custom decoders).

## Structure

- **Extraction scripts**: Scripts to extract embeddings from GFMs for the FTW dataset
  - `extract_ftw_galileo_emb.py`
  - `extract_ftw_croma_emb.py`
  - `extract_ftw_decur_emb.py`
  - `extract_ftw_dofa_emb.py`
  - `extract_ftw_prithvi_emb.py`
  - `extract_ftw_satlas_emb.py`
  - `extract_ftw_softcon_emb.py`

- **Model wrappers**: Python modules for loading and using GFM backbones
  - These can be imported by segmentation models that want to use GFM features

## Migration from `prue-galileo-decode/GFMs`

The extraction scripts and GFM integration code should be copied from:
- `prue-galileo-decode/GFMs/` → `src/pretrained/gfms/`

## Usage in Segmentation Models

GFMs are typically used as backbones **inside** Segmenter adapters:

```python
# Example: DECODE model using Galileo embeddings
from pretrained.gfms import load_galileo_encoder

class DecodeWithGalileoSegmenter:
    def __init__(self, decode_weights, galileo_weights):
        self.galileo_encoder = load_galileo_encoder(galileo_weights)
        self.decode_head = load_decode_head(decode_weights)
    
    def predict(self, batch):
        # Extract features with Galileo
        features = self.galileo_encoder(batch)
        # Decode with DECODE head
        logits = self.decode_head(features)
        return SemanticOutput(logits=logits)
```

## Note

GFMs are **optional** - segmentation models can work without them. They're a way to leverage pretrained geospatial foundation models as feature extractors.

