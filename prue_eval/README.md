# Source code for evaluation suite

- `src/detections.py`: defines the core `Detections` data structure with mask,
  polygon, and COCO conversion helpers. All scripts serialize/deserialize this
  class, so keeping schema compatibility is critical.
- `src/evaluator.py`: houses the `Evaluator` class that computes pixel/object/
  COCO metrics. `evaluate_by_country.py` instantiates it and passes the masks
  or detections.
- `src/converters.py` and `src/intermediate_formats.py`: adapters that turn
  raw model outputs (SAM, DECODE, etc.) into the unified formats
  used downstream.
- `src/models/`: contains model-specific loading utilities referenced by
  `run_model_inference.py`.

If you need to extend the evaluation logic (e.g., add a new metric), update
`src/` first, then expose the option via the relevant script CLI.