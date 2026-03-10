"""
FTW panoptic metadata for Detectron2/Mask2Former.
Matches ag-seg trainer/metadata.py and ref_CATEGORY_ID_GUIDELINES.md:
- thing_dataset_id_to_contiguous_id[1] = 0 (ag_field)
- stuff_dataset_id_to_contiguous_id for both ag_field and background.
"""

CUSTOM_CATEGORIES = [
    {"color": [100, 204, 25], "isthing": 1, "id": 1, "name": "ag_field"},
    {"color": [153, 30, 76], "isthing": 0, "id": 2, "name": "background"},
]


def get_metadata():
    meta = {}
    thing_classes = [x["name"] for x in CUSTOM_CATEGORIES if x["isthing"] == 1]
    thing_colors = [x["color"] for x in CUSTOM_CATEGORIES if x["isthing"] == 1]
    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    stuff_classes = [x["name"] for x in CUSTOM_CATEGORIES]
    stuff_colors = [x["color"] for x in CUSTOM_CATEGORIES]
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors
    thing_dataset_id_to_contiguous_id = {1: 0}  # ag_field: 1 -> 0
    stuff_dataset_id_to_contiguous_id = {cat["id"]: i for i, cat in enumerate(CUSTOM_CATEGORIES)}
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    return meta
