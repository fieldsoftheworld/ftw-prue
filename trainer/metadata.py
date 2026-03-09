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

    # For semantic segmentation, we need all classes
    stuff_classes = [x["name"] for x in CUSTOM_CATEGORIES]  # all categories
    stuff_colors = [x["color"] for x in CUSTOM_CATEGORIES]  # all colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    # For things: map dataset ID 1 -> contiguous ID 0
    thing_dataset_id_to_contiguous_id[1] = 0  # ag_field: 1 -> 0

    # For semantic segmentation:
    # Map dataset ID 1 -> contiguous ID 0 (ag_field)
    # Map dataset ID 2 -> contiguous ID 1 (background)
    for i, cat in enumerate(CUSTOM_CATEGORIES):
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta
