# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class COCOPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                # segments_info = [self._convert_category_id(x) for x in segments_info] # naughty naughty
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def evaluate(self):
        comm.synchronize()

        # import pdb; pdb.set_trace()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute
            with contextlib.redirect_stdout(io.StringIO()):
                # import pdb; pdb.set_trace()
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


class FilteredCOCOPanopticEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }
        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)
        self.logger = logging.getLogger(__name__)
        
    def reset(self):
        self._predictions = []
        self._processed_image_ids = set()  # Track which images were successfully processed
        self._skipped_image_ids = set()  # Track which images were skipped
        
    def _validate_prediction(self, panoptic_img, segments_info):
        """Validate that all segments in segments_info appear in the panoptic image."""
        if segments_info is None:
            return False
            
        unique_ids = set(np.unique(panoptic_img))
        for segment in segments_info:
            if segment['id'] not in unique_ids:
                return False
        return True
        
    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb
        
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            try:
                panoptic_img, segments_info = output["panoptic_seg"]
                panoptic_img = panoptic_img.cpu().numpy()
                
                # Skip if validation fails
                if not self._validate_prediction(panoptic_img, segments_info):
                    self._skipped_image_ids.add(image_id)
                    self.logger.info(f"Skipping image {image_id} due to invalid prediction")
                    continue
                    
                file_name = os.path.basename(input["file_name"])
                file_name_png = os.path.splitext(file_name)[0] + ".png"
                
                with io.BytesIO() as out:
                    Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                    self._predictions.append({
                        "image_id": image_id,
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    })
                self._processed_image_ids.add(image_id)
                
            except Exception as e:
                self._skipped_image_ids.add(image_id)
                self.logger.warning(f"Error processing image {image_id}: {str(e)}")
                continue
                
    def evaluate(self):
        """Evaluate using only the successfully processed images."""
        comm.synchronize()
        
        # Gather predictions and processed/skipped image IDs from all workers
        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        
        processed_ids = comm.gather(self._processed_image_ids)
        processed_ids = set().union(*processed_ids)
        
        skipped_ids = comm.gather(self._skipped_image_ids)
        skipped_ids = set().union(*skipped_ids)
        
        if not comm.is_main_process():
            return
            
        self.logger.info(f"Successfully processed {len(processed_ids)} images")
        self.logger.info(f"Skipped {len(skipped_ids)} images")
        
        # Read and filter ground truth
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)
        
        with open(gt_json, 'r') as f:
            gt_data = json.load(f)
            
        # Filter images and annotations in ground truth
        gt_data['images'] = [img for img in gt_data['images'] 
                           if img['id'] in processed_ids]
        gt_data['annotations'] = [ann for ann in gt_data['annotations'] 
                                if ann['image_id'] in processed_ids]
        
        # Create temporary directory for evaluation
        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            self.logger.info(f"Writing filtered predictions to {pred_dir}")
            
            # Write prediction PNGs
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))
            
            # Create filtered predictions JSON
            filtered_gt_json = os.path.join(pred_dir, "filtered_gt.json")
            with open(filtered_gt_json, 'w') as f:
                json.dump(gt_data, f)
            
            # Create predictions JSON
            pred_data = gt_data.copy()
            pred_data["annotations"] = self._predictions
            predictions_json = os.path.join(pred_dir, "predictions.json")
            with open(predictions_json, 'w') as f:
                json.dump(pred_data, f)
            
            # Run evaluation on filtered dataset
            try:
                from panopticapi.evaluation import pq_compute
                with contextlib.redirect_stdout(io.StringIO()):
                    pq_res = pq_compute(
                        filtered_gt_json,
                        predictions_json,
                        gt_folder=gt_folder,
                        pred_folder=pred_dir
                    )
            except Exception as e:
                self.logger.error(f"Evaluation failed: {str(e)}")
                return None
        
        # Format results
        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
        
        # Add information about dataset coverage
        res["images_evaluated"] = len(processed_ids)
        res["images_skipped"] = len(skipped_ids)
        res["dataset_coverage"] = len(processed_ids) / (len(processed_ids) + len(skipped_ids))
        
        return OrderedDict({"panoptic_seg": res})

if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
