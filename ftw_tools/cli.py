import datetime
import enum
import json
import os
from pathlib import Path
from typing import Optional

import click
import wget

from ftw_tools.settings import (
    ALL_COUNTRIES,
    LULC_COLLECTIONS,
    S2_COLLECTIONS,
    SUPPORTED_POLY_FORMATS_TXT,
    TEMPORAL_OPTIONS,
)
from ftw_tools.utils import parse_bbox

# Imports are in the functions below to speed-up CLI startup time
# Some of the ML related imports (presumable torch) are very slow
# See https://github.com/fieldsoftheworld/ftw-baselines/issues/40

COUNTRIES_CHOICE = ALL_COUNTRIES.copy()
COUNTRIES_CHOICE.append("all")


class ModelVersions(enum.StrEnum):
    """Mapping from short_name to .ckpt file in github."""

    TWO_CLASS_CCBY = "2_Class_CCBY_FTW_Pretrained.ckpt"
    TWO_CLASS_FULL = "2_Class_FULL_FTW_Pretrained.ckpt"
    THREE_CLASS_CCBY = "3_Class_CCBY_FTW_Pretrained.ckpt"
    THREE_CLASS_FULL = "3_Class_FULL_FTW_Pretrained.ckpt"


# All commands are meant to use dashes as separator for words.
# All parameters are meant to use underscores as separator for words.


# Common parameter definitions for shared CLI options
def common_bbox_option():
    """Common bbox option for inference commands."""
    return click.option(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box to use for the download in the format 'minx,miny,maxx,maxy'",
        callback=parse_bbox,
    )


def common_year_option():
    """Common year option for inference commands."""
    return click.option(
        "--year",
        type=click.IntRange(min=2015, max=datetime.date.today().year),
        required=True,
        help="Year to run model inference over",
    )


def common_cloud_cover_option():
    """Common cloud cover option for inference commands."""
    return click.option(
        "--cloud_cover_max",
        "-ccx",
        type=click.IntRange(min=0, max=100),
        default=20,
        show_default=True,
        help="Maximum percentage of cloud cover allowed in the Sentinel-2 scene",
    )


def common_buffer_days_option():
    """Common buffer days option for inference commands."""
    return click.option(
        "--buffer_days",
        "-b",
        type=click.IntRange(min=0),
        default=14,
        show_default=True,
        help="Number of days to buffer the date for querying to help balance decreasing cloud cover "
        "and selecting a date near the crop calendar indicated date.",
    )


def common_stac_host_option():
    """Common STAC host option for inference commands."""
    return click.option(
        "--stac_host",
        "-h",
        type=click.Choice(["mspc", "earthsearch"]),
        default="mspc",
        show_default=True,
        help="The host to download the imagery from. mspc = Microsoft Planetary Computer, earthsearch = EarthSearch (Element84/AWS).",
    )


def common_s2_collection_option():
    """Common S2 collection option for inference commands."""
    return click.option(
        "--s2_collection",
        "-s2",
        type=click.Choice(list(S2_COLLECTIONS.keys())),
        default="c1",
        show_default=True,
        help="Sentinel-2 collection to use with EarthSearch only: 'old-baseline' = sentinel-2-l2a, 'c1' = sentinel-2-c1-l2a (default). Ignored when using MSPC.",
    )


def common_verbose_option():
    """Common verbose option for inference commands."""
    return click.option(
        "--verbose",
        "-v",
        is_flag=True,
        default=False,
        show_default=True,
        help="Enable verbose output showing STAC calls, scene details, and download URLs.",
    )


@click.group()
def ftw():
    """Fields of The World (FTW) - Command Line Interface"""
    pass


## Data group


@ftw.group()
def data():
    """Downloading, unpacking, and preparing the FTW dataset."""
    pass


@data.command("download", help="Download and unpack the FTW dataset.")
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default="./data",
    show_default=True,
    help="Folder where the files will be downloaded to.",
)
@click.option(
    "--clean_download",
    "--clean",
    "-f",
    is_flag=True,
    default=False,
    show_default=True,
    help="If set, the script will delete the folder before downloading.",
)
@click.option(
    "--countries",
    type=click.Choice(COUNTRIES_CHOICE, case_sensitive=False),
    default="all",
    show_default=True,
    help="Comma-separated list of countries to download. The default value 'all' downloads all available countries.",
)


@click.option(
    "--no-unpack",  # deprecated
    "--no_unpack",
    is_flag=True,
    default=False,
    show_default=True,
    help="If set, the script will NOT unpack the downloaded files.",
)
def data_download(out, clean_download, countries, no_unpack):
    from ftw_tools.download.download_ftw import download
    from ftw_tools.download.unpack import unpack

    download(out, clean_download, countries)
    if not no_unpack:
        unpack(out)


@data.command(
    "unpack",
    help="Unpack the downloaded FTW dataset. Specify the folder where the data is located via INPUT, which defaults to './data'.",
)
@click.argument(
    "input",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="./data",
    required=False,
)
def data_unpack(input):
    from ftw_tools.download.unpack import unpack

    unpack(input)


### Model group


@ftw.group()
def model():
    """Training and testing FTW models."""
    pass


@model.command("fit", help="Fit the model")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the config file",
)
@click.option(
    "--ckpt_path",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    show_default=True,
    help="Path to a checkpoint file to resume training from",
)
@click.argument(
    "cli_args", nargs=-1, type=click.UNPROCESSED
)  # Capture all remaining arguments
def model_fit(config, ckpt_path, cli_args):
    from ftw_tools.models.baseline_eval import fit

    fit(config, ckpt_path, cli_args)


@model.command("test", help="Test the model")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to model checkpoint",
)
@click.option(
    "--countries",
    "-c",
    type=click.Choice(COUNTRIES_CHOICE, case_sensitive=False),
    multiple=True,
    required=True,
    help="Countries to evaluate on",
)
@click.option(
    "--dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./data/ftw",
    show_default=True,
    help="Directory of the FTW dataset",
)
@click.option(
    "--gpu",
    type=int,
    default=0,
    show_default=True,
    help="GPU to use, zero-based index. Set to -1 to use CPU. CPU is also always used if CUDA is not available.",
)
@click.option(
    "--iou_threshold",
    "-iou",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help="IoU threshold for matching predictions to ground truths",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default="metrics.json",
    show_default=True,
    help="Output file for metrics",
)
@click.option(
    "--model_predicts_3_classes",
    "-p3",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether the model predicts 3 classes or 2 classes (default)",
)
@click.option(
    "--test_on_3_classes",
    "-t3",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to test on 3 classes or 2 classes (default)",
)
@click.option(
    "--temporal_options",
    "-t",
    type=click.Choice(TEMPORAL_OPTIONS),
    default="stacked",
    show_default=True,
    help="Temporal option",
)
@click.option(
    "--swap_order",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to run inference on (window_a, window_b) instead of the default (window_b, window_a).",
)


@click.option(
    "--input_type",
    type=str,
    default="images",
    show_default=True)

@click.option(
    "--backbone",
    type=str,
    default=None,
    show_default=True)

@click.option(
    "--encoder_ckpt_path",
    type=str,
    default=None,
    show_default=True)


@click.option(
    "--feat_root",
    type=str,
    default=None,
    show_default=True)

@click.option(
    "--test_split",
    type=str,
    default="test",
    show_default=True)

def test(
    model,
    backbone,
    test_split,
    dir,
    gpu,
    countries,
    iou_threshold,
    out,
    model_predicts_3_classes,
    test_on_3_classes,
    temporal_options,
    swap_order,
    input_type,
    feat_root,
    encoder_ckpt_path
):
    from ftw_tools.models.baseline_eval import test

    test(
        model_path=model,
        test_split=test_split,
        dir=dir,
        gpu=gpu,
        countries=countries,
        iou_threshold=iou_threshold,
        out=out,
        model_predicts_3_classes=model_predicts_3_classes,
        test_on_3_classes=test_on_3_classes,
        temporal_options=temporal_options,
        swap_order=swap_order,
        input_type=input_type,
        feat_root=feat_root,
        backbone=backbone,
        encoder_ckpt_path=encoder_ckpt_path
    )


@model.command("download", help="Download model checkpoints")
@click.option(
    "--type",
    type=click.Choice(ModelVersions),
    required=True,
    help="Short model name corresponding to a .ckpt file in github.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default=None,
    show_default=True,
    help="File where the file will be stored to. Defaults to the original filename of the selected model.",
)
def model_download(type: ModelVersions, out: Optional[str] = None):
    github_url = f"https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v1/{type.value}"
    target = Path(out or type.value)
    if target.exists():
        print(f"File {target} already exists, skipping download.")
        return

    print(f"Downloading {github_url} to {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    wget.download(github_url, str(target.resolve()))


if __name__ == "__main__":
    ftw()
