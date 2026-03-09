"""Tests for utility functions."""

import pytest
import pandas as pd

from ftw_tools.utils import harvest_to_datetime, parse_bbox, compute_md5


def test_harvest_to_datetime():
    result = harvest_to_datetime(1, 2024)
    assert result == pd.Timestamp("2024-01-01")

    result = harvest_to_datetime(365, 2024)
    assert result == pd.Timestamp("2024-12-30")


def test_compute_md5_missing_file():
    assert compute_md5("/nonexistent/path/file.txt") is None


def test_parse_bbox_valid():
    result = parse_bbox(None, None, "1.0,2.0,3.0,4.0")
    assert result == [1.0, 2.0, 3.0, 4.0]


def test_parse_bbox_none():
    assert parse_bbox(None, None, None) is None


def test_parse_bbox_invalid_count():
    with pytest.raises(Exception):
        parse_bbox(None, None, "1.0,2.0")


def test_parse_bbox_invalid_value():
    with pytest.raises(Exception):
        parse_bbox(None, None, "a,b,c,d")
