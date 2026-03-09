"""Tests for settings and constants."""

from ftw_tools.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS


def test_countries_list():
    assert len(ALL_COUNTRIES) == 25
    assert "austria" in ALL_COUNTRIES
    assert "vietnam" in ALL_COUNTRIES
    assert all(c == c.lower() for c in ALL_COUNTRIES)
    assert len(set(ALL_COUNTRIES)) == len(ALL_COUNTRIES), "duplicate countries"


def test_temporal_options():
    assert "stacked" in TEMPORAL_OPTIONS
    assert "sam2" in TEMPORAL_OPTIONS
