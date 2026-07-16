"""Tests for the layered config loader: shipped base + user override merge."""
from __future__ import annotations

import json

import pytest

from xcell import config as user_config


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Each test starts from a clean module cache."""
    user_config._USER_CONFIG = None
    user_config._CONFIG_PATH = None
    user_config._CONFIG_ERROR = None
    yield
    user_config._USER_CONFIG = None
    user_config._CONFIG_PATH = None
    user_config._CONFIG_ERROR = None


def test_deep_merge_overrides_leaves_and_keeps_siblings():
    base = {"scanpy": {"filter_cells": {"min_genes": 10, "max_genes": 8000}},
            "display": {"point_size": 1, "color_scale": "reds"}}
    override = {"scanpy": {"filter_cells": {"min_genes": 25}}}
    merged = user_config._deep_merge(base, override)
    # overridden leaf wins
    assert merged["scanpy"]["filter_cells"]["min_genes"] == 25
    # sibling leaf from base is retained
    assert merged["scanpy"]["filter_cells"]["max_genes"] == 8000
    # untouched sections from base are retained
    assert merged["display"]["point_size"] == 1
    assert merged["display"]["color_scale"] == "reds"
    # base is not mutated
    assert base["scanpy"]["filter_cells"]["min_genes"] == 10


def test_deep_merge_non_dict_replaces():
    base = {"a": {"b": 1}, "list": [1, 2]}
    override = {"a": 5, "list": [9]}
    merged = user_config._deep_merge(base, override)
    assert merged["a"] == 5
    assert merged["list"] == [9]


def test_shipped_base_loaded_when_no_user_config(monkeypatch, tmp_path):
    # No user override anywhere: env unset, home points to an empty dir.
    monkeypatch.delenv("XCELL_CONFIG_PATH", raising=False)
    monkeypatch.setattr(user_config.Path, "home", staticmethod(lambda: tmp_path))
    cfg = user_config.load_user_config()
    # The shipped config.yaml must exist and provide the requested defaults.
    assert cfg, "shipped config.yaml should be loaded as the base layer"
    assert cfg["display"]["point_size"] == 1
    assert cfg["scanpy"]["filter_cells"]["min_genes"] == 10
    assert cfg["scanpy"]["filter_genes"]["min_cells"] == 10


def test_user_override_deep_merges_over_shipped_base(monkeypatch, tmp_path):
    # A partial user config overrides one key; everything else comes from base.
    override_file = tmp_path / "config.json"
    override_file.write_text(json.dumps({"display": {"point_size": 7}}))
    monkeypatch.setenv("XCELL_CONFIG_PATH", str(override_file))
    cfg = user_config.load_user_config()
    # user key wins
    assert cfg["display"]["point_size"] == 7
    # shipped defaults for other display keys survive the merge
    assert cfg["display"]["color_scale"] == "reds"
    # shipped scanpy defaults untouched by the display-only override
    assert cfg["scanpy"]["filter_cells"]["min_genes"] == 10
