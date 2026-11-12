"""Strict config loading: all train configs parse, ablations differ only where they should, and
the held-out rule (no dev/test file as training data) is enforced in code. CPU-only; no data
file is ever read here."""

import copy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from lora_sql.config import config_from_dict, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "train"
REFERENCE = "qv_r8"
ALL_CONFIGS = ["qv_r8", "qv_r4", "qv_r16", "attn_r8", "all7_r8"]
ABLATIONS = [n for n in ALL_CONFIGS if n != REFERENCE]
# Seed repeats: name -> (the config it repeats, its seed). They are not ablations: nothing but
# the seed changes.
SEED_REPEATS = {"qv_r16_s1": ("qv_r16", 1), "qv_r16_s2": ("qv_r16", 2)}


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{prefix}{k}."))
        else:
            flat[f"{prefix}{k}"] = v
    return flat


def _reference_dict() -> dict:
    with open(CONFIG_DIR / f"{REFERENCE}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("name", [*ALL_CONFIGS, *SEED_REPEATS])
def test_config_parses(name):
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    assert cfg.run_name == name


@pytest.mark.parametrize("name", ABLATIONS)
def test_ablation_differs_only_in_lora_and_names(name):
    ref = _flatten(asdict(load_config(CONFIG_DIR / f"{REFERENCE}.yaml")))
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    flat = _flatten(asdict(cfg))

    assert flat.keys() == ref.keys()
    differing = {k for k in flat if flat[k] != ref[k]}
    allowed = {"run_name", "train.out_dir"} | {k for k in flat if k.startswith("lora.")}
    assert differing <= allowed, f"unexpected differences: {sorted(differing - allowed)}"
    assert cfg.lora.alpha == 2 * cfg.lora.rank
    assert cfg.train.out_dir == f"runs/{cfg.run_name}"


@pytest.mark.parametrize("name", SEED_REPEATS)
def test_seed_repeat_differs_only_in_seed_and_names(name):
    parent, seed = SEED_REPEATS[name]
    ref = _flatten(asdict(load_config(CONFIG_DIR / f"{parent}.yaml")))
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    flat = _flatten(asdict(cfg))

    assert flat.keys() == ref.keys()
    differing = {k for k in flat if flat[k] != ref[k]}
    assert differing == {"seed", "run_name", "train.out_dir"}, f"differs in {sorted(differing)}"
    assert cfg.seed == seed != ref["seed"]
    assert cfg.train.out_dir == f"runs/{cfg.run_name}"


@pytest.mark.parametrize("held_out", ["dev.json", "test.json"])
def test_rejects_non_train_file(held_out):
    d = _reference_dict()
    d["data"]["train_files"] = ["train_spider.json", held_out]
    with pytest.raises(ValueError, match="train_files"):
        config_from_dict(d)


def test_rejects_unknown_key():
    d = _reference_dict()
    d["train"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="unknown config key.*train.learning_rate"):
        config_from_dict(d)


def test_rejects_missing_key():
    d = _reference_dict()
    del d["lora"]["dropout"]
    with pytest.raises(ValueError, match="missing config key.*lora.dropout"):
        config_from_dict(d)


def test_coerces_string_float():
    d = copy.deepcopy(_reference_dict())
    d["train"]["lr"] = "2e-4"  # what YAML 1.1 gives for an exponent with no dot
    cfg = config_from_dict(d)
    assert isinstance(cfg.train.lr, float) and cfg.train.lr == 2e-4
