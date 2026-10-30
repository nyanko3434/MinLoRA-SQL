"""LoRA checkpoint save / load on the 0.5B fp32 fixture: a roundtrip is bit-exact, and every way
strict=False could silently load nothing (prefix mismatch, wrong rank, zero adapter, wrong base)
raises instead."""

import json
import struct

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from lora_sql.checkpoint import ADAPTER_FILE, SIDECAR_FILE, load_lora, save_lora
from lora_sql.lora import LoRALinear, inject_lora

META = {"step": 0, "config_sha256": "test"}


def _saved(fresh_model, tmp_path, randomize=True):
    model = inject_lora(fresh_model(), target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)
    if randomize:
        torch.manual_seed(0)
        for m in model.modules():
            if isinstance(m, LoRALinear):
                nn.init.normal_(m.lora_B, std=0.02)
    return model, save_lora(model, tmp_path / "ckpt", META)


def _edit_sidecar(ckpt, **changes):
    path = ckpt / SIDECAR_FILE
    sidecar = json.loads(path.read_text())
    sidecar.update(changes)
    path.write_text(json.dumps(sidecar))


def test_roundtrip_logits_equal(fresh_model, batch, tmp_path):
    model, ckpt = _saved(fresh_model, tmp_path)
    loaded = load_lora(fresh_model(), ckpt)

    model.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.equal(model(**batch).logits, loaded(**batch).logits)


def test_file_has_only_lora_keys_and_expected_size(fresh_model, tmp_path):
    _, ckpt = _saved(fresh_model, tmp_path)
    sidecar = json.loads((ckpt / SIDECAR_FILE).read_text())

    with safe_open(ckpt / ADAPTER_FILE, framework="pt") as f:
        keys = list(f.keys())
    assert keys and all("lora_" in k for k in keys)
    assert len(keys) == sidecar["n_keys"]

    # safetensors layout: u64 header length, JSON header, raw tensor bytes (fp32 = 4 bytes each).
    data = (ckpt / ADAPTER_FILE).read_bytes()
    (header_len,) = struct.unpack("<Q", data[:8])
    assert header_len < 64 * 1024
    assert len(data) == 8 + header_len + sidecar["n_params"] * 4


def test_prefix_trap_raises(fresh_model, tmp_path):
    _, ckpt = _saved(fresh_model, tmp_path)
    state = load_file(ckpt / ADAPTER_FILE)
    save_file({f"base_model.model.{k}": v for k, v in state.items()}, ckpt / ADAPTER_FILE)
    with pytest.raises(AssertionError, match="unexpected"):
        load_lora(fresh_model(), ckpt)


def test_rank_mismatch_raises(fresh_model, tmp_path):
    _, ckpt = _saved(fresh_model, tmp_path)
    _edit_sidecar(ckpt, rank=4)
    with pytest.raises(AssertionError, match="don't match"):
        load_lora(fresh_model(), ckpt)


def test_zero_lora_B_raises(fresh_model, tmp_path):
    _, ckpt = _saved(fresh_model, tmp_path, randomize=False)
    with pytest.raises(AssertionError, match="no-op"):
        load_lora(fresh_model(), ckpt)


def test_base_model_mismatch_raises(fresh_model, tmp_path):
    _, ckpt = _saved(fresh_model, tmp_path)
    _edit_sidecar(ckpt, base_model="Qwen/Qwen2.5-Coder-7B-Instruct")
    with pytest.raises(AssertionError, match="trained on"):
        load_lora(fresh_model(), ckpt)
