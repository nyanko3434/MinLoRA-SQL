"""Injection-correctness tests for lora_sql.lora against the real Qwen2.5-0.5B fixture model.

Run in fp32 on CUDA: bf16 only carries ~3 significant decimal digits, which can't support the
exact-equality assertions here (bit-identical logits at init, exact param/layer counts).
"""

import pytest
import torch
import torch.nn as nn

from lora_sql.lora import LoRALinear, inject_lora, lora_param_stats, set_lora_scale

TARGET_SETS = [
    pytest.param(("q_proj", "v_proj"), 48, 540_672, id="qv"),
    pytest.param(("q_proj", "k_proj", "v_proj", "o_proj"), 96, 1_081_344, id="qkvo"),
    pytest.param(
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        168,
        4_399_104,
        id="all7",
    ),
]


@pytest.mark.parametrize("target_modules,expected_layers,expected_params", TARGET_SETS)
def test_wrap_count(fresh_model, target_modules, expected_layers, expected_params):
    model = fresh_model()
    inject_lora(model, target_modules=target_modules, rank=8, alpha=16, dropout=0.0)
    stats = lora_param_stats(model)
    assert stats["n_lora_layers"] == expected_layers


@pytest.mark.parametrize("target_modules,expected_layers,expected_params", TARGET_SETS)
def test_param_count(fresh_model, target_modules, expected_layers, expected_params):
    model = fresh_model()
    inject_lora(model, target_modules=target_modules, rank=8, alpha=16, dropout=0.0)
    stats = lora_param_stats(model)
    assert stats["lora"] == expected_params


def test_bit_identical_at_init(base_model_and_tokenizer, fresh_model, batch):
    base_model, _ = base_model_and_tokenizer
    with torch.no_grad():
        base_logits = base_model(**batch).logits.clone()

    model = fresh_model()
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)
    with torch.no_grad():
        new_logits = model(**batch).logits

    assert torch.equal(base_logits, new_logits)


def test_freeze_check(fresh_model):
    model = fresh_model()
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)
    stats = lora_param_stats(model)

    assert stats["trainable"] == stats["lora"]
    assert stats["trainable"] > 0

    leaked = [name for name, p in model.named_parameters() if p.requires_grad and "lora_" not in name]
    assert leaked == [], f"non-LoRA parameters require grad: {leaked}"


def test_device_dtype_placement(fresh_model):
    model = fresh_model()
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)

    checked = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            checked += 1
            # Adapters are fp32 regardless of base dtype (peft's autocast_adapter_dtype=True).
            for adapter in (m.lora_A, m.lora_B):
                assert adapter.device == m.base_layer.weight.device
                assert adapter.dtype == torch.float32

    assert checked > 0


def test_scaling():
    layer = LoRALinear(nn.Linear(4, 4), r=8, alpha=16)
    assert layer.scaling == 2.0


def test_set_lora_scale_interpolates_between_base_and_full():
    torch.manual_seed(0)
    # alpha / r = 1.5: not a power of two, so a wrong multiplication order would show.
    model = nn.Sequential(LoRALinear(nn.Linear(6, 5), r=8, alpha=12), nn.Tanh(),
                          LoRALinear(nn.Linear(5, 3), r=8, alpha=12))
    for m in model:
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.5)  # zero-init B would make every scale equal the base
    first = model[0]
    x = torch.randn(4, 6)
    with torch.no_grad():
        base, full = first.base_layer(x), first(x)
        assert not torch.allclose(base, full)
        for factor in (0.5, 0.25):
            assert set_lora_scale(model, factor) == 2
            assert all(m.scaling == 1.5 * factor for m in model if isinstance(m, LoRALinear))
            assert first.alpha == 12  # the adapter is still the one its sidecar describes
            assert torch.allclose(first(x), base + factor * (full - base), atol=1e-6)
        set_lora_scale(model, 0.5)
        set_lora_scale(model, 0.5)  # absolute, not cumulative
        assert first.scaling == 0.75
        set_lora_scale(model, 1.0)
        assert torch.equal(first(x), full)


@pytest.mark.parametrize("factor", [0.0, -0.5, float("inf"), float("nan")])
def test_set_lora_scale_rejects_bad_factors(factor):
    with pytest.raises(ValueError, match="finite number > 0"):
        set_lora_scale(nn.Sequential(LoRALinear(nn.Linear(4, 4))), factor)


def test_set_lora_scale_needs_adapters():
    with pytest.raises(ValueError, match="no LoRALinear"):
        set_lora_scale(nn.Sequential(nn.Linear(4, 4)), 0.5)
