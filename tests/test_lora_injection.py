"""Injection-correctness tests for lora_sql.lora against the real Qwen2.5-0.5B fixture model.

Run in fp32 on CUDA: bf16 only carries ~3 significant decimal digits, which can't support the
exact-equality assertions here (bit-identical logits at init, exact param/layer counts).
"""

import copy

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_sql.lora import LoRALinear, inject_lora, lora_param_stats

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

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


@pytest.fixture(scope="module")
def base_model_and_tokenizer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).cuda().eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def batch(base_model_and_tokenizer):
    _, tokenizer = base_model_and_tokenizer
    return tokenizer("SELECT * FROM singer WHERE age > 30;", return_tensors="pt").to("cuda")


def fresh_model(base_model_and_tokenizer):
    """Deep-copy the pristine module-scoped base model so injection (in-place) can't leak state."""
    base_model, _ = base_model_and_tokenizer
    return copy.deepcopy(base_model)


@pytest.mark.parametrize("target_modules,expected_layers,expected_params", TARGET_SETS)
def test_wrap_count(base_model_and_tokenizer, target_modules, expected_layers, expected_params):
    model = fresh_model(base_model_and_tokenizer)
    inject_lora(model, target_modules=target_modules, rank=8, alpha=16, dropout=0.0)
    stats = lora_param_stats(model)
    assert stats["n_lora_layers"] == expected_layers


@pytest.mark.parametrize("target_modules,expected_layers,expected_params", TARGET_SETS)
def test_param_count(base_model_and_tokenizer, target_modules, expected_layers, expected_params):
    model = fresh_model(base_model_and_tokenizer)
    inject_lora(model, target_modules=target_modules, rank=8, alpha=16, dropout=0.0)
    stats = lora_param_stats(model)
    assert stats["lora"] == expected_params


def test_bit_identical_at_init(base_model_and_tokenizer, batch):
    base_model, _ = base_model_and_tokenizer
    with torch.no_grad():
        base_logits = base_model(**batch).logits.clone()

    model = fresh_model(base_model_and_tokenizer)
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)
    with torch.no_grad():
        new_logits = model(**batch).logits

    assert torch.equal(base_logits, new_logits)


def test_freeze_check(base_model_and_tokenizer):
    model = fresh_model(base_model_and_tokenizer)
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)
    stats = lora_param_stats(model)

    assert stats["trainable"] == stats["lora"]
    assert stats["trainable"] > 0

    leaked = [name for name, p in model.named_parameters() if p.requires_grad and "lora_" not in name]
    assert leaked == [], f"non-LoRA parameters require grad: {leaked}"


def test_device_dtype_placement(base_model_and_tokenizer):
    model = fresh_model(base_model_and_tokenizer)
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)

    checked = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            checked += 1
            expected_dtype = getattr(m.base_layer, "compute_dtype", m.base_layer.weight.dtype)
            assert m.lora_A.device == m.base_layer.weight.device
            assert m.lora_A.dtype == expected_dtype

    assert checked > 0


def test_scaling():
    layer = LoRALinear(nn.Linear(4, 4), r=8, alpha=16)
    assert layer.scaling == 2.0
