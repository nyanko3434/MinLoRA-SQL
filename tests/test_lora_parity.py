"""Numerical parity of our from-scratch LoRALinear against HuggingFace peft.

Run in fp32 on CUDA: bf16 only carries ~3 significant decimal digits, which can't reach
atol=1e-6. A missing GPU is a broken environment here, not a reason to skip.
"""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear as PeftLoraLinear

from lora_sql.lora import LoRALinear, inject_lora

TARGET_CASES = [
    pytest.param(4, 8, ("q_proj", "v_proj"), id="r4-qv"),
    pytest.param(8, 16, ("q_proj", "v_proj"), id="r8-qv"),
    pytest.param(16, 32, ("q_proj", "v_proj"), id="r16-qv"),
    pytest.param(
        8,
        16,
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        id="r8-all7",
    ),
]


def build_parity_pair(fresh_model, rank, alpha, target_modules):
    """Build matched peft/ours LoRA models with peft's adapter transplanted into ours.

    lora_B is zero-initialized on both sides, so a naive comparison at init is vacuously
    true (base == base). Randomizing peft's lora_B before transplanting makes the adapter
    contribution non-trivial, so forward/backward parity actually exercises the LoRA math.
    """
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
    )
    peft_model = get_peft_model(fresh_model(), cfg).eval()

    our_model = inject_lora(
        fresh_model(),
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=0.0,
    ).eval()

    peft_layers = {
        n.removeprefix("base_model.model."): m
        for n, m in peft_model.named_modules()
        if isinstance(m, PeftLoraLinear)
    }
    mine = {n: m for n, m in our_model.named_modules() if isinstance(m, LoRALinear)}
    assert peft_layers.keys() == mine.keys()

    torch.manual_seed(0)
    for name, pl in peft_layers.items():
        nn.init.normal_(pl.lora_B["default"].weight, std=0.02)
        mine[name].lora_A.data.copy_(pl.lora_A["default"].weight.data)
        mine[name].lora_B.data.copy_(pl.lora_B["default"].weight.data)

    return peft_model, our_model, peft_layers, mine


@pytest.mark.parametrize("rank,alpha,target_modules", TARGET_CASES)
def test_forward_parity(fresh_model, batch, rank, alpha, target_modules):
    peft_model, our_model, _, _ = build_parity_pair(
        fresh_model, rank, alpha, target_modules
    )

    with torch.no_grad():
        peft_logits = peft_model(**batch).logits
        our_logits = our_model(**batch).logits

    assert torch.allclose(peft_logits, our_logits, atol=1e-6)


@pytest.mark.parametrize("rank,alpha,target_modules", TARGET_CASES)
def test_backward_parity(fresh_model, batch, rank, alpha, target_modules):
    peft_model, our_model, peft_layers, mine = build_parity_pair(
        fresh_model, rank, alpha, target_modules
    )

    peft_logits = peft_model(**batch).logits
    torch.manual_seed(1)
    target = torch.randn_like(peft_logits)
    peft_loss = ((peft_logits - target) ** 2).mean()
    peft_loss.backward()

    our_logits = our_model(**batch).logits
    our_loss = ((our_logits - target) ** 2).mean()
    our_loss.backward()

    for name, pl in peft_layers.items():
        m = mine[name]
        assert torch.allclose(
            pl.lora_A["default"].weight.grad, m.lora_A.grad, atol=1e-6
        ), f"lora_A grad mismatch at {name}"
        assert torch.allclose(
            pl.lora_B["default"].weight.grad, m.lora_B.grad, atol=1e-6
        ), f"lora_B grad mismatch at {name}"
