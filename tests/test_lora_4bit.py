"""The 4-bit (QLoRA) path: LoRALinear over bitsandbytes Linear4bit, and parity against peft.

Base is nf4 with bf16 compute; adapters are fp32. Our forward mirrors peft's bnb Linear4bit cast
order exactly (x cast to the adapter dtype before dropout, scaling applied in fp32 before the
downcast), so parity is asserted with torch.equal, not a tolerance. Run with -s to see the max abs
diffs that the README quotes.
"""

import bitsandbytes as bnb
import pytest
import torch
import torch.nn as nn
from conftest import MODEL_NAME, requires_gpu
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.bnb import Linear4bit as PeftLoraLinear4bit
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from lora_sql.lora import LoRALinear, inject_lora

pytestmark = requires_gpu

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# alpha=12 gives scaling=1.5. At scaling=2.0 (a power of two) multiplying by scaling is exact
# in bf16, so applying it before vs after the downcast is indistinguishable and the cast-order
# claim would go untested: the pre-Session-5 order (scale after downcast) still passes r8-qv,
# but differs from peft by 0.32 in the logits at alpha=12.
QLORA_CASES = [
    pytest.param(16, ("q_proj", "v_proj"), id="r8-qv"),
    pytest.param(12, ("q_proj", "v_proj"), id="r8-a12-qv"),
    pytest.param(
        16,
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        id="r8-all7",
    ),
]


def make_base_layer():
    torch.manual_seed(0)
    # Quantization happens on the move to CUDA.
    return bnb.nn.Linear4bit(256, 128, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()


def make_input():
    torch.manual_seed(2)
    return torch.randn(2, 5, 256, device="cuda", dtype=torch.bfloat16)


def load_4bit():
    # Always a fresh from_pretrained: Params4bit does not survive copy.deepcopy safely.
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=BNB_CONFIG, device_map={"": 0}
    ).eval()


def test_reads_logical_dims():
    base = make_base_layer()
    assert base.weight.shape != (128, 256), "premise: Linear4bit weight should be packed"

    wrapped = LoRALinear(base, r=8, alpha=16)
    assert wrapped.lora_A.shape == (8, 256)
    assert wrapped.lora_B.shape == (128, 8)


def test_adapter_and_output_dtypes():
    wrapped = LoRALinear(make_base_layer(), r=8, alpha=16)
    assert wrapped.lora_A.dtype == torch.float32
    assert wrapped.lora_B.dtype == torch.float32

    with torch.no_grad():
        assert wrapped(make_input()).dtype == torch.bfloat16


def test_identity_at_init():
    base = make_base_layer()
    wrapped = LoRALinear(base, r=8, alpha=16)
    x = make_input()
    with torch.no_grad():
        assert torch.equal(wrapped(x), base(x))


def test_backward_grads():
    wrapped = LoRALinear(make_base_layer(), r=8, alpha=16)
    torch.manual_seed(0)
    nn.init.normal_(wrapped.lora_B, std=0.02)

    wrapped(make_input()).float().pow(2).mean().backward()

    for name, adapter in (("lora_A", wrapped.lora_A), ("lora_B", wrapped.lora_B)):
        assert adapter.grad is not None, f"{name} has no grad"
        assert adapter.grad.dtype == torch.float32
        assert adapter.grad.abs().sum() > 0, f"{name} grad is all zero"

    with_grad = [n for n, p in wrapped.base_layer.named_parameters() if p.grad is not None]
    assert with_grad == [], f"base params received grad: {with_grad}"


def build_qlora_pair(batch, alpha, target_modules):
    """Build matched 4-bit peft/ours models with peft's adapter transplanted into ours.

    The two models come from two separate loads, so first check that those loads are
    themselves bit-identical. lora_B is randomized before transplanting so the comparison
    exercises the LoRA math instead of comparing base against base.
    """
    peft_base = load_4bit()
    our_base = load_4bit()
    with torch.no_grad():
        base_logits = peft_base(**batch).logits.clone()
        assert torch.equal(base_logits, our_base(**batch).logits), (
            "two 4-bit loads of the same checkpoint are not bit-identical; "
            "parity would be meaningless"
        )

    cfg = LoraConfig(
        r=8,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
    )
    peft_model = get_peft_model(peft_base, cfg).eval()
    our_model = inject_lora(
        our_base, target_modules=target_modules, rank=8, alpha=alpha, dropout=0.0
    ).eval()

    peft_layers = {
        n.removeprefix("base_model.model."): m
        for n, m in peft_model.named_modules()
        if isinstance(m, PeftLoraLinear4bit)
    }
    mine = {n: m for n, m in our_model.named_modules() if isinstance(m, LoRALinear)}
    assert peft_layers.keys() == mine.keys()

    for name, pl in peft_layers.items():
        assert pl.lora_A["default"].weight.dtype == torch.float32, f"premise: peft lora_A not fp32 at {name}"
        assert pl.lora_B["default"].weight.dtype == torch.float32, f"premise: peft lora_B not fp32 at {name}"

    torch.manual_seed(0)
    for name, pl in peft_layers.items():
        nn.init.normal_(pl.lora_B["default"].weight, std=0.02)
        mine[name].lora_A.data.copy_(pl.lora_A["default"].weight.data)
        mine[name].lora_B.data.copy_(pl.lora_B["default"].weight.data)

    return peft_model, our_model, peft_layers, mine, base_logits


@pytest.mark.parametrize("alpha,target_modules", QLORA_CASES)
def test_qlora_forward_parity(request, batch, alpha, target_modules):
    peft_model, our_model, _, _, base_logits = build_qlora_pair(batch, alpha, target_modules)

    with torch.no_grad():
        peft_logits = peft_model(**batch).logits
        our_logits = our_model(**batch).logits

    max_diff = (peft_logits.float() - our_logits.float()).abs().max().item()
    print(f"\n[{request.node.callspec.id}] logits max abs diff: {max_diff:.3e}")

    assert not torch.equal(our_logits, base_logits), "adapters had no effect; parity is vacuous"
    assert torch.equal(peft_logits, our_logits)


@pytest.mark.parametrize("alpha,target_modules", QLORA_CASES)
def test_qlora_backward_parity(request, batch, alpha, target_modules):
    peft_model, our_model, peft_layers, mine, _ = build_qlora_pair(batch, alpha, target_modules)

    peft_logits = peft_model(**batch).logits
    torch.manual_seed(1)
    # Logits are bf16 on the 4-bit path; take the MSE in fp32 on both sides.
    target = torch.randn(peft_logits.shape, device="cuda")
    ((peft_logits.float() - target) ** 2).mean().backward()

    our_logits = our_model(**batch).logits
    ((our_logits.float() - target) ** 2).mean().backward()

    max_diff = 0.0
    for name, pl in peft_layers.items():
        m = mine[name]
        for which, peft_grad, our_grad in (
            ("lora_A", pl.lora_A["default"].weight.grad, m.lora_A.grad),
            ("lora_B", pl.lora_B["default"].weight.grad, m.lora_B.grad),
        ):
            max_diff = max(max_diff, (peft_grad - our_grad).abs().max().item())
            assert torch.equal(peft_grad, our_grad), f"{which} grad mismatch at {name}"
    print(f"\n[{request.node.callspec.id}] lora grad max abs diff: {max_diff:.3e}")

    with_grad = [n for n, p in our_model.named_parameters() if p.grad is not None and "lora_" not in n]
    assert with_grad == [], f"non-LoRA params received grad: {with_grad}"
