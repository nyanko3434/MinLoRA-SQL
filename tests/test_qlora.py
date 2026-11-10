"""Completion-only loss and the QLoRA training path.

The loss tests compare against HF's own `model(ids, labels=...)` loss, with the prompt masked
to -100, on the 0.5B fp32 fixture with randomized lora_B and real training rows. They run in two
tiers:
- Tier 1 runs our CE indexing on full-sequence logits. It must match HF at 1e-6: this verifies
  the shift / alignment.
- Tier 2 runs the real `logits_to_keep` path end to end. Its tolerance is explained in the test.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from conftest import MODEL_NAME

from lora_sql.config import ModelConfig
from lora_sql.data import build_train_example, load_examples
from lora_sql.lora import LoRALinear, inject_lora
from lora_sql.qlora import (
    assert_training_dtypes,
    completion_ce,
    completion_loss,
    load_model,
    prepare_for_training,
)

TRAIN_PATH = Path(__file__).resolve().parents[1] / "data" / "spider_data" / "train_spider.json"


@pytest.fixture(scope="module")
def train_rows(base_model_and_tokenizer):
    if not TRAIN_PATH.exists():
        pytest.skip("Spider is not downloaded to data/spider_data/")
    _, tokenizer = base_model_and_tokenizer
    return [build_train_example(tokenizer, ex) for ex in load_examples(TRAIN_PATH)[:3]]


def _lora_model(fresh_model):
    model = prepare_for_training(fresh_model())
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)
    lora = [m for m in model.modules() if isinstance(m, LoRALinear)]
    torch.manual_seed(0)
    for m in lora:
        nn.init.normal_(m.lora_B, std=0.02)
    return model, lora


def _take_grads(model, lora):
    grads = [t.grad.clone() for m in lora for t in (m.lora_A, m.lora_B)]
    model.zero_grad(set_to_none=True)
    return grads


def _hf_loss(model, x, n):
    labels = x.clone()
    labels[:, :-n] = -100
    return model(input_ids=x, labels=labels).loss


def test_completion_ce_alignment_matches_hf(fresh_model, train_rows):
    model, lora = _lora_model(fresh_model)
    for ids, n in train_rows:
        x = torch.tensor([ids], device="cuda")

        hf = _hf_loss(model, x, n)
        hf.backward()
        hf_grads = _take_grads(model, lora)

        full_logits = model(input_ids=x).logits
        ours = completion_ce(full_logits[:, -(n + 1):], x, n) / n
        ours.backward()
        our_grads = _take_grads(model, lora)

        assert torch.allclose(ours, hf, atol=1e-6), (ours.item(), hf.item())
        for g_ours, g_hf in zip(our_grads, hf_grads):
            assert torch.allclose(g_ours, g_hf, atol=1e-6)

        # Negative control: a one-token misalignment must be glaring, not within tolerance.
        with torch.no_grad():
            shifted = completion_ce(full_logits[:, -(n + 2):-1], x, n) / n
        assert (shifted - hf).abs() > 1.0, "off-by-one is indistinguishable; the check is vacuous"


def test_completion_loss_matches_hf(fresh_model, train_rows):
    # Tolerance is NOT 1e-6, by mechanism. With logits_to_keep=n+1, lm_head runs on n+1 rows
    # instead of the full sequence, and cuBLAS picks a different fp32 GEMM kernel with a
    # different reduction order over hidden=896. Kept logits differ from the full-logit slice
    # by up to 2.5e-5 at |logit|~32 (a few ulps). Measured vs HF: loss <= 2.7e-6, LoRA grads
    # <= 5.5e-6 abs (grads up to ~2.7). Tier 1 shows the same CE code on full logits matches
    # HF at 1e-6 with bit-identical grads. A one-token misalignment shifts the loss by >10.
    model, lora = _lora_model(fresh_model)
    for ids, n in train_rows:
        x = torch.tensor([ids], device="cuda")

        hf = _hf_loss(model, x, n)
        hf.backward()
        hf_grads = _take_grads(model, lora)

        loss_sum, n_tokens = completion_loss(model, x, n)
        assert n_tokens == n
        assert loss_sum.dtype == torch.float32
        ours = loss_sum / n_tokens
        ours.backward()
        our_grads = _take_grads(model, lora)

        print(f"\n[len={len(ids)} n={n}] |loss diff| = {abs(ours.item() - hf.item()):.2e}")
        assert torch.allclose(ours, hf, atol=1e-5), (ours.item(), hf.item())
        for g_ours, g_hf in zip(our_grads, hf_grads):
            assert torch.allclose(g_ours, g_hf, rtol=1e-5, atol=1e-5)


def _nf4_model():
    cfg = ModelConfig(
        name_or_path=MODEL_NAME, quant="nf4", double_quant=True, dtype="bfloat16", gradient_checkpointing=True
    )
    model = prepare_for_training(load_model(cfg))
    return inject_lora(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)


def test_nf4_training_path_dtypes_and_grads(train_rows):
    model = _nf4_model()
    assert_training_dtypes(model, torch.bfloat16)

    lora = [m for m in model.modules() if isinstance(m, LoRALinear)]
    torch.manual_seed(0)
    for m in lora:
        nn.init.normal_(m.lora_B, std=0.02)

    ids, n = train_rows[0]
    loss_sum, n_tokens = completion_loss(model, torch.tensor([ids], device="cuda"), n)
    (loss_sum / n_tokens).backward()

    # Non-reentrant checkpointing on a 4-bit base: grads reach the adapters with no
    # input-require-grads hook.
    for m in lora:
        for t in (m.lora_A, m.lora_B):
            assert t.grad is not None and t.grad.dtype == torch.float32
            assert t.grad.abs().sum() > 0


def test_assert_training_dtypes_rejects_leak(fresh_model):
    model = inject_lora(fresh_model(), target_modules=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.0)
    assert_training_dtypes(model, torch.float32)

    model.get_input_embeddings().weight.requires_grad_(True)
    with pytest.raises(AssertionError, match="non-LoRA params require grad"):
        assert_training_dtypes(model, torch.float32)
