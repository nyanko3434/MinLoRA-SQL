"""QLoRA training plumbing: model loading, training prep, dtype guard, completion-only loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from lora_sql.config import ModelConfig


def load_model(model_cfg: ModelConfig) -> nn.Module:
    """nf4: 4-bit base with bf16 compute (embed / lm_head / norms stay bf16).
    none: plain from_pretrained in the config dtype (e.g. the 0.5B in fp32 for smoke tests)."""
    if model_cfg.quant == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=model_cfg.double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_cfg.name_or_path,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
    return AutoModelForCausalLM.from_pretrained(
        model_cfg.name_or_path,
        dtype=getattr(torch, model_cfg.dtype),
        device_map={"": 0},
    )


def prepare_for_training(model: nn.Module, gradient_checkpointing: bool = True) -> nn.Module:
    """Deliberately NOT peft.prepare_model_for_kbit_training: it casts every non-Params4bit bf16
    param to fp32, and on the 7B's untied embed + lm_head that is 4.36 GB. No base param is
    upcast here. No input-require-grads hook either: with use_reentrant=False, gradients reach
    the adapters through checkpointed blocks even though the embeddings don't require grad
    (peft's own source skips the hook in this case)."""
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def assert_training_dtypes(model: nn.Module, base_dtype: torch.dtype) -> None:
    """Only fp32 LoRA params train, and embed / lm_head are still in the base dtype (on the nf4
    path that is bf16, i.e. nothing was upcast to fp32)."""
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no parameter requires grad"
    leaked = [n for n, _ in trainable if "lora_" not in n]
    assert leaked == [], f"non-LoRA params require grad: {leaked}"
    not_fp32 = [n for n, p in trainable if p.dtype != torch.float32]
    assert not_fp32 == [], f"LoRA params not fp32: {not_fp32}"

    for name, module in (("embed_tokens", model.get_input_embeddings()), ("lm_head", model.get_output_embeddings())):
        assert module.weight.dtype == base_dtype, (
            f"{name} is {module.weight.dtype}, expected {base_dtype} (upcast base params?)"
        )


def completion_ce(logits: torch.Tensor, input_ids: torch.Tensor, n_completion: int) -> torch.Tensor:
    """Summed fp32 cross-entropy over the completion. `logits` holds the LAST n_completion + 1
    positions: position i predicts token i + 1, so dropping the final position leaves exactly the
    n_completion predictions for the last n_completion ids."""
    logits = logits[:, :-1].float()
    targets = input_ids[:, -n_completion:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")


def completion_loss(model: nn.Module, input_ids: torch.Tensor, n_completion: int) -> tuple[torch.Tensor, int]:
    """(summed fp32 completion loss, number of loss tokens). Only n_completion + 1 positions of
    logits are materialized; no labels go to HF, so it never builds full-sequence fp32 logits."""
    logits = model(input_ids=input_ids, logits_to_keep=n_completion + 1).logits
    return completion_ce(logits, input_ids, n_completion), n_completion * input_ids.shape[0]
