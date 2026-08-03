import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def base_model_and_tokenizer():
    assert torch.cuda.is_available(), "CUDA not available: parity tests require a real GPU"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).cuda().eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def batch(base_model_and_tokenizer):
    _, tokenizer = base_model_and_tokenizer
    return tokenizer("SELECT * FROM singer WHERE age > 30;", return_tensors="pt").to("cuda")


def fresh_model(base_model_and_tokenizer):
    """Deep-copy the pristine module-scoped base model so injection (in-place) can't leak state."""
    base_model, _ = base_model_and_tokenizer
    return copy.deepcopy(base_model)
