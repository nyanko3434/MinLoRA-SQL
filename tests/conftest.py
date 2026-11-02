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


@pytest.fixture
def fresh_model(base_model_and_tokenizer):
    """Factory returning a fresh deepcopy of the pristine session-scoped base model on
    each call, so in-place injection (inject_lora / get_peft_model) can't leak state
    across tests."""
    base_model, _ = base_model_and_tokenizer

    def _fresh():
        return copy.deepcopy(base_model)

    return _fresh
