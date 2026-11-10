import copy
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]

# A machine without a GPU or without the datasets gets skips, not failures. Tests that take the
# model fixtures below skip through them; the rest carry one of these marks.
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
requires_spider = pytest.mark.skipif(not (ROOT / "data" / "spider_data" / "dev.json").exists(),
                                     reason="Spider is not downloaded to data/spider_data/")


@pytest.fixture(scope="session")
def base_model_and_tokenizer():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (the 0.5B fixture runs on cuda)")

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
