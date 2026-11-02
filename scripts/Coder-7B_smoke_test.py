import os
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoConfig, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"  # or your local snapshot path

cfg = AutoConfig.from_pretrained(MODEL)
print(type(cfg).__name__)
print("layers      ", cfg.num_hidden_layers)
print("hidden       ", cfg.hidden_size)
print("q heads      ", cfg.num_attention_heads)
print("kv heads     ", cfg.num_key_value_heads)   # GQA: should be < q heads

tok = AutoTokenizer.from_pretrained(MODEL)
print("vocab", tok.vocab_size, "| eos", tok.eos_token, "| pad", tok.pad_token)

import glob, os
pat = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/*/*.safetensors"
)
for s in sorted(glob.glob(pat)):
    print(os.path.basename(s), f"{os.path.getsize(s)/1e9:.2f} GB")