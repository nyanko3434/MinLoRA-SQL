"""Session 4: plot ours vs peft toy-training loss curves and report max per-step delta."""
import json

import matplotlib.pyplot as plt

with open("results/toy_loss_ours.json") as f:
    ours = json.load(f)
with open("results/toy_loss_peft.json") as f:
    peft = json.load(f)

max_delta = max(abs(o - p) for o, p in zip(ours, peft))
print(f"max per-step |loss_ours - loss_peft| = {max_delta:.6f}")

steps = range(len(ours))
plt.figure(figsize=(8, 5))
plt.plot(steps, ours, label="ours (inject_lora)")
plt.plot(steps, peft, label="peft", linestyle="--")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Toy LoRA training: ours vs peft")
plt.legend()
plt.tight_layout()
plt.savefig("results/toy_loss_curves.png", dpi=150)
print("saved results/toy_loss_curves.png")
