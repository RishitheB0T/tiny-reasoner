"""
pass@1 vs pass@k for the RL starting point, the longer-trained SFT model and the GRPO model.

    python -m tinyreasoner.passk

pass@1 (greedy): the single most-likely answer is right.
pass@k (k samples at temperature 1): at least ONE of k tries is right.

A well-known finding about RL on LLMs is that it mostly "sharpens" the model:
it turns answers the model could ALREADY sometimes find (pass@k) into answers
it finds on the first try (pass@1). Comparing both numbers shows whether that
happened here.
"""

import json
import os

import torch

from .data import load_jsonl
from .evaluate import pass_at_k, score, solve
from .utils import load_model, load_tokenizer, setup_torch


def main(k=8, n_per_level=100, levels=(4, 5, 6)):
    setup_torch()
    tok = load_tokenizer()
    out = {}
    models = [("sft_step1500 (RL start)", "runs/sft/ckpt_step1500.pt"),
              ("sft_step3000 (+1500 SFT steps)", "runs/sft/ckpt.pt"),
              ("sft_step1500 + grpo", "runs/grpo/ckpt.pt")]
    for name, ckpt in models:
        model = load_model(ckpt)
        out[name] = {}
        for lv in levels:
            problems = load_jsonl(f"data/test_L{lv}.jsonl")[:n_per_level]
            greedy = score(problems, solve(model, tok, problems, batch_size=100))["accuracy"]
            gen = torch.Generator().manual_seed(lv)
            samples = solve(model, tok, problems, batch_size=128, temperature=1.0, n_samples=k, generator=gen)
            out[name][f"L{lv}"] = {"pass@1_greedy": greedy, f"pass@{k}": pass_at_k(problems, samples),
                                   "mean_sample_acc": round(sum(score(problems, [[s[j]] for s in samples])["accuracy"]
                                                                for j in range(k)) / k, 4)}
        print(name, json.dumps(out[name]))
    os.makedirs("results", exist_ok=True)
    with open("results/passk.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
