"""
Measure how often the model gets the right answer.

    python -m tinyreasoner.evaluate --ckpt=runs/sft/ckpt.pt --out=results/eval_sft.json

"Accuracy" here = the text inside <answer>...</answer> exactly equals the true answer.
We decode greedily (temperature 0: always pick the most likely token), so the
result is deterministic. pass@k samples k answers per question and counts the
question as solved if ANY of them is right.
"""

import json
import os
import sys
from collections import defaultdict

import torch

from .data import extract_answer, is_correct, load_jsonl
from .model import generate
from .utils import cut_at_eot, left_pad, load_model, load_tokenizer, setup_torch


@torch.no_grad()
def solve(model, tok, problems, batch_size=64, temperature=0.0, top_k=None, top_p=None,
          n_samples=1, max_new_tokens=None, generator=None, bf16=False):
    """returns the model's completions: a list (one per problem) of lists (n_samples strings)"""
    model.eval()
    prompts = [tok.encode(p["prompt"]) for p in problems]
    # sort by length so each batch needs little padding; remember original order
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results = [[] for _ in problems]
    jobs = [i for i in order for _ in range(n_samples)]
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b: b + batch_size]
        idx, mask = left_pad([prompts[i] for i in chunk], tok.eot_id)
        new = max_new_tokens or model.config.block_size - idx.shape[1]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            out = generate(model, idx, mask, max_new_tokens=new, temperature=temperature,
                           top_k=top_k, top_p=top_p, eos_id=tok.eot_id, generator=generator)
        for i, row in zip(chunk, out.tolist()):
            results[i].append(tok.decode(cut_at_eot(row, tok.eot_id)))
    return results


def score(problems, completions):
    """accuracy overall / per level / per kind, plus how often the answer format was valid"""
    by_level, by_kind = defaultdict(list), defaultdict(list)
    fmt = []
    for p, cs in zip(problems, completions):
        ok = is_correct(cs[0], p["answer"])
        by_level[p["level"]].append(ok)
        by_kind[p["kind"]].append(ok)
        fmt.append(extract_answer(cs[0]) is not None)
    mean = lambda xs: round(sum(xs) / max(len(xs), 1), 4)  # noqa: E731
    return {
        "accuracy": mean([x for v in by_level.values() for x in v]),
        "by_level": {str(k): mean(v) for k, v in sorted(by_level.items())},
        "by_kind": {k: mean(v) for k, v in sorted(by_kind.items())},
        "format_ok": mean(fmt),
        "n": len(problems),
    }


def pass_at_k(problems, completions):
    """fraction of problems where at least one of the k samples is correct"""
    hits = [any(is_correct(c, p["answer"]) for c in cs) for p, cs in zip(problems, completions)]
    return round(sum(hits) / len(hits), 4)


def evaluate_levels(model, tok, data_dir="data", levels=(1, 2, 3, 4, 5, 6), n_per_level=None, **kw):
    problems = []
    for lv in levels:
        rows = load_jsonl(os.path.join(data_dir, f"test_L{lv}.jsonl"))
        problems += rows[:n_per_level] if n_per_level else rows
    completions = solve(model, tok, problems, **kw)
    return score(problems, completions), problems, completions


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:])
    setup_torch()
    model = load_model(args["ckpt"])
    tok = load_tokenizer(args.get("tokenizer", "data/tokenizer.json"))
    res, problems, completions = evaluate_levels(model, tok, args.get("data_dir", "data"))
    print(json.dumps(res, indent=2))
    if "out" in args:
        os.makedirs(os.path.dirname(args["out"]), exist_ok=True)
        with open(args["out"], "w") as f:
            json.dump(res, f, indent=2)
        # also keep a few example outputs, right and wrong, for the README
        samples = [{"prompt": p["prompt"], "model": c[0], "answer": p["answer"],
                    "correct": is_correct(c[0], p["answer"])} for p, c in zip(problems[::50], completions[::50])]
        with open(args["out"].replace(".json", "_samples.json"), "w") as f:
            json.dump(samples, f, indent=2)


if __name__ == "__main__":
    main()
