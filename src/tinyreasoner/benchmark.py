"""
Inference speed experiments: KV cache, quantization, speculative decoding.

    python -m tinyreasoner.benchmark --ckpt=runs/grpo/ckpt.pt --draft=runs/draft/ckpt.pt

All numbers are measured on this machine's CPU, so treat them as relative
(A vs B), not absolute.
"""

import copy
import io
import json
import os
import statistics
import sys
import time

import torch

from .data import load_jsonl
from .evaluate import score, solve
from .model import generate
from .quantize import quantization_error, quantize_model
from .speculative import speculative_generate
from .utils import cut_at_eot, left_pad, load_model, load_tokenizer, setup_torch


def timed(fn, repeats=3):
    """median wall-clock seconds of fn() over a few runs (after one warm-up run)"""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench_kv_cache(model, tok, prompts, n_new=120):
    """tokens/second with and without the KV cache, for batch size 1 and 32"""
    res = {}
    for bs in (1, 32):
        idx, mask = left_pad([tok.encode(p["prompt"]) for p in prompts[:bs]], tok.eot_id)
        for use_cache in (True, False):
            t = timed(lambda: generate(model, idx, mask, max_new_tokens=n_new, temperature=0, use_cache=use_cache))
            res[f"batch{bs}_{'cache' if use_cache else 'no_cache'}_tok_per_s"] = round(bs * n_new / t, 1)
        res[f"batch{bs}_speedup"] = round(res[f"batch{bs}_cache_tok_per_s"] / res[f"batch{bs}_no_cache_tok_per_s"], 2)
    return res


def saved_mb(model):
    """size of the saved weights file, in MB"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def bench_quantization(model, tok, problems, prompts):
    """
    fp32 vs bf16 vs our int8 (from scratch, de-quantizes on the fly)
    vs PyTorch's dynamic int8 (real int8 matrix-multiply kernels)
    """
    variants = {
        "fp32": model,
        "bf16": copy.deepcopy(model).to(torch.bfloat16),
        "int8": quantize_model(copy.deepcopy(model)),
        "int8_pytorch_kernels": torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8),
    }
    idx = torch.tensor([tok.encode(prompts[0]["prompt"])])
    res = {}
    for name, m in variants.items():
        acc = score(problems, solve(m, tok, problems, batch_size=100))
        t = timed(lambda: generate(m, idx, max_new_tokens=100, temperature=0))
        res[name] = {"size_mb": round(saved_mb(m), 3), "accuracy": acc["accuracy"],
                     "by_level": acc["by_level"], "batch1_tok_per_s": round(100 / t, 1)}
    res["int8"]["mean_weight_rel_error"] = round(quantization_error(model, variants["int8"]), 5)
    return res


def bench_speculative(target, draft, tok, problems, k=4):
    """plain greedy vs speculative decoding, one prompt at a time"""
    prompts = [tok.encode(p["prompt"]) for p in problems]
    agree, acc_rates, calls, toks = 0, [], 0, 0

    def run_plain():
        outs = []
        for p in prompts:
            o = generate(target, torch.tensor([p]), max_new_tokens=200, temperature=0, eos_id=tok.eot_id)
            outs.append(cut_at_eot(o[0].tolist(), tok.eot_id))
        return outs

    def run_spec():
        return [speculative_generate(target, draft, p, max_new_tokens=200, k=k, eos_id=tok.eot_id) for p in prompts]

    t_plain = timed(run_plain, repeats=1)
    t_spec = timed(run_spec, repeats=1)
    plain, spec = run_plain(), run_spec()
    for a, (b, st) in zip(plain, spec):
        agree += cut_at_eot(b, tok.eot_id) == a
        acc_rates.append(st["acceptance_rate"])
        calls += st["target_calls"]
        toks += st["tokens"]
    t_draft_only = timed(lambda: [generate(draft, torch.tensor([p]), max_new_tokens=200, temperature=0,
                                           eos_id=tok.eot_id) for p in prompts], repeats=1)
    return {
        "k": k,
        "n_prompts": len(prompts),
        "identical_outputs": f"{agree}/{len(prompts)}",
        "mean_acceptance_rate": round(sum(acc_rates) / len(acc_rates), 3),
        "tokens_per_target_call": round(toks / calls, 2),
        "plain_greedy_s": round(t_plain, 2),
        "speculative_s": round(t_spec, 2),
        "draft_alone_s": round(t_draft_only, 2),
        "speedup": round(t_plain / t_spec, 2),
        "target_params_M": round(target.num_params() / 1e6, 2),
        "draft_params_M": round(draft.num_params() / 1e6, 2),
    }


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:])
    setup_torch(int(args.get("threads", 4)))
    torch.manual_seed(0)
    tok = load_tokenizer()
    model = load_model(args.get("ckpt", "runs/sft/ckpt.pt"))
    tests = {lv: load_jsonl(f"data/test_L{lv}.jsonl") for lv in range(1, 7)}
    problems = [p for lv in tests for p in tests[lv][:100]]
    long_prompts = tests[6]

    results = {}
    print("== KV cache"); results["kv_cache"] = bench_kv_cache(model, tok, long_prompts); print(results["kv_cache"])
    print("== quantization"); results["quantization"] = bench_quantization(model, tok, problems, long_prompts)
    print(json.dumps(results["quantization"], indent=1))
    if "draft" in args:
        draft = load_model(args["draft"])
        spec_problems = [p for lv in tests for p in tests[lv][:10]]
        for k in (2, 4, 6):
            print(f"== speculative decoding k={k}")
            results[f"speculative_k{k}"] = bench_speculative(model, draft, tok, spec_problems, k=k)
            print(results[f"speculative_k{k}"])
    os.makedirs("results", exist_ok=True)
    with open("results/benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
