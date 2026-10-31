"""
Collect every result file into one Markdown page: results/RESULTS.md

    python -m tinyreasoner.report

(The README's numbers are copied from this file, so they always match the runs.)
"""

import json
import os

from .utils import read_metrics


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def pct(x):
    return f"{100 * x:.1f}%"


def accuracy_table():
    rows = [("SFT, 1500 steps (RL starting point)", "results/eval_sft_step1500.json"),
            ("SFT, 1500 steps + GRPO", "results/eval_grpo.json"),
            ("SFT, 3000 steps (1500 more SFT steps)", "results/eval_sft.json")]
    data = [(name, _load(p)) for name, p in rows if _load(p)]
    if not data:
        return ""
    levels = sorted(data[0][1]["by_level"])
    out = ["| Model | " + " | ".join(f"L{lv}" for lv in levels) + " | all |",
           "|---|" + "---|" * (len(levels) + 1)]
    for name, d in data:
        out.append(f"| {name} | " + " | ".join(pct(d["by_level"][lv]) for lv in levels) + f" | {pct(d['accuracy'])} |")
    kinds = sorted(data[0][1]["by_kind"])
    out += ["", "By problem kind (all levels together):", "",
            "| Model | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for name, d in data:
        out.append(f"| {name} | " + " | ".join(pct(d["by_kind"][k]) for k in kinds) + " |")
    return "\n".join(out)


def timing_table():
    sft = read_metrics("runs/sft/metrics.jsonl") if os.path.exists("runs/sft/metrics.jsonl") else []
    grpo = read_metrics("runs/grpo/metrics.jsonl") if os.path.exists("runs/grpo/metrics.jsonl") else []
    t = {r["step"]: r["time_s"] for r in sft if "val_loss" in r}
    out = ["| Stage | Steps | Wall-clock time |", "|---|---|---|"]
    if 1500 in t and 3000 in t:
        out.append(f"| SFT, steps 0 → 1500 | 1500 | {t[1500] / 60:.0f} min |")
        out.append(f"| SFT, steps 1500 → 3000 | 1500 | {(t[3000] - t[1500]) / 60:.0f} min |")
    if grpo:
        last = grpo[-1]
        out.append(f"| GRPO from step 1500 | {last['step']} | {last['time_s'] / 60:.0f} min |")
    return "\n".join(out)


def passk_table():
    d = _load("results/passk.json")
    if not d:
        return ""
    levels = list(next(iter(d.values())).keys())
    out = ["| Model | " + " | ".join(f"{lv} pass@1 | {lv} pass@8" for lv in levels) + " |",
           "|---|" + "---|" * (2 * len(levels))]
    for name, v in d.items():
        cells = []
        for lv in levels:
            k8 = [k for k in v[lv] if k.startswith("pass@") and k != "pass@1_greedy"][0]
            cells += [pct(v[lv]["pass@1_greedy"]), pct(v[lv][k8])]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def lora_table():
    return "\n\n".join(t for t in (_lora_table("results/lora.json", "**Fine-tuned on division only:**"),
                                    _lora_table("results/lora_replay.json", "**Division + 20% old problems (replay):**"),
                                    _all_levels_after_finetune()) if t)


def _all_levels_after_finetune():
    rows = [("Base model (SFT, 3000 steps)", "results/eval_sft.json"),
            ("LoRA + replay, merged", "results/eval_lora_replay_merged.json"),
            ("Full fine-tuning + replay", "results/eval_full_replay.json")]
    data = [(n, _load(p)) for n, p in rows if _load(p)]
    if len(data) < 2:
        return ""
    levels = sorted(data[0][1]["by_level"])
    out = ["**Every level, after fine-tuning with replay** (the forgetting check above only covered levels 2-4):", "",
           "| Model | " + " | ".join(f"L{lv}" for lv in levels) + " |", "|---|" + "---|" * len(levels)]
    for n, d in data:
        out.append(f"| {n} | " + " | ".join(pct(d["by_level"][lv]) for lv in levels) + " |")
    return "\n".join(out)


def _lora_table(path, title):
    d = _load(path)
    if not d:
        return ""
    total = d["full"]["trainable_params"]
    out = [title, "", "| Method | Trained weights | New task (long division) | Old tasks (levels 2-4) | Saved file |",
           "|---|---|---|---|---|",
           f"| Before fine-tuning | - | {pct(d['before']['div_acc'])} | {pct(d['before']['old_tasks_acc'])} | - |",
           f"| LoRA (rank 8) | {d['lora']['trainable_params']:,} ({100 * d['lora']['trainable_params'] / total:.1f}%) | "
           f"{pct(d['lora']['div_acc'])} | {pct(d['lora']['old_tasks_acc'])} | {d['lora']['adapter_file_kb'] / 1024:.2f} MB adapter |",
           f"| LoRA merged into W | - | {pct(d['lora']['merged']['div_acc'])} | {pct(d['lora']['merged']['old_tasks_acc'])} | - |",
           f"| Full fine-tuning | {total:,} (100%) | {pct(d['full']['div_acc'])} | {pct(d['full']['old_tasks_acc'])} | "
           f"{d['full']['checkpoint_file_kb'] / 1024:.2f} MB model |"]
    return "\n".join(out)


def bench_tables():
    d = _load("results/benchmarks.json")
    if not d:
        return ""
    kv = d["kv_cache"]
    out = ["**KV cache** (generating 120 tokens after a level-6 prompt):", "",
           "| Batch | Without cache | With cache | Speed-up |", "|---|---|---|---|"]
    for bs in (1, 32):
        out.append(f"| {bs} | {kv[f'batch{bs}_no_cache_tok_per_s']} tok/s | {kv[f'batch{bs}_cache_tok_per_s']} tok/s | "
                   f"{kv[f'batch{bs}_speedup']}x |")
    q = d["quantization"]
    out += ["", "**Weight precision** (accuracy on 100 test problems per level):", "",
            "| Variant | File size | Accuracy, levels 1-5 | Speed (batch 1) |", "|---|---|---|---|"]
    names = {"fp32": "float32 (normal)", "bf16": "bfloat16", "int8": "int8, ours (from scratch)",
             "int8_pytorch_kernels": "int8, PyTorch kernels"}
    for k, label in names.items():
        if k in q:
            acc15 = sum(v for lv, v in q[k]["by_level"].items() if lv != "6") / 5
            out.append(f"| {label} | {q[k]['size_mb']:.2f} MB | {pct(acc15)} | {q[k]['batch1_tok_per_s']} tok/s |")
    if "int8" in q:
        out.append(f"\nAverage relative rounding error of the int8 weights: {100 * q['int8']['mean_weight_rel_error']:.2f}%")
    specs = sorted((k for k in d if k.startswith("speculative_k")), key=lambda k: int(k.split("_k")[1]))
    if specs:
        s0 = d[specs[0]]
        out += ["", f"**Speculative decoding** (target {s0['target_params_M']}M params, draft {s0['draft_params_M']}M, "
                    f"{s0['n_prompts']} prompts, one at a time):", "",
                "| k (draft guesses) | Same output as greedy? | Guesses accepted | Tokens per big-model call | Time | Speed-up |",
                "|---|---|---|---|---|---|",
                f"| plain greedy (no draft) | - | - | 1.0 | {s0['plain_greedy_s']} s | 1.00x |"]
        for k in specs:
            s = d[k]
            out.append(f"| {s['k']} | {s['identical_outputs']} | {pct(s['mean_acceptance_rate'])} | "
                       f"{s['tokens_per_target_call']} | {s['speculative_s']} s | {s['speedup']}x |")
    return "\n".join(out)


def main():
    parts = ["# Results\n", "All numbers below were produced by the code in this repo on a 4-core CPU.\n",
             "## Accuracy on held-out test sets (300 problems per level, greedy decoding)\n", accuracy_table(),
             "\n## Training time\n", timing_table(),
             "\n## pass@1 vs pass@8 (100 problems per level; 8 samples at temperature 1)\n", passk_table(),
             "\n## LoRA vs full fine-tuning (new task: long division)\n", lora_table(),
             "\n## Inference speed experiments\n", bench_tables()]
    text = "\n".join(p for p in parts if p is not None) + "\n"
    with open("results/RESULTS.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
