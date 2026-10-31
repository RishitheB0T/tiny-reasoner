"""
Draw every figure in results/ from the logged metrics.

    python -m tinyreasoner.plots
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .utils import read_metrics  # noqa: E402

# validated palette (categorical slots 1-3, and a light->dark blue ramp for ordered levels)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LEVEL_RAMP = {"1": "#86b6ef", "2": "#5598e7", "3": "#2a78d6", "4": "#1c5cab", "5": "#104281"}  # SFT: levels 1-5
RL_LEVEL_RAMP = {"3": "#86b6ef", "4": "#3987e5", "5": "#1c5cab", "6": "#0d366b"}  # GRPO: levels 3-6
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12, "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
    "lines.linewidth": 2, "text.color": INK,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join("results", name), dpi=150)
    plt.close(fig)
    print("wrote results/" + name)


def _end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)


def plot_sft(path="runs/sft/metrics.jsonl", name="sft_curves.png"):
    rows = read_metrics(path)
    train = [r for r in rows if "loss" in r]
    evals = [r for r in rows if "val_acc" in r]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.plot([r["step"] for r in train], _rolling([r["loss"] for r in train], 5), color=BLUE)
    a.set_yscale("log")
    a.set(title="Stage 1 (SFT): training loss", xlabel="step", ylabel="cross-entropy, 5-point average (log)")
    for lv, color in LEVEL_RAMP.items():
        xs = [r["step"] for r in evals if f"val_acc_L{lv}" in r]
        ys = [100 * r[f"val_acc_L{lv}"] for r in evals if f"val_acc_L{lv}" in r]
        if xs:
            b.plot(xs, ys, color=color, marker="o", markersize=4, label=f"level {lv}")
    b.set(title="Validation accuracy by difficulty", xlabel="step", ylabel="accuracy (%)", ylim=(-3, 103))
    b.legend(loc="lower right")
    _save(fig, name)


def _rolling(xs, k=10):
    return [sum(xs[max(0, i - k + 1): i + 1]) / len(xs[max(0, i - k + 1): i + 1]) for i in range(len(xs))]


def plot_grpo(path="runs/grpo/metrics.jsonl"):
    rows = read_metrics(path)
    train = [r for r in rows if "reward" in r]
    evals = [r for r in rows if "val_acc" in r]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    steps = [r["step"] for r in train]
    acc = [100 * r["sample_acc"] for r in train]
    a.plot(steps, acc, color=BLUE, alpha=0.25, linewidth=1)
    a.plot(steps, _rolling(acc), color=BLUE, label="10-step average")
    a.set(title="Stage 2 (GRPO): sampled answers that were right", xlabel="step", ylabel="correct samples (%)")
    a.legend(loc="lower right")
    for lv, color in RL_LEVEL_RAMP.items():
        xs = [r["step"] for r in evals if f"val_acc_L{lv}" in r]
        ys = [100 * r[f"val_acc_L{lv}"] for r in evals if f"val_acc_L{lv}" in r]
        if xs:
            b.plot(xs, ys, color=color, marker="o", markersize=4, label=f"level {lv}" + (" (never trained on)" if lv == "6" else ""))
    b.set(title="Greedy accuracy on held-out RL problems", xlabel="step", ylabel="accuracy (%)", ylim=(-3, 103))
    b.legend(loc="center right")
    _save(fig, "grpo_curves.png")

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(steps, [r["kl"] for r in train], color=ORANGE)
    ax.set(title="KL distance from the SFT model", xlabel="step", ylabel="mean per-token KL")
    _save(fig, "grpo_kl.png")


def _grouped_bars(ax, groups, series, colors):
    """series: {label: [value per group]}"""
    n = len(series)
    width = 0.8 / n
    for i, ((label, vals), color) in enumerate(zip(series.items(), colors)):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ax.bar(xs, vals, width=width - 0.03, color=color, label=label, zorder=2)
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="x", visible=False)


def plot_before_after(start="results/eval_sft_step1500.json", sft="results/eval_sft.json",
                      grpo="results/eval_grpo.json"):
    a, s, g = (json.load(open(p)) for p in (start, sft, grpo))
    t_sft = {r["step"]: r["time_s"] for r in read_metrics("runs/sft/metrics.jsonl") if "val_loss" in r}
    t_grpo = read_metrics("runs/grpo/metrics.jsonl")[-1]["time_s"]
    levels = sorted(a["by_level"])
    series = {"start: SFT, 1500 steps": [100 * a["by_level"][lv] for lv in levels],
              f"+ GRPO, 300 steps ({t_grpo / 60:.0f} min)": [100 * g["by_level"][lv] for lv in levels],
              f"+ 1500 more SFT steps ({(t_sft[3000] - t_sft[1500]) / 60:.0f} min)": [100 * s["by_level"][lv] for lv in levels]}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _grouped_bars(ax, [f"level {lv}" + ("\n(never trained on)" if lv == "6" else "") for lv in levels],
                  series, [BLUE, ORANGE, AQUA])
    for i, lv in enumerate(levels):
        if all(v[i] == 0 for v in series.values()):
            ax.annotate("0% for all three", (i, 0), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK2)
    ax.annotate(f"{series[list(series)[0]][levels.index('5')]:.0f}%", (levels.index("5") - 0.267, series[list(series)[0]][levels.index("5")]),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set(title="Test accuracy: RL vs more supervised training, from the same start",
           ylabel="accuracy (%)", ylim=(0, 105))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    _save(fig, "accuracy_before_after.png")


def plot_lora(paths=("results/lora.json", "results/lora_replay.json")):
    runs = [(p, json.load(open(p))) for p in paths if os.path.exists(p)]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.2), squeeze=False)
    for ax, (path, r) in zip(axes[0], runs):
        total = r["full"]["trainable_params"]
        labels = {"before": "before fine-tuning",
                  "lora": f"LoRA ({100 * r['lora']['trainable_params'] / total:.0f}% of weights trained)",
                  "full": "full fine-tuning (100% of weights)"}
        series = {labels[k]: [100 * r[k]["div_acc"], 100 * r[k]["old_tasks_acc"]] for k in ("before", "lora", "full")}
        _grouped_bars(ax, ["new task:\nlong division", "old tasks:\nlevels 2-4"], series, [BLUE, ORANGE, AQUA])
        for gi in range(2):
            for si, vals in enumerate(series.values()):
                if vals[gi] < 3:
                    ax.annotate(f"{vals[gi]:.0f}%", (gi + (si - 1) * 0.8 / 3, vals[gi]), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
        rf = r.get("replay_frac", 0)
        ax.set(title="Fine-tuned on division only" if not rf else
               f"Division + {100 * rf:.0f}% old problems (replay)", ylabel="accuracy (%)", ylim=(0, 105))
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join("results", "lora.png"), dpi=150)
    plt.close(fig)
    print("wrote results/lora.png")


def plot_benchmarks(path="results/benchmarks.json"):
    r = json.load(open(path))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    kv = r["kv_cache"]
    _grouped_bars(axes[0], ["batch 1", "batch 32"],
                  {"no KV cache": [kv["batch1_no_cache_tok_per_s"], kv["batch32_no_cache_tok_per_s"]],
                   "with KV cache": [kv["batch1_cache_tok_per_s"], kv["batch32_cache_tok_per_s"]]}, [BLUE, ORANGE])
    for i, bs in enumerate((1, 32)):
        axes[0].annotate(f"{kv[f'batch{bs}_speedup']:.1f}x faster", (i + 0.2, kv[f"batch{bs}_cache_tok_per_s"]),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    axes[0].set(title="KV cache: generation speed", ylabel="tokens per second")
    axes[0].legend(loc="upper left")

    q = r["quantization"]
    names = [n for n in ("fp32", "bf16", "int8", "int8_pytorch_kernels") if n in q]
    axes[1].bar([n.replace("_pytorch_kernels", "\n(PyTorch)") for n in names], [q[n]["size_mb"] for n in names],
                color=BLUE, width=0.55, zorder=2)
    for i, n in enumerate(names):
        acc15 = sum(v for lv, v in q[n]["by_level"].items() if lv != "6") / 5
        axes[1].annotate(f"L1-5 acc {100 * acc15:.1f}%", (i, q[n]["size_mb"]), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    axes[1].set(title="Weight precision: model size", ylabel="megabytes")
    axes[1].grid(axis="x", visible=False)

    ks = sorted(int(k.split("_k")[1]) for k in r if k.startswith("speculative_k"))
    if ks:
        sp = [r[f"speculative_k{k}"]["speedup"] for k in ks]
        acc = [r[f"speculative_k{k}"]["mean_acceptance_rate"] for k in ks]
        axes[2].bar([str(k) for k in ks], sp, color=ORANGE, width=0.55, zorder=2)
        for i, (s_, a_) in enumerate(zip(sp, acc)):
            axes[2].annotate(f"{100 * a_:.0f}% accepted", (i, s_), xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=8.5, color=INK2)
        axes[2].axhline(1.0, color=AXIS, linewidth=1, zorder=1)
        axes[2].set(title="Speculative decoding speed-up", xlabel="draft tokens per step (k)",
                    ylabel="speed-up vs plain greedy (x)")
        axes[2].grid(axis="x", visible=False)
    _save(fig, "benchmarks.png")


def main():
    os.makedirs("results", exist_ok=True)
    steps = [
        (plot_sft, "runs/sft/metrics.jsonl"),
        (plot_grpo, "runs/grpo/metrics.jsonl"),
        (plot_before_after, "results/eval_grpo.json"),
        (plot_lora, "results/lora.json"),
        (plot_benchmarks, "results/benchmarks.json"),
    ]
    for fn, needed in steps:
        if os.path.exists(needed):
            fn()
        else:
            print(f"skip {fn.__name__}: {needed} not found yet")
    if os.path.exists("runs/draft/metrics.jsonl"):
        plot_sft("runs/draft/metrics.jsonl", "draft_curves.png")


if __name__ == "__main__":
    main()
