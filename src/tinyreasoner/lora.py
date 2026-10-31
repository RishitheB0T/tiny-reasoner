"""
LoRA (Low-Rank Adaptation), written from scratch.

Problem: fine-tuning ALL weights of a big model is expensive (memory for
gradients + optimizer state for every weight) and you must store a full copy
of the model for every new task.

LoRA's idea: keep the original weight W frozen and learn a small correction
    W' = W + (alpha / r) * B @ A        A: (r, in)   B: (out, r)   with r tiny (e.g. 8)
For a 256x256 matrix that's 2*8*256 = 4,096 trainable numbers instead of 65,536.
B starts at zero, so at the start the model behaves EXACTLY like the original.
After training you can "merge" B@A into W, so inference costs nothing extra.

    python -m tinyreasoner.lora --config=configs/lora.json
"""

import copy
import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import load_jsonl
from .evaluate import score, solve
from .train_sft import get_lr, make_batch
from .utils import (MetricsLogger, encode_example, load_config, load_model, load_tokenizer,
                    save_checkpoint, set_seed, setup_torch)


class LoRALinear(nn.Module):
    """wraps a frozen nn.Linear and adds the trainable low-rank path x -> A -> B"""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))  # zero -> starts as a no-op
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    def merged(self):
        """a plain nn.Linear with W + scale * B @ A baked in"""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=False)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight + self.scale * self.lora_B @ self.lora_A)
        return lin


DEFAULT_TARGETS = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")


def apply_lora(model, r=8, alpha=16, targets=DEFAULT_TARGETS, train_token_ids=()):
    """
    freeze the whole model, then wrap every Linear whose name is in `targets` with LoRA.
    train_token_ids: ids of tokens the base model never saw (their embedding rows are
    still random), which we let train too - like PEFT's `modules_to_save`, but just those rows.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    if train_token_ids:
        emb = model.wte.weight
        emb.requires_grad_(True)
        row_mask = torch.zeros(emb.shape[0], 1)
        row_mask[list(train_token_ids)] = 1.0
        emb.register_hook(lambda g: g * row_mask)  # gradient only flows into the new rows
    return model


def merge_lora(model):
    """replace every LoRALinear by its merged nn.Linear (no extra inference cost)"""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def count_trainable(model, train_token_ids=()):
    n = sum(p.numel() for n_, p in model.named_parameters() if p.requires_grad and "wte" not in n_)
    if model.wte.weight.requires_grad:
        n += len(train_token_ids) * model.wte.weight.shape[1] if train_token_ids else model.wte.weight.numel()
    return n


# ----------------------------------------------------------------------------- experiment


@dataclass
class LoRAConfig:
    out_dir: str = "runs/lora"
    init_from: str = "runs/sft/ckpt.pt"
    data_dir: str = "data"
    tokenizer: str = "data/tokenizer.json"
    rank: int = 8
    alpha: int = 16
    iters: int = 600
    batch_size: int = 64
    lora_lr: float = 2e-3
    full_lr: float = 3e-4
    warmup_iters: int = 30
    min_lr_ratio: float = 0.1
    eval_interval: int = 100
    n_eval: int = 300
    n_forget: int = 200  # original-task problems per level to measure forgetting
    replay_frac: float = 0.0  # share of each batch taken from the ORIGINAL training data ("replay")
    results_name: str = "lora"  # results/<results_name>.json
    bf16: bool = False
    seed: int = 1337
    threads: int = 4


def sample_batch(rng, train, old, cfg):
    """indices into train+old: (1 - replay_frac) new-task examples, replay_frac old-task examples"""
    n_old = int(round(cfg.batch_size * cfg.replay_frac)) if old else 0
    return ([rng.randrange(len(train)) for _ in range(cfg.batch_size - n_old)] +
            [len(train) + rng.randrange(len(old)) for _ in range(n_old)])


def finetune(model, train, cfg, lr, logger, tag, eval_fn, tok, old=()):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = copy.copy(cfg)
    sched.learning_rate, sched.max_iters = lr, cfg.iters
    rng = random.Random(cfg.seed)
    model.train()
    for it in range(cfg.iters + 1):
        if it % cfg.eval_interval == 0 or it == cfg.iters:
            logger.log(method=tag, step=it, **eval_fn(model))
            model.train()
            if it == cfg.iters:
                break
        for g in opt.param_groups:
            g["lr"] = get_lr(it, sched)
        x, y = make_batch(list(train) + list(old), sample_batch(rng, train, old, cfg), tok.eot_id)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % 50 == 0:
            logger.log(method=tag, step=it, loss=round(loss.item(), 4))


def main():
    cfg = load_config(LoRAConfig)
    setup_torch(cfg.threads)
    set_seed(cfg.seed)
    tok = load_tokenizer(cfg.tokenizer)
    train = [encode_example(tok, r) for r in load_jsonl(os.path.join(cfg.data_dir, "div_train.jsonl"))]
    div_test = load_jsonl(os.path.join(cfg.data_dir, "div_test.jsonl"))[: cfg.n_eval]
    old_test = []
    for lv in (2, 3, 4):
        old_test += load_jsonl(os.path.join(cfg.data_dir, f"test_L{lv}.jsonl"))[: cfg.n_forget]

    # tokens that only the new task uses ('/' and 'r'): the base model never trained their embeddings
    old_rows = load_jsonl(os.path.join(cfg.data_dir, "train.jsonl"))[:20000]
    seen = set()
    for r in old_rows:
        seen.update(tok.encode(r["prompt"] + r["completion"]))
    old = [encode_example(tok, r) for r in old_rows] if cfg.replay_frac > 0 else []
    new_ids = sorted({i for p, c in train[:5000] for i in p + c} - seen - {tok.eot_id})
    print("new token ids:", new_ids, [tok.decode([i]) for i in new_ids])

    def eval_fn(model):
        d = score(div_test, solve(model, tok, div_test, batch_size=100, bf16=cfg.bf16))
        o = score(old_test, solve(model, tok, old_test, batch_size=100, bf16=cfg.bf16))
        return {"div_acc": d["accuracy"], "old_tasks_acc": o["accuracy"]}

    logger = MetricsLogger(cfg.out_dir, cfg)
    results = {}

    # A) LoRA
    model = apply_lora(load_model(cfg.init_from), cfg.rank, cfg.alpha, train_token_ids=new_ids)
    n_lora = count_trainable(model, new_ids)
    finetune(model, train, cfg, cfg.lora_lr, logger, "lora", eval_fn, tok, old)
    adapter = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    adapter["wte_new_rows"] = model.wte.weight[new_ids].detach().clone()
    torch.save(adapter, os.path.join(cfg.out_dir, "lora_adapter.pt"))
    lora_final = eval_fn(model)
    # merging must not change the outputs
    merged = merge_lora(model).eval()
    merged_final = eval_fn(merged)
    save_checkpoint(os.path.join(cfg.out_dir, "merged_ckpt.pt"), merged)
    results["lora"] = {"trainable_params": n_lora, **lora_final, "merged": merged_final,
                       "adapter_file_kb": round(os.path.getsize(os.path.join(cfg.out_dir, "lora_adapter.pt")) / 1024, 1)}

    # B) full fine-tuning, same data and steps
    model = load_model(cfg.init_from).train()
    n_full = count_trainable(model)
    finetune(model, train, cfg, cfg.full_lr, logger, "full", eval_fn, tok, old)
    save_checkpoint(os.path.join(cfg.out_dir, "full_ckpt.pt"), model)
    results["full"] = {"trainable_params": n_full, **eval_fn(model),
                       "checkpoint_file_kb": round(os.path.getsize(os.path.join(cfg.out_dir, "full_ckpt.pt")) / 1024, 1)}
    # C) the starting point, for reference
    results["before"] = eval_fn(load_model(cfg.init_from))

    import json
    os.makedirs("results", exist_ok=True)
    results["replay_frac"] = cfg.replay_frac
    with open(f"results/{cfg.results_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
