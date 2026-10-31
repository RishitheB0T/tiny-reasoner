"""
Stage 1: Supervised Fine-Tuning (SFT) - learn by imitating worked solutions.

    python -m tinyreasoner.train_sft --config=configs/sft.json

The model reads "Q:347+586\\nA:" and is trained to predict the worked solution
one token at a time (next-token prediction, exactly like in "Let's build GPT").
The only difference from pre-training: the loss is computed ONLY on the answer
tokens, never on the question (we mask the question targets with -100), because
we want the model to learn to answer, not to invent questions.
"""

import math
import os
import random
import time
from dataclasses import dataclass

import torch

from .data import load_jsonl
from .evaluate import score, solve
from .model import GPT, GPTConfig
from .utils import (MetricsLogger, encode_example, load_config, load_model, load_tokenizer,
                    right_pad, save_checkpoint, set_seed, setup_torch)


@dataclass
class SFTConfig:
    out_dir: str = "runs/sft"
    data_dir: str = "data"
    tokenizer: str = "data/tokenizer.json"
    # model
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 2
    n_embd: int = 256
    block_size: int = 256
    dropout: float = 0.0
    # training
    batch_size: int = 64
    max_iters: int = 3000
    learning_rate: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_train_examples: int = 0  # 0 = all
    # evaluation / logging
    log_interval: int = 25
    eval_interval: int = 250
    eval_problems: int = 300  # validation problems for the quick accuracy check
    bf16: bool = False  # bfloat16 autocast; measured no faster than fp32 on our CPU, so off
    seed: int = 1337
    threads: int = 4
    init_from: str = ""  # start from an existing checkpoint instead of random weights
    snapshot_steps: str = ""  # e.g. "1500": also keep ckpt_step1500.pt (the RL experiment starts from it)


def get_lr(it, cfg):
    """linear warmup, then cosine decay down to min_lr (same schedule as GPT-2 / nanoGPT)"""
    max_lr, min_lr = cfg.learning_rate, cfg.learning_rate * cfg.min_lr_ratio
    if it < cfg.warmup_iters:
        return max_lr * (it + 1) / cfg.warmup_iters
    progress = min(1.0, (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def make_batch(examples, idxs, pad_id):
    """inputs x and targets y (shifted by one); prompt targets are -100 so they get no loss"""
    xs, ys = [], []
    for i in idxs:
        p, c = examples[i]
        full = p + c
        y = list(full[1:])
        y[: len(p) - 1] = [-100] * (len(p) - 1)
        xs.append(full[:-1])
        ys.append(y)
    x = right_pad(xs, pad_id)
    y = right_pad(ys, -100)
    return x, y


@torch.no_grad()
def val_loss(model, examples, pad_id, batch_size, bf16, n_batches=10):
    model.eval()
    losses = []
    for b in range(n_batches):
        idxs = list(range(b * batch_size, min((b + 1) * batch_size, len(examples))))
        if not idxs:
            break
        x, y = make_batch(examples, idxs, pad_id)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    cfg = load_config(SFTConfig)
    setup_torch(cfg.threads)
    set_seed(cfg.seed)
    tok = load_tokenizer(cfg.tokenizer)

    t0 = time.time()
    train_rows = load_jsonl(os.path.join(cfg.data_dir, "train.jsonl"))
    if cfg.max_train_examples:
        train_rows = train_rows[: cfg.max_train_examples]
    val_rows = load_jsonl(os.path.join(cfg.data_dir, "val.jsonl"))
    train = [encode_example(tok, r) for r in train_rows]
    val = [encode_example(tok, r) for r in val_rows]
    longest = max(len(p) + len(c) for p, c in train)
    print(f"tokenized {len(train)} train / {len(val)} val examples in {time.time() - t0:.1f}s, longest={longest} tokens")
    assert longest <= cfg.block_size

    if cfg.init_from:
        model = load_model(cfg.init_from).train()
    else:
        mcfg = GPTConfig(block_size=cfg.block_size, vocab_size=tok.vocab_size, n_layer=cfg.n_layer,
                         n_head=cfg.n_head, n_kv_head=cfg.n_kv_head, n_embd=cfg.n_embd, dropout=cfg.dropout)
        model = GPT(mcfg)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters, config={model.config}")
    opt = model.configure_optimizer(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2))
    logger = MetricsLogger(cfg.out_dir, cfg)

    rng = random.Random(cfg.seed)
    model.train()
    t_last = time.time()
    tokens_seen = 0
    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            vl = val_loss(model, val, tok.eot_id, cfg.batch_size, cfg.bf16)
            res = score(val_rows[: cfg.eval_problems],
                        solve(model, tok, val_rows[: cfg.eval_problems], batch_size=100, bf16=cfg.bf16))
            logger.log(step=it, val_loss=round(vl, 4), val_acc=res["accuracy"],
                       **{f"val_acc_L{k}": v for k, v in res["by_level"].items()})
            save_checkpoint(os.path.join(cfg.out_dir, "ckpt.pt"), model, {"step": it})
            if str(it) in str(cfg.snapshot_steps).split(","):
                save_checkpoint(os.path.join(cfg.out_dir, f"ckpt_step{it}.pt"), model, {"step": it})
            model.train()
            if it == cfg.max_iters:
                break

        lr = get_lr(it, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = make_batch(train, [rng.randrange(len(train)) for _ in range(cfg.batch_size)], tok.eot_id)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        tokens_seen += x.numel()

        if it % cfg.log_interval == 0:
            dt = (time.time() - t_last) / (cfg.log_interval if it else 1)
            t_last = time.time()
            logger.log(step=it, loss=round(loss.item(), 6), lr=lr, grad_norm=round(norm.item(), 3),
                       ms_per_step=round(dt * 1000), tokens_seen=tokens_seen)


if __name__ == "__main__":
    main()
