"""
Stage 2: GRPO (Group Relative Policy Optimization) - learn by practising with rewards.

    python -m tinyreasoner.grpo --config=configs/grpo.json

This is the reinforcement learning (RL) algorithm DeepSeek used for DeepSeek-R1.
One training step:

  1. take a few questions, and let the model answer EACH question G times
     (sampling with temperature 1, so the G answers differ)
  2. score every answer with code (rewards.py): right answer -> 1, wrong -> 0
  3. advantage = how much better an answer is than ITS OWN GROUP's average:
         A = (reward - mean(group)) / std(group)
     (PPO needs a second "critic" network to estimate the average; GRPO simply
      uses the group, which is why it is cheaper)
  4. nudge the model to make above-average answers more likely and
     below-average answers less likely, with two safety belts:
       - clipping: never change an answer's probability by more than ~20% per step
       - KL penalty: stay close to the frozen starting (SFT) model, the "reference"
"""

import copy
import os
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import load_jsonl
from .evaluate import score, solve
from .model import generate
from .rewards import correctness_reward, total_reward
from .utils import (MetricsLogger, cut_at_eot, left_pad, load_config, load_model, load_tokenizer,
                    right_pad, save_checkpoint, set_seed, setup_torch)


@dataclass
class GRPOConfig:
    out_dir: str = "runs/grpo"
    init_from: str = "runs/sft/ckpt.pt"
    data_dir: str = "data"
    tokenizer: str = "data/tokenizer.json"
    iters: int = 300
    prompts_per_step: int = 16  # questions per step
    group_size: int = 8  # G: answers sampled per question
    temperature: float = 1.0
    learning_rate: float = 2e-5
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    ppo_epochs: int = 2  # optimisation passes over each batch of answers
    minibatch_size: int = 64
    grad_clip: float = 1.0
    rl_levels: str = ""  # e.g. "4,5": train only on these levels ("" = all levels in rl_prompts.jsonl)
    n_val: int = 300  # held-out RL prompts for tracking greedy accuracy
    eval_interval: int = 25
    log_interval: int = 1
    bf16: bool = False
    seed: int = 1337
    threads: int = 4


def token_logprobs(model, seqs, bf16):
    """log p(token_t | tokens_<t) for every position t>=1. seqs is right-padded, shape (N, T)"""
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        logits, _ = model(seqs[:, :-1])
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, seqs[:, 1:, None]).squeeze(-1)  # (N, T-1)


def build_sequences(prompts, completions, pad_id):
    """prompt+completion, right padded, plus a mask that is 1 on completion tokens (the 'actions')"""
    seqs = [p + c for p, c in zip(prompts, completions)]
    x = right_pad(seqs, pad_id)
    mask = torch.zeros(x.shape[0], x.shape[1] - 1)
    for i, (p, c) in enumerate(zip(prompts, completions)):
        mask[i, len(p) - 1: len(p) - 1 + len(c)] = 1.0  # target positions of completion tokens
    return x, mask


def grpo_loss(new_logp, old_logp, ref_logp, adv, mask, clip_eps, kl_beta):
    """
    the clipped GRPO objective (to MINIMISE), averaged per answer then over answers.
    new/old/ref_logp, mask: (N, T); adv: (N,)
    """
    ratio = torch.exp(new_logp - old_logp)
    a = adv[:, None]
    surrogate = torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    # k3 estimator of KL(new || ref): always >= 0, and 0 only when they agree
    d = ref_logp - new_logp
    kl = torch.exp(d) - d - 1
    per_token = -(surrogate - kl_beta * kl)
    per_answer = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    stats = {
        "kl": ((kl * mask).sum() / mask.sum()).item(),
        "clip_frac": ((((ratio - 1).abs() > clip_eps).float() * mask).sum() / mask.sum()).item(),
    }
    return per_answer.mean(), stats


def group_advantages(rewards, group_size, eps=1e-4):
    """(reward - group mean) / group std. A group where every answer scored the same gets all zeros."""
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + eps)
    return adv.view(-1)


def main():
    cfg = load_config(GRPOConfig)
    setup_torch(cfg.threads)
    set_seed(cfg.seed)
    tok = load_tokenizer(cfg.tokenizer)
    gen = torch.Generator().manual_seed(cfg.seed)

    policy = load_model(cfg.init_from).train()
    ref = copy.deepcopy(policy).eval()  # frozen copy of the SFT model
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.99), weight_decay=0.0)

    rows = load_jsonl(os.path.join(cfg.data_dir, "rl_prompts.jsonl"))
    # validation always covers every level (so we can watch level 6 too); training may be filtered
    val_rows, train_rows = rows[: cfg.n_val], rows[cfg.n_val:]
    if cfg.rl_levels:
        keep_levels = {int(x) for x in str(cfg.rl_levels).split(",")}
        train_rows = [r for r in train_rows if r["level"] in keep_levels]
    rng = random.Random(cfg.seed)
    logger = MetricsLogger(cfg.out_dir, cfg)
    G = cfg.group_size

    for it in range(cfg.iters + 1):
        if it % cfg.eval_interval == 0 or it == cfg.iters:
            res = score(val_rows, solve(policy, tok, val_rows, batch_size=100, bf16=cfg.bf16))
            logger.log(step=it, val_acc=res["accuracy"], **{f"val_acc_L{k}": v for k, v in res["by_level"].items()},
                       val_format_ok=res["format_ok"])
            save_checkpoint(os.path.join(cfg.out_dir, "ckpt.pt"), policy, {"step": it})
            if it == cfg.iters:
                break

        # 1) sample G answers for each question
        batch = [rng.choice(train_rows) for _ in range(cfg.prompts_per_step)]
        prompts = [tok.encode(p["prompt"]) for p in batch for _ in range(G)]
        policy.eval()
        idx, amask = left_pad(prompts, tok.eot_id)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16):
            out = generate(policy, idx, amask, max_new_tokens=policy.config.block_size - idx.shape[1],
                           temperature=cfg.temperature, eos_id=tok.eot_id, generator=gen)
        completions = []
        for row in out.tolist():
            c = cut_at_eot(row, tok.eot_id)
            completions.append(c + [tok.eot_id] if len(c) < len(row) else c)  # keep the stop token as an action
        texts = [tok.decode(cut_at_eot(c, tok.eot_id)) for c in completions]

        # 2) score them
        answers = [p["answer"] for p in batch for _ in range(G)]
        rewards = torch.tensor([total_reward(t, a) for t, a in zip(texts, answers)])
        correct = torch.tensor([correctness_reward(t, a) for t, a in zip(texts, answers)])

        # 3) group-relative advantages; groups where all answers tie teach nothing -> drop them
        adv = group_advantages(rewards, G)
        keep = (rewards.view(-1, G).std(-1) > 0).repeat_interleave(G)
        frac_zero = 1 - keep.float().mean().item()

        stats = {"kl": 0.0, "clip_frac": 0.0}
        n_updates = 0
        if keep.any():
            ids = keep.nonzero().squeeze(-1).tolist()
            x, mask = build_sequences([prompts[i] for i in ids], [completions[i] for i in ids], tok.eot_id)
            adv_k = adv[ids]
            with torch.no_grad():
                old_logp = token_logprobs(policy, x, cfg.bf16)
                ref_logp = token_logprobs(ref, x, cfg.bf16)

            # 4) a few clipped policy-gradient updates on these answers
            policy.train()
            for _ in range(cfg.ppo_epochs):
                perm = torch.randperm(len(ids), generator=gen)
                for s in range(0, len(ids), cfg.minibatch_size):
                    mb = perm[s: s + cfg.minibatch_size]
                    new_logp = token_logprobs(policy, x[mb], cfg.bf16)
                    loss, st = grpo_loss(new_logp, old_logp[mb], ref_logp[mb], adv_k[mb], mask[mb],
                                         cfg.clip_eps, cfg.kl_beta)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                    opt.step()
                    for k in stats:
                        stats[k] += st[k]
                    n_updates += 1
            stats = {k: v / n_updates for k, v in stats.items()}

        if it % cfg.log_interval == 0:
            logger.log(step=it, reward=round(rewards.mean().item(), 4), sample_acc=round(correct.mean().item(), 4),
                       frac_zero_std_groups=round(frac_zero, 3), kl=round(stats["kl"], 5),
                       clip_frac=round(stats["clip_frac"], 4),
                       mean_len=round(sum(len(c) for c in completions) / len(completions), 1))


if __name__ == "__main__":
    main()
