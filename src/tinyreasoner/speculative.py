"""
Speculative decoding (greedy version), written from scratch.

Generating text is slow because the big model must run once PER TOKEN, one
after another. Trick: let a small, fast "draft" model guess the next k tokens,
then let the big "target" model check all k guesses in ONE forward pass
(checking k tokens at once costs about the same as generating one).

    draft guesses:     7  +  6  +  0  =  1
    target agrees:     7  +  6  +  0  =  (target says 1? yes) -> accept all, plus 1 bonus token
    first mismatch:    keep the agreed prefix, then use the target's own token there

With greedy decoding the output is EXACTLY what the target alone would have
written - we only save time, never change the answer.
"""

import torch

from .model import KVCache


@torch.no_grad()
def speculative_generate(target, draft, prompt_ids, max_new_tokens=128, k=4, eos_id=None):
    """batch size 1. returns (new_token_ids, stats)"""
    device = next(target.parameters()).device
    t_cache, d_cache = KVCache(target.config.n_layer), KVCache(draft.config.n_layer)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    # prefill both caches with everything except the last prompt token
    if prompt.shape[1] > 1:
        target(prompt[:, :-1], cache=t_cache)
        draft(prompt[:, :-1], cache=d_cache)
    last = prompt_ids[-1]  # the newest token, not yet in either cache
    out, proposed, accepted, target_calls = [], 0, 0, 0
    block = target.config.block_size

    while len(out) < max_new_tokens:
        committed = t_cache.length  # number of tokens both caches agree on
        kk = min(k, max_new_tokens - len(out), block - committed - 2)
        if kk <= 0:
            break
        # 1) draft proposes kk tokens, one at a time (cheap)
        guesses, tok = [], last
        for _ in range(kk):
            logits, _ = draft(torch.tensor([[tok]], device=device), cache=d_cache)
            tok = int(logits[0, -1].argmax())
            guesses.append(tok)
        # 2) target scores [last, g1..gk] in ONE call -> its own choice after each position
        logits, _ = target(torch.tensor([[last] + guesses], device=device), cache=t_cache)
        target_calls += 1
        choice = logits[0].argmax(-1).tolist()  # choice[i] = target's token after position i
        # 3) accept the longest prefix where draft and target agree
        n = 0
        while n < kk and guesses[n] == choice[n]:
            n += 1
        new = guesses[:n] + [choice[n]]  # agreed tokens + target's correction (or a bonus token)
        proposed += kk
        accepted += n
        # 4) roll caches back to: committed + last + accepted guesses
        t_cache.truncate(committed + 1 + n)
        d_cache.truncate(committed + 1 + n)
        if d_cache.length < committed + 1 + n:  # all kk accepted: draft never saw its own last guess
            draft(torch.tensor([[guesses[-1]]], device=device), cache=d_cache)
        for t in new:
            out.append(t)
            if (eos_id is not None and t == eos_id) or len(out) >= max_new_tokens:
                break
        if eos_id is not None and out[-1] == eos_id:
            break
        last = out[-1]

    stats = {"proposed": proposed, "accepted": accepted, "target_calls": target_calls,
             "acceptance_rate": accepted / max(proposed, 1), "tokens": len(out)}
    return out, stats
