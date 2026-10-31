"""
A small Llama-style language model, written from scratch.

If you built GPT along with "Let's build GPT" / nanoGPT, this is the same
skeleton (token embedding -> N transformer blocks -> final norm -> lm_head)
with the four upgrades that modern open models (Llama, Mistral, Qwen) use:

  1. RoPE        positions are injected by ROTATING the query/key vectors,
                 instead of adding a learned position embedding (wpe).
  2. RMSNorm     a simpler LayerNorm: only rescales, no mean subtraction, no bias.
  3. SwiGLU      the MLP gets a "gate": w2( silu(w1 x) * w3 x ) instead of
                 c_proj( gelu(c_fc x) ).
  4. GQA         grouped-query attention: many query heads SHARE a few key/value
                 heads, so the KV cache is several times smaller.

Plus a KV cache for fast generation, and support for left-padded batches.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 256  # max sequence length
    vocab_size: int = 320
    n_layer: int = 6
    n_head: int = 8  # query heads
    n_kv_head: int = 2  # key/value heads (GQA). n_kv_head == n_head is normal attention
    n_embd: int = 256
    ffn_hidden: int = 0  # 0 -> about 8/3 * n_embd, like Llama
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        if not self.ffn_hidden:
            self.ffn_hidden = 32 * math.ceil(8 * self.n_embd / 3 / 32)


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2)) * weight   (LayerNorm without the mean and the bias)"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ----------------------------------------------------------------------------- RoPE


def precompute_rope(head_dim, max_len, theta=10000.0):
    """cos/sin tables of shape (max_len, head_dim). Each pair of dimensions spins at its own speed."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (max_len, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """rotate x (B, nh, T, hd) by the angle of each token's position"""
    return x * cos + rotate_half(x) * sin


# ----------------------------------------------------------------------------- KV cache


class KVCache:
    """
    Remembers the keys and values of tokens we already processed, so that when
    generating token t+1 we only run the model on ONE new token instead of
    re-running all t previous ones. Stores n_kv_head heads (not n_head): that is
    exactly the memory saving GQA buys you.
    """

    def __init__(self, n_layer):
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    @property
    def length(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer, k, v):
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)
            v = torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v

    def truncate(self, n):
        """forget everything after the first n tokens (used by speculative decoding)"""
        self.k = [k[:, :, :n] if k is not None else None for k in self.k]
        self.v = [v[:, :, :n] if v is not None else None for v in self.v]


# ----------------------------------------------------------------------------- blocks


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_kv_head = config.n_head, config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, mask, cache, layer):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, nkv, T, hd)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer, k, v)  # now (B, nkv, past+T, hd)
        # GQA: every group of n_head/n_kv_head query heads reads the same k/v head
        rep = self.n_head // self.n_kv_head
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        drop = self.dropout if self.training else 0.0
        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))


class SwiGLU(nn.Module):
    """w2( silu(w1 x) * w3 x ): w1 makes a 'gate' that decides how much of w3 x passes through"""

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.w3 = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.w2 = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))"""

    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, mask, cache, layer):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask, cache, layer)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ----------------------------------------------------------------------------- the model


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying, as in GPT-2

        cos, sin = precompute_rope(config.n_embd // config.n_head, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 trick: scale down the projections that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weights are counted once

    def forward(self, idx, targets=None, attn_mask=None, cache=None, last_only=False):
        """
        idx:       (B, T) token ids
        targets:   (B, T) next-token ids, -100 where we don't want a loss
        attn_mask: (B, past+T) 1 for real tokens, 0 for padding (needed for left-padded batches)
        cache:     a KVCache; tokens in idx come right after the ones already in the cache
        """
        B, T = idx.shape
        past = cache.length if cache is not None else 0
        assert past + T <= self.config.block_size, f"sequence too long: {past + T} > {self.config.block_size}"

        # each token's position. With left padding, real tokens start counting at 0 after the pads
        if attn_mask is not None:
            pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)[:, past:]
        else:
            pos = torch.arange(past, past + T, device=idx.device).expand(B, T)
        cos = self.rope_cos[pos].unsqueeze(1).to(self.wte.weight.dtype)  # (B, 1, T, hd)
        sin = self.rope_sin[pos].unsqueeze(1).to(self.wte.weight.dtype)

        mask = None  # None -> plain causal attention (fast path)
        if attn_mask is not None or past > 0:
            mask = self._attention_mask(attn_mask, past, T, idx.device)

        x = self.drop(self.wte(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, mask, cache, i)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            return logits, loss
        if last_only:
            x = x[:, [-1], :]  # during generation we only need the very last position
        return self.lm_head(x), None

    @staticmethod
    def _attention_mask(attn_mask, past, T, device):
        """(B or 1, 1, T, past+T) boolean: True = this query may look at this key"""
        q_idx = torch.arange(past, past + T, device=device)[:, None]
        k_idx = torch.arange(past + T, device=device)[None, :]
        allowed = (k_idx <= q_idx)[None, None]  # causal: never look into the future
        if attn_mask is not None:
            allowed = allowed & attn_mask.bool()[:, None, None, :]  # never look at padding
            allowed = allowed | (k_idx == q_idx)[None, None]  # (a pad token may see itself: avoids NaN)
        return allowed

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """AdamW; weight decay only on matrices (not on norms), as in nanoGPT"""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)


# ----------------------------------------------------------------------------- sampling


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """pick the next token from logits (B, V). temperature 0 = greedy (always the most likely)"""
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and top_p < 1.0:
        # keep the smallest set of tokens whose probabilities add up to top_p ("nucleus")
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        drop = sorted_p.cumsum(-1) - sorted_p > top_p
        sorted_p = sorted_p.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_i, sorted_p)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(model, idx, attn_mask=None, max_new_tokens=128, temperature=1.0, top_k=None,
             top_p=None, eos_id=None, use_cache=True, generator=None):
    """
    idx: (B, T) LEFT-padded prompts (padding on the left so every prompt ends at
    the same column and new tokens line up). Returns (B, n_new) new tokens; after a
    sequence emits eos_id it keeps emitting eos_id.
    """
    B = idx.shape[0]
    if attn_mask is None:
        attn_mask = torch.ones_like(idx)
    cache = KVCache(model.config.n_layer) if use_cache else None
    seq, mask, inp = idx, attn_mask, idx
    finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
    out = []
    for _ in range(max_new_tokens):
        if seq.shape[1] >= model.config.block_size:
            break
        if use_cache:
            logits, _ = model(inp, attn_mask=mask, cache=cache, last_only=True)
        else:  # recompute everything from scratch every step (slow; for comparison)
            logits, _ = model(seq, attn_mask=mask, last_only=True)
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        if eos_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            finished |= nxt == eos_id
        out.append(nxt)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones_like(nxt)[:, None]], dim=1)
        inp = nxt[:, None]
        if eos_id is not None and finished.all():
            break
    return torch.stack(out, dim=1) if out else idx.new_zeros(B, 0)
