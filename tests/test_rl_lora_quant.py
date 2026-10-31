import copy

import torch

from tinyreasoner.grpo import build_sequences, group_advantages, grpo_loss, token_logprobs
from tinyreasoner.lora import LoRALinear, apply_lora, count_trainable, merge_lora
from tinyreasoner.model import GPT, GPTConfig, generate
from tinyreasoner.quantize import Int8Linear, model_bytes, quantize_model
from tinyreasoner.rewards import correctness_reward, format_reward, total_reward
from tinyreasoner.speculative import speculative_generate
from tinyreasoner.utils import cut_at_eot


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=50, block_size=96, n_layer=2, n_embd=32, n_head=4, n_kv_head=2)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg)).eval()


# ----------------------------------------------------------------------------- rewards


def test_rewards():
    good = "1+2+0=3|=3;<answer>3</answer>"
    assert correctness_reward(good, "3") == 1.0
    assert format_reward(good) == 0.1
    assert total_reward(good, "3") == 1.1
    assert total_reward(good, "4") == 0.1  # nicely formatted but wrong: only the small format reward
    assert format_reward("<answer>3</answer> extra") == 0.0
    assert format_reward("<answer>3</answer><answer>3</answer>") == 0.0
    assert total_reward("garbage", "3") == 0.0


# ----------------------------------------------------------------------------- GRPO pieces


def test_group_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    a = group_advantages(r, 4)
    assert a[0] > 0 and (a[1:4] < 0).all()  # the only right answer in group 1 is pushed up
    assert torch.allclose(a[4:], torch.zeros(4))  # all-correct group: nothing to learn
    assert abs(a[:4].sum().item()) < 1e-5  # advantages are centred within a group


def test_build_sequences_mask():
    x, mask = build_sequences([[1, 2, 3], [4]], [[5, 6], [7, 8, 9]], pad_id=0)
    assert x.tolist() == [[1, 2, 3, 5, 6], [4, 7, 8, 9, 0]]
    # mask marks the positions whose TARGET is a completion token
    assert mask.tolist() == [[0, 0, 1, 1], [1, 1, 1, 0]]


def test_grpo_loss_direction():
    """one gradient step must raise the probability of an answer with positive advantage"""
    m = tiny()
    ref = copy.deepcopy(m)
    x, mask = build_sequences([[1, 2, 3]], [[4, 5, 6]], pad_id=0)
    with torch.no_grad():
        old = token_logprobs(m, x, False)
        refl = token_logprobs(ref, x, False)
    loss, stats = grpo_loss(token_logprobs(m, x, False), old, refl, torch.tensor([1.0]), mask, 0.2, 0.04)
    assert stats["kl"] < 1e-6 and stats["clip_frac"] == 0.0  # identical policies at the start
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        new = token_logprobs(m, x, False)
    assert ((new - old) * mask).sum() > 0


# ----------------------------------------------------------------------------- LoRA


def test_lora_starts_as_identity_and_merges_exactly():
    base = tiny()
    x = torch.randint(0, 50, (2, 10))
    ref_logits, _ = base(x)
    m = apply_lora(copy.deepcopy(base), r=4, alpha=8)
    assert torch.allclose(m(x)[0], ref_logits, atol=1e-5)  # B = 0 -> no change at start
    # train a little so B != 0, then check that merging gives the same outputs
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.lora_B, std=0.05)
    lora_logits, _ = m(x)
    assert not torch.allclose(lora_logits, ref_logits, atol=1e-4)
    merged = merge_lora(m)
    assert torch.allclose(merged(x)[0], lora_logits, atol=1e-4)


def test_lora_trainable_params_and_frozen_base():
    m = apply_lora(tiny(), r=4, alpha=8, train_token_ids=[7])
    n = count_trainable(m, [7])
    total = sum(p.numel() for p in tiny().parameters())
    assert 0 < n < total * 0.5
    x = torch.randint(0, 50, (2, 10))
    _, loss = m.train()(x, x)
    loss.backward()
    g = m.wte.weight.grad
    assert g[7].abs().sum() > 0 and g[torch.arange(50) != 7].abs().sum() == 0  # only the new row learns
    for name, p in m.named_parameters():
        if "base" in name:
            assert not p.requires_grad


# ----------------------------------------------------------------------------- quantization


def test_int8_quantization():
    m = tiny()
    q = quantize_model(copy.deepcopy(m))
    assert any(isinstance(mod, Int8Linear) for mod in q.modules())
    assert model_bytes(q) < model_bytes(m)
    x = torch.randint(0, 50, (2, 10))
    a, _ = m(x)
    b, _ = q(x)
    assert (a - b).abs().max() < 0.05 * a.abs().max()  # small rounding error only


# ----------------------------------------------------------------------------- speculative decoding


def test_speculative_decoding_equals_greedy():
    target, draft = tiny(0), tiny(1, n_layer=1, n_embd=16, n_head=2, n_kv_head=1)
    prompt = [1, 2, 3, 4]
    plain = generate(target, torch.tensor([prompt]), max_new_tokens=40, temperature=0)[0].tolist()
    for k in (1, 3, 5):
        spec, stats = speculative_generate(target, draft, prompt, max_new_tokens=40, k=k)
        assert spec == plain
        assert stats["tokens"] == 40
    # a draft identical to the target gets every guess accepted
    spec, stats = speculative_generate(target, copy.deepcopy(target), prompt, max_new_tokens=40, k=4)
    assert spec == plain and stats["acceptance_rate"] == 1.0


def test_speculative_respects_eos():
    target = tiny(0)
    prompt = [1, 2, 3]
    plain = generate(target, torch.tensor([prompt]), max_new_tokens=20, temperature=0)[0].tolist()
    eos = plain[5]
    spec, _ = speculative_generate(target, tiny(2), prompt, max_new_tokens=20, k=3, eos_id=eos)
    assert cut_at_eot(spec, eos) == cut_at_eot(plain, eos)
