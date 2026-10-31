import torch

from tinyreasoner.model import GPT, GPTConfig, KVCache, apply_rope, generate, precompute_rope, sample_next
from tinyreasoner.utils import left_pad


def tiny(**kw):
    torch.manual_seed(0)
    cfg = dict(vocab_size=50, block_size=64, n_layer=2, n_embd=32, n_head=4, n_kv_head=2)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg)).eval()


def test_shapes_and_loss():
    m = tiny()
    x = torch.randint(0, 50, (3, 10))
    y = torch.randint(0, 50, (3, 10))
    logits, loss = m(x, y)
    assert logits.shape == (3, 10, 50)
    # with random weights every token is about equally likely, so loss ~ ln(vocab_size)
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.5


def test_ignore_index_masks_loss():
    m = tiny()
    x = torch.randint(0, 50, (2, 8))
    y = torch.full_like(x, -100)
    y[:, -1] = x[:, -1]
    _, loss_one = m(x, y)
    logits, _ = m(x)
    expected = torch.nn.functional.cross_entropy(logits[:, -1], x[:, -1])
    assert torch.allclose(loss_one, expected, atol=1e-5)


def test_weight_tying():
    m = tiny()
    assert m.lm_head.weight is m.wte.weight


def test_causality():
    """changing a future token must not change earlier outputs"""
    m = tiny()
    x = torch.randint(0, 50, (1, 12))
    y = x.clone()
    y[0, 8] = (y[0, 8] + 1) % 50
    a, _ = m(x)
    b, _ = m(y)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)
    assert not torch.allclose(a[:, 8:], b[:, 8:], atol=1e-5)


def test_rope_depends_only_on_relative_position():
    """<rot(q, m), rot(k, n)> depends on m - n only: the whole point of RoPE"""
    cos, sin = precompute_rope(16, 64)
    q, k = torch.randn(16), torch.randn(16)

    def score(m, n):
        qm = apply_rope(q.view(1, 1, 1, 16), cos[m], sin[m]).flatten()
        kn = apply_rope(k.view(1, 1, 1, 16), cos[n], sin[n]).flatten()
        return (qm @ kn).item()

    assert abs(score(5, 2) - score(40, 37)) < 1e-4
    assert abs(score(5, 2) - score(5, 3)) > 1e-4


def test_kv_cache_matches_full_forward():
    m = tiny()
    x = torch.randint(0, 50, (2, 15))
    full, _ = m(x)
    cache = KVCache(m.config.n_layer)
    out = [m(x[:, :6], cache=cache)[0]]
    for t in range(6, 15):
        out.append(m(x[:, t: t + 1], cache=cache)[0])
    assert torch.allclose(full, torch.cat(out, 1), atol=1e-4)
    # GQA: the cache stores n_kv_head (2) heads, not n_head (4)
    assert cache.k[0].shape == (2, 2, 15, 8)


def test_generate_cache_equals_no_cache_and_padding_is_invisible():
    m = tiny()
    idx, mask = left_pad([[1, 2, 3, 4, 5], [6, 7]], pad_id=0)
    a = generate(m, idx, mask, max_new_tokens=12, temperature=0, use_cache=True)
    b = generate(m, idx, mask, max_new_tokens=12, temperature=0, use_cache=False)
    assert torch.equal(a, b)
    alone = generate(m, torch.tensor([[6, 7]]), max_new_tokens=12, temperature=0)
    assert torch.equal(a[1], alone[0])


def test_generate_stops_at_eos():
    m = tiny()
    out = generate(m, torch.tensor([[1, 2]]), max_new_tokens=30, temperature=0, eos_id=int(
        generate(m, torch.tensor([[1, 2]]), max_new_tokens=1, temperature=0)[0, 0]))
    assert out.shape[1] == 1  # the very first greedy token was the "eos"


def test_sampling_filters():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    g = torch.Generator().manual_seed(0)
    picks = {int(sample_next(logits, 1.0, top_k=2, generator=g)) for _ in range(200)}
    assert picks <= {3, 4}
    assert int(sample_next(logits, 0)) == 4  # greedy
    peaked = torch.tensor([[0.0, 0.0, 0.0, 10.0, 0.0]])
    picks = {int(sample_next(peaked, 1.0, top_p=0.5, generator=g)) for _ in range(100)}
    assert picks == {3}
