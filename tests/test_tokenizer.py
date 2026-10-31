from tinyreasoner.tokenizer import EOT, Tokenizer, get_stats, merge

TEXT = """The quick brown fox jumps over the lazy dog. The dog sleeps; the fox runs!
Q:347+586\nA:347+586:7+6+0=13|4+8+1=13|3+5+1=9|=933;<answer>933</answer>"""


def test_get_stats_and_merge():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_roundtrip_and_compression():
    tok = Tokenizer()
    tok.train(TEXT * 3, vocab_size=300)
    assert len(tok.merges) > 0
    for s in [TEXT, "hello world 12345", "ünïcödé ✓", ""]:
        assert tok.decode(tok.encode(s)) == s
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8"))  # BPE compresses


def test_digits_are_never_merged():
    tok = Tokenizer()
    tok.train(["1234 1234 1234 1234 5555 5555"] * 10, vocab_size=300)
    ids = tok.encode("1234")
    assert len(ids) == 4  # one token per digit, always


def test_special_token():
    tok = Tokenizer()
    tok.train(TEXT, vocab_size=280)
    ids = tok.encode("abc" + EOT)
    assert ids[-1] == tok.eot_id
    assert tok.decode(ids) == "abc" + EOT
    assert tok.vocab_size == 256 + len(tok.merges) + 1


def test_save_load(tmp_path):
    tok = Tokenizer()
    tok.train(TEXT, vocab_size=290)
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = Tokenizer.load(path)
    assert tok2.merges == tok.merges
    assert tok2.encode(TEXT) == tok.encode(TEXT)
    assert tok2.eot_id == tok.eot_id
