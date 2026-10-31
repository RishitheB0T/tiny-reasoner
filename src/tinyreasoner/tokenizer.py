"""
Byte Pair Encoding (BPE) tokenizer, written from scratch (in the style of minbpe).

The idea in one line: start with raw bytes (256 "letters"), then repeatedly
glue the most frequent neighbouring pair into a brand new token.

    "<answer>"  ->  bytes  < a n s w e r >          (8 tokens)
                ->  after a few merges: "<" "answer" ">"   (3 tokens)

Before BPE runs, text is split into chunks with a regex (like GPT-2/GPT-4 do),
and merges never cross a chunk boundary. Our split pattern puts EVERY DIGIT in
its own chunk, so numbers are always one-token-per-digit. GPT-4 instead groups
up to 3 digits ("\\p{N}{1,3}"); Llama splits single digits like we do. Single
digits make arithmetic much easier to learn: "347" is always 3, 4, 7.
"""

import json
from collections import Counter
from functools import lru_cache

import regex as re

# GPT-4 style split pattern, except \p{N} matches ONE digit instead of {1,3}
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

EOT = "<|endoftext|>"  # marks the end of an answer; also used as padding


def get_stats(ids, counts=None, weight=1):
    """count consecutive pairs: [1, 2, 3, 1, 2] -> {(1,2): 2, (2,3): 1, (3,1): 1}"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def merge(ids, pair, idx):
    """replace every occurrence of `pair` in ids with the new token idx"""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self):
        self.pattern = SPLIT_PATTERN
        self.compiled = re.compile(self.pattern)
        self.merges = {}  # (int, int) -> int
        self.special_tokens = {}  # str -> int
        self.vocab = {i: bytes([i]) for i in range(256)}  # int -> bytes
        self._encode_chunk_cached = lru_cache(maxsize=100_000)(self._encode_chunk)

    # ------------------------------------------------------------------ train
    def train(self, text, vocab_size, verbose=False):
        """text: one string, or a list of separate documents (merges never cross documents)"""
        assert vocab_size >= 256
        docs = [text] if isinstance(text, str) else text
        # count each distinct chunk once (with its frequency) - much faster than
        # looping over the whole text for every merge, and gives the same result
        chunk_counts = Counter(ch for doc in docs for ch in self.compiled.findall(doc))
        chunks = [list(ch.encode("utf-8")) for ch in chunk_counts]
        freqs = list(chunk_counts.values())
        for i in range(vocab_size - 256):
            stats = {}
            for ids, f in zip(chunks, freqs):
                get_stats(ids, stats, f)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            if stats[pair] < 2:  # nothing left worth merging
                break
            idx = 256 + i
            chunks = [merge(ids, pair, idx) for ids in chunks]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}: {pair} -> {idx} ({self.vocab[idx]!r}) had {stats[pair]} occurrences")
        self.register_special_tokens({EOT: 256 + len(self.merges)})
        self._encode_chunk_cached.cache_clear()

    def register_special_tokens(self, special_tokens):
        self.special_tokens = dict(special_tokens)
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_id(self):
        return self.special_tokens[EOT]

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, chunk):
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # apply the merge that was learned EARLIEST (lowest new id) first
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return tuple(ids)

    def encode_ordinary(self, text):
        """encode text that contains no special tokens"""
        ids = []
        for chunk in self.compiled.findall(text):
            ids.extend(self._encode_chunk_cached(chunk))
        return ids

    def encode(self, text, allow_special=True):
        if not allow_special or not self.special_tokens:
            return self.encode_ordinary(text)
        special_re = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        ids = []
        for part in re.split(special_re, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    # ----------------------------------------------------------------- decode
    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.inverse_special:
                parts.append(self.inverse_special[i].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    # -------------------------------------------------------------- save/load
    def save(self, path):
        """merges are saved in order, which is all we need to rebuild the vocab"""
        with open(path, "w") as f:
            json.dump({
                "pattern": self.pattern,
                "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
                "special_tokens": self.special_tokens,
            }, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        tok = cls()
        tok.pattern = d["pattern"]
        tok.compiled = re.compile(tok.pattern)
        for a, b, idx in d["merges"]:
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        tok.register_special_tokens(d["special_tokens"])
        return tok


def main():
    import argparse

    from .data import load_jsonl

    ap = argparse.ArgumentParser(description="train the BPE tokenizer on the training data")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--vocab_size", type=int, default=320)
    ap.add_argument("--out", default="data/tokenizer.json")
    args = ap.parse_args()

    rows = load_jsonl(args.data)[:50_000]
    docs = [r["prompt"] + r["completion"] for r in rows]
    tok = Tokenizer()
    tok.train(docs, args.vocab_size, verbose=True)
    tok.save(args.out)

    ex = rows[0]["prompt"] + rows[0]["completion"]
    ids = tok.encode(ex)
    print(f"\nvocab size {tok.vocab_size} ({len(tok.merges)} merges + 256 bytes + {len(tok.special_tokens)} special)")
    print(f"example: {len(ex)} characters -> {len(ids)} tokens")
    print(" | ".join(repr(tok.decode([i])) for i in ids))
    assert tok.decode(ids) == ex


if __name__ == "__main__":
    main()
