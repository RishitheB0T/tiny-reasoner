"""small helpers shared by all the training scripts"""

import dataclasses
import json
import os
import random
import sys
import time

import numpy as np
import torch

from .model import GPT, GPTConfig
from .tokenizer import Tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_torch(threads=4):
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("high")


# ----------------------------------------------------------------------------- config


def load_config(cls, argv=None):
    """
    Build a dataclass config from (1) its defaults, (2) an optional --config file.json,
    (3) --key=value overrides on the command line. Like nanoGPT's configurator.py.
    """
    argv = sys.argv[1:] if argv is None else argv
    values = {}
    for arg in argv:
        if arg.startswith("--config="):
            with open(arg.split("=", 1)[1]) as f:
                values.update(json.load(f))
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for arg in argv:
        if arg.startswith("--config="):
            continue
        assert arg.startswith("--") and "=" in arg, f"expected --key=value, got {arg}"
        key, val = arg[2:].split("=", 1)
        assert key in fields, f"unknown config key {key}"
        default = fields[key].default
        try:
            values[key] = json.loads(val)
        except json.JSONDecodeError:
            values[key] = val  # plain strings
        if isinstance(default, float) and isinstance(values[key], int):
            values[key] = float(values[key])
    return cls(**values)


# ----------------------------------------------------------------------------- batching


def encode_example(tok, problem):
    """token ids of prompt and of completion (+ end-of-text) for one problem"""
    p = tok.encode(problem["prompt"])
    c = tok.encode(problem["completion"]) + [tok.eot_id]
    return p, c


def right_pad(seqs, pad_id):
    """[[1,2,3],[4]] -> [[1,2,3],[4,pad,pad]] (used for training)"""
    T = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_id] * (T - len(s)) for s in seqs], dtype=torch.long)


def left_pad(seqs, pad_id):
    """[[1,2,3],[4]] -> [[1,2,3],[pad,pad,4]] plus a mask of real tokens (used for generation)"""
    T = max(len(s) for s in seqs)
    idx = torch.tensor([[pad_id] * (T - len(s)) + s for s in seqs], dtype=torch.long)
    mask = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs], dtype=torch.long)
    return idx, mask


def cut_at_eot(ids, eot_id):
    """generated ids up to (not including) the first end-of-text token"""
    ids = list(ids)
    return ids[: ids.index(eot_id)] if eot_id in ids else ids


# ----------------------------------------------------------------------------- checkpoints


def save_checkpoint(path, model, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(model.config),
                **(extra or {})}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_tokenizer(path="data/tokenizer.json"):
    return Tokenizer.load(path)


# ----------------------------------------------------------------------------- logging


class MetricsLogger:
    """appends one JSON line per log call to <run_dir>/metrics.jsonl (simple experiment tracking)"""

    def __init__(self, run_dir, config=None):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "metrics.jsonl")
        open(self.path, "w").close()
        if config is not None:
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(dataclasses.asdict(config), f, indent=2)
        self.t0 = time.time()

    def log(self, **metrics):
        metrics["time_s"] = round(time.time() - self.t0, 1)
        with open(self.path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)


def read_metrics(path):
    with open(path) as f:
        return [json.loads(line) for line in f]
