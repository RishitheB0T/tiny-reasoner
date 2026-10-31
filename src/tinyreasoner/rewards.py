"""
Reward functions for reinforcement learning. A reward is just a number that says
"how good was this answer?". Ours are VERIFIABLE: plain code checks the answer,
no human and no learned reward model needed (the DeepSeek-R1 recipe).

  correctness:  1.0 if the final <answer> is exactly right, else 0.0
  format:       0.1 if the output has exactly one well-formed <answer>N</answer>
                at the very end, else 0.0

The format reward is deliberately small: if it were big, the model could
"reward hack" by producing nicely formatted WRONG answers.
"""

import re

from .data import extract_answer

FORMAT_RE = re.compile(r"^[^<>]*<answer>[0-9r\-]+</answer>$")

CORRECT_REWARD = 1.0
FORMAT_REWARD = 0.1


def correctness_reward(completion, answer):
    return CORRECT_REWARD if extract_answer(completion) == answer else 0.0


def format_reward(completion):
    return FORMAT_REWARD if FORMAT_RE.match(completion) else 0.0


def total_reward(completion, answer):
    return correctness_reward(completion, answer) + format_reward(completion)
