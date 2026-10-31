import random

import pytest

from tinyreasoner.data import (DIV_LEVELS, LEVELS, add_steps, div1_steps, extract_answer, generate,
                               is_correct, make_problem, mul1_steps, sub_steps)


def test_add_steps_carry():
    steps, r = add_steps(999, 1)
    assert r == 1000
    assert steps == ["9+1+0=10", "9+0+1=10", "9+0+1=10"]


def test_sub_steps_borrow():
    steps, r = sub_steps(905, 387)
    assert r == 518
    assert steps == ["5-7-0=8b1", "0-8-1=1b1", "9-3-1=5b0"]


def test_mul1_and_div1():
    assert mul1_steps(347, 6) == (["7*6+0=42", "4*6+4=28", "3*6+2=20"], 2082)
    assert div1_steps(947, 4) == (["9/4=2r1", "14/4=3r2", "27/4=6r3"], "236r3")


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_every_problem_is_correct(level):
    """the final answer in the worked solution must match Python's own arithmetic"""
    rng = random.Random(level)
    for _ in range(300):
        p = make_problem(rng, level)
        q = p["question"]
        assert extract_answer(p["completion"]) == p["answer"] == str(eval(q))
        assert p["prompt"] == f"Q:{q}\nA:"


def test_division_problems_are_correct():
    rng = random.Random(0)
    for _ in range(300):
        p = make_problem(rng, rng.choice(list(DIV_LEVELS)), DIV_LEVELS)
        a, m = map(int, p["question"].split("/"))
        assert p["answer"] == f"{a // m}r{a % m}"


def test_generate_is_reproducible_and_respects_exclude():
    a = generate(50, {1: 1, 2: 1}, seed=7)
    b = generate(50, {1: 1, 2: 1}, seed=7)
    assert a == b
    c = generate(50, {1: 1, 2: 1}, seed=7, exclude={p["question"] for p in a[:10]})
    assert not {p["question"] for p in a[:10]} & {p["question"] for p in c}


def test_answer_checking():
    assert is_correct("1+2+0=3|=3;<answer>3</answer>", "3")
    assert not is_correct("1+2+0=3|=3;<answer>4</answer>", "3")
    assert extract_answer("no answer tag") is None
