import json
import os

import pytest
import torch

os.environ["TINYREASONER_NO_AUTOLOAD"] = "1"  # don't load the real checkpoint on import

from fastapi.testclient import TestClient  # noqa: E402

from tinyreasoner.model import GPT, GPTConfig  # noqa: E402
from tinyreasoner.serve import create_app, expected_answer  # noqa: E402
from tinyreasoner.tokenizer import Tokenizer  # noqa: E402

TOK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tokenizer.json")


@pytest.fixture(scope="module")
def client():
    tok = Tokenizer.load(TOK_PATH)
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, n_layer=1, n_embd=32, n_head=4, n_kv_head=2))
    return TestClient(create_app(model, tok))


def test_expected_answer():
    assert expected_answer("347+586") == "933"
    assert expected_answer("12+34-5") == "41"
    assert expected_answer("47*36") == "1692"
    assert expected_answer("947/4") == "236r3"
    for bad in ["9**9**9", "import os", "1/0", ""]:
        with pytest.raises(ValueError):
            expected_answer(bad)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_generate(client):
    r = client.post("/generate", json={"question": "12+34", "max_new_tokens": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["expected"] == "46"
    assert body["tokens"] <= 10
    assert set(body) >= {"completion", "answer", "correct", "latency_ms"}


def test_generate_rejects_bad_input(client):
    assert client.post("/generate", json={"question": "hello"}).status_code == 422
    assert client.post("/generate", json={"question": "1+1", "temperature": 9}).status_code == 422


def test_stream(client):
    with client.stream("GET", "/generate/stream", params={"question": "12+34"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [e for e in text.split("\n\n") if e.strip()]
    assert events[-1].startswith("event: done")
    done = json.loads(events[-1].split("data: ", 1)[1])
    assert done["expected"] == "46"


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "TinyReasoner" in r.text
