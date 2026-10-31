"""
Serve the model as a web API (FastAPI = Python's equivalent of a Spring Boot REST controller).

    uvicorn tinyreasoner.serve:app --port 8000      (then open http://localhost:8000)

Endpoints
    GET  /health                       is the server up? which model is loaded?
    POST /generate                     {"question": "347+586"} -> the full worked answer as JSON
    GET  /generate/stream?question=..  the answer token-by-token as Server-Sent Events (like ChatGPT typing)
    GET  /                             a tiny web page to try it
"""

import json
import os
import re
import time

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .data import extract_answer, make_prompt
from .model import KVCache, sample_next
from .utils import load_model, load_tokenizer

CHAIN_RE = re.compile(r"^\d{1,6}([+\-]\d{1,6}){1,4}$")  # 12+345-6
MULDIV_RE = re.compile(r"^\d{1,6}[*/]\d{1,3}$")  # 347*6, 347*36, 947/4


def expected_answer(question):
    """compute the true answer WITHOUT eval(), so the API can say whether the model was right"""
    if CHAIN_RE.match(question):
        nums = [int(n) for n in re.findall(r"\d+", question)]
        ops = re.findall(r"[+\-]", question)
        total = nums[0]
        for op, n in zip(ops, nums[1:]):
            total = total + n if op == "+" else total - n
        return str(total)
    if MULDIV_RE.match(question):
        a, op, b = re.match(r"(\d+)([*/])(\d+)", question).groups()
        a, b = int(a), int(b)
        if op == "*":
            return str(a * b)
        if b == 0:
            raise ValueError("division by zero")
        return f"{a // b}r{a % b}"
    raise ValueError("unsupported question")


class GenerateRequest(BaseModel):
    question: str = Field(..., examples=["347+586"], max_length=40)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    max_new_tokens: int = Field(220, ge=1, le=250)


def stream_tokens(model, tok, question, temperature=0.0, top_k=None, top_p=None, max_new_tokens=220):
    """yields the generated text one token at a time (KV cache: one tiny forward pass per token)"""
    ids = tok.encode(make_prompt(question))
    cache = KVCache(model.config.n_layer)
    x = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(min(max_new_tokens, model.config.block_size - len(ids))):
            logits, _ = model(x, cache=cache, last_only=True)
            nxt = int(sample_next(logits[:, -1, :], temperature, top_k, top_p))
            if nxt == tok.eot_id:
                return
            yield tok.decode([nxt])
            x = torch.tensor([[nxt]])


def create_app(model=None, tok=None):
    app = FastAPI(title="TinyReasoner", description="A tiny from-scratch LLM that shows its working.")
    ckpt = os.environ.get("MODEL_CKPT", "runs/sft/ckpt.pt")
    if model is None:
        model = load_model(ckpt)
    if tok is None:
        tok = load_tokenizer(os.environ.get("TOKENIZER", "data/tokenizer.json"))
    model.eval()

    def check(question):
        q = question.replace(" ", "")
        try:
            return q, expected_answer(q)
        except ValueError as e:
            raise HTTPException(422, f"{e}. Try things like 347+586, 9021-4567, 347*6, 47*36, 52+34-17, 947/4")

    @app.get("/health")
    def health():
        return {"status": "ok", "params": model.num_params(), "checkpoint": ckpt}

    @app.post("/generate")
    def generate_endpoint(req: GenerateRequest):
        q, expected = check(req.question)
        t0 = time.perf_counter()
        pieces = list(stream_tokens(model, tok, q, req.temperature, req.top_k, req.top_p, req.max_new_tokens))
        completion = "".join(pieces)
        answer = extract_answer(completion)
        return {"question": q, "completion": completion, "answer": answer, "expected": expected,
                "correct": answer == expected, "tokens": len(pieces),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    @app.get("/generate/stream")
    def generate_stream(question: str = Query(..., max_length=40), temperature: float = Query(0.0, ge=0, le=2)):
        q, expected = check(question)

        def events():
            text = ""
            for piece in stream_tokens(model, tok, q, temperature):
                text += piece
                yield f"data: {json.dumps({'token': piece})}\n\n"
            answer = extract_answer(text)
            yield f"event: done\ndata: {json.dumps({'answer': answer, 'expected': expected, 'correct': answer == expected})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    return app


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TinyReasoner</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e4e2dc;--accent:#2f6fde;--ok:#1f8a4c;--bad:#c2392b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a09f99;--card:#1f1f1d;--line:#34332f;--accent:#7aa7ff;--ok:#5cc98a;--bad:#ff7b6b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:32px 16px}h1{margin:0 0 4px;font-size:1.6rem}p{color:var(--muted);margin:0 0 20px}
form{display:flex;gap:8px;flex-wrap:wrap}input{flex:1;min-width:0;padding:10px 12px;font:inherit;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
button{padding:10px 16px;font:inherit;border:0;border-radius:8px;background:var(--accent);color:#fff;cursor:pointer}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;white-space:pre-wrap;word-break:break-all;min-height:4em;font:14px/1.6 ui-monospace,monospace}
#verdict{font-weight:600}.ok{color:var(--ok)}.bad{color:var(--bad)}.chips{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}
.chips button{background:transparent;color:var(--accent);border:1px solid var(--line);padding:4px 10px;font-size:.9rem}
</style></head><body><main>
<h1>TinyReasoner</h1><p>A 4.3M-parameter language model built from scratch and trained to show its working.
It only knows the problem shapes it was trained on: +/- with equal-length numbers, multiply by 1 or 2 digits, and
a+b-c with 2-digit numbers.</p>
<form id="f"><input id="q" value="4729+8356" aria-label="question"><button>Solve</button></form>
<div class="chips"></div>
<pre id="out"></pre><div id="verdict"></div>
<script>
const ex=["4729+8356","90210-45678","3476*7","47*36","52+34-17","123*45"];
const chips=document.querySelector(".chips");
ex.forEach(e=>{const b=document.createElement("button");b.type="button";b.textContent=e;b.onclick=()=>{q.value=e;run()};chips.appendChild(b)});
let es;
function run(){
  if(es)es.close();out.textContent="";verdict.textContent="";verdict.className="";
  es=new EventSource("/generate/stream?question="+encodeURIComponent(q.value));
  es.onmessage=m=>{out.textContent+=JSON.parse(m.data).token.replace(/\\|/g,"|\\n").replace(/;/g,";\\n")};
  es.addEventListener("done",m=>{const d=JSON.parse(m.data);verdict.textContent=(d.correct?"Correct: ":"Wrong: expected ")+d.expected;verdict.className=d.correct?"ok":"bad";es.close()});
  es.onerror=()=>{es.close();if(!out.textContent)out.textContent="error: check the question format"};
}
f.onsubmit=e=>{e.preventDefault();run()};
</script></main></body></html>"""


try:
    app = create_app() if os.environ.get("TINYREASONER_NO_AUTOLOAD") != "1" else None
except FileNotFoundError:
    app = None  # no trained checkpoint yet; tests build their own app with create_app()
