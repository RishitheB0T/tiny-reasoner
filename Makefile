# Shortcuts. Run `make all` to rebuild everything from scratch (about 2.5 hours on a 4-core CPU).
PY ?= .venv/bin/python
RUN = PYTHONPATH=src $(PY) -m

.PHONY: setup data tokenizer sft draft grpo lora bench eval passk plots train all test serve docker clean

setup:            ## create the virtual environment and install dependencies
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:             ## generate all synthetic datasets into data/
	$(RUN) tinyreasoner.data --out data

tokenizer:        ## train the BPE tokenizer
	$(RUN) tinyreasoner.tokenizer

sft:              ## stage 1: supervised fine-tuning (~1 hour)
	$(RUN) tinyreasoner.train_sft --config=configs/sft.json

draft:            ## small draft model for speculative decoding (~5 min)
	$(RUN) tinyreasoner.train_sft --config=configs/draft.json

grpo:             ## stage 2: reinforcement learning with GRPO (~45 min)
	$(RUN) tinyreasoner.grpo --config=configs/grpo.json

lora:             ## LoRA vs full fine-tuning on a new task (long division), without and with replay
	$(RUN) tinyreasoner.lora --config=configs/lora.json
	$(RUN) tinyreasoner.lora --config=configs/lora_replay.json
	$(RUN) tinyreasoner.evaluate --ckpt=runs/lora_replay/merged_ckpt.pt --out=results/eval_lora_replay_merged.json
	$(RUN) tinyreasoner.evaluate --ckpt=runs/lora_replay/full_ckpt.pt --out=results/eval_full_replay.json

eval:             ## accuracy of the SFT and GRPO models on the held-out test sets
	$(RUN) tinyreasoner.evaluate --ckpt=runs/sft/ckpt_step1500.pt --out=results/eval_sft_step1500.json
	$(RUN) tinyreasoner.evaluate --ckpt=runs/sft/ckpt.pt --out=results/eval_sft.json
	$(RUN) tinyreasoner.evaluate --ckpt=runs/grpo/ckpt.pt --out=results/eval_grpo.json

passk:            ## pass@1 vs pass@8 for the three models
	$(RUN) tinyreasoner.passk

bench:            ## KV cache / quantization / speculative decoding benchmarks
	$(RUN) tinyreasoner.benchmark --ckpt=runs/sft/ckpt.pt --draft=runs/draft/ckpt.pt

plots:            ## draw all figures into results/ and write results/RESULTS.md
	$(RUN) tinyreasoner.plots
	$(RUN) tinyreasoner.report

train: data tokenizer sft draft grpo eval passk lora bench plots

all: train test

test:             ## run the unit tests (fast, no trained model needed)
	$(PY) -m pytest -q

serve:            ## start the web app on http://localhost:8000
	PYTHONPATH=src $(PY) -m uvicorn tinyreasoner.serve:app --port 8000

docker:           ## build and run the web app in Docker
	docker build -t tinyreasoner . && docker run -p 8000:8000 tinyreasoner

clean:
	rm -rf runs results/*.png data/*.jsonl
