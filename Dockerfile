# Serves the trained model. Build AFTER training (it copies runs/sft/ckpt.pt, the best model).
FROM python:3.12-slim

WORKDIR /app
# CPU-only PyTorch is much smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/tokenizer.json data/tokenizer.json
COPY runs/sft/ckpt.pt runs/sft/ckpt.pt

ENV PYTHONPATH=/app/src \
    MODEL_CKPT=runs/sft/ckpt.pt \
    TOKENIZER=data/tokenizer.json
EXPOSE 8000
CMD ["uvicorn", "tinyreasoner.serve:app", "--host", "0.0.0.0", "--port", "8000"]
