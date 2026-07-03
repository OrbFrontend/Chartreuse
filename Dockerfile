# GPU inference image. torch 2.6.0 + CUDA 12.4 match the trained checkpoint
# (see docs/HANDOVER-phase4.md). Run with: docker run --gpus all -p 8000:8000 ...
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app.py index.html text_segmentation.py ./

# Which trained classifier to bake in. Tracks the encoder slug from core/paths.py;
# override per encoder/axis at build:  docker build --build-arg MODEL_DIR=models/ettin150m-purple ...
ARG MODEL_DIR=models/ettin400m-purple

# Only the files the model needs at inference + its OWN calibrated threshold (the one
# calibrate.py writes INSIDE the model dir) — not the checkpoint-* subdirs, training_args.bin,
# or the ONNX export. Baked to a fixed internal path so the image stops depending on the slug;
# app.py reads it via the MODEL_DIR env set below. The threshold COPY also fails the build loud
# if you forgot to calibrate — better than silently shipping the wrong operating point.
COPY ${MODEL_DIR}/config.json \
     ${MODEL_DIR}/model.safetensors \
     ${MODEL_DIR}/tokenizer.json \
     ${MODEL_DIR}/tokenizer_config.json \
     ${MODEL_DIR}/special_tokens_map.json \
     ${MODEL_DIR}/threshold.json \
     models/classifier/
ENV MODEL_DIR=models/classifier

# Model is baked in; never reach out to the Hub at load time.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 8000
# ponytail: single uvicorn worker — one model copy on the GPU serves all requests
# (sync endpoint runs in a threadpool; GPU serializes the forwards). Add workers
# or a batching queue only if one GPU worker measurably saturates.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
