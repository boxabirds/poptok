# Model downloader: plain Python + huggingface_hub, no GPU (so poptok's GPU-busy checks ignore it).
#   docker build -t hfdl docker/hfdl
#   docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v ~/models:/models hfdl python -c "from huggingface_hub import snapshot_download; ..."
FROM docker.io/library/python:3.12-slim
RUN pip install --no-cache-dir "huggingface_hub[hf_transfer]"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
