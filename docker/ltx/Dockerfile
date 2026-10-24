# LTX-2 / LTX-2.5 reference pipelines (github.com/Lightricks/LTX-2) in a uv venv. Code mounted at /work.
#   docker build -t ltx2 -f docker/ltx/Dockerfile ref/LTX-2
# Base = the CUDA image zoommorph already uses (local); uv supplies its own Python and torch wheels.
ARG BASE=nvidia/cuda:12.6.3-runtime-ubuntu22.04
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/uvpython
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
WORKDIR /ltx
COPY . /ltx
RUN uv sync --python 3.12 --package ltx-pipelines && chmod -R a+rX /opt/venv /opt/uvpython
ENV PATH=/opt/venv/bin:$PATH
RUN uv pip install --python /opt/venv/bin/python opencv-python-headless numexpr scikit-image
