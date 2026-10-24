# Analysis image: Demucs stems, tempo, whisper-timestamped (pipeline/analyse.py), beat grid, Whisper scoring.
#   docker build -t m2v-pre docker/analysis
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python
RUN pip3 install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install "numpy<2" librosa soundfile tqdm requests \
    "git+https://github.com/boxabirds/demucs#egg=demucs" \
    "git+https://github.com/linto-ai/whisper-timestamped#egg=whisper-timestamped" openai-whisper
WORKDIR /m2v
