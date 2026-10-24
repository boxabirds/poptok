# LongCat-Video-Avatar 1.5 (Meituan; audio-driven singing/talking video from a photo, Whisper-large-v3 audio, 8-step
# distilled) for the DGX Spark (ARM64, GB10). Built on the infinitetalk image, which already has torch cu130 and xformers
# compiled for sm_121. LongCat defaults to flash-attn 2 (not built for GB10); LONGCAT_ATTN=xformers switches it.
#   docker build -t longcat docker/longcat                (needs the infinitetalk image; a few minutes)
# Weights in ~/models/longcat: LongCat-Video (tokenizer, text_encoder, vae, scheduler) and LongCat-Video-Avatar-1.5
# (base_model, lora, whisper-large-v3, vocal_separator). The demo scripts expect them side by side.
FROM infinitetalk
RUN pip3 install "audio-separator==0.30.2" onnxruntime av loguru psutil \
    && python3 -c "import audio_separator.separator, onnxruntime, av"
RUN git clone https://github.com/meituan-longcat/LongCat-Video.git /opt/LongCat-Video && cd /opt/LongCat-Video \
    && git checkout 6b3f4b8
WORKDIR /opt/LongCat-Video
COPY attn_override.py /tmp/attn_override.py
RUN python3 /tmp/attn_override.py longcat_video/modules/avatar/attention.py longcat_video/modules/attention.py
ENV LONGCAT_ATTN=xformers PYTHONPATH=/opt/LongCat-Video
