# Blender, headless, for low-poly world blockouts (pipeline/world/): scripted scenes and camera paths rendered to
# clay/layout videos that drive LTX-2.5's Layout-To-Render IC-LoRA. CPU (Eevee/Cycles CPU or Workbench) or NVIDIA.
#   docker build -t blender docker/blender
#   docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work blender -b -P script.py -- ARGS
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates ffmpeg \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libgl1 libegl1 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sL https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz | tar -xJ -C /opt \
    && ln -s /opt/blender-4.2.3-linux-x64/blender /usr/local/bin/blender
ENV HOME=/tmp
ENTRYPOINT ["blender"]
