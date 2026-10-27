# Blender 5.2.0, headless, the same version on every render box (2026-10-27, Ice Walk whole song split across gruntus,
# thing1, thing2 and quintus: "if you use different versions of blender you might get different results"). The DGX
# Sparks run the community aarch64 build of 5.2.0 (github.com/CoconutMacaroon/blender-arm64, docker/blender52-arm),
# quintus the official macOS build.
#   docker build -t blender52 docker/blender52
#   docker run --rm --gpus all --user $(id -u):$(id -g) -v $PWD:/work -w /work blender52 -b -P script.py -- ARGS
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates ffmpeg \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libgl1 libegl1 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sL https://download.blender.org/release/Blender5.2/blender-5.2.0-linux-x64.tar.xz | tar -xJ -C /opt \
    && ln -s /opt/blender-5.2.0-linux-x64/blender /usr/local/bin/blender
ENV HOME=/tmp
ENTRYPOINT ["blender"]
