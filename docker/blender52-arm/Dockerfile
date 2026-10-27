# Blender 5.2.0 on the DGX Sparks (aarch64, GB10; 2026-10-27, Ice Walk whole song across four boxes): the community build
# github.com/CoconutMacaroon/blender-arm64 (release v14-5.2.0, CUDA + OptiX), unpacked in ~/poptok-blender/cmake-make on
# the host and mounted at /opt/b; this image only adds the Ubuntu 24.04 libraries it links against (the rest are in its
# libExt/). Same version as docker/blender52 (gruntus) and quintus's official macOS build.
# The community build ships no OIDN CPU device, so it rendered un-denoised: OIDN 2.5.0 (the version official 5.2.0 ships)
# is built from source for aarch64 (RenderKit/oidn v2.5.0, ISPC 1.32 aarch64, -DOIDN_DEVICE_CPU=ON) into
# ~/poptok-blender/oidn-lib and mounted first on the library path; then a Spark's frame matches gruntus's to 0.03/255.
#   docker build -t blender52 docker/blender52-arm
#   docker run --rm --gpus all -v ~/poptok-blender/cmake-make:/opt/b:ro -v ~/poptok-blender/oidn-lib:/opt/oidn:ro \
#     -e LD_LIBRARY_PATH=/opt/oidn:/opt/b/libExt:/opt/b/lib --user $(id -u):$(id -g) -v $PWD:/work -w /work \
#     blender52 -b -P script.py -- ARGS
FROM nvidia/cuda:13.0.1-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
      libgoogle-glog0v6t64 libgflags2.2 libmetis5 libpng16-16t64 libjpeg-turbo8 libwebp7 libfreetype6 libvulkan1 libspnav0 \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libgl1 libegl1 libopenjp2-7 libpotrace0 \
      libosdcpu3.5.0t64 libosdgpu3.5.0t64 libpugixml1v5 libepoxy0 libfftw3-single3 libfftw3-double3 libopenexr-3-1-30 \
      libimath-3-1-29t64 libblosc1 libboost-iostreams1.83.0 libavcodec60 libavformat60 libavutil58 libswscale7 libgif7 \
      libheif1 libraw23t64 libtiff6 libwebpdemux2 libwebpmux3 libexpat1 libyaml-cpp0.8 \
    && (apt-get install -y --no-install-recommends libpystring0 || true) && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/b/libExt:/opt/b/lib HOME=/tmp NVIDIA_DRIVER_CAPABILITIES=all
ENTRYPOINT ["/opt/b/bin/blender"]
