FROM rocm/dev-ubuntu-22.04:5.3.3

COPY ./rocm-hip-cmake-clang_rt-builtins.patch /

RUN patch -p1 < rocm-hip-cmake-clang_rt-builtins.patch
