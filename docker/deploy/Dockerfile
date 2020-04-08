ARG BUILD_ENV

FROM $BUILD_ENV as builder

# Build DLA-Future
COPY . /DLA-Future

ENV MKL_LIB=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64

RUN mkdir /DLA-Future/build && cd /DLA-Future/build && \
    CC=mpicc CXX=mpicxx cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DDLAF_WITH_CUDA=ON \
      -DDLAF_WITH_MKL=ON \
      -DDLAF_WITH_TEST=ON \
      -DDLAF_BUILD_MINIAPPS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr && \
      make -j$(nproc)
#      /root/bundler/bundler \
#        -d /root/DLA-Future.bundle/ \ 
#        -e test/unit/DLA-Futuretest_blas_tile \
#        -e test/unit/test_lapack_tile \
#        -e test/unit/test_matrix \
#        -e test/unit/test_tile \
#        -e test/unit/test_util_math \
#        -l ${MKL_LIB}/libmkl_tbb_thread.so \
#        -l ${MKL_LIB}/libmkl_intel_thread.so \
#        -l ${MKL_LIB}/libmkl_sequential.so \
#        -l ${MKL_LIB}/libmkl_def.so \
#        -l ${MKL_LIB}/libmkl_vml_def.so \
#        -l ${MKL_LIB}/libmkl_vml_cmpt.so \
#        -l ${MKL_LIB}/libmkl_mc.so \
#        -l ${MKL_LIB}/libmkl_vml_mc.so \
#        -l ${MKL_LIB}/libmkl_mc3.so \
#        -l ${MKL_LIB}/libmkl_vml_mc3.so \
#        -l ${MKL_LIB}/libmkl_avx.so \
#        -l ${MKL_LIB}/libmkl_vml_avx.so \
#        -l ${MKL_LIB}/libmkl_avx2.so \
#        -l ${MKL_LIB}/libmkl_vml_avx2.so \
#        -l ${MKL_LIB}/libmkl_avx512_mic.so \
#        -l ${MKL_LIB}/libmkl_vml_avx512_mic.so \
#        -l ${MKL_LIB}/libmkl_avx512.so \
#        -l ${MKL_LIB}/libmkl_vml_avx512.so \
#        -l ${MKL_LIB}/libmkl_core.so && \
#      rm -rf /DLA-Future
#
#FROM ubuntu:18.04
#
## This is the only thing necessary really from nvidia/cuda's ubuntu18.04 runtime image
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
#ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"
#
#COPY --from=builder /root/DLA-Future.bundle /root/DLA-Future.bundle
#
## Make it easy to call our binaries.
#ENV PATH="/root/DLA-Future.bundle/usr/bin:$PATH"
#
#RUN echo "/root/DLA-Future.bundle/usr/lib/" > /etc/ld.so.conf.d/cosma.conf && ldconfig
#
#WORKDIR /root/DLA-Future.bundle/usr/bin
