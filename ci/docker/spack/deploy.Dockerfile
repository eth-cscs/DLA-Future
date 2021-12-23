ARG BUILD_IMAGE

# This is the folder where the project is built
ARG BUILD=/DLA-Future-build
# Where a bunch of shared libs live
ARG DEPLOY=/root/DLA-Future.bundle

FROM $BUILD_IMAGE as builder

# This is where we copy the sources to
ARG SOURCE=/DLA-Future

ARG BUILD
ARG DEPLOY

# Build DLA-Future
COPY . ${SOURCE}

SHELL ["/bin/bash", "-c"]

RUN spack repo rm --scope site dlaf && \
    spack repo add ${SOURCE}/spack && \
    spack -e ci develop --no-clone -p ${SOURCE} dla-future@develop && \
    spack -e ci install --keep-stage

# Prune and bundle binaries
RUN export SPACK_BUILD=`spack -e ci location -b dla-future` && \
    mkdir ${BUILD} && cd ${SPACK_BUILD} && \
    export TEST_BINARIES=`ctest --show-only=json-v1 | jq '.tests | map(.command[0]) | .[]' | tr -d \"` && \
    libtree -d ${DEPLOY} ${TEST_BINARIES} && \
    rm -rf ${DEPLOY}/usr/bin && \
    libtree -d ${DEPLOY} $(which ctest addr2line) && \
    cp -L ${SOURCE}/ci/mpi-ctest ${DEPLOY}/usr/bin && \
    echo "$TEST_BINARIES" | xargs -I{file} find -samefile {file} -exec cp --parents '{}' ${BUILD} ';' && \
    find -name CTestTestfile.cmake -exec sed -i "s|${SPACK_BUILD}|${BUILD}|g" '{}' ';' -exec cp --parents '{}' ${BUILD} ';' && \
    rm -rf ${SPACK_BUILD}

# Deploy MKL separately, since it dlopen's some libs
RUN export MKL_LIB=`spack -e ci location -i intel-mkl`/mkl/lib/intel64 && \
    libtree -d ${DEPLOY} \
    ${MKL_LIB}/libmkl_avx.so \
    ${MKL_LIB}/libmkl_avx2.so \
    ${MKL_LIB}/libmkl_core.so \
    ${MKL_LIB}/libmkl_def.so \
    ${MKL_LIB}/libmkl_intel_thread.so \
    ${MKL_LIB}/libmkl_mc.so \
    ${MKL_LIB}/libmkl_mc3.so \
    ${MKL_LIB}/libmkl_sequential.so \
    ${MKL_LIB}/libmkl_tbb_thread.so \
    ${MKL_LIB}/libmkl_vml_avx.so \
    ${MKL_LIB}/libmkl_vml_avx2.so \
    ${MKL_LIB}/libmkl_vml_cmpt.so \
    ${MKL_LIB}/libmkl_vml_def.so \
    ${MKL_LIB}/libmkl_vml_mc.so \
    ${MKL_LIB}/libmkl_vml_mc3.so

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

ARG BUILD
ARG DEPLOY

# tzdata is needed to print correct time
RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends \
      tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder ${BUILD} ${BUILD}
COPY --from=builder ${DEPLOY} ${DEPLOY}

# Make it easy to call our binaries.
ENV PATH="${DEPLOY}/usr/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2"

# Automatically print stacktraces on segfault
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libSegFault.so

RUN echo "${DEPLOY}/usr/lib/" > /etc/ld.so.conf.d/dlaf.conf && ldconfig

WORKDIR ${BUILD}
