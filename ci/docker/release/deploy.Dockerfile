ARG BUILD_IMAGE

# This is the folder where the project is built
ARG BUILD=/DLA-Future-build

# This is where we copy the sources to
ARG SOURCE=/DLA-Future

# Where a bunch of shared libs live
ARG DEPLOY=/root/DLA-Future.bundle

FROM $BUILD_IMAGE as builder

ARG BUILD
ARG SOURCE
ARG DEPLOY

# With or without CUDA
ARG DLAF_WITH_CUDA=OFF

# Build DLA-Future
COPY . ${SOURCE}

SHELL ["/bin/bash", "-c"]

RUN mkdir ${BUILD} && cd ${BUILD} && \
    source /opt/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64 && \
    CC=/usr/local/mpich/bin/mpicc CXX=/usr/local/mpich/bin/mpicxx cmake ${SOURCE} \
      -DMKL_ROOT=/opt/intel/compilers_and_libraries/linux/mkl \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-Werror" \
      -DDLAF_WITH_CUDA=${DLAF_WITH_CUDA} \
      -DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES=/usr/local/cuda/targets/x86_64-linux/include \
      -DCMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES="/usr/local/cuda/targets/x86_64-linux/lib/stubs;/usr/local/cuda/targets/x86_64-linux/lib;/usr/lib/gcc/x86_64-linux-gnu/7;/usr/lib/x86_64-linux-gnu;/usr/lib;/lib/x86_64-linux-gnu;/lib;/usr/local/cuda/lib64/stubs" \
      -DDLAF_WITH_MKL=ON \
      -DDLAF_BUILD_TESTING=ON \
      -DDLAF_BUILD_MINIAPPS=ON \
      -DMPIEXEC_EXECUTABLE=srun \
      -DDLAF_CI_RUNNER_USES_MPIRUN=1 && \
      make -j$(nproc)

# Prune and bundle binaries
RUN mkdir ${BUILD}-tmp && cd ${BUILD} && \
    source /opt/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64 && \
    export TEST_BINARIES=`ctest --show-only=json-v1 | jq '.tests | map(.command[0]) | .[]' | tr -d \"` && \
    libtree -d ${DEPLOY} ${TEST_BINARIES} && \
    rm -rf ${DEPLOY}/usr/bin && \
    libtree -d ${DEPLOY} $(which ctest addr2line) && \
    cp -L ${SOURCE}/ci/mpi-ctest ${DEPLOY}/usr/bin && \
    echo "$TEST_BINARIES" | xargs -I{file} find -samefile {file} -exec cp --parents '{}' ${BUILD}-tmp ';' && \
    find -name CTestTestfile.cmake -exec cp --parent '{}' ${BUILD}-tmp ';' && \
    rm -rf ${BUILD} && \
    mv ${BUILD}-tmp ${BUILD}

# Deploy MKL separately, since it dlopen's some libs
RUN source /opt/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64 && \
    export MKL_LIB=$MKLROOT/lib/intel64 && \
    libtree -d ${DEPLOY} \
    --chrpath \
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
