ARG BUILD_IMAGE
ARG DEPLOY_BASE_IMAGE

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

# Build DLA-Future
COPY . ${SOURCE}

SHELL ["/bin/bash", "-c"]

# Disable error on warning (HIPBLAS deprecation warning)
RUN sed -i '/Werror/d' ${SOURCE}/spack/packages/dla-future/package.py

# Note: we force spack to build in ${BUILD} creating a link to it
RUN spack repo rm --scope site dlaf && \
    spack repo add ${SOURCE}/spack && \
    spack -e ci develop --no-clone -p ${SOURCE} dla-future@develop && \
    spack -e ci concretize -f && \
    mkdir ${BUILD} && \
    ln -s ${BUILD} `spack -e ci location -b dla-future` && \
    spack -e ci install --keep-stage

WORKDIR ${BUILD}

# Prune and bundle binaries
RUN mkdir ${BUILD}-tmp && cd ${BUILD} && \
    export TEST_BINARIES=`ctest --show-only=json-v1 | jq '.tests | map(.command[0]) | .[]' | tr -d \"` && \
    libtree -d ${DEPLOY} ${TEST_BINARIES} && \
    rm -rf ${DEPLOY}/usr/bin && \
    libtree -d ${DEPLOY} $(which ctest addr2line) && \
    cp -L ${SOURCE}/ci/mpi-ctest ${DEPLOY}/usr/bin && \
    echo "$TEST_BINARIES" | xargs -I{file} find -samefile {file} -exec cp --parents '{}' ${BUILD}-tmp ';' && \
    find -name CTestTestfile.cmake -exec cp --parents '{}' ${BUILD}-tmp ';' && \
    rm -rf ${BUILD} && \
    mv ${BUILD}-tmp ${BUILD}

# Deploy MKL separately, since it dlopen's some libs
ARG USE_MKL=ON
RUN if [ "$USE_MKL" = "ON" ]; then \
      export MKL_LIB=`spack -e ci location -i intel-mkl`/mkl/lib/intel64 && \
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
      ${MKL_LIB}/libmkl_vml_mc3.so ; \
    fi

# Deploy Extra RocBlas files separately.
ARG USE_ROCBLAS=OFF
RUN if [ "$USE_ROCBLAS" = "ON" ]; then \
      cp -r `spack -e ci location -i rocblas`/lib/library ${DEPLOY}/usr/lib ; \
    fi

# Multistage build, this is the final small image
FROM $DEPLOY_BASE_IMAGE

# set jfrog autoclean policy
LABEL com.jfrog.artifactory.retention.maxDays="7"
LABEL com.jfrog.artifactory.retention.maxCount="10"

ENV DEBIAN_FRONTEND noninteractive

ARG BUILD
ARG DEPLOY

ARG EXTRA_APTGET_DEPLOY
# tzdata is needed to print correct time
RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends \
      ${EXTRA_APTGET_DEPLOY} \
      tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder ${BUILD} ${BUILD}
COPY --from=builder ${DEPLOY} ${DEPLOY}

# Make it easy to call our binaries.
ENV PATH="${DEPLOY}/usr/bin:$PATH"

# Automatically print stacktraces on segfault
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libSegFault.so

RUN echo "${DEPLOY}/usr/lib/" > /etc/ld.so.conf.d/dlaf.conf && ldconfig

WORKDIR ${BUILD}
