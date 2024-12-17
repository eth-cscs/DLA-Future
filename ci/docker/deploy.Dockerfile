ARG DEPS_IMAGE
FROM $DEPS_IMAGE

LABEL com.jfrog.artifactory.retention.maxDays="7"
LABEL com.jfrog.artifactory.retention.maxCount="10"

# Directory where the project is built
ARG BUILD=/DLA-Future-build
# Directory where the miniapps are built as separate project
ARG BUILD_MINIAPP=/DLA-Future-miniapp-build
# Directory where the sources are copied to
ARG SOURCE=/DLA-Future
# Directory for some helper executables
ARG BIN=/DLA-Future-build/bin

# Build DLA-Future
COPY . ${SOURCE}

SHELL ["/bin/bash", "-c"]

ARG NUM_PROCS
# Note: we force spack to build in ${BUILD} creating a link to it
RUN spack repo rm --scope site dlaf && \
    spack repo add ${SOURCE}/spack && \
    spack -e ci develop --no-clone --path ${SOURCE} --build-directory ${BUILD} dla-future@master && \
    spack -e ci concretize -f && \
    spack -e ci --config "config:flags:keep_werror:all" install --jobs ${NUM_PROCS} --keep-stage --verbose && \
    find ${BUILD} -name CMakeFiles -exec rm -rf {} +

# Test deployment with miniapps as independent project
RUN mkdir ${BUILD_MINIAPP} && cd ${BUILD_MINIAPP} && \
    spack -e ci build-env dla-future@master -- \
    bash -c "cmake -DCMAKE_PREFIX_PATH=`spack -e ci location -i dla-future` ${SOURCE}/miniapp && make -j ${NUM_PROCS}"

RUN mkdir -p ${BIN} && cp -L ${SOURCE}/ci/{mpi-ctest,check-threads} ${BIN}

# Make it easy to call our binaries.
ENV PATH="${BIN}:$PATH"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2"

# Automatically print stacktraces on segfault
ARG DLAF_LD_PRELOAD
ENV LD_PRELOAD=${DLAF_LD_PRELOAD}

WORKDIR ${BUILD}
