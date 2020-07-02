# Build environment image
ARG BUILD_ENV

# This is the folder where the project is built
ARG BUILD=/DLA-Future-build

# This is where we copy the sources to
ARG SOURCE=/DLA-Future

# Where "make install" should go / we copy the bare minimum
# of binaries to here
ARG DEPLOY=/root/DLA-Future.bundle

FROM $BUILD_ENV as builder

ARG BUILD
ARG SOURCE
ARG DEPLOY
ARG DEPLOY_IMAGE

# Build DLA-Future
COPY . $SOURCE

SHELL ["/bin/bash", "-c"]

# Build the project with coverage symbols
RUN mkdir ${BUILD} && cd ${BUILD} && \
    CC=/usr/local/mpich/bin/mpicc CXX=/usr/local/mpich/bin/mpicxx cmake ${SOURCE} \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-O0 -Werror -fprofile-arcs -ftest-coverage" \
      -DCMAKE_EXE_LINKER_FLAGS="-fprofile-arcs -ftest-coverage" \
      -DLAPACK_CUSTOM_TYPE=Custom \
      -DLAPACK_CUSTOM_INCLUDE_DIR=/usr/local/include \
      -DLAPACK_CUSTOM_LIBRARY=openblas \
      -DDLAF_WITH_CUDA=OFF \
      -DDLAF_WITH_MKL=OFF \
      -DDLAF_WITH_TEST=ON \
      -DDLAF_BUILD_MINIAPPS=ON \
      -DMPIEXEC_EXECUTABLE=srun \
      -DDLAF_CI_RUNNER_USES_MPIRUN=1 && \
      make -j$(nproc)

# Prune and bundle binaries
RUN mkdir ${BUILD}-tmp && cd ${BUILD} && \
    export TEST_BINARIES=`ctest --show-only=json-v1 | jq '.tests | map(.command[0]) | .[]' | tr -d \"` && \
    libtree -d ${DEPLOY} ${TEST_BINARIES} && \
    rm -rf ${DEPLOY}/usr/bin && \
    libtree -d ${DEPLOY} $(which ctest gcov addr2line) && \
    cp -L ${SOURCE}/ci/{mpi-ctest,upload_codecov} ${DEPLOY}/usr/bin && \
    cp -L $(which lcov) $(which geninfo) ${DEPLOY}/usr/bin && \
    echo "$TEST_BINARIES" | xargs -I{file} find -samefile {file} -exec cp --parents '{}' ${BUILD}-tmp ';' && \
    find '(' -name CTestTestfile.cmake -o -iname "*.gcno" ')' -exec cp --parent '{}' ${BUILD}-tmp ';' && \
    rm -rf ${BUILD} && \
    mv ${BUILD}-tmp ${BUILD}

# Generate the gitlab-ci yml file
RUN cd ${BUILD} && \
    ${SOURCE}/ci/ctest_to_gitlab_codecov.sh "${DEPLOY_IMAGE}" > ${DEPLOY}/pipeline.yml

# Multistage build, this is the final small image
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

ARG BUILD
ARG SOURCE
ARG DEPLOY

# Install perl to make lcov happy
# codecov upload needs curl + ca-certificates
# tzdata is needed to print correct time
# TODO: remove git after https://github.com/codecov/codecov-bash/pull/291
#       or https://github.com/codecov/codecov-bash/pull/265 is merged
RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends \
      perl \
      curl \
      ca-certificates \
      git \
      tzdata && \
    rm -rf /var/lib/apt/lists/*

# Copy the executables and the codecov gcno files
COPY --from=builder ${BUILD} ${BUILD}
COPY --from=builder ${DEPLOY} ${DEPLOY}

# Copy the source files into the image as well.
# This is necessary for code coverage of MPI tests: gcov has to have write temporary
# data into the source folder. In distributed applications we can therefore not mount
# the git repo folder at runtime in the container, because it is shared and would
# cause race conditions in gcov. When PR #291 or #265 (see above) is merged
# we can at the very least remove all remnants of git, in particular the `.git` folder...
COPY --from=builder ${SOURCE} ${SOURCE}

# Make it easy to call our binaries.
ENV PATH="${DEPLOY}/usr/bin:$PATH"

# Used in our ctest wrapper to upload reports
ENV ENABLE_COVERAGE="YES"

# Automatically print stacktraces on segfault
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libSegFault.so

RUN echo "${DEPLOY}/usr/lib/" > /etc/ld.so.conf.d/dlaf.conf && ldconfig

WORKDIR ${BUILD}
